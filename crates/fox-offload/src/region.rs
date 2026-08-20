//! Safe shared-memory access from a kernel: each thread gets a disjoint slice.
//!
//! The paper's `Region`/`PartitioningStrategy` idea, with one thing added.
//!
//! Upstream has a version of the trait in review (rust-lang/rust#158076): an
//! `unsafe trait` with `get`/`get_mut` over raw pointers, no concrete strategies
//! beyond a `Dummy` test impl, and disjointness written down as a safety obligation
//! on whoever implements it. This module keeps the same idea in safe code, ships
//! strategies that are actually usable, and — the part neither the paper nor the PR
//! has — makes the obligation checkable.
//!
//! Both the paper and the PR say each strategy *guarantees* disjoint access between
//! threads, and that guarantee is the whole reason a kernel body needs no `unsafe`.
//! It is stated in prose. A strategy with an off-by-one produces two threads
//! writing the same cell — silent corruption on a GPU, the worst class of bug
//! there is, and nothing in the type system catches it.
//!
//! So here a strategy is a **proof obligation that gets discharged by execution**:
//! [`verify_disjoint`] enumerates an entire launch grid and fails if any element is
//! claimed twice or left unclaimed. It runs in `cargo test` with no GPU, which is
//! the point — the property is about the strategy, not about the hardware.
//!
//! Note this module contains no `unsafe` at all. [`Region::lane`] borrows `&mut
//! self`, so at most one lane is live at a time and the compiler enforces
//! exclusivity directly; strided lanes index through the parent slice rather than
//! aliasing it. On the device each thread holds its own `Region` over the same
//! memory, and *that* is where the strategy's disjointness is what keeps the whole
//! thing sound — hence the verifier.

use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use crate::thread::{GridDim, ThreadId};

/// The set of element indices one thread owns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    /// This thread owns nothing (grid larger than the data).
    Empty,
    /// `start .. start + len`.
    Contiguous { start: usize, len: usize },
    /// `start, start + stride, start + 2*stride, ...`, `count` of them.
    Strided {
        start: usize,
        stride: usize,
        count: usize,
    },
}

impl Lane {
    /// How many elements this lane covers.
    pub const fn len(&self) -> usize {
        match *self {
            Lane::Empty => 0,
            Lane::Contiguous { len, .. } => len,
            Lane::Strided { count, .. } => count,
        }
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `i`-th element index in this lane, or `None` past the end.
    pub const fn nth(&self, i: usize) -> Option<usize> {
        match *self {
            Lane::Empty => None,
            Lane::Contiguous { start, len } => {
                if i < len {
                    Some(start + i)
                } else {
                    None
                }
            }
            Lane::Strided {
                start,
                stride,
                count,
            } => {
                if i < count {
                    Some(start + i * stride)
                } else {
                    None
                }
            }
        }
    }
}

/// How the elements of a buffer are split between the threads of a launch.
///
/// # Contract
///
/// For a given `len` and `grid`, the lanes of any two distinct threads must be
/// disjoint, and their union must be exactly `0..len`. Implementors are expected to
/// prove this with [`verify_disjoint`] in their own tests — the trait cannot
/// enforce it, which is precisely why it is checked rather than trusted.
pub trait PartitioningStrategy {
    /// Which elements `tid` owns, out of `len`, under `grid`.
    fn lane(len: usize, tid: ThreadId, grid: &GridDim) -> Lane;
}

/// Contiguous blocks: thread `k` of `n` owns `[k*chunk, (k+1)*chunk)`.
///
/// Good locality per thread, poor coalescing across threads — the right choice when
/// a thread reads its whole range (a per-thread reduction), the wrong one for a
/// streaming elementwise pass.
#[derive(Clone, Copy, Debug)]
pub struct Linear1D;

impl PartitioningStrategy for Linear1D {
    fn lane(len: usize, tid: ThreadId, grid: &GridDim) -> Lane {
        let n = grid.total_threads();
        if n == 0 {
            return Lane::Empty;
        }
        let k = tid.linear(grid);
        if k >= n {
            return Lane::Empty;
        }
        // Spread the remainder over the first `len % n` threads so the union is
        // exactly `0..len` with no thread more than one element heavier.
        let base = (len as u64) / n;
        let extra = (len as u64) % n;
        let start = k * base + core::cmp::min(k, extra);
        let this = base + if k < extra { 1 } else { 0 };
        if this == 0 {
            Lane::Empty
        } else {
            Lane::Contiguous {
                start: start as usize,
                len: this as usize,
            }
        }
    }
}

/// Grid-stride: thread `k` of `n` owns `k, k+n, k+2n, ...`.
///
/// Adjacent threads touch adjacent addresses, which is what a GPU memory
/// controller wants. The standard shape for elementwise work.
#[derive(Clone, Copy, Debug)]
pub struct Strided1D;

impl PartitioningStrategy for Strided1D {
    fn lane(len: usize, tid: ThreadId, grid: &GridDim) -> Lane {
        let n = grid.total_threads();
        if n == 0 {
            return Lane::Empty;
        }
        let k = tid.linear(grid);
        if k >= n || k >= len as u64 {
            return Lane::Empty;
        }
        // ceil((len - k) / n)
        let count = ((len as u64 - k) + n - 1) / n;
        Lane::Strided {
            start: k as usize,
            stride: n as usize,
            count: count as usize,
        }
    }
}

/// A buffer partitioned between the threads of a launch.
///
/// `lane` takes `&mut self`, so only one lane can be live at a time and the borrow
/// checker guarantees exclusivity *within* one thread with no `unsafe`. Across
/// threads, exclusivity rests on `S` — see [`verify_disjoint`].
pub struct Region<'a, T, S: PartitioningStrategy> {
    data: &'a mut [T],
    grid: GridDim,
    _strategy: PhantomData<S>,
}

impl<'a, T, S: PartitioningStrategy> Region<'a, T, S> {
    pub fn new(data: &'a mut [T], grid: GridDim) -> Self {
        Self {
            data,
            grid,
            _strategy: PhantomData,
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn grid(&self) -> &GridDim {
        &self.grid
    }

    /// The elements `tid` owns, as a writable view.
    pub fn lane(&mut self, tid: ThreadId) -> LaneMut<'_, T> {
        let lane = S::lane(self.data.len(), tid, &self.grid);
        LaneMut {
            data: self.data,
            lane,
        }
    }

    /// Which elements `tid` owns, without borrowing them.
    pub fn lane_of(&self, tid: ThreadId) -> Lane {
        S::lane(self.data.len(), tid, &self.grid)
    }
}

/// One thread's writable view of its own elements.
pub struct LaneMut<'a, T> {
    data: &'a mut [T],
    lane: Lane,
}

impl<T> LaneMut<'_, T> {
    pub fn len(&self) -> usize {
        self.lane.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lane.is_empty()
    }

    /// The global index of this lane's `i`-th element.
    pub fn index_of(&self, i: usize) -> Option<usize> {
        self.lane.nth(i)
    }

    pub fn get(&self, i: usize) -> Option<&T> {
        self.lane.nth(i).and_then(|j| self.data.get(j))
    }

    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        match self.lane.nth(i) {
            Some(j) => self.data.get_mut(j),
            None => None,
        }
    }

    /// Iterate `(global index, value)` over this lane.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &T)> + '_ {
        (0..self.len()).filter_map(move |i| {
            let j = self.lane.nth(i)?;
            Some((j, self.data.get(j)?))
        })
    }
}

/// What [`verify_disjoint`] found wrong.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DisjointError {
    /// Two threads claim the same element. Silent corruption on a device.
    Overlap {
        index: usize,
        first: u64,
        second: u64,
    },
    /// No thread claims this element. Not unsound, but the kernel skips data.
    Unclaimed { index: usize },
    /// A lane names an element past the end of the buffer.
    OutOfBounds {
        index: usize,
        len: usize,
        thread: u64,
    },
}

/// Discharge a strategy's disjointness obligation for one `(len, grid)` by
/// enumerating the whole grid.
///
/// This is the check the paper leaves as prose. It is exhaustive rather than
/// sampled — for the sizes a test uses, a full sweep is cheap and a property test
/// that misses the one colliding index is worthless.
///
/// Returns the number of threads examined.
pub fn verify_disjoint<S: PartitioningStrategy>(
    len: usize,
    grid: &GridDim,
) -> Result<u64, DisjointError> {
    let mut owner: Vec<Option<u64>> = vec![None; len];
    let total = grid.total_threads();

    for wz in 0..grid.workgroups[2] {
        for wy in 0..grid.workgroups[1] {
            for wx in 0..grid.workgroups[0] {
                for tz in 0..grid.threads[2] {
                    for ty in 0..grid.threads[1] {
                        for tx in 0..grid.threads[0] {
                            let tid = ThreadId::new([tx, ty, tz], [wx, wy, wz]);
                            let k = tid.linear(grid);
                            let lane = S::lane(len, tid, grid);
                            for i in 0..lane.len() {
                                let Some(idx) = lane.nth(i) else { continue };
                                if idx >= len {
                                    return Err(DisjointError::OutOfBounds {
                                        index: idx,
                                        len,
                                        thread: k,
                                    });
                                }
                                if let Some(first) = owner[idx] {
                                    return Err(DisjointError::Overlap {
                                        index: idx,
                                        first,
                                        second: k,
                                    });
                                }
                                owner[idx] = Some(k);
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(index) = owner.iter().position(|o| o.is_none()) {
        return Err(DisjointError::Unclaimed { index });
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grids() -> alloc::vec::Vec<GridDim> {
        alloc::vec![
            GridDim::linear(1, 1),
            GridDim::linear(1, 32),
            GridDim::linear(4, 64),
            GridDim::linear(7, 5),
            GridDim {
                workgroups: [3, 2, 1],
                threads: [4, 2, 1]
            },
        ]
    }

    #[test]
    fn linear1d_partitions_exactly() {
        for grid in grids() {
            for len in [0usize, 1, 5, 63, 64, 65, 1000] {
                assert_eq!(
                    verify_disjoint::<Linear1D>(len, &grid),
                    Ok(grid.total_threads()),
                    "Linear1D len={len} grid={grid:?}"
                );
            }
        }
    }

    #[test]
    fn strided1d_partitions_exactly() {
        for grid in grids() {
            for len in [0usize, 1, 5, 63, 64, 65, 1000] {
                assert_eq!(
                    verify_disjoint::<Strided1D>(len, &grid),
                    Ok(grid.total_threads()),
                    "Strided1D len={len} grid={grid:?}"
                );
            }
        }
    }

    /// The verifier has to actually catch a bad strategy, or it proves nothing.
    #[test]
    fn a_strategy_that_overlaps_is_caught() {
        struct EveryoneOwnsEverything;
        impl PartitioningStrategy for EveryoneOwnsEverything {
            fn lane(len: usize, _tid: ThreadId, _grid: &GridDim) -> Lane {
                Lane::Contiguous { start: 0, len }
            }
        }
        let err = verify_disjoint::<EveryoneOwnsEverything>(8, &GridDim::linear(1, 4));
        assert!(
            matches!(err, Err(DisjointError::Overlap { .. })),
            "got {err:?}"
        );
    }

    /// The classic off-by-one: `len / n` with the remainder dropped.
    #[test]
    fn a_strategy_that_drops_the_remainder_is_caught() {
        struct TruncatingChunks;
        impl PartitioningStrategy for TruncatingChunks {
            fn lane(len: usize, tid: ThreadId, grid: &GridDim) -> Lane {
                let n = grid.total_threads().max(1);
                let chunk = (len as u64) / n; // remainder silently lost
                let k = tid.linear(grid);
                Lane::Contiguous {
                    start: (k * chunk) as usize,
                    len: chunk as usize,
                }
            }
        }
        // 10 elements over 4 threads: chunk = 2, so 8..10 is claimed by nobody.
        let err = verify_disjoint::<TruncatingChunks>(10, &GridDim::linear(1, 4));
        assert_eq!(err, Err(DisjointError::Unclaimed { index: 8 }));
    }

    #[test]
    fn a_strategy_that_runs_off_the_end_is_caught() {
        struct OneTooFar;
        impl PartitioningStrategy for OneTooFar {
            fn lane(len: usize, _tid: ThreadId, _grid: &GridDim) -> Lane {
                Lane::Contiguous {
                    start: 0,
                    len: len + 1,
                }
            }
        }
        let err = verify_disjoint::<OneTooFar>(4, &GridDim::linear(1, 1));
        assert!(
            matches!(err, Err(DisjointError::OutOfBounds { .. })),
            "got {err:?}"
        );
    }

    #[test]
    fn lanes_write_through_to_the_buffer() {
        let grid = GridDim::linear(1, 4);
        let mut data = [0i32; 10];
        let mut region: Region<'_, i32, Linear1D> = Region::new(&mut data, grid);
        for k in 0..4u32 {
            let tid = ThreadId::new([k, 0, 0], [0, 0, 0]);
            let mut lane = region.lane(tid);
            for i in 0..lane.len() {
                let global = lane.index_of(i).unwrap();
                *lane.get_mut(i).unwrap() = global as i32;
            }
        }
        assert_eq!(data, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn strided_lanes_interleave() {
        let grid = GridDim::linear(1, 3);
        let mut data = [0i32; 7];
        let mut region: Region<'_, i32, Strided1D> = Region::new(&mut data, grid);
        let tid = ThreadId::new([1, 0, 0], [0, 0, 0]);
        assert_eq!(
            region.lane_of(tid),
            Lane::Strided {
                start: 1,
                stride: 3,
                count: 2
            }
        );
        let mut lane = region.lane(tid);
        assert_eq!(lane.len(), 2);
        *lane.get_mut(0).unwrap() = 11;
        *lane.get_mut(1).unwrap() = 44;
        assert_eq!(data, [0, 11, 0, 0, 44, 0, 0]);
    }
}
