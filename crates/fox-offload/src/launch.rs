//! Running a kernel — on a device, or on the CPU as a reference oracle.
//!
//! The paper measures its kernels against RAJA and CUDA baselines: it answers *is
//! the Rust version fast?* It does not answer *does the Rust version compute what
//! the code it replaced computed?* — nothing in the setup can, because the two
//! implementations only ever run on different machines.
//!
//! A [`Kernel`] here is written once against [`ThreadId`]/[`GridDim`] and can be
//! executed two ways: by `core::offload::offload!` on a device, or by
//! [`launch_cpu`] over an emulated grid. The CPU path visits threads in
//! [`ThreadId::linear`] order, so it is deterministic and can be diffed against the
//! CPU implementation a kernel is meant to replace.
//!
//! It also means this crate is developed and tested on stable rustc with no GPU —
//! which is not a consolation prize while `clang-linker-wrapper` is missing
//! upstream (see `docs/design/rust-gpu-offload.md`), it is how the work gets done
//! at all.

use crate::thread::{GridDim, ThreadId};

/// A kernel: what one thread does.
///
/// The body must be written as if every thread runs concurrently — [`launch_cpu`]
/// running them in sequence is an execution *schedule*, not a licence to depend on
/// one. Use a [`crate::region::Region`] for anything shared and mutable.
pub trait Kernel {
    /// What the kernel operates on. A GAT so args can borrow.
    type Args<'a>;

    /// One thread's work.
    fn thread(args: &mut Self::Args<'_>, tid: ThreadId, grid: &GridDim);
}

/// Execute `K` over `grid` on the CPU, one thread at a time, in linear order.
///
/// Returns the number of threads run.
pub fn launch_cpu<K: Kernel>(args: &mut K::Args<'_>, grid: &GridDim) -> u64 {
    let mut n = 0u64;
    for wz in 0..grid.workgroups[2] {
        for wy in 0..grid.workgroups[1] {
            for wx in 0..grid.workgroups[0] {
                for tz in 0..grid.threads[2] {
                    for ty in 0..grid.threads[1] {
                        for tx in 0..grid.threads[0] {
                            K::thread(args, ThreadId::new([tx, ty, tz], [wx, wy, wz]), grid);
                            n += 1;
                        }
                    }
                }
            }
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::{Linear1D, Region};

    struct Doubler;
    impl Kernel for Doubler {
        type Args<'a> = &'a mut [i32];
        fn thread(args: &mut Self::Args<'_>, tid: ThreadId, grid: &GridDim) {
            let mut region: Region<'_, i32, Linear1D> = Region::new(args, *grid);
            let mut lane = region.lane(tid);
            for i in 0..lane.len() {
                if let Some(v) = lane.get_mut(i) {
                    *v *= 2;
                }
            }
        }
    }

    #[test]
    fn every_element_is_touched_exactly_once() {
        let grid = GridDim::linear(2, 8);
        let mut data: alloc::vec::Vec<i32> = (0..37).collect();
        let mut args: &mut [i32] = &mut data;
        assert_eq!(launch_cpu::<Doubler>(&mut args, &grid), 16);
        assert!(data.iter().enumerate().all(|(i, &v)| v == 2 * i as i32));
    }

    #[test]
    fn grid_shape_does_not_change_the_result() {
        let shapes = [
            GridDim::linear(1, 1),
            GridDim::linear(1, 64),
            GridDim::linear(8, 8),
            GridDim::linear(5, 3),
        ];
        let expected: alloc::vec::Vec<i32> = (0..37).map(|i| 2 * i).collect();
        for grid in shapes {
            let mut data: alloc::vec::Vec<i32> = (0..37).collect();
            let mut args: &mut [i32] = &mut data;
            launch_cpu::<Doubler>(&mut args, &grid);
            assert_eq!(data, expected, "grid {grid:?}");
        }
    }
}
