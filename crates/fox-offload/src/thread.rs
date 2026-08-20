//! Where a thread is in the launch grid — portably.
//!
//! This module exists because the first line of every GPU kernel ever written is
//! not portable. Verified by compiling, 2026-08-20:
//!
//! | | AMD (`core::arch::amdgpu`) | NVIDIA (`core::arch::nvptx`) |
//! |---|---|---|
//! | thread index | `workitem_id_x()` — safe | `_thread_idx_x()` — `unsafe` |
//! | block index | `workgroup_id_x()` — safe | `_block_idx_x()` — `unsafe` |
//! | block size | **does not exist** | `_block_dim_x()` — `unsafe` |
//!
//! Different names, different safety, and AMD cannot ask how large its own
//! workgroup is. (The names used by the example in `core/src/offload.md` —
//! `thread_idx_x`, `block_idx_x`, `block_dim_x` — exist on *neither* target, so
//! that example does not compile.)
//!
//! Because AMD has no workgroup-size query, [`GridDim`] carries the dimensions
//! rather than reading them back from hardware. That is not a workaround: the host
//! chose those dimensions when it launched, so passing them is strictly more
//! reliable than asking the device to recall them.

/// The shape of a launch: how many workgroups, and how many threads in each.
///
/// Mirrors the `workgroup_dim` / `thread_dim` arguments of `core::offload::offload!`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridDim {
    pub workgroups: [u32; 3],
    pub threads: [u32; 3],
}

impl GridDim {
    /// A 1-D launch of `workgroups` groups of `threads` threads.
    pub const fn linear(workgroups: u32, threads: u32) -> Self {
        Self {
            workgroups: [workgroups, 1, 1],
            threads: [threads, 1, 1],
        }
    }

    /// Total threads in the launch. `u64` because a large 3-D grid overflows `u32`.
    pub const fn total_threads(&self) -> u64 {
        (self.workgroups[0] as u64)
            * (self.workgroups[1] as u64)
            * (self.workgroups[2] as u64)
            * (self.threads[0] as u64)
            * (self.threads[1] as u64)
            * (self.threads[2] as u64)
    }

    /// Threads per workgroup — the number AMD cannot ask the hardware for.
    pub const fn threads_per_group(&self) -> u32 {
        self.threads[0] * self.threads[1] * self.threads[2]
    }

    /// Number of workgroups.
    pub const fn group_count(&self) -> u32 {
        self.workgroups[0] * self.workgroups[1] * self.workgroups[2]
    }
}

/// One thread's position in the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThreadId {
    /// Position within the workgroup.
    pub thread: [u32; 3],
    /// Which workgroup.
    pub workgroup: [u32; 3],
}

impl ThreadId {
    pub const fn new(thread: [u32; 3], workgroup: [u32; 3]) -> Self {
        Self { thread, workgroup }
    }

    /// Global index along X. The `i` in `if i < n { out[i] = ... }`.
    pub const fn global_x(&self, grid: &GridDim) -> u32 {
        self.thread[0] + self.workgroup[0] * grid.threads[0]
    }

    /// Flat index over the whole grid, workgroup-major then thread-major.
    ///
    /// This is the ordering a partitioning strategy is indexed by, and it is also
    /// the order [`crate::launch::launch_cpu`] visits threads in — which is what
    /// makes CPU execution a deterministic oracle.
    pub const fn linear(&self, grid: &GridDim) -> u64 {
        let g = (self.workgroup[2] as u64 * grid.workgroups[1] as u64 + self.workgroup[1] as u64)
            * grid.workgroups[0] as u64
            + self.workgroup[0] as u64;
        let t = (self.thread[2] as u64 * grid.threads[1] as u64 + self.thread[1] as u64)
            * grid.threads[0] as u64
            + self.thread[0] as u64;
        g * grid.threads_per_group() as u64 + t
    }
}

/// Read this thread's position from the hardware.
///
/// Only meaningful inside a kernel. `grid` supplies the workgroup size, which AMD
/// has no intrinsic for. On the host this returns thread 0 of workgroup 0, so a
/// kernel body compiled for the host is a single-threaded execution of itself
/// rather than a compile error.
#[cfg(target_arch = "amdgpu")]
pub fn current(_grid: &GridDim) -> ThreadId {
    use core::arch::amdgpu::{
        workgroup_id_x, workgroup_id_y, workgroup_id_z, workitem_id_x, workitem_id_y, workitem_id_z,
    };
    ThreadId {
        thread: [workitem_id_x(), workitem_id_y(), workitem_id_z()],
        workgroup: [workgroup_id_x(), workgroup_id_y(), workgroup_id_z()],
    }
}

#[cfg(target_arch = "nvptx64")]
#[allow(unsafe_code)] // NVIDIA's intrinsics are `unsafe fn`; AMD's are not.
pub fn current(_grid: &GridDim) -> ThreadId {
    use core::arch::nvptx::{
        _block_idx_x, _block_idx_y, _block_idx_z, _thread_idx_x, _thread_idx_y, _thread_idx_z,
    };
    // SAFETY: these read special registers and are only unsafe because the nvptx
    // module predates `safe fn` in extern blocks. They have no preconditions.
    unsafe {
        ThreadId {
            thread: [_thread_idx_x(), _thread_idx_y(), _thread_idx_z()],
            workgroup: [_block_idx_x(), _block_idx_y(), _block_idx_z()],
        }
    }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub fn current(_grid: &GridDim) -> ThreadId {
    ThreadId {
        thread: [0, 0, 0],
        workgroup: [0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_index_covers_the_grid_exactly_once() {
        let grid = GridDim {
            workgroups: [3, 2, 1],
            threads: [4, 2, 1],
        };
        assert_eq!(grid.total_threads(), 48);
        let mut seen = alloc::vec![false; 48];
        for wz in 0..1 {
            for wy in 0..2 {
                for wx in 0..3 {
                    for tz in 0..1 {
                        for ty in 0..2 {
                            for tx in 0..4 {
                                let id = ThreadId::new([tx, ty, tz], [wx, wy, wz]);
                                let i = id.linear(&grid) as usize;
                                assert!(!seen[i], "thread {id:?} collided at {i}");
                                seen[i] = true;
                            }
                        }
                    }
                }
            }
        }
        assert!(seen.iter().all(|&s| s), "grid not fully covered");
    }

    #[test]
    fn global_x_matches_the_usual_kernel_prologue() {
        let grid = GridDim::linear(4, 256);
        let id = ThreadId::new([7, 0, 0], [2, 0, 0]);
        assert_eq!(id.global_x(&grid), 7 + 2 * 256);
    }

    #[test]
    fn threads_per_group_is_what_amd_cannot_ask_for() {
        assert_eq!(
            GridDim {
                workgroups: [1, 1, 1],
                threads: [16, 4, 2]
            }
            .threads_per_group(),
            128
        );
    }
}
