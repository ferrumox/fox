//! Portable GPU kernels for fox, built on rustc's LLVM Offload backend.
//!
//! # Why this exists
//!
//! rustc grew a GPU offload backend (`-Zoffload`, `#[offload_kernel]`,
//! `core::offload::offload!`) from the work in *GPU Offload in Rust: Portable,
//! Safe, and Fast* (arXiv:2608.13759), upstreamed by its own authors over ~22
//! months. What shipped is the **launch mechanism**. Two things the paper
//! describes did not ship, and the paper says both are ordinary Rust needing no
//! compiler support:
//!
//! * `Region` + `PartitioningStrategy` — hands each thread a disjoint slice, so a
//!   kernel body needs no `unsafe`. This is the entire safety claim of the work.
//! * `Preload`/`PreloadMut` — device residency as a type, so a chain of kernels
//!   does not round-trip through the host between launches.
//!
//! A third gap turned up by compiling rather than reading: the thread-index API is
//! **not portable**. AMD spells it `workitem_id_x()` (safe), NVIDIA
//! `_thread_idx_x()` (`unsafe`), AMD has no workgroup-size query at all, and the
//! names in `core/src/offload.md` exist on neither target. See [`thread`].
//!
//! # What this crate adds on top of the paper
//!
//! 1. **Disjointness is checked, not asserted.** Both the paper and #158076 state
//!    that a strategy guarantees disjoint access, and that claim is the only thing
//!    making the kernel body safe.
//!    [`region::verify_disjoint`] enumerates a whole launch grid and fails if any
//!    element is claimed twice or left unclaimed. Runs in `cargo test`, no GPU.
//!
//! 2. **Residency is paged and ref-counted, not two-state.** `PreloadMut` models
//!    "resident until dropped". A server needs "resident across millions of
//!    launches, in fixed-size blocks, shared between sequences, copied on write" —
//!    see [`resident`], which mirrors fox's own `src/kv_cache/`.
//!
//! 3. **A CPU backend that runs the same kernel source.** The paper measures its
//!    kernels against RAJA/CUDA baselines, but has no oracle: nothing establishes
//!    that a Rust kernel computes what the code it replaced computed.
//!    [`launch::launch_cpu`] executes a [`Kernel`] over an emulated grid,
//!    deterministically, so a kernel is differentially testable against fox's
//!    existing CPU implementation — and so this crate is developed and tested on
//!    stable rustc with no device at all.
//!
//! # Status
//!
//! Nothing here is wired into fox's engine, deliberately. End-to-end offload does
//! not execute on any machine we have (`scripts/probe_rust_offload.sh` says where
//! it stops), and decode on the target hardware is bandwidth-bound anyway. This is
//! the half that can be built and tested *now*, so it is ready when the toolchain
//! catches up.

#![cfg_attr(not(test), no_std)]
// Los intrínsecos de índice de hilo de `thread` viven tras estas puertas. Sólo se
// activan al compilar PARA un dispositivo, así que el crate sigue construyéndose en
// estable para el host.
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]
#![deny(unsafe_code)]

// `alloc` no existe en un target de GPU. Todo lo que lo usa —el pool de residencia,
// el verificador de disjunción, el atajo de argmax en host— es de host por naturaleza.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
extern crate alloc;

pub mod kernels;
pub mod launch;
pub mod region;
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub mod resident;
pub mod thread;

pub use launch::{launch_cpu, Kernel};
pub use region::{Lane, PartitioningStrategy, Region};
pub use thread::{GridDim, ThreadId};
