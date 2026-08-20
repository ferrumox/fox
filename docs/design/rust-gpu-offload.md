# Rust GPU offload — feasibility for fox

**Status:** investigation, no code in fox. **Measured:** 2026-08-20 on the target
machine (Ryzen + Radeon 890M / gfx1150, no discrete GPU attached).
**Re-run with:** `scripts/probe_rust_offload.sh [gfx1100|sm_120]`.

Every claim below was produced by running the toolchain on this host. Where
something is inferred rather than observed, it says so.

## What this is about

*GPU Offload in Rust: Portable, Safe, and Fast* (arXiv:2608.13759 — Drehwald,
Domínguez, Sala, Aspuru-Guzik, Doerfert) describes a GPU compilation path built
into rustc on top of LLVM's Offload infrastructure. The interesting part is not the
paper: **the work is upstream in rustc**, tracking issue
[rust-lang/rust#131513](https://github.com/rust-lang/rust/issues/131513), opened
2024-10-10 by `@ZuseZ4` — Drehwald himself. The paper documents a compiler feature
that has been landing in pieces for ~22 months, not a separate prototype. That is
the reason to care: this is on a path to stabilisation, not a research artifact.

fox has no stake in it *yet*. fox does not write kernels — it wraps llama.cpp,
whose kernels are hand-written CUDA/HIP/Vulkan/Metal. This work lives one layer
below where fox lives. It matters to fox only if it eventually becomes a way to
own that layer.

## What works on this machine today

Installing the toolchain piece is one command; the earlier conclusion that the
device pass was blocked was wrong, and the cause was simply not having run it:

```
rustup component add offload --toolchain nightly
```

That ships `libRustOffload-23.so`, `libLLVMOffload.so`, `libomptarget.so`, and the
device-side runtime bitcode for both vendors (`libomptarget-amdgpu.bc`,
`libomptarget-nvptx.bc`, `libompdevice.a`). No ROCm and no CUDA toolkit are needed
to *compile*.

| Stage | Result |
|---|---|
| Host metadata pass (`-Zoffload=HostMetadata=`) | works — writes a manifest naming each kernel |
| Device codegen → `amdgcn-amd-amdhsa`, `gfx1100` | works — 3.4 KB offload image |
| Device codegen → `nvptx64-nvidia-cuda`, `sm_120` | works — from the *same source* |
| Host link (`-Zoffload=Host=`) | links, emits the `__tgt_*` runtime calls |
| Device image embedded in the binary | **no** — see below |
| End-to-end execution | **no** — fails at kernel launch |

Both device passes need `-Zbuild-std=core` (the GPU targets have no prebuilt std),
`-Zunstable-options`, and `lto = "fat"` on the host pass.

### The generated code is good

Release build, AMD, from safe Rust with an `if i < x.len()` guard:

```llvm
define amdgpu_kernel void @_RNvC6probe34fill(ptr ... %0, ptr ... writeonly %1) {
  %3 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %4 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %5 = shl i32 %4, 8
  %6 = add i32 %5, %3
  %7 = icmp ult i32 %6, 256
  ...
  store float %12, ptr addrspace(1) %14, align 4
```

Correct calling convention, intrinsics inlined, the `* 256` strength-reduced to a
shift, correct AMDGPU address spaces, and **zero panic paths**. The NVIDIA image is
the equivalent `ptx_kernel` using `llvm.nvvm.read.ptx.sreg.tid.x`. This is
indistinguishable from what HIP/CUDA would emit for the same loop, and it supports
the paper's central performance claim on the codegen side.

In a **debug** build the same kernel carries `llvm.umul.with.overflow.i32` and calls
to `core::panicking::panic_const_mul_overflow` / `panic_bounds_check`. Rust's
arithmetic and bounds checks are compiled into device code, where a panic has
nowhere to go — it lowers to a call followed by `unreachable`. This is the concrete
form of the register-pressure gap the paper reports (33 vs 28 registers vs
RAJA-CUDA) and it is a real design question nobody has answered: *what should a
panic on a GPU do?*

### The container format confirms the plumbing

`device.bin` is an LLVM offload binary (magic `OFFLOAD`):

```
OFFLOADING IMAGE [0]:
kind      llvm ir
arch      gfx1100
triple    amdgcn-amd-amdhsa
producer  openmp
```

`producer openmp` is the honest label: this is rustc → LLVM Offload → the OpenMP
target runtime, exactly as the paper describes. Note `kind llvm ir` — final ISA
codegen happens at load time, which is why `-Ctarget-cpu` matters and why the image
is portable across a target family.

## What does not work, and why

**The device image is never embedded in the host binary.** The linked executable
imports `__tgt_register_lib`, `__tgt_target_kernel`, `__tgt_target_data_begin_mapper`
and friends, but `readelf -SW` finds no offload section and `strings` finds no
`gfx1100`. So the program registers an empty image, and at launch:

```
omptarget device 0 info: Entering OpenMP data region ... to(unknown)[1024]
omptarget device 0 info: Entering OpenMP kernel ... alloc(unknown)[1024]
omptarget error: Host ptr 0x... does not have a matching target pointer.
omptarget fatal error 1: failure of target construct while offloading is mandatory
```

The embedding step is normally `clang-linker-wrapper`/`clang-offload-packager`, and
the `offload` rustup component ships **only libraries** — no tools. That is what
"partial offload component" means in
[PR #160991](https://github.com/rust-lang/rust/issues/131513). Inferred, not
observed: supplying the wrapper by hand may be enough to close this. Not attempted.

Unrelated but worth recording: `libhsa-runtime64.so.1` (1.11.0, from Ubuntu) is
present on this host, so the AMD plugin has something to load if we get that far.
Whether gfx1150-under-`HSA_OVERRIDE_GFX_VERSION=11.0.0` works through libomptarget
is **untested** — we never reached a real launch.

## Two gaps in the shipped feature

These are the substance of any project fox might build here, and both were
confirmed by compiling, not by reading.

### 1. The documented example does not compile

`core/src/offload.md` uses `thread_idx_x()`, `block_idx_x()`, `block_dim_x()`.
Those names exist on neither target:

| | AMD (`core::arch::amdgpu`) | NVIDIA (`core::arch::nvptx`) |
|---|---|---|
| thread index | `workitem_id_x()` — safe | `_thread_idx_x()` — `unsafe` |
| block index | `workgroup_id_x()` — safe | `_block_idx_x()` — `unsafe` |
| block size | **does not exist** | `_block_dim_x()` — `unsafe` |

So the thread-indexing API — the first line of every kernel ever written — is not
portable across the two vendors the paper is about: different names, different
safety, and AMD has no workgroup-size query at all (the probe hard-codes 256).
Writing one kernel for both targets requires a shim today. That shim is trivial and
belongs in a library, not in every kernel.

### 2. The paper's safe abstractions are not in the toolchain

What shipped is the launch mechanism. The example in `offload.md` takes
`*mut [f64; 256]` and its body is `unsafe` — by-hand C with Rust syntax. Two things
the paper describes are absent from `core`, and the paper states both are ordinary
Rust needing no compiler support:

- **`Region` + `PartitioningStrategy`** — hands each thread a disjoint slice so the
  kernel body needs no `unsafe`. This is the entire safety claim of the work.
- **`Preload` / `PreloadMut`** — device residency as a type, so a chain of kernels
  does not round-trip through the host between launches.

`#[offload_kernel]` does accept `&mut [f32; 256]`, so the ownership-derived transfer
direction (`&T` → to, `&mut T` → to-from) is reachable; it is the *within-kernel*
safety and the *across-launch* residency that have no library.

## Where fox would fit

If fox builds anything here, the defensible contribution is not "an inference engine
on Rust kernels" — that is a much larger project with a worse risk profile, and on
this hardware decode is bandwidth-bound anyway (19 GB × 3.9 tok/s ≈ 74 of ~120 GB/s;
see the Qwen3.8-27B measurements). It is the **memory model**.

The paper's model is per-launch: map in, compute, map out. `PreloadMut` widens that
to two states, resident or dropped. An inference server needs neither: its KV cache
is tens of GB resident across millions of launches, in fixed-size blocks, ref-counted,
shared between sequences, copied on write. fox already implements exactly that in
`src/kv_cache/` — `PageTable`, `allocate`/`free_blocks`/`retain_block`,
`copy_on_write`, `is_shared`.

Expressing *that* discipline in the offload type system — paged, ref-counted,
copy-on-write device residency enforced by ownership — is a contribution the paper
does not have and its future-work section does not claim (it lists async transfers,
multi-device, ABI validation). vLLM's PagedAttention is a runtime data structure with
no static guarantees; this would be a compile-time one.

Two smaller pieces would support it:

- **Checked disjointness.** A `PartitioningStrategy` is a safety proof obligation
  stated in prose. Enumerating a launch grid and failing when an index is claimed
  twice or left unclaimed discharges it by execution, with no GPU involved.
- **A CPU reference backend.** The paper compares kernel *performance* against RAJA
  and CUDA. Nothing establishes that a Rust kernel computes what the CPU code it
  replaced computed. Executing the same `Kernel` over an emulated grid gives a
  differential-testing oracle — and lets the whole thing be developed and tested on
  stable rustc with no device, which is what makes progress possible while the
  embedding step is missing upstream.

## Decision

**Do not wire anything into fox's engine.** The end-to-end path does not execute
here, and even when it does, this is a platform bet rather than a speedup — see
`microbenchmarks-lie`: a 4.6× sampling micro-benchmark produced no measurable
throughput change on real traffic.

What is defensible now is building the *library* half — the parts that need no GPU —
so it is ready when the embedding step lands: the portable thread-index shim, the
`Region`/partitioning layer with checked disjointness, the residency types, and a CPU
oracle to test them against fox's existing `sample_greedy`. Re-run
`scripts/probe_rust_offload.sh` on each nightly bump; when the "offload section
present" check flips to PASS, the runtime half is unblocked.

The one hardware note: the discrete NVIDIA card in this machine is detached. The
`nvptx64`/`sm_120` device pass already compiles, and NVIDIA is the better-supported
libomptarget target — reattaching it is probably the cheapest way to reach a first
real execution.
