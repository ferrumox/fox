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
| Device image reaches the binary | **yes**, with an LLVM 23 `clang-linker-wrapper` (see below) |
| End-to-end execution | **no** — the kernel entry is never bound; host registration is a placeholder |

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

## The link step, solved

`clang-linker-wrapper` is not optional and not a workaround: the paper states the
flow is "two cargo commands and one `clang-linker-wrapper` invocation". rustc emits
the image into a `.llvm.offloading` section flagged `SHF_EXCLUDE`, so a plain linker
is *required* to drop it; the wrapper is what extracts it, compiles it for the
device, and emits a host object with the image and a populated descriptor.

The rustup `offload` component ships libraries only, so the tool has to come from
somewhere else. Two sources were tried:

- **ROCm 7.2's** (`/opt/rocm/llvm/bin/clang-linker-wrapper`, AMD clang 22.0.0git).
  Runs standalone on the host, but cannot read rust's output:
  `error: Invalid data was encountered while parsing the file`. The offload binary
  header declares version 2; the LLVM 22 reader does not accept it. Worse, driven
  through `rustc -Clinker=`, it swallows that failure and produces a binary with no
  image rather than erroring.
- **apt.llvm.org's `llvm-toolchain-noble-23`** — works. The packages are
  23.1.0 (2026-08-18), the same LLVM rust's nightly is built against, and they
  extract without root, the same trick used for the Vulkan SDK:

  ```
  curl -sL -o ct23.deb https://apt.llvm.org/noble/pool/main/l/llvm-toolchain-23/clang-tools-23_*.deb
  dpkg-deb -x ct23.deb llvm23/      # also: libllvm23, libclang-cpp23, clang-23, lld-23
  export LD_LIBRARY_PATH=$PWD/llvm23/usr/lib/x86_64-linux-gnu
  export PATH=$PWD/llvm23/usr/lib/llvm-23/bin:$PATH
  ```

  Invoked by hand on rustc's `host.o`, it extracts the image, compiles it for
  `gfx1100` and links. The resulting executable **does** contain the `OFFLOAD`
  magic and the `gfx1100` string. Driving it through `rustc -Clinker=` does not yet
  work — rustc adds `--gc-sections`, `-nodefaultlibs` and `-fuse-ld=lld` and the
  image is lost again — so the working recipe is a manual final link.

So the packaging blocker is closed locally. What it uncovered is a different and
more interesting problem.

## The real blocker: the kernel is never bound to its device entry

With a real image embedded the failure does not change, which rules out the empty
descriptor as *the* cause. The error is:

```
Entering OpenMP data region ... to(unknown)[1024]          <- succeeds
Entering OpenMP kernel ... alloc(unknown)[1024]
omptarget error: Host ptr 0x555555597280 does not have a matching target pointer.
```

**That pointer is the kernel, not the data.** Run under `setarch -R`, the PIE base is
`0x555555554000` and `nm` puts `._RNvC6probe44fill.region_id` at `0x43280`; the sum is
exactly the reported address. So this is libomptarget's `getTableMap()` failing to
find the *kernel entry*, not a data-mapping failure. (An earlier revision of this
document said the launch was handed a different pointer than the one mapped. That was
wrong: disassembling the final binary shows both `__tgt_target_data_begin_mapper` and
`KernelArgs.ArgBasePtrs`/`ArgPtrs` pointing at the same two stack slots, both holding
`&x`.)

Everything the lookup needs looks correct:

| | |
|---|---|
| `llvm_offload_entries` in the binary | present, allocated, 56 bytes = one entry |
| entry `Address` | `0x43280` — equals `region_id` |
| entry `SymbolName` | `_RNvC6probe44fill` |
| device image exports | `_RNvC6probe44fill`, `.kd`, `.region_id` |

So the host table is right, the names match, and the image is there. What is wrong is
the registration. The binary contains **two** descriptors and **two** ctor pairs:

| | rustc's, `0x43210` | wrapper's, `0x4eb80` |
|---|---|---|
| `NumDeviceImages` | `0` | `1` |
| `DeviceImages` | `NULL` | `0x4eb60` |
| `HostEntriesBegin`/`End` | `NULL`/`NULL` | `0x4fa38`/`0x4fa70` |

rustc registers an all-null descriptor and then calls `__tgt_init_all_rtls()`,
initialising every plugin before the real image is registered. That is not an
accident, and `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs` says so
outright — it builds the null struct unconditionally, with the correct form sitting
next to it in a comment:

```rust
let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
// @.omp_offloading.descriptor = ... { i32 1, ptr @.omp_offloading.device_images,
//                                     ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// @.omp_offloading.descriptor = ... { i32 0, ptr null, ptr null, ptr null }
```

and, immediately above the `__tgt_init_all_rtls` declaration:

```rust
// FIXME(offload): Drop this, once we fully automated our offload compilation pipeline,
// since LLVM will initialize them for us if it sees gpu kernels being registered.
```

Two binary-patch experiments, so the conclusion is not just source reading:

- **Neutralising rustc's ctor** (`ret` at its entry) removes the fatal error — and the
  program prints `x[42]=0`, silently producing wrong results with no omptarget output
  at all. Which exposes a second defect: rustc **ignores the return value** of
  `__tgt_target_kernel`, so an offload that fails for any reason does not fall back
  and does not complain. There is no host fallback branch after the call.
- **Reordering `.init_array`** so the wrapper's registration runs first changes
  nothing.

So the host-side registration is a placeholder that the feature's own source says is
waiting on the rest of the pipeline. That is consistent with rustup not shipping the
linker wrapper: the host/link half is simply not finished yet, and no amount of
getting the image into the binary compensates.

Ruled out along the way, so a bug report can be narrow: not a stale build (all passes
rebuilt clean, image verified in the final binary), not the argument form (`*mut` and
`&mut` behave identically), not a function boundary (array declared at the call site
behaves identically), and not the AMD plugin — libomptarget enumerates four devices,
allocates on device 0, and prints a correct host-device mapping table, so device
memory allocation works.

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
anywhere we can reach, and even when it does, this is a platform bet rather than a speedup — see
`microbenchmarks-lie`: a 4.6× sampling micro-benchmark produced no measurable
throughput change on real traffic.

What is defensible now is building the *library* half — the parts that need no GPU —
so it is ready when the embedding step lands: the portable thread-index shim, the
`Region`/partitioning layer with checked disjointness, the residency types, and a CPU
oracle to test them against fox's existing `sample_greedy`. Re-run
`scripts/probe_rust_offload.sh` on each nightly bump; the check that matters now is
whether the mapped pointer and the launched pointer agree.

The one hardware note: the discrete NVIDIA card in this machine is detached, so the
only reachable runtime is the 890M. The `nvptx64`/`sm_120` device pass already
compiles, and NVIDIA is the better-supported libomptarget target — but since the
blocker is now host-side pointer bookkeeping rather than anything device-specific,
reattaching the card would most likely reproduce the same error, not get past it.
