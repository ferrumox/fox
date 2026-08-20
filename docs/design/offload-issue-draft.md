# Draft: rust-lang/rust issue

Not part of fox. Kept here so the analysis behind it stays with
`rust-gpu-offload.md`.

Ready to file. The toolchain-mixing explanation was checked and ruled out — see
"Reproduced with a fully matched toolchain" below.

**Title:** `std::offload`: kernels never launch, the host `__tgt_bin_desc` is registered empty

---

The example in the dev guide does not build on current nightly, so this is a
minimal one. Three separate reasons, all on `f7d782a3b`:

- `#[offload_kernel]` expands to something referencing `std`, and the example is
  `#![no_std]`, so the `HostMetadata` pass fails with ``cannot find `std` in the list
  of imported crates``.
- `assert_eq!(x[i], 2.5)` on the generic kernel's output gives `E0283: type
  annotations needed`.
- The AMD branch imports only `workgroup_id_x as block_idx_x` and
  `workitem_id_x as thread_idx_x`, but the kernel body calls `block_dim_x()`, which
  is never imported for `amdgpu` and does not exist in `core::arch::amdgpu` at all.

So, the code I actually ran:

```rust
#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]
#![feature(gpu_offload, abi_gpu_kernel)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn gid_x() -> u32 {
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    workitem_id_x() + workgroup_id_x() * 256
}
#[cfg(not(target_arch = "amdgpu"))]
#[inline(always)]
fn gid_x() -> u32 { 0 }

#[core::offload::offload_kernel]
fn fill(x: &mut [f32; 256]) {
    let i = gid_x() as usize;
    if i < x.len() { x[i] = i as f32 * 2.0; }
}

pub fn run() -> [f32; 256] {
    let mut x = [0.0f32; 256];
    core::offload::offload! {
        kernel = fill,
        workgroup_dim = [1, 1, 1],
        thread_dim = [256, 1, 1],
        args = (&mut x,),
    }
    x
}
```

built with

```
RUSTFLAGS="-Zunstable-options -Zoffload=HostMetadata=$PWD/meta.bin" \
  cargo +nightly build --release --lib
RUSTFLAGS="-Zunstable-options -Zoffload=Device=$PWD/meta.bin -Ctarget-cpu=gfx1100" \
  cargo +nightly build --release --lib -Zbuild-std=core --target amdgcn-amd-amdhsa
RUSTFLAGS="-Zunstable-options -Zoffload=Host=$DEVICE_BIN -L native=$TOOLCHAIN_LIB \
  -Clink-arg=-lomptarget" cargo +nightly build --release
```

and a final link through `clang-linker-wrapper` 23.1.0 from apt.llvm.org's
`llvm-toolchain-noble-23`, since the `offload` rustup component ships libraries but no
tools:

```
clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/cc \
  -L/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/x86_64-linux-gnu -L$TOOLCHAIN_LIB \
  host.o -lomptarget -Wl,-rpath,$TOOLCHAIN_LIB -o app
```

I expected `x[42] == 84.0`.

Instead:

```
omptarget device 0 info: Entering OpenMP data region with being_mapper at unknown:0:0 with 1 arguments:
omptarget device 0 info: to(unknown)[1024]
omptarget device 0 info: Creating new map entry with HstPtrBase=0x00007ffc3f4d4408, ...
                         TgtPtrBegin=0x000062bf9597f6e0, Size=1024, DynRefCount=1
omptarget device 0 info: Entering OpenMP kernel at unknown:0:0 with 1 arguments:
omptarget device 0 info: alloc(unknown)[1024]
omptarget error: Host ptr 0x5d4dc9f01280 does not have a matching target pointer.
omptarget fatal error 1: failure of target construct while offloading is mandatory
```

## The failing pointer is the kernel, not the argument

This bit is easy to misread, so to be explicit: the address in the error is the
kernel's `region_id`, not the mapped array. Re-running under `setarch -R` to pin the
PIE base at `0x555555554000` gives `Host ptr 0x555555597280`, and `nm` puts
`._RNvC6probe44fill.region_id` at `0x43280`. `0x555555554000 + 0x43280` is that
address exactly. So this is `getTableMap()` in libomptarget failing to
find the kernel entry, and the data mapping above it succeeded.

Everything that lookup needs looks right in the final binary:

| | |
|---|---|
| `llvm_offload_entries` | present, `WA`, 56 bytes, one entry |
| entry `Address` | `0x43280`, equal to `region_id` |
| entry `SymbolName` | `_RNvC6probe44fill` |
| device image exports | `_RNvC6probe44fill`, `.kd`, `.region_id` |

Disassembling the final binary (not just the object) also rules out an argument
mixup: `__tgt_target_data_begin_mapper` gets `%rcx`/`%r8` pointing at `0x48(%rsp)` and
`0x40(%rsp)`, `KernelArgs.ArgBasePtrs`/`ArgPtrs` get the same two slots, and both
slots hold `&x`.

## `.omp_offloading.descriptor` is registered all-zero

The binary ends up with two descriptors and two ctor pairs, one from
`register_offload` and one from the linker wrapper:

```
rustc's,   .rodata      @0x43210:  00 00 00 00 ... all zero
wrapper's, .data.rel.ro @0x4eb80:  NumDeviceImages=1, DeviceImages=0x4eb60,
                                   HostEntriesBegin=0x4fa38, HostEntriesEnd=0x4fa70
```

`register_offload` in `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs` builds
the null one unconditionally, with the populated form sitting next to it in a comment:

```rust
let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
let omp_descriptor = add_global(cx, ".omp_offloading.descriptor", const_struct, InternalLinkage);
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 0, ptr null, ptr null, ptr null }
```

and registers it, then calls `__tgt_init_all_rtls()` under the existing

```rust
// FIXME(offload): Drop this, once we fully automated our offload compilation pipeline,
// since LLVM will initialize them for us if it sees gpu kernels being registered.
```

I assume the null descriptor is intended as a placeholder while the wrapper supplies
the real one. If so, registering it and eagerly initialising the RTLs from it seems to
be what breaks the binding, but I could not confirm the mechanism from the outside.
Two things I tried, both by patching the linked binary rather than rebuilding rustc:

- Turning `register_offload`'s ctor into a bare `ret` removes the fatal error. The
  program then prints `x[42]=0` and produces no omptarget output at all.
- Swapping the two entries in `.init_array` so the wrapper registers first changes
  nothing.

Adding the device runtime the way the dev guide's wrapper invocation does —
`--should-extract=gfx1100 --device-linker=amdgcn-amd-amdhsa=-lompdevice` against the
`libompdevice.a` in the rustup component, plus `-lomp` — also changes nothing.

## Reproduced with a fully matched toolchain

Since the `offload` component ships no tools, the first runs mixed rustup's rustc and
libomptarget with apt.llvm.org's `clang-linker-wrapper`, and "you mixed LLVM builds"
would be a fair thing to suspect. So I built rustc from source at the same commit this
nightly comes from (`f7d782a3b`) with `--enable-llvm-offload --enable-clang
--enable-lld`, and relinked using only artifacts from that build:

- `clang-linker-wrapper` and `clang`, `23.1.0-rust-1.100.0-nightly`, from
  `rust-lang/llvm-project` at `21cf28432798952d942bacc6bcee3a328faa3638` — the same
  commit string that appears inside the `device.bin` rustc produced
- `libomptarget.so`, `libomp.so`, `libLLVM.so` from that build's `offload/` and
  `llvm/` output
- `ld.lld` = `rust-lld` from the toolchain, `LLD 23.1.0` at the same commit

Same failure, byte for byte in behaviour. So this is not a mixed-toolchain artifact.

One caveat, stated because it is the only piece that is not from that build:
`libompdevice.a` still comes from the rustup component, because bootstrap's
`llvm::OmpOffload` step fails on this machine — it configures the amdgcn device
runtime with `-DCMAKE_C_COMPILER=cc -DCMAKE_C_COMPILER_TARGET=amdgcn-amd-amdhsa` and
dies on `Host compiler does not support '-fuse-ld=lld'`. That looks like the problem
#161118 is already about, so I have not filed it separately. The host half of offload
built and installed fine.

## Two smaller things

**The return value of `__tgt_target_kernel` is dropped.** In the generated code the
instruction after the call is `xorps %xmm0,%xmm0`; there is no branch on the result and
no host fallback. That is why the first experiment above produced silently wrong
results instead of an error. Whatever happens with the descriptor, an offload that
fails at runtime should probably not be indistinguishable from one that worked.

**Neither documented example compiles.** Besides the dev guide one above,
`library/core/src/offload.md` uses `thread_idx_x()`, `block_idx_x()` and
`block_dim_x()`, and none of those exist on either target: `core::arch::amdgpu` has `workitem_id_x()`/`workgroup_id_x()` as safe
fns and no workgroup-size query at all, while `core::arch::nvptx` has
`_thread_idx_x()`/`_block_idx_x()`/`_block_dim_x()` as `unsafe` fns. Happy to send a
PR for the doc if that is useful, though it is really the portability gap underneath
that matters: a kernel that builds for both targets needs a `cfg` shim today. I can
open that separately if you would rather keep this issue narrow.

## Possibly the same thing as #150391

#150391 is filed about the missing debug locations, but the paste in it, from the
rustc-dev-guide usage example, contains the same `does not have a matching target
pointer` line and the same "value never came back" symptom (`The first element is zero
0.000000`). The maptypes differ from mine (`tofrom` there, `to` + `alloc` here), so I
am not certain it is one bug rather than two.

It also looks like nothing executes an offloaded kernel in CI yet — #158817 says
"We can check LLVM IRs in tests/codegen-llvm/gpu_offload without a gpu runner so far"
— which would explain why an IR-level regression suite stays green through this.

## Meta

```
rustc 1.100.0-nightly (f7d782a3b 2026-08-19)
rustup component add offload --toolchain nightly
libomptarget from that component
clang-linker-wrapper 23.1.0 from apt.llvm.org llvm-toolchain-noble-23
AMD Radeon 890M (gfx1150) with HSA_OVERRIDE_GFX_VERSION=11.0.0, so gfx1100
Ubuntu 24.04, libhsa-runtime64 1.11.0
```

Device codegen itself looks fine, for what it's worth. The release IR for the kernel
above is

```llvm
define amdgpu_kernel void @_RNvC6probe44fill(ptr nofree readnone captures(none) %0,
                                             ptr nofree noundef writeonly captures(none) %1) {
  %3 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %4 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %5 = shl i32 %4, 8
  %6 = add i32 %5, %3
  %7 = icmp ult i32 %6, 256
  ...
  store float %12, ptr addrspace(1) %14, align 4
}
```

with the bounds check folded, the multiply strength-reduced and no panic paths left,
and the `nvptx64`/`sm_120` pass produces the equivalent `ptx_kernel`. It is only the
host side that does not come together.
