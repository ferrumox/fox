# Issue listo para pegar

> **SUPERADO — NO PUBLICAR. 2026-08-20, noche.** Este documento acusa a rustc de un
> fallo que no es suyo. El kernel no se lanzaba porque el `libhsa-runtime64` 1.11 de
> Ubuntu no reconoce gfx1150 y enumera cero GPUs; con HSA 1.18 el rustc de serie de
> rustup ejecuta el kernel. Ver `rustc-offload-fix-results.md`. Se conserva como
> registro de la investigación, no como borrador vivo.

Las frases en prosa son tuyas, traducidas del español palabra por palabra. Lo demás
son datos (código, comandos, trazas, direcciones, citas de la fuente).

FALTAN DOS COSAS, ver el final del fichero.

**Título:** std::offload: no offloaded kernel launches, the kernel's region_id has no matching target pointer

---

I tried `std::offload` end to end with my own code, because the dev guide example does not
compile, for three separate reasons: `std` in a `no_std` crate, an unannotated generic, and
`block_dim_x()`, which does not exist on `amdgpu`.

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

and a final link through `clang-linker-wrapper` 23.1.0 from apt.llvm.org, since the
`offload` rustup component ships libraries but no tools:

```
clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/cc \
  -L/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/x86_64-linux-gnu -L$TOOLCHAIN_LIB \
  host.o -lomptarget -Wl,-rpath,$TOOLCHAIN_LIB -o app
```

I expected `x[42]` to be `84.0`, that is, each thread writing its global index times two.

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

The address in the error is the kernel's `region_id`, not the mapped array, even though
the message sounds like the second.

I checked it by fixing the PIE base with `setarch -R` and adding the offset `nm` gives,
which matches exactly:

```
base                                 0x555555554000
._RNvC6probe44fill.region_id       + 0x000000043280
                                   = 0x555555597280   <- the address in the error
```

Data mapping and device codegen work. What fails is the kernel lookup in libomptarget's
table.

Everything that lookup needs looks right in the final binary:

| | |
|---|---|
| `llvm_offload_entries` | present, `WA`, 56 bytes, one entry |
| entry `Address` | `0x43280`, equal to `region_id` |
| entry `SymbolName` | `_RNvC6probe44fill` |
| device image exports | `_RNvC6probe44fill`, `.kd`, `.region_id` |

I also ruled out an argument mixup by disassembling the final binary; I can give the
details if they are useful.

## `.omp_offloading.descriptor` is registered all-zero

I think rustc registers an all-zero descriptor and on top of that initialises the RTLs
from it, which breaks the wrapper's good registration. But I could not see the exact
mechanism from the outside: patching the ctor removes the fatal error and leaves the
program running nothing.

The binary ends up with two descriptors, rustc's all-zero one and the linker wrapper's
populated one. `register_offload` in
`compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs` builds the null one
unconditionally, with the populated form sitting next to it in a comment:

```rust
let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
let omp_descriptor = add_global(cx, ".omp_offloading.descriptor", const_struct, InternalLinkage);
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 0, ptr null, ptr null, ptr null }
```

and then calls `__tgt_init_all_rtls()` under the existing

```rust
// FIXME(offload): Drop this, once we fully automated our offload compilation pipeline,
// since LLVM will initialize them for us if it sees gpu kernels being registered.
```

## Not a mixed-toolchain artifact

I rebuilt rustc from source at the same commit and relinked using only artifacts from
that build, and it fails the same.

Specifically: `--enable-llvm-offload --enable-clang --enable-lld` at `f7d782a3b`, then
relinked with that build's `clang-linker-wrapper` and `clang`
(`23.1.0-rust-1.100.0-nightly`, `rust-lang/llvm-project` at
`21cf28432798952d942bacc6bcee3a328faa3638`, the same commit string that appears inside
the `device.bin`), its `libomptarget.so` / `libomp.so` / `libLLVM.so`, and `rust-lld` as
`ld.lld`.

One caveat: `libompdevice.a` still comes from the rustup component, because bootstrap's
`llvm::OmpOffload` step fails here with `Host compiler does not support '-fuse-ld=lld'`,
which looks like what #161118 is already about.

## What this adds

TODO — ver abajo, hueco 1.

What this report adds is the diagnosis of what that address is, a repro that compiles
today, the proof that this is not toolchain mixing, the ignored return value of
`__tgt_target_kernel` (a separate problem), and the reason CI stays green.

## Meta

```
rustc 1.100.0-nightly (f7d782a3b 2026-08-19)   # master HEAD at the time of writing
rustup component add offload --toolchain nightly
libomptarget from that component
clang-linker-wrapper 23.1.0 from apt.llvm.org llvm-toolchain-noble-23
AMD Radeon 890M (gfx1150) with HSA_OVERRIDE_GFX_VERSION=11.0.0, so gfx1100
Ubuntu 24.04, libhsa-runtime64 1.11.0
```

---

# LO QUE FALTA

## Hueco 1 — reconocer los dos PRs (media frase tuya)

Tu respuesta 7 contesta «qué aporto», pero no dice «sé que existen». Sin eso, la primera
respuesta que recibes es un enlace al #152777. Hace falta una línea antes del párrafo
«What this adds», del tipo: que sabes que el #152777 y el #149827 arreglarían esto y que
llevan meses parados.

## Hueco 2 — el #150391 (una línea tuya)

Sin contestar. La pregunta era: ¿qué tiene que ver con el #150391 y por qué no estás
seguro de que sea lo mismo? Los datos: misma línea `does not have a matching target
pointer` en su pegado, mismo síntoma de valor que no vuelve, pero allí los maptypes son
`tofrom` y aquí `to` + `alloc`.

## Hueco 3 — la declaración de uso de LLM

Va al final, envuelta en `<!-- homu-ignore:start -->` y `<!-- homu-ignore:end -->` para
que no acabe en el historial de git. Tiene que decir dos hechos, con tus palabras:

- que usaste un LLM para investigar el fallo y para revisar, y que verificaste tú lo que
  se afirma;
- que escribiste el texto en español y está traducido automáticamente.
