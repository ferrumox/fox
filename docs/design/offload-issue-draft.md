# Apuntes para el issue de rust-lang/rust

> **SUPERADO — NO PUBLICAR. 2026-08-20, noche.** Este documento acusa a rustc de un
> fallo que no es suyo. El kernel no se lanzaba porque el `libhsa-runtime64` 1.11 de
> Ubuntu no reconoce gfx1150 y enumera cero GPUs; con HSA 1.18 el rustc de serie de
> rustup ejecuta el kernel. Ver `rustc-offload-fix-results.md`. Se conserva como
> registro de la investigación, no como borrador vivo.

**Esto son APUNTES, no texto para pegar.** Los datos (comandos, trazas, direcciones,
citas de la fuente) son hechos y se pegan tal cual. Las frases que los unen tienen que
reescribirse con tus palabras: la política de rust-lang prohíbe cuerpos de issue
originados en un LLM. La versión larga anterior quedó en
`offload-issue-draft-largo.md` por si necesitas consultar algo de lo que se cortó.

Verificado el 2026-08-20 contra `f7d782a3b`, que es el nightly instalado y el HEAD de
master. Sin duplicados: `does not have a matching target pointer` sólo aparece en
#150391, y el tracking #131513 no tiene comentarios.

**Título:** std::offload: no offloaded kernel launches, the kernel's region_id has no matching target pointer

---

## 1. Contexto  ← FRASE TUYA (2 líneas)

Que estabas siguiendo la tubería documentada en el dev guide sobre el nightly actual, y
que el ejemplo de allí no compila hoy por motivos aparte, así que usas uno mínimo.

## 2. El repro  ← SE PEGA

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

```
RUSTFLAGS="-Zunstable-options -Zoffload=HostMetadata=$PWD/meta.bin" \
  cargo +nightly build --release --lib
RUSTFLAGS="-Zunstable-options -Zoffload=Device=$PWD/meta.bin -Ctarget-cpu=gfx1100" \
  cargo +nightly build --release --lib -Zbuild-std=core --target amdgcn-amd-amdhsa
RUSTFLAGS="-Zunstable-options -Zoffload=Host=$DEVICE_BIN -L native=$TOOLCHAIN_LIB \
  -Clink-arg=-lomptarget" cargo +nightly build --release
```

Enlace final con `clang-linker-wrapper` 23.1.0 de apt.llvm.org, porque el componente
`offload` de rustup trae bibliotecas pero no herramientas:

```
clang-linker-wrapper --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/cc \
  -L/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/x86_64-linux-gnu -L$TOOLCHAIN_LIB \
  host.o -lomptarget -Wl,-rpath,$TOOLCHAIN_LIB -o app
```

## 3. Esperado vs real  ← FRASE TUYA (1 línea) + traza pegada

Esperabas `x[42] == 84.0`.

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

## 4. El puntero que falla es el kernel  ← LA PARTE FUERTE. FRASES TUYAS (3-4)

Esto es lo que justifica el issue. Tiene que dejar claro que la dirección del error es el
`region_id` del kernel, no el array mapeado, y por tanto que lo que falla es la búsqueda
de la entrada del kernel en `getTableMap()`, no el mapeo de datos, que sí funcionó.

Datos que se pegan:

- Bajo `setarch -R` la base PIE queda en `0x555555554000` y el error dice
  `Host ptr 0x555555597280`.
- `nm` sitúa `._RNvC6probe44fill.region_id` en `0x43280`.
- `0x555555554000 + 0x43280 = 0x555555597280`. Exacto.

| | |
|---|---|
| `llvm_offload_entries` | presente, `WA`, 56 bytes, una entrada |
| entrada `Address` | `0x43280`, igual al `region_id` |
| entrada `SymbolName` | `_RNvC6probe44fill` |
| exports de la imagen | `_RNvC6probe44fill`, `.kd`, `.region_id` |

Y una frase diciendo que también descartaste una confusión de argumentos desmontando el
binario final, sin desarrollarlo. Si preguntan, lo cuentas en un comentario.

## 5. El descriptor nulo  ← FRASE TUYA (2)

El binario acaba con dos descriptores: el de rustc, todo a cero, y el del wrapper,
poblado. `register_offload` en `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs`
construye el nulo incondicionalmente, con la forma buena al lado en un comentario:

```rust
let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
let omp_descriptor = add_global(cx, ".omp_offloading.descriptor", const_struct, InternalLinkage);
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 0, ptr null, ptr null, ptr null }
```

y luego llama a `__tgt_init_all_rtls()` bajo este `FIXME` que ya estaba:

```rust
// FIXME(offload): Drop this, once we fully automated our offload compilation pipeline,
// since LLVM will initialize them for us if it sees gpu kernels being registered.
```

**Tus dos frases:** qué supones (que el nulo es un marcador de posición mientras el
wrapper aporta el real) y qué NO pudiste confirmar (el mecanismo exacto, desde fuera del
compilador). No afirmes causa raíz.

## 6. No es mezcla de toolchains  ← FRASE TUYA (2)

Reconstruiste rustc desde fuente en `f7d782a3b` con `--enable-llvm-offload
--enable-clang --enable-lld` y relinkaste usando sólo artefactos de esa compilación:
`clang-linker-wrapper` y `clang` `23.1.0-rust-1.100.0-nightly` de `rust-lang/llvm-project`
en `21cf28432798952d942bacc6bcee3a328faa3638`, el mismo commit que aparece dentro del
`device.bin`; sus `libomptarget.so`, `libomp.so`, `libLLVM.so`; y `rust-lld` como
`ld.lld`. Mismo fallo.

Salvedad honesta: el `libompdevice.a` sigue viniendo del componente de rustup, porque el
paso `llvm::OmpOffload` de bootstrap falla aquí con `Host compiler does not support
'-fuse-ld=lld'`, que parece ser de lo que va el #161118.

## 7. Los dos PRs parados  ← FRASE TUYA (2). LO MÁS IMPORTANTE DESPUÉS DEL PUNTO 4

- **#152777**, «offload: automate additional steps». Borra el descriptor nulo, el
  `__tgt_register_lib` y el `__tgt_init_all_rtls`. Borrador, `S-waiting-on-author`, con
  conflictos, sin tocar desde el 2026-05-28.
- **#149827**, «automate offload, part 3 - clang-linker-wrapper». La otra mitad.
  Borrador, parado desde el 2026-04-28.

**Tus dos frases:** que los conoces, que llevan meses parados, y que lo que aportas es un
caso de fallo concreto contra el que probarlos cuando se retomen. Sin esto, la primera
respuesta que recibes es un enlace a ellos.

## 8. Posible relación con #150391  ← 1 LÍNEA TUYA

Misma línea `does not have a matching target pointer` y mismo síntoma, pero allí los
maptypes son `tofrom` y aquí `to` + `alloc`, así que no afirmas que sea el mismo bug.

## 9. Meta  ← SE PEGA

```
rustc 1.100.0-nightly (f7d782a3b 2026-08-19)   # master HEAD at the time of writing
rustup component add offload --toolchain nightly
libomptarget from that component
clang-linker-wrapper 23.1.0 from apt.llvm.org llvm-toolchain-noble-23
AMD Radeon 890M (gfx1150) with HSA_OVERRIDE_GFX_VERSION=11.0.0, so gfx1100
Ubuntu 24.04, libhsa-runtime64 1.11.0
```

Y una frase final diciendo que el codegen de dispositivo se ve bien y que es sólo el lado
del host lo que no encaja.

---

## Guardado para comentarios de seguimiento, NO en el issue

- Los dos experimentos de parcheo del binario (neutralizar el ctor da `x[42]=0`;
  reordenar `.init_array` no cambia nada).
- El desmontaje completo de `ArgBasePtrs`/`ArgPtrs`.
- El bloque de IR de dispositivo.

## Issues aparte, NO en este

- El valor de retorno de `__tgt_target_kernel` se ignora, sin fallback ni error. Es otro
  bug, más pequeño y más fácil de arreglar, y suelto tiene más recorrido.
- Los tres motivos por los que el ejemplo del dev guide no compila hoy.
- El ejemplo de `library/core/src/offload.md`: eso es el PR de la rama
  `offload-doc-example`, que enlazará a este issue.
