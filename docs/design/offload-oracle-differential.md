# El oráculo de CPU contra la GPU, por fin ejecutado

**2026-08-20, tarde.** Primer contraste real entre `crates/fox-offload` y una GPU, posible
sólo desde que se resolvió que el bloqueo era el runtime de HSA y no rustc — ver
`rustc-offload-fix-results.md`.

## El resultado

```
hilos lanzados (CPU): 256
parciales comparados: 256
discrepancias:        0
argmax GPU: Candidate { index: 123, value: 77.0 }
argmax CPU: Candidate { index: 123, value: 77.0 }
VEREDICTO: COINCIDEN
```

Ejecutado con `OMP_TARGET_OFFLOAD=MANDATORY`, o sea que corrió de verdad en la Radeon 890M
y no cayó al host.

Lo que se comparó no es «un kernel equivalente»: es **el mismo `BlockArgmax::thread` del
crate**, compilado dos veces —`x86_64` y `amdgcn`/`gfx1100`— desde el mismo fuente. Los 256
parciales coinciden uno a uno, no sólo el ganador.

La entrada es deliberadamente hostil: valores negativos, un **empate** en 12.5 entre los
índices 17 y 200, y un **NaN** en el 99.

- La regla de desempate del crate (gana el índice mayor, como `Iterator::max_by`) se
  cumple en GPU.
- La regla «un NaN nunca gana» se cumple en GPU.
- `Candidate { index: u32, value: f32 }` **atraviesa el mapeo de dispositivo intacto**.
  Es un dato sobre la brecha de ABI que el paper reconoce sin implementar (§10 de la
  crítica): para un struct de dos campos de 4 bytes, host y amdgcn coinciden.

Esto es exactamente el test diferencial que la crítica del paper señala que nadie hace: el
paper compara *rendimiento* contra RAJA y CUDA, y nada establece que un kernel en Rust
calcule lo que calculaba el código al que sustituye.

## Cinco defectos del crate que sólo aparecen compilando para device

Ninguno se ve en `cargo test`. Todos salieron al intentar la pasada de `amdgcn` por primera
vez. Los cuatro primeros se arreglaron en una copia de trabajo; **no se ha tocado
`crates/fox-offload`**, que tiene cambios de otra sesión.

**1. `extern crate alloc;` incondicional** en `lib.rs:54`. Basta eso para que el crate no
compile para GPU: `can't find crate for alloc`. Y contradice el comentario de su propio
`Cargo.toml` — *"Kernel code has to compile for a target whose entire standard library is
`core`; a dependency tree is how that stops being true"*.

**2. `region.rs` importa `alloc::vec` a nivel de módulo**, para `verify_disjoint`, que es
una herramienta de host. Mismo efecto.

**3. `thread.rs` llama a `workitem_id_x()` y compañía sin activar `stdarch_amdgpu`.** El
crate apunta a estable, así que no puede activarla; hace falta
`#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]`. Es decir: la función
`ThreadId::current()`, que existe **sólo** para el dispositivo, nunca ha compilado para uno.

**4. `resident.rs`, `verify_disjoint` y `argmax_via_kernel` son de host** y necesitan
quedar fuera en targets de GPU. El primero usa `Vec` en todo el pool de bloques.

**5. El más serio: los métodos de trait de un crate dependencia no se instancian para el
dispositivo.** Con el kernel llamando a `BlockArgmax::thread`, el enlace de device falla:

```
ld.lld: error: undefined symbol:
  <fox_offload::kernels::argmax::BlockArgmax as fox_offload::launch::Kernel>::thread
```

Se resolvió poniendo `#[inline(always)]` en `thread` y en `Linear1D::lane`. Funciona, pero
es una restricción de diseño real y no está documentada en ningún sitio: **si el kernel
llama a algo de otro crate que no se inlinea, no enlaza.** Para una biblioteca de kernels,
que es justo lo que `fox-offload` quiere ser, eso condiciona toda la API.

## Cómo reproducirlo

El arnés vive en el scratchpad de la sesión (`oracle/`) y no en el repo, porque necesita
una copia parcheada del crate. Las cuatro piezas del entorno:

- `rustc +nightly` de serie, sin parchear.
- `clang-linker-wrapper` de LLVM 23 y `ld.lld` en el `PATH`.
- Runtime de device: `--device-linker=amdgcn-amd-amdhsa=-L<...>/amdgcn-amd-amdhsa -lompdevice`.
- **`libhsa-runtime64` de ROCm 7.2.4** en `LD_LIBRARY_PATH`. Sin esto no hay GPU y todo lo
  anterior da ceros en silencio.

Pasadas: `HostMetadata`, luego `Device` con `-Zbuild-std=core,alloc` — nótese el `alloc`,
que hoy hace falta por el defecto 1 —, luego `Host`, y enlace final a mano.

## Qué hacer con esto

Los cinco defectos son arreglos pequeños y concretos en `crates/fox-offload`, y el
diferencial merece ser un test del crate en vez de un arnés suelto. Pero el crate tiene
cambios sin commitear de otra sesión, así que no se tocó.

Y hay una consecuencia para la crítica del paper: su frase *"as of 2026-08-20 we could not
execute a single offloaded kernel on a stock nightly"* (línea 272) **ya es falsa**, y era
una de las dos advertencias marcadas como obligatorias si algo de eso se publicaba.
