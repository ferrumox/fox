# Prompt to hand this to a fresh session

Copy everything below the line into a new session. It is deliberately
self-contained: it carries the facts inline so it still works if the branch
`research/rust-gpu-offload` has not been pushed and the documents are not
reachable.

---

Trabajo de contribución a `rust-lang/rust`, a partir de una investigación que ya está
hecha y verificada. No la repitas: lo que sigue se comprobó **ejecutándolo** el
2026-08-20 en esta máquina (Ryzen + Radeon 890M / gfx1150, sin GPU discreta).

## Contexto: qué se investigó y qué se encontró

El paper *GPU Offload in Rust: Portable, Safe, and Fast* (arXiv:2608.13759, Drehwald et
al., 13 ago 2026) describe un backend de offload GPU integrado en rustc sobre LLVM
Offload. No es sólo un paper: **está upstream**, tracking issue
[rust-lang/rust#131513](https://github.com/rust-lang/rust/issues/131513), abierto el
2024-10-10 por `@ZuseZ4`, que es Manuel Drehwald, el primer autor.

Se verificó el estado real del pipeline en esta máquina:

- `rustup component add offload --toolchain nightly` trae `libRustOffload-23.so`,
  `libomptarget.so` y el runtime de device de los dos fabricantes. **No hace falta ROCm
  ni CUDA para compilar.**
- La pasada de metadatos host y las dos pasadas de device **funcionan**: `amdgcn`/`gfx1100`
  y `nvptx64`/`sm_120` compilan **desde el mismo fuente**. El IR de release es limpio
  (intrínsecos inlineados, `*256` reducido a `shl`, address spaces correctos, cero panics).
- Flags obligatorias: `-Zunstable-options`, `-Zbuild-std=core` en device, y `lto = "fat"`
  en el perfil para la pasada host.
- **El binario no ejecuta.** rustc marca la sección `.llvm.offloading` como `SHF_EXCLUDE`,
  así que el enlazador debe descartarla; hace falta `clang-linker-wrapper`, que rustup no
  distribuye. El de ROCm 7.2 (LLVM 22) no sabe leer la salida de LLVM 23. El que sirve es
  el de `apt.llvm.org` / `llvm-toolchain-noble-23` (23.1.0), extraíble sin root con
  `dpkg-deb -x` (paquetes `clang-tools-23`, `libllvm23`, `libclang-cpp23`, `clang-23`,
  `lld-23`). Con él la imagen **sí** entra en el binario.
- **Y aun así el kernel no se lanza.** El puntero que reporta `omptarget` es el
  `region_id` **del kernel**, no un argumento — comprobado con `setarch -R`: base PIE
  `0x555555554000` + `0x43280` = la dirección exacta reportada. La tabla de entradas del
  host es correcta y la imagen exporta el símbolo. Lo que falla es el **registro**: el
  binario acaba con **dos descriptores**, el del wrapper poblado y **el de rustc todo a
  ceros**, que además llama a `__tgt_init_all_rtls()`.
  `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs` construye el nulo
  **incondicionalmente**, con la forma correcta al lado en un comentario, y lleva un
  `FIXME(offload): Drop this, once we fully automated our offload compilation pipeline`.
- **No es mezcla de toolchains.** Se reprodujo compilando rustc desde fuente en el mismo
  commit del nightly (`f7d782a3b`) con `--enable-llvm-offload --enable-clang --enable-lld`
  y reenlazando sólo con artefactos de ese build. Fallo idéntico.
- Descartado también: build viejo, forma del argumento (`*mut` y `&mut` fallan igual),
  frontera de función, orden de `.init_array`, y el plugin AMD (libomptarget enumera 4
  dispositivos, reserva en el 0 y saca una tabla de mapeo correcta).

**Si el repo está disponible**, la rama `research/rust-gpu-offload` de
`/home/manuelslemos/Documents/ferrumox/fox` tiene `docs/design/rust-offload-handoff.md`
(resumen y decisiones), `rust-gpu-offload.md` (viabilidad), `offload-paper-critique.md`
(lectura del paper), `offload-issue-draft.md` (el reporte) y
`scripts/probe_rust_offload.sh` (reproduce el pipeline). **Ojo: la rama puede no estar
empujada.** Si no la encuentras, trabaja con los datos de arriba, que bastan.

## Tres entregables, en este orden. El orden es la parte importante

### 1. El issue a rust-lang/rust

Existe un borrador terminado en `docs/design/offload-issue-draft.md`. Si no lo alcanzas,
reconstrúyelo con los datos del contexto.

Título: *`std::offload`: kernels never launch, the host `__tgt_bin_desc` is registered
empty*.

Ya se comprobó que **no es duplicado**, con un matiz que hay que mantener en el texto: el
mismo error (`Host ptr … does not have a matching target pointer`) aparece **dentro** del
issue #150391, abierto por el propio ZuseZ4 en dic-2025 ejecutando el ejemplo del
rustc-dev-guide — pero lo tituló *"std::offload lacks debug locations"*, o sea reportó los
`unknown:0:0` de la traza y el fallo de lanzamiento nunca tuvo issue propio. Los maptypes
difieren (`tofrom` allí, `to` + `alloc` aquí), así que el borrador dice que **puede o no
ser el mismo bug**, sin afirmarlo. Mantén esa cautela.

Contexto adicional que explica por qué esto sobrevive: la PR #158817 ("ci: Enable offload
tests in CI"), aún en borrador, dice *"We can check LLVM IRs in
tests/codegen-llvm/gpu_offload without a gpu runner so far"*. **Nadie ejecuta un kernel de
offload en CI.**

Tu trabajo: revisarlo con ojo crítico, comprobar que no ha aparecido un duplicado desde el
2026-08-20 (busca `"does not have a matching target pointer"`, `__tgt_register_lib` y
#131513), y decir si hay algo que templar. **No lo publiques tú** — dame el texto final y
lo pego yo desde mi cuenta.

### 2. El PR de documentación

`library/core/src/offload.md` en rust-lang/rust tiene un ejemplo que **no compila**. Usa
`thread_idx_x()`, `block_idx_x()` y `block_dim_x()`, y ninguno de los tres existe:

| | AMD (`core::arch::amdgpu`) | NVIDIA (`core::arch::nvptx`) |
|---|---|---|
| índice de hilo | `workitem_id_x()` — segura | `_thread_idx_x()` — `unsafe` |
| índice de bloque | `workgroup_id_x()` — segura | `_block_idx_x()` — `unsafe` |
| tamaño de bloque | **no existe** | `_block_dim_x()` — `unsafe` |

Se pudrió porque el bloque está marcado ```` ```rust,ignore ```` y los doctests nunca lo
compilan. Verificado con `E0425: cannot find function` en nightly 1.100.0.

Prepara el PR entero: rama, el arreglo, mensaje de commit y texto del PR. **Déjalo listo
para que yo lo empuje** — no empujes ni abras el PR tú.

El ejemplo corregido tiene que compilar de verdad para los dos targets. Si eso obliga a un
shim con `#[cfg]`, enséñalo tal cual: la falta de portabilidad de la indexación de hilos es
el problema de fondo, no un detalle a esconder. Nota también que el ejemplo del
rustc-dev-guide (`src/offload/usage.md`) está roto por tres motivos distintos —
`#[offload_kernel]` emite rutas de `std` en un crate `#![no_std]`, un `E0283` en el
`assert_eq!`, y la rama AMD llama a `block_dim_x()` sin importarlo — por si quieres
mencionarlo o abrir eso aparte.

### 3. Las estrategias para la PR #158076 — **sólo después de que contesten al issue**

[PR #158076](https://github.com/rust-lang/rust/pull/158076), "Offload safe mutable args
with `Region` and `PartitioningStrategy`", de `Sa4dUs`, asignada a `ZuseZ4`, abierta desde
junio de 2026 y esperando revisión. Lo que trae:

```rust
pub unsafe trait PartitioningStrategy {
    type View<'a, T: 'a>;
    type ViewMut<'a, T: 'a>;
    fn index() -> usize;
    unsafe fn get<'a, T>(ptr: *const T, len: usize) -> Option<Self::View<'a, T>>;
    unsafe fn get_mut<'a, T>(ptr: *mut T, len: usize) -> Option<Self::ViewMut<'a, T>>;
}
```

**Ninguna estrategia concreta** más allá de un `Dummy` de test, y la disjunción escrita
como obligación de seguridad sobre quien implemente el trait, sin nada que la compruebe.

El paper invita explícitamente: *"By releasing our interface as a standalone crate, we hope
to encourage users to explore additional partitioning schemes."* Y la §3.2 admite:
*"neither of these requirements can be verified by the compiler, therefore both trait and
functions are unsafe"* — cierto, y no es lo mismo que "no se puede verificar": enumerar la
malla lo decide en milisegundos y sin GPU.

Lo que tenemos y encaja en ese hueco está en `crates/fox-offload/` (rama
`research/rust-gpu-offload`, 27 tests, cero dependencias, compila en stable):
`Linear1D` y `Strided1D` como estrategias concretas sin `unsafe`, y `verify_disjoint`, que
recorre una malla entera y falla si un elemento se reclama dos veces o queda sin reclamar
— con tests que demuestran que caza un reparto solapado, un resto perdido y un
desbordamiento.

**Esto NO es un drop-in.** Hay una PR de otra persona a medio revisar. El camino es
comentar en la PR ofreciéndolo, no aparecer con código. Y no lo abordes hasta que el issue
del punto 1 tenga respuesta: si dicen que el enlace host va en camino, cambia el encuadre.

## Restricciones

- El issue y el PR van **en inglés**, en voz de ingeniero: sin listas de viñetas donde va
  prosa, sin negritas de más, con las dudas reales dichas como dudas. Salen con mi nombre.
- **No infles el PR de docs.** Es un arreglo de documentación y cualquiera puede ver su
  tamaño.
- **El orden importa y no es negociable.** El issue lleva un día de análisis y establece
  que sé de qué hablo; el PR detrás se lee como *"y de paso te arreglo esto que encontré"*.
  Al revés es un arreglo tipográfico de un desconocido.
- **No empujes nada ni abras nada en GitHub.** Prepara y entrega; yo publico.
- No toques nada de fox fuera de estos entregables. La investigación está cerrada.

## Qué quiero al terminar

Los textos listos para pegar, qué cambiaste de los borradores y por qué, y si algo de lo
que te di como verificado ya no se sostiene.
