# Prompt: fix the rustc bug that stops offloaded kernels from launching

Copy everything below the line into a fresh session. Self-contained on purpose.

Companion to `rust-offload-contribution-prompt.md`, which covers filing the report,
the docs PR, and the offer to PR #158076. This is the escalation of the same line:
that one **reports** the bug, this one **fixes** it.

---

Trabajo sobre `rust-lang/rust`. Objetivo concreto y comprobable: **hacer que un kernel
offloaded se lance de verdad**, que hoy no ocurre.

Todo lo que va como "verificado" se comprobó **ejecutándolo** el 2026-08-20 en esta misma
máquina (Ryzen + Radeon 890M / gfx1150, sin GPU discreta conectada). No lo repitas por
rutina.

## Criterio de éxito, para que no haya ambigüedad

`scripts/probe_rust_offload.sh` (en el repo fox, rama `research/rust-gpu-offload`) llega
hoy hasta el lanzamiento y muere. **Éxito es que imprima `x[42]=84 x[255]=510`** en vez de
`omptarget fatal error 1`.

Si al final del día no lo imprime, el resultado es igual de válido: escribe qué
descartaste y por qué. Es un experimento, no un encargo.

## Paso 0 — dos minutos, antes de nada

Busca en `rust-lang/rust` **PRs abiertas que toquen
`compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs`**, o que mencionen
`register_offload`, `__tgt_bin_desc` o `__tgt_register_lib`.

El 2026-08-20 había cinco PRs de offload en vuelo (#158032 selección de dispositivo,
#161118 build del runtime, #158817 tests en CI, #158076 `Region`, #152011 LLVM-22) y
**ninguna iba del descriptor host**. Si eso ha cambiado, **para y dímelo**: puede que el
arreglo ya esté en camino y el trabajo sobre.

## El bug

`rustc` compila kernels para AMD y NVIDIA correctamente. El binario final no lanza nada:

```
omptarget error: Host ptr 0x555555597280 does not have a matching target pointer.
omptarget fatal error 1: failure of target construct while offloading is mandatory
```

Ese puntero es el **`region_id` del kernel**, no un argumento — comprobado con `setarch -R`
para fijar la base PIE en `0x555555554000`, más `0x43280` que es donde `nm` sitúa
`._RNvC…fill.region_id`. Suma exacta. Así que libomptarget no consigue **ligar la entrada
del kernel**, y el mapeo de datos previo sí funciona.

La tabla del host es correcta: sección `llvm_offload_entries` presente y allocated, su
única entrada con `Address` = `region_id` y `SymbolName` = el símbolo del kernel, y la
imagen de device exporta ese símbolo más `.kd` y `.region_id`.

**Lo que falla es el registro.** El binario acaba con **dos descriptores**:

| | rustc, `.rodata` | wrapper, `.data.rel.ro` |
|---|---|---|
| `NumDeviceImages` | **0** | 1 |
| `DeviceImages` | **NULL** | ok |
| `HostEntriesBegin`/`End` | **NULL** / **NULL** | `__start`/`__stop` correctos |

Y `register_offload` en `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs` construye
el nulo **incondicionalmente**, con la forma correcta al lado en un comentario:

```rust
let const_struct = cx.const_struct(&[cx.get_const_i32(0), ptr_null, ptr_null, ptr_null], false);
let omp_descriptor = add_global(cx, ".omp_offloading.descriptor", const_struct, InternalLinkage);
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 0, ptr null, ptr null, ptr null }
```

Y registra ese nulo, y acto seguido llama a `__tgt_init_all_rtls()`, bajo este comentario
que ya está en el fichero:

```rust
// FIXME(offload): Drop this, once we fully automated our offload compilation pipeline,
// since LLVM will initialize them for us if it sees gpu kernels being registered.
```

**No es mezcla de toolchains.** Se reprodujo compilando rustc desde fuente en el mismo
commit del nightly (`f7d782a3b`) y reenlazando sólo con artefactos de ese build. Descartado
además: build viejo, forma del argumento (`*mut` y `&mut` fallan igual), frontera de
función, orden de `.init_array`, y el plugin AMD (libomptarget enumera 4 dispositivos,
reserva en el 0 y saca una tabla de mapeo correcta).

## Dos candidatos de arreglo

**(1) Emitir el descriptor poblado.** Literalmente la forma del comentario: `i32 1`,
`@.omp_offloading.device_images`, y los símbolos `__start`/`__stop_llvm_offload_entries`.
Requiere que rustc conozca el símbolo de las imágenes de device en ese punto — comprueba si
lo tiene o si hay que declararlo como `extern`.

**(2) Que rustc no registre nada** y deje el trabajo entero al `clang-linker-wrapper`, que
ya emite su propio descriptor poblado y su propio ctor. El `FIXME` sugiere que es hacia
donde van ellos. Ojo: al quitar `register_offload` se va también
`__tgt_init_all_rtls()`, y **eso rompió** cuando lo probamos parcheando el binario — el
programa dejó de fallar pero tampoco ejecutó el kernel y no imprimió nada de omptarget. O
sea que si vas por (2), hay que ver quién inicializa los RTLs.

Prueba **(1) primero**: es aditivo y no quita nada que esté sosteniendo el tinglado.

Dato para calibrar: parchear el binario ya enlazado no bastó ni quitando el ctor de rustc
ni reordenando `.init_array`. Eso sugiere que el arreglo tiene que estar en la generación,
no en el enlace.

## Lo que ya está construido y NO hay que repetir

**`~/src/rust-offload/rust`, 9,1 GB.** Es un clon de `rust-lang/rust` en el commit
`f7d782a3be4` — el mismo del nightly instalado — ya configurado con:

```
./configure --enable-llvm-link-shared --release-channel=nightly --enable-llvm-assertions \
  --enable-llvm-offload --enable-clang --enable-lld --enable-option-checking \
  --enable-ninja --disable-docs
```

**LLVM, clang y lld ya están compilados e instalados** (27 minutos que no hay que repetir),
y la mitad host de offload también: `libomptarget.so`, `libLLVMOffload.so`, `libomp.so` y
los plugins de amdgpu y cuda están en `build/x86_64-unknown-linux-gnu/offload/lib/`.

`ninja` no está en el sistema; hay uno en `~/.local/bin/ninja` (1.13.2).

### El único paso que falla, y su causa

`./x build --stage 1 library` se cae en `llvm::OmpOffload`:

```
"-DCMAKE_C_COMPILER=cc" "-DCMAKE_CXX_COMPILER=c++"
"-DCMAKE_C_COMPILER_TARGET=amdgcn-amd-amdhsa" "-DLLVM_USE_LINKER=lld"
...
CMake Error: Host compiler does not support '-fuse-ld=lld'
```

Construye el runtime **de device** (target amdgcn) con **gcc**, que ni apunta a amdgcn ni
encuentra lld. Es exactamente el problema de la PR #161118 (*"we fail to build offload on a
fresh os, since it picks gcc which can't build all of offload"*), así que **es conocido y no
hay que reportarlo**.

**Arreglo sugerido, no probado:** poner `cc`/`cxx` en `bootstrap.toml` apuntando al clang
recién construido —
`build/x86_64-unknown-linux-gnu/llvm/bin/clang` y `clang++` — o un shim en el `PATH` que
llame `cc`/`c++` a esos. Y `ld.lld` en el `PATH`: no hay uno suelto, pero
`~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld`
es el lld del fork de Rust y sirve enlazado con ese nombre.

Si cambiar `cc`/`cxx` dispara una recompilación completa de LLVM, es aceptable: son ~30
minutos. Pero mira antes si se puede evitar.

**Alternativa si el runtime de device sigue sin construirse:** el `libompdevice.a` del
componente `offload` de rustup sirvió para los enlaces manuales. Se puede tomar prestado.

## Cómo probar el parche

Tras reconstruir stage1, engancharlo con `rustup toolchain link` y correr la sonda. Si no
alcanzas el repo de fox, el pipeline mínimo es: pasada `-Zoffload=HostMetadata=`, pasada
`-Zoffload=Device=` con `-Zbuild-std=core --target amdgcn-amd-amdhsa -Ctarget-cpu=gfx1100`,
pasada `-Zoffload=Host=<device.bin>` con `lto = "fat"` en el perfil, y enlace final a mano
con `clang-linker-wrapper` sobre el `host.o`. El wrapper emparejado está en
`build/x86_64-unknown-linux-gnu/llvm/bin/`.

El kernel de prueba escribe `x[i] = i as f32 * 2.0` sobre `[f32; 256]`.

## Restricciones

- **No empujes nada ni abras nada en GitHub.** Si el parche funciona, prepara la rama, el
  commit y el texto del PR y déjalo listo para que yo lo empuje desde mi cuenta.
- El repo de fox no se toca. Esto es trabajo sobre `rust-lang/rust`.
- Si en algún momento te ves reescribiendo cosas de fox o midiendo rendimiento, te has
  desviado: el objetivo es una línea de salida concreta.

## Entregable

Si funciona: el parche, el texto del PR, y la salida de la sonda mostrando `x[42]=84`.

Si no: qué probaste, qué descartaste y dónde se queda ahora — con el mismo nivel de detalle
que el diagnóstico de arriba, que es lo que hace que el siguiente intento no empiece de
cero.
