# Resultados: intentar arreglar el lanzamiento de kernels offloaded

Ejecutado el **2026-08-20**, tarde, siguiendo `rustc-offload-fix-prompt.md`.
Toolchain construido desde fuente en `~/src/rust-offload/rust` (commit `f7d782a3be4`),
enganchado como `rustup toolchain link offload-fix`.

**Criterio de éxito: `x[42]=84`. ALCANZADO** — pero no por el arreglo que se buscaba.

## RESOLUCIÓN: no hay bug de rustc

`x[0]=0 x[42]=84 x[255]=510`, con el **rustc de serie de rustup, sin parchear**, y el
`libomptarget` del componente `offload`. Tres ejecuciones seguidas, y también con
`OMP_TARGET_OFFLOAD=MANDATORY`, o sea que corre de verdad en la GPU.

**Lo que fallaba era el `libhsa-runtime64` del sistema**: el de Ubuntu, 1.11.0 de abril de
2024, que no reconoce gfx1150 y enumera cero GPUs. Sin dispositivos no hay
`TranslationTable`, y el lanzamiento muere con `does not have a matching target pointer`.

Arreglo, sin root, extrayendo de los .deb de AMD igual que con el SDK de Vulkan:

```
curl -sL -o hsa.deb https://repo.radeon.com/rocm/apt/latest/pool/main/h/hsa-rocr/hsa-rocr_1.18.0.70204-93~24.04_amd64.deb
curl -sL -o rpr.deb https://repo.radeon.com/rocm/apt/latest/pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb
dpkg-deb -x hsa.deb root/ ; dpkg-deb -x rpr.deb root/
export LD_LIBRARY_PATH=$PWD/root/opt/rocm-7.2.4/lib:$LD_LIBRARY_PATH
```

Con HSA 1.18 la GPU aparece **nativamente como gfx1150**, sin `HSA_OVERRIDE_GFX_VERSION`.

Control que lo cierra: el **mismo binario** con la HSA vieja del sistema vuelve a fallar
con el error de siempre.

Todo lo que sigue queda como registro de cómo se llegó hasta aquí, con tres diagnósticos
equivocados por el camino.

> **AVISO, escrito al final del día.** Todo lo que sigue hasta la sección «Lo que
> realmente bloquea esta máquina» sigue siendo correcto como observación, pero su
> interpretación es **falsa**. En esta máquina **el plugin AMDGPU de libomptarget no ve
> la GPU**, así que ninguno de los experimentos estaba midiendo lo que creíamos. Lee esa
> sección primero.

Pero el experimento **refuta la hipótesis principal** que teníamos, que es un resultado
más valioso que el que se buscaba, y corrige lo que íbamos a publicar en el issue.

## Resumen en una línea

El descriptor nulo **no es la causa** del fallo. Se eliminó por completo, se corrigió
además un problema de orden de constructores que apareció por el camino, y **el kernel
sigue sin ligarse a su entrada**.

## Los tres experimentos

Los tres sobre el mismo kernel (`x[i] = i as f32 * 2.0` sobre `[f32; 256]`), gfx1100,
enlace final con el `clang-linker-wrapper` emparejado del propio build.

### A — replicar la PR #152777

`register_offload` vaciado por completo: sin descriptor nulo, sin `__tgt_register_lib`,
sin `atexit`, sin `__tgt_init_all_rtls`, sin entrada en `llvm.global_ctors`. Es
literalmente lo que hace la #152777 (`// Let LLVM handle it for us` + `return`).

**Resultado: `x[0]=0 x[42]=0 x[255]=0`.** Sin error, sin salida de omptarget, sin
ejecutar el kernel. Con `OMP_TARGET_OFFLOAD=MANDATORY` sí aborta:
*failure of target construct*.

O sea que **la #152777 tal cual no arregla el bug**, y lo empeora en un sentido: el
fallo pasa de abortar a ser silencioso.

Diagnóstico: sin el registro de rustc, `PM` (el `PluginManager` de libomptarget) nunca
existe. La aserción `PM && "Runtime not initialized"` en `interface.cpp:121` lo confirma
cuando se ejecuta sin pasar por el wrapper.

### B — sólo conservar la inicialización de los RTLs

Igual que A, pero manteniendo un constructor que llama únicamente a
`__tgt_init_all_rtls()`, con la prioridad original 101.

**Resultado: aserción `PM && "Runtime not initialized"` en `interface.cpp:102`**, dentro
del propio `__tgt_init_all_rtls`.

**Hallazgo:** `__tgt_init_all_rtls()` **exige que `PM` ya exista**, y quien lo crea es
`__tgt_register_lib`. La inicialización no puede preceder al registro.

Eso explica por qué el código actual registra un descriptor nulo: **no es un placeholder
gratuito, es lo que hace posible una llamada prematura a `init_all_rtls`.** El nulo es
consecuencia del orden, no la causa del fallo.

### C — inicializar después del registro del wrapper

Igual que B, pero con prioridad de constructor **200** en vez de 101.

El motivo: el ctor del wrapper usa **también 101**
(`llvm/lib/Frontend/Offloading/OffloadWrapper.cpp`, líneas 265 y 632). Con la misma
prioridad, el orden entre ambos **queda indefinido**. Con 200, el de rustc corre después,
cuando `__tgt_register_lib` ya ha creado el `PluginManager`.

**Resultado: el runtime arranca, los datos se mapean, y el kernel sigue sin encontrarse.**

```
omptarget device 0 info: Entering OpenMP data region ... to(unknown)[1024]
omptarget device 0 info: Entering OpenMP kernel ... alloc(unknown)[1024]
omptarget error: Host ptr 0x555555597258 does not have a matching target pointer.
```

Es exactamente el fallo original.

## Lo que queda descartado

Con el experimento C todo lo que la búsqueda necesita está **verificado correcto en el
binario final**, y aun así falla:

| | |
|---|---|
| Descriptores registrados | **uno solo**, el del wrapper. El nulo ya no existe |
| Contenido del descriptor | `NumDeviceImages=1`, y los tres punteros no nulos |
| Rango de entradas | `__start`=`0x4fa38`, `__stop`=`0x4fa70` — 56 bytes, una entrada |
| Sección | `llvm_offload_entries`, presente |
| Entrada: `Version` / `Kind` | `1` / `1` |
| Entrada: `Address` | `0x43258` |
| `region_id` según `nm` | `0x43258` — **coincide** |
| Entrada: `SymbolName` → | `"_RNvC13offload_probe4fill"` |
| Exporta la imagen de device | `_RNvC13offload_probe4fill` y su `.kd` — **coincide** |
| Orden de constructores | registro primero, inicialización después |
| Mapeo de datos | funciona, aparece en la traza |
| Aritmética del puntero fallido | `0x555555554000 + 0x43258 = 0x555555597258` — es el `region_id` |

**Todo cuadra y `getTableMap()` sigue sin encontrar la entrada.** El fallo está más
abajo de lo que suponíamos: no en qué registra rustc, sino en el contenido de la imagen de
device. Ver la comparación con el control de C++, más abajo.

## Consecuencia para el issue

**Hay que corregir `offload-issue-final.md` antes de publicarlo.** Su sección del
descriptor dice, en palabras del autor:

> «Creo que rustc registra un descriptor a ceros y encima inicializa los RTLs desde él,
> lo cual rompe el registro bueno del wrapper»

**Eso queda refutado por el experimento C.** Sin descriptor nulo, con el registro del
wrapper como único registro, y con la inicialización en el orden correcto, el fallo es
idéntico. Publicarlo tal cual habría sido afirmar una causa falsa, que es justo lo que el
propio borrador advertía de no hacer.

La versión honesta ahora es más fuerte, no más débil: se han eliminado tres explicaciones
candidatas por experimento, no por lectura.

## Lo que realmente bloquea esta máquina

Al instrumentar `registerLib` apareció el dato que lo cambia todo:

```
[DBG] AMDGPU  init() OK, 0 dispositivos visibles
[DBG] CUDA    init() OK, 0 dispositivos visibles
[DBG] x86_64  init() OK, 4 dispositivos visibles
```

**El plugin AMDGPU se inicializa sin error y encuentra cero GPUs.** Sin dispositivos no se
crea `TranslationTable`, y `getTableMap` falla — que es el error que llevamos persiguiendo
desde por la mañana.

### La cadena de correcciones que esto obliga

**«libomptarget enumera cuatro dispositivos, reserva en el 0 y saca una tabla de mapeo
correcta».** Escrito esta mañana en `rust-gpu-offload.md` para descartar el plugin de AMD.
**Falso.** Esos cuatro dispositivos son los del plugin **`x86_64`**, o sea CPU. La traza
instrumentada lo enseña sin ambigüedad. El mapeo de datos que «funcionaba» ocurría en
memoria del host.

**«El offload de C++ funciona, luego el fallo está en rustc».** Escrito esta misma tarde.
**Falso.** Ese binario nunca tocó la GPU: OpenMP cayó a la ejecución en host, que da el
resultado correcto y engaña. Con `OMP_TARGET_OFFLOAD=MANDATORY` falla igual que Rust. La
única diferencia real entre clang y rustc es que **clang emite vuelta al host y rustc no**.

**«El descriptor nulo rompe el registro del wrapper».** Refutado por el experimento C, y
ahora se entiende por qué el descriptor nulo existe: `__tgt_init_all_rtls` no puede correr
sin `PM`, y `PM` lo crea `__tgt_register_lib`. Es el arranque del runtime, no un placeholder
inútil.

### Por qué no ve la GPU

El hardware y el driver están bien: `/sys/class/kfd/kfd/topology/nodes/1` existe con
`gfx_target_version 110500`, o sea gfx1150.

Y HSA llega a verla, pero **sólo con el override**. Con un programa mínimo que hace
`hsa_init` + `hsa_iterate_agents`:

```
sin override:  Agent creation failed. The GPU node has an unrecognized id.
               1 agente, 0 GPU
con override:  2 agentes, 1 GPU (gfx1100)
```

El `libhsa-runtime64` del sistema es el de Ubuntu, **1.11.0 de abril de 2024**, y no conoce
gfx1150. No hay ROCm instalado.

Dos cosas se probaron y **no** bastaron:

- **Permisos.** `/dev/kfd` es `root:render` y el usuario sólo está en `video`. Se concedió
  acceso con `pkexec setfacl -m u:$USER:rw /dev/kfd` — el plugin siguió en 0.
- **El nombre de la biblioteca.** El plugin se construye «for dlopened libhsa» y busca
  `libhsa-runtime64.so`, sin versión; en el sistema sólo está `.so.1`, porque el enlace sin
  versión vive en el paquete `-dev`. Se creó el enlace en un directorio propio y se puso el
  primero en `LD_LIBRARY_PATH` — el plugin siguió en 0.

Así que queda una hipótesis por confirmar, y es la que más encaja: **el runtime de HSA de
2024 es demasiado antiguo para el plugin AMDGPU de LLVM 23**, que consulta propiedades por
agente que esa versión no expone, y descarta el agente en silencio.

### Lo que esto significa para todo el hilo

**La afirmación central de la investigación —«el offload de rustc compila pero no
ejecuta»— no está establecida.** Nunca se ha probado en esta máquina contra una pila de
GPU funcional, porque no la hay. Todo lo observado es compatible con que rustc esté bien
y el entorno no.

**El issue no se puede publicar.** Habría reportado a rust-lang un bug del compilador que
aquí es, como mínimo, indistinguible de un problema de la máquina.

Lo que **sí** sigue en pie, porque no depende de la GPU:

- El ejemplo de `core/src/offload.md` llama a funciones inexistentes. Es el PR #161408, ya
  enviado, y no se ve afectado por nada de esto.
- rustc **ignora el retorno de `__tgt_target_kernel`** y no emite vuelta al host. Eso es
  real, verificado leyendo el código generado, y es exactamente por lo que un entorno roto
  produce ceros en silencio en vez de un error. Es reportable por sí solo.
- Que el ctor de rustc y el del `clang-linker-wrapper` compartan **prioridad 101** es una
  fragilidad real de orden, aunque no sea la causa de este fallo.

### Cómo continuar, si se continúa

1. **Instalar ROCm 7.x** y repetir la sonda. Es la única forma de saber si el offload de
   rustc ejecuta. Hasta entonces, ninguna conclusión sobre ejecución es válida.
2. Alternativa: reconectar la NVIDIA discreta y probar por `nvptx64`, que evita HSA entero.
3. Con GPU funcional, volver a correr los experimentos A–D, que entonces sí medirán algo.

## El material del día que sigue siendo válido

## El control que nadie había hecho: C++ OpenMP **sí funciona**

Después de los tres experimentos se hizo lo que faltaba desde el principio: comprobar si
**algún** offload funciona en esta máquina. Un programa mínimo de OpenMP en C++, compilado
con el clang de este mismo árbol y enlazado contra el mismo `libomptarget` y el mismo
`libompdevice.a`:

```cpp
#pragma omp target map(tofrom: x[0:256])
{ for (int i = 0; i < 256; ++i) x[i] = i * 2.0f; }
```

```
x[0]=0 x[42]=84 x[255]=510
```

**Funciona.** Es exactamente el criterio de éxito que el kernel de Rust no alcanza.

Eso descarta de golpe todo lo que no es rustc: la GPU 890M con
`HSA_OVERRIDE_GFX_VERSION=11.0.0` ejecuta kernels offloaded, el plugin de AMD funciona,
`libomptarget` funciona, `libompdevice.a` funciona, y la tubería del
`clang-linker-wrapper` funciona. **El fallo está en lo que emite rustc, y en nada más.**

Y deja algo aún más útil: un **binario de referencia que funciona** contra el que comparar.

Nota de fontanería: hace falta `-nogpulib` (no hay ROCm en la máquina) y pasar el runtime
de device a mano con `-Xoffload-linker -L.../amdgcn-amd-amdhsa -Xoffload-linker -lompdevice`,
más `libLLVM.so` en el `LD_LIBRARY_PATH`.

## La comparación de los dos binarios

### Idéntico: la entrada de host

Decodificando los 56 bytes de `llvm_offload_entries` en ambos:

| campo | C++ (funciona) | Rust (falla) |
|---|---|---|
| `Version` | 1 | 1 |
| `Kind` | 1 | 1 |
| `Flags` | 0 | 0 |
| `Size` / `Data` / `AuxAddr` | 0 / 0 / 0 | 0 / 0 / 0 |
| `Address` | `0x2008` | `0x43258` = el `region_id` |
| `SymbolName` | `__omp_offloading_25_7cb2_main_l5` | `_RNvC13offload_probe4fill` |

**Estructuralmente idénticas.** La entrada de host no es la diferencia.

### Idéntico: la cabecera de la imagen de device

Ambas `kind elf`, `arch gfx1100`, `triple amdgcn-amd-amdhsa`, `producer openmp`, OS/ABI
`AMD HSA`, ABI version 4, y **las mismas flags ELF `0x41, gfx1100`**. Así que la
comprobación de compatibilidad imagen-dispositivo no debería rechazarla.

### **Distinto: lo que hay dentro de la imagen**

Símbolos de cada imagen de device:

```
C++ (funciona)                                Rust (falla)
──────────────────────────────────            ─────────────────────────────
__omp_offloading_..._main_l5                  _RNvC13offload_probe4fill
__omp_offloading_..._main_l5.kd               _RNvC13offload_probe4fill.kd
__omp_offloading_..._main_l5_kernel_environment       (nada más)
__omp_offloading_..._main_l5_dynamic_environment
__omp_rtl_device_environment
__kmpc_target_init
__kmpc_target_deinit
```

**La imagen de Rust lleva el kernel y su descriptor, y nada del entorno que el runtime de
OpenMP espera.** Faltan cuatro cosas: `_kernel_environment`, `_dynamic_environment`,
`__omp_rtl_device_environment` y el par `__kmpc_target_init`/`__kmpc_target_deinit`.

Dos consecuencias localizadas en la fuente de libomptarget:

- `GenericKernelTy::init` (`PluginInterface.cpp:79`) busca `<nombre>_kernel_environment`.
  Su ausencia **no es fatal**: cae a *"default Bare (0) execution mode"*.
- `DeviceTy::loadBinary` (`device.cpp:215-219`) busca `__omp_rtl_device_environment` y, si
  no está, **retorna antes de tiempo**, saltándose la tabla de llamadas indirectas y la
  escritura del entorno de dispositivo. El comentario dice *"This symbol is optional"*.

Y la tabla que después falla sólo se construye si el plugin acepta la imagen, el
dispositivo la acepta y `initializeDevice` tiene éxito (`PluginManager.cpp:226-268`). Si
cualquiera de esos pasos se cae, **no hay `TranslationTable`**, y `getTableMap` devuelve
`nullptr` — que es literalmente el error que se ve.

**Cuál de esos pasos falla es lo que queda por determinar.** Lo que ya no está en duda es
que la causa vive en el contenido de la imagen que emite rustc, no en el registro del host.

## Qué probar después

- **Averiguar cuál de los cuatro pasos de `registerLib` se cae** con la imagen de Rust:
  `isPluginCompatible`, `initializePlugin`, `isDeviceCompatible` o `initializeDevice`.
  Un `printf` en cada uno lo resuelve en una compilación de libomptarget.
- **Hacer que rustc emita `__omp_rtl_device_environment`** y ver si con eso basta. Es la
  hipótesis más barata de probar y la que más encaja con el retorno anticipado de
  `loadBinary`.
- Si eso no basta, **añadir el `_kernel_environment`** y el par
  `__kmpc_target_init`/`__kmpc_target_deinit`, que es lo que emite clang y rustc no.
- El binario C++ que funciona queda como **referencia para diferenciar**: mismo runtime,
  misma GPU, mismo enlazador.

## Estado de la máquina

- `~/src/rust-offload/rust`: contiene el parche del **experimento C** sin commitear, en
  `compiler/rustc_codegen_llvm/src/builder/gpu_offload.rs`. `bootstrap.toml` lleva
  `compress-debuginfo = "off"` añadido, con copia en `bootstrap.toml.bak`.
- `rustup toolchain link offload-fix` apunta al stage1 de ese árbol.
- Pipeline reproducible en el scratchpad de la sesión, directorio `run/`.

### Cómo desbloquear el build, que el prompt daba por imposible

Tres cosas, ninguna necesita root:

1. **`ld.lld` en el `PATH`** — un enlace simbólico al `rust-lld` del toolchain de rustup
   basta. Sólo hace falta para los pasos de offload; **quitarlo después**, porque a
   `std` le hace usar un lld sin zlib.
2. **Borrar el caché de CMake de `build/<host>/offload/build/`** — conservaba
   `CMAKE_C_COMPILER_TARGET=amdgcn-amd-amdhsa` de una pasada de GPU anterior, así que al
   configurar el host le pedía a `cc` compilar apuntando a amdgcn. El mensaje resultante
   (*"Host compiler does not support -fuse-ld=lld"*) manda a investigar en la dirección
   equivocada. Encaja con lo que dice #161118 sobre que amdgcn y nvptx comparten
   directorio.
3. **`compress-debuginfo = "off"`** en `bootstrap.toml` — este árbol pidió
   `LLVM_ENABLE_ZLIB=ON` pero no hay `zlib.h` en el sistema, así que su `ld.lld` rechaza
   `--compress-debug-sections=zlib`, que bootstrap añade al enlazar `std`.

Con eso, `./x build --stage 1 library --warnings warn` completa. El `--warnings warn`
hace falta porque vaciar `register_offload` deja `get_function` sin usar, y el perfil
deniega avisos. **La #152777 se topará con lo mismo.**
