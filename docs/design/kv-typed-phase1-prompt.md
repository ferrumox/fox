# Prompt for phase 1 of the typed-KV thesis

Copy everything below the line into a fresh session. Self-contained on purpose.

---

Trabajo de diseño y clasificación en el repo fox, `/home/manuelslemos/Documents/ferrumox/fox`.
Es la **fase 1** de una idea que ya está encuadrada: no hay que re-discutir si merece la
pena, hay que **medir si merece la pena**. Dos días como mucho.

## La idea, en un párrafo

La caché KV de fox es un pool de bloques de tamaño fijo. Las secuencias tienen tablas de
bloques, dos secuencias con prefijo común los **comparten**, y escribir en un bloque
compartido obliga a **copiarlo antes**. Hoy todo eso son reglas de tiempo de ejecución:
`KVCacheManager` te da `allocate`, `free_blocks`, `retain_block`, `copy_on_write`,
`is_shared`, y nada te impide llamarlas mal.

La tesis es llevar esas reglas al **sistema de tipos**, de forma que llamarlas mal deje de
compilar. En un motor esto importa más de lo normal porque los bugs de KV son
**silenciosos**: no revientan, producen tokens equivocados, respuestas truncadas, o
corrompen la conversación *del siguiente usuario*.

**No da velocidad.** Cambia el coste de tocar ese subsistema, que hoy sólo puede modificar
con seguridad quien escribió el scheduler porque las reglas viven en comentarios.

## El diseño a contrastar

```rust
struct Owned<'p>  { … }   // lo poseo yo solo. Escribible.
struct Shared<'p> { … }   // lo referencia alguien más. SIN método de escritura.
```

- `Shared` no tiene `write`: escribir en un bloque compartido **no es un error, es un
  programa que no se puede expresar**.
- `Shared::make_mut(self, pool) -> Owned` — único camino de compartido a escribible.
- `Owned::share(self) -> (Shared, Shared)` — consume el propio: no sigues escribiendo
  después de compartir.
- Ninguno `Copy` ni `Clone`; `Drop` devuelve al pool. No puedes olvidar soltar ni soltar
  dos veces.
- `'p` ata el handle al pool: no puede sobrevivirlo.
- Estados en el tipo para el orden: una secuencia `NeedsTrim` **no tiene** `prefill`;
  `apply_trims` la mueve a `Ready`, que sí.

Límite conocido y aceptado: Rust no tiene tipos dependientes, así que *"este bloque cubre
las posiciones 128..160"* se queda en ejecución. El alcance es **posesión, compartición y
copy-on-write**, no aritmética de posiciones.

Hay una implementación **en runtime** ya escrita y probada en
`crates/fox-offload/src/resident.rs` (rama `research/rust-gpu-offload`, 27 tests, cero
dependencias, compila en stable): `BlockPool`, `BlockTable`, `share_prefix`,
`copy_on_write`, `Residency`. Es el punto de partida, no el resultado.

## La tarea

Coge los bugs de **ciclo de vida del KV** del historial de fox y clasifica **uno a uno**:
*¿el sistema de tipos lo habría rechazado en compilación?*

Candidatos localizados (rama `feature/prompt-cache-determinism`, todos con `git show`):

| commit | qué era |
|---|---|
| `7f735e0` | `trim_sequence` devolvía éxito sin retroceder; reutilización aceptada sobre esa mentira → respuestas vacías en modelos recurrentes |
| `5141b1c` | la admisión expropiaba; livelock + corrupción, en pareja |
| `d5140b0` | peticiones con acierto de caché decodificaban en celdas KV ya ocupadas |
| `3f9b6e8` | prefijo KV donado sin recortar; secuencias sin limpiar al fallar el decode |
| `a4171eb` | donación después de un roll de contexto |
| `42e4416` | la caché de prompts elegía checkpoint por orden de inserción |
| `173c4d7` | crash del prefix-cache en modelos híbridos por `seq_cp` |
| `f41b223` + `aa54463` | contabilidad duplicada del prefijo compartido: 282 bloques donde hacían falta 72 |

**Verifica la lista antes de usarla.** Puede sobrar alguno — `06aaf6d` por ejemplo cerró
una fuga del prefix-cache que resultó ser **un falso positivo**, según `STATUS.md`, así que
no cuenta como bug. Y puede faltar alguno; busca en el historial y en `CHANGELOG.md`.
Descarta lo que sea dimensionado de memoria (`9c8d7d6`, `901b079`) y no ciclo de vida.

Para cada uno quiero: **qué salió mal exactamente** (léete el commit, no el asunto),
**qué tipo o qué ausencia de método lo habría rechazado**, o **por qué no lo habría
rechazado**. Y si sólo lo habría rechazado *como clase* —no ese bug concreto, pero sí la
familia entera— dilo así, que es una respuesta distinta y legítima.

## La regla de decisión, pactada de antemano

Está pactada **antes** de empezar precisamente para que no se pueda racionalizar después:

- **Rechaza 4 o 5 de los ~6-8** → la disciplina se justifica sola. Se hace el refactor, y
  la clasificación queda escrita como sección de evaluación de un posible paper.
- **Rechaza 1** → no compensa. Se descarta habiendo gastado dos días en vez de un mes.
- Entre medias → dilo con esas palabras y expón el argumento en las dos direcciones. No
  redondees hacia arriba.

## Lo que más me importa de todo esto

**Que el resultado pueda ser que no.** Es un ejercicio para matar una idea barata si es
mala, no para justificarla. Hay una estimación previa que hice yo y que te doy **para que
la contrastes, no para que la creas**:

> el orden de los recortes se atrapa entero; el retroceso de recurrentes no como ese bug
> pero sí como clase (`trim` sólo en secuencias `Rewindable`); la contabilidad duplicada
> sólo a medias.

**Si tu clasificación coincide demasiado bien con esa estimación, sospecha de ti mismo y
enséñame el razonamiento por bug.** Y si sale que no compensa, dilo claramente y sin
suavizarlo: eso es el ejercicio funcionando, no fallando.

## Restricciones

- **No hace falta GPU, ni que el offload de rustc ejecute.** Esto es un sistema de tipos.
  Si algo te lleva hacia el offload, te has desviado.
- **No empieces el refactor.** La fase 2 depende de este resultado. Como mucho, escribe los
  tipos lo justo para comprobar que compilan y que rechazan lo que dices que rechazan —
  un test que **no** compile es evidencia válida aquí (`compile_fail` en un doctest, o
  `trybuild`).
- No toques `src/kv_cache/` ni el scheduler.
- `make ci` tiene que seguir verde.

## Entregable

Un documento en `docs/design/` con la tabla de clasificación, el razonamiento por bug, el
veredicto contra la regla, y —si sale que sí— qué tendría que cambiar en la API de
`KVCacheManager`. Más los tipos mínimos en `crates/fox-offload/src/resident.rs` que
sostengan lo que afirmes.

Y dime al final si algo de lo que te di como encuadre no se sostuvo al mirarlo de cerca.
