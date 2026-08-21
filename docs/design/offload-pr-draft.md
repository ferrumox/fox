# PR — versión final

Falta una sola cosa: la declaración de LLM del final.

════════════════════════════════════════════════════════════
TÍTULO
════════════════════════════════════════════════════════════

Fix the kernel example in core's offload docs

════════════════════════════════════════════════════════════
CUERPO  (todo lo que sigue, hasta el final)
════════════════════════════════════════════════════════════

The kernel example in `library/core/src/offload.md` calls three names that do not exist
on any target: `thread_idx_x()`, `block_idx_x()` and `block_dim_x()`.

| | `core::arch::amdgpu` | `core::arch::nvptx` |
|---|---|---|
| thread index | `workitem_id_x()`, safe | `_thread_idx_x()`, `unsafe` |
| block index | `workgroup_id_x()`, safe | `_block_idx_x()`, `unsafe` |
| workgroup size | not available | `_block_dim_x()`, `unsafe` |

Renaming is not enough, so this is fixed with a `cfg` shim plus a constant. Verified by
compiling for `gfx1100` and `sm_120`, with no warnings.

The `ignore` stays, because `offload!` does not compile without `-Zoffload=` or
`-Clto=fat`.

r? @ZuseZ4

<!-- homu-ignore:start -->
[AQUÍ VA TU DECLARACIÓN DE LLM, TRADUCIDA]
<!-- homu-ignore:end -->

════════════════════════════════════════════════════════════
FIN DEL CUERPO
════════════════════════════════════════════════════════════

## Notas

- Quitada la línea del `#NNNNN`, porque el PR va antes que el issue. Cuando abras el
  issue, los enlazas con un comentario.
- El título lo puse yo; si quieres otro, se cambia.
- Rama lista: https://github.com/ManuelSLemos/rust/pull/new/offload-doc-example
- Base: `rust-lang/rust:master`. NO pongas `Fixes #`.
