# Design — MLA / recurrent KV sizing

Status: **Shipped (0.18)**

> Current per-feature status lives in [`STATUS.md`](../../STATUS.md); the
> comparison vs Ollama/vLLM lives in
> [`vllm-gap-analysis.md`](vllm-gap-analysis.md); the broader "stop computing
> memory ourselves" rework this closes part of is
> [`model-architecture-rework.md`](model-architecture-rework.md) §4.2. This doc
> records *why* the fix was built the way it was, and a real bug real-model
> verification found along the way.

## Why this shape

fox capped the llama.cpp context's `n_ctx` at load time with a hand-rolled
formula: `bytes_per_token = 2 * n_head_kv * head_dim * 2(f16) * n_layer`. This
is only correct for plain positional attention:

- **MLA (DeepSeek-V2/V3, GGUF arch `deepseek2`/`deepseek2-ocr`)** compresses KV
  into a small latent vector (GGUF key `<arch>.attention.kv_lora_rank`, only
  present on MLA models) — the formula massively **over-estimates** bytes/token,
  capping `n_ctx` far smaller than the model actually needs.
- **Recurrent/hybrid (Mamba, RWKV, Jamba, LFM2)** have no per-token KV at all —
  state is roughly constant per sequence. The formula's inputs aren't
  meaningful here at all.

This was **partially** already fixed before this task: the block *pool*
already followed llama.cpp's real `llama_n_ctx()` after context creation
(`Model::kv_cache_capacity()`, wired into the real serving paths). What
remained broken was the **pre-creation cap** — the formula still ran *before*
`llama_init_from_model` was ever called, so a bad guess could shrink `n_ctx`
needlessly (MLA) or produce a number that just happened to be whatever
(recurrent) before llama.cpp ever got a say.

## Key decision: ask llama.cpp by trying, don't predict

`llama_init_from_model` already returns a null pointer on allocation failure,
and fox already turned that into a clean error. Instead of predicting whether
`n_ctx` tokens fit, fox now **asks by trying**: attempt context creation at the
full desired `n_ctx` (`effective_max_ctx * n_seq`, not pre-capped by any
per-token formula); on failure, halve `n_ctx` (floor: `effective_max_ctx`, the
minimum viable single-sequence context) and retry; if even the floor fails,
surface a clear load-time error. This is the exact *shape* of the 0.16
decode-time OOM bisection retry (`bisection_split` in `batch.rs`, triggered on
`llama_decode` returning `ret==1`), applied one layer earlier, at context
creation (`shrink_n_ctx` in `engine/model/llama_cpp/mod.rs`). No per-architecture
branching anywhere — MLA models simply succeed at a larger real `n_ctx` than
the old formula would have allowed; recurrent models are no longer sized by a
formula whose inputs don't apply to them at all.

`--gpu-memory-fraction` keeps its documented, user-facing meaning: the old
formula still runs, but only as a soft **ceiling** on the first attempt (avoid
an obviously wasteful huge first try on constrained hardware), not a precise
predictor. Being conservative for MLA under a tight memory budget (the
ceiling overestimates MLA's real bytes/token, so it may request less than
MLA could actually use) is an accepted v1 limitation — documented, not silently
eaten. The bug this closes is *crashing/mis-sizing*, not *squeezing out the
last usable token of headroom*.

## `KvMemoryClass` — lightweight, fact-derived classification

Not the full `ArchClass`/`KvModel` enums `model-architecture-rework.md` §4.1
sketched — just enough for observability (`fox probe`, load-time logs):

```rust
pub enum KvMemoryClass { Standard, Latent, Recurrent }
```

- `Latent` when `arch_name` (the model's own declared `general.architecture`)
  is `"deepseek2"` or `"deepseek2-ocr"` — confirmed exact identifiers in
  `vendor/llama.cpp/src/llama-arch.cpp`; DeepSeek-V3 reuses the `deepseek2`
  tag, there is no separate `deepseek3`. This is matching the model's own
  authoritative identity against a known table, the same kind of fact-lookup
  `arch_name` itself already is — not a content-sniffing heuristic.
- `Recurrent` when `!supports_seq_copy` — see the bug below for what that
  actually checks now.
- `Standard` otherwise.

## A real bug, found by testing against a real model

The plan for this task assumed recurrent detection already worked correctly —
`Model::supports_seq_copy()` was implemented via `llama_memory_can_shift`,
and STATUS.md dated this to a "historic fix (v0.3.1)". **Verification against
a real Mamba model (`Felladrin/gguf-mamba-130m-hf`, 2026-08-01) proved this
premise wrong**: `fox probe` reported `KV seq-copy: yes` for a genuine
recurrent model, and `fox serve`'s startup log confirmed prefix caching was
being **enabled**, not disabled, for it.

Root cause, found by reading llama.cpp's own source
(`vendor/llama.cpp/src/llama-memory-recurrent.cpp`):

```cpp
bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}
```

`llama_memory_can_shift` returns `true` for recurrent memory — not because
fox's block-level KV copy-on-write (the prefix cache's actual mechanism) is
valid for it, but because *repositioning* happens to be a cheap no-op for
recurrent state. Fox's existing code conflated two different llama.cpp
concepts: "can positions be shifted" (what `can_shift` literally answers) and
"does fox's sequence-copy-based prefix cache apply to this architecture"
(what `supports_seq_copy` needs to answer) — for standard attention these
happen to coincide, which is presumably why the original v0.3.1 fix looked
correct at the time; they diverge for recurrent models, and nothing caught it
because no real recurrent model had ever been run through fox end-to-end
before this task.

**Fix**: `supports_seq_copy()` now uses `llama_model_is_recurrent`/
`llama_model_is_hybrid` — public llama.cpp APIs that answer the
architecture-level question directly (`llama.h`, confirmed available via the
existing `allowlist_function("llama_.*")` bindgen wildcard, zero new
FFI/build-system work). `roll_context()`'s existing `llama_memory_can_shift`
check is **left alone, with the same model-level check added defensively
alongside it** (not instead of it) — for context-*rolling* specifically,
`can_shift` genuinely does answer the right question (can llama.cpp's
position-shift primitive be used at all), and per-cache-type values elsewhere
confirmed this (see below); the recurrent case was the one architecture where
"trivial to shift" and "safe for fox's prefix-cache/rolling assumptions"
diverge, so the added guard is specifically about not relying on `can_shift`
alone in the one place it was proven misleading.

## A second, correctly-detected llama.cpp limitation (not a fox bug)

Verifying against `mradermacher/DeepSeek-V2-Lite-GGUF` (real MLA) surfaced a
*different*, genuine llama.cpp limitation — not something to fix in fox.
`llama_kv_cache_dsv4::get_can_shift()` (the specialized cache class MLA models
use):

```cpp
bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}
```

Context-rolling is not yet implemented upstream for MLA's compressed cache.
fox's **existing, unmodified** `roll_context()` check already detects this
correctly via `llama_memory_can_shift` and fails the individual request
cleanly (`finish_reason: "error"`) rather than corrupting output or crashing —
verified directly: a context-filling e2e request against DeepSeek-V2-Lite
produced a clean error after 1 token, and the *next* request on the same
server succeeded normally (the disconnect-recovery e2e check passed). This is
llama.cpp's own documented TODO, out of fox's control; not addressed here.

## Real-model verification (2026-08-01)

**Recurrent** — `Felladrin/gguf-mamba-130m-hf` (Mamba, 130M, added to
`registry.json` as `mamba-130m`): `fox probe` reports `KV memory class:
recurrent`, `KV seq-copy: no (prefix cache disabled)`; `fox serve`'s log
confirms prefix caching disabled; context creation succeeds (the byte-budget
formula degenerates to `bytes_per_token = 0` for a model with `n_head_kv = 0`,
falling through to the existing "use the full desired ctx" branch — no
regression, no artificial cap); a real chat completion decodes successfully.
`make e2e` on this fixture: **18/22 pass**. The 4 failures (checks 9-12,
generation stopping a few tokens after `min_tokens` should have suppressed
EOS) trace to this specific community GGUF conversion's vocab metadata —
`eog_tokens` is derived purely from `llama_vocab_is_eog` across the vocab
(`engine/model/llama_cpp/mod.rs`), a property of the file's own metadata,
untouched by this task's changes — not a KV-sizing or recurrent-detection
regression.

**MLA** — `mradermacher/DeepSeek-V2-Lite-GGUF` (DeepSeek-V2-Lite, 16B total /
2.4B active MoE, added to `registry.json` as `deepseek-v2-lite`): `fox probe`
reports `KV memory class: latent (MLA)`, correctly flags the expected
head_dim/n_embd "contradictions" (192 vs. formula's 128; MLA's compressed
geometry doesn't satisfy `n_embd == n_head*head_dim` — this is `fox probe`'s
pre-existing contradiction detector doing exactly its job). Context creation
succeeds on the **first attempt** at the model's full trained context
(163,840 tokens) — no retry needed. `make e2e`: **20/22 pass**; the 2 failures
are the DSV4 shift limitation above, a real llama.cpp gap correctly and
safely detected, not a crash.

## v1 scope cuts (deliberate, not oversights)

- **No full `ArchClass`/`KvModel` struct** from `model-architecture-rework.md`
  §4.1 — `KvMemoryClass` is a lighter, purpose-built enum for this task.
  P3/P4 of that doc's migration plan remain unaddressed.
- **`--gpu-memory-fraction` ceiling still uses the old formula** as a
  heuristic starting guess — can be conservative for MLA under tight memory,
  never incorrect/crash-prone (see key decision above).
- **DSV4 (MLA) context-rolling** is a real llama.cpp-side gap, not fixed here
  (out of fox's control) — correctly detected and fails cleanly.

## Where to look

| Concern | File |
|---|---|
| `KvMemoryClass` + classifier | `engine/model/model_info.rs` |
| Empirical context-size retry loop, `shrink_n_ctx` | `engine/model/llama_cpp/mod.rs` (`load()`, near `resolve_context_len`) |
| Recurrent/hybrid detection fix | `engine/model/llama_cpp/mod.rs` (`supports_seq_copy`, `roll_context`) |
| `fox probe` output | `cli/probe.rs` |
| Real end-to-end verification fixtures | `registry.json` (`mamba-130m`, `deepseek-v2-lite`) |
