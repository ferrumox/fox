# Design — LoRA adapter support

Status: **Shipped (0.18)**

> Current per-feature status lives in [`STATUS.md`](../../STATUS.md); the
> comparison vs Ollama/vLLM lives in
> [`vllm-gap-analysis.md`](vllm-gap-analysis.md). This doc records *why* LoRA
> support was built the way it was, for whoever extends it next (multi-base-model
> LoRA, per-sequence mixed-adapter batching, hot-reload of adapters).

## Why this shape

llama.cpp's C API for LoRA is a single function that mutates the whole context:

```c
int32_t llama_set_adapters_lora(
    struct llama_context * ctx,
    struct llama_adapter_lora ** adapters,
    size_t n_adapters,
    const float * scales);
```

This is the load-bearing constraint the whole design has to respect: **the active
adapter set is a property of the `llama_context`, not of a sequence.** Unlike
vLLM's punica/SGMV kernels, there is no way for two sequences batched into the
same `llama_decode()` call to use two different adapters. llama.cpp's own
reference server (`tools/server/`) handles this by grouping in-flight requests
by adapter configuration and only batching requests whose configuration
matches, switching the context's active set between groups. fox mirrors that
exactly rather than attempting the (currently impossible, for llama.cpp-backed
inference) per-sequence mix.

## Key decision: one base model, N named adapter aliases

A client selects an adapter the same way it already selects a model: via the
OpenAI/Ollama `model` field — no new, non-standard request field. This mirrors
vLLM's own `--lora-modules name=path` convention and fox's existing
`--draft-model`/`--mmproj` precedent of "one global pairing against whatever the
primary model is."

`--lora-modules name=path.gguf[:scale][,name2=path2.gguf[:scale2]...]`
(`FOX_LORA_MODULES`, parsed by `parse_lora_modules` in `cli/serve.rs`) loads
each adapter onto the primary model (`--model-path`) at startup via
`llama_adapter_lora_init`. `ModelRegistry::resolve_for_request` (`model_registry/mod.rs`)
is the new resolution layer this needed — `EngineEntry` carried no
alias/name metadata before this feature: if the requested `model` name matches
a configured adapter, it resolves to the **primary** model's `EngineEntry` plus
a `LoraSelection { name, scale }`; otherwise it behaves exactly like the
pre-existing `get_or_load`, returning `None`.

**v1 scope cut**: serving LoRA on top of *multiple different* base models
concurrently is out of scope. One `--lora-modules` set applies to one primary
model per server process — a real limitation, not an oversight.

## Group-and-switch: extending vision's split pattern to decode too

Vision already established "partition the batch by a per-request special
requirement, handle each partition separately" in `do_prefill`. LoRA needed the
same pattern in **both** `do_prefill` and `do_decode` — unlike vision (whose
special handling is confined to a request's first prefill turn), a LoRA
selection affects every decode step for a request's whole lifetime.

`llama_cpp/batch.rs`: both functions now partition `requests` by
`req.lora.as_ref().map(|l| (&l.name, l.scale))` (no-adapter is its own group)
via `group_by_lora`, then for each group call `apply_lora_group` —
`llama_set_adapters_lora` with that group's adapter (or an empty array to
clear adapters for the no-LoRA group) — immediately before building and
decoding that group's sub-batch. The original single-batch bodies were renamed
to `do_prefill_batch`/`do_decode_batch` and now operate per-group; OOM
bisection-retry composes unchanged, since it already operates within one
sub-batch.

**Accepted cost, not solved in v1**: llama.cpp marks `sched_need_reserve` on
every adapter switch (a compute-graph re-reservation). Heavy adapter churn
across many concurrently-active adapter configs has a real per-step overhead —
the same tradeoff llama.cpp's own server accepts. Deployments expecting to
serve many adapters with high request-level interleaving should budget for
this; it is not something fox's design fixes, because llama.cpp itself doesn't
expose a cheaper switch primitive.

## Prefix cache: explicit `skip_prefix_cache`, not a free ride

KV computed under one adapter's weights is invalid input for a different
adapter (or no adapter) at the same token positions; reusing it would silently
corrupt generation. Vision got prefix-cache exclusion "for free" because
multimodal requests carry an empty `prompt_tokens` (nothing to hash or match).
LoRA requests carry real text tokens, so this needed an explicit flag:
`InferenceRequest::with_lora()` sets `skip_prefix_cache: true` alongside the
selection. `scheduler/schedule.rs` checks this flag at both existing
prefix-cache call sites — the block-hash lookup (skipped entirely, computing
`Vec::new()` instead) and the donate-on-completion path (an early return) —
rather than reusing vision's "empty vec" trick, since a LoRA request's tokens
are real and must still be scheduled/decoded normally.

## v1 scope cuts (deliberate, not oversights)

- **One base model per `--lora-modules` set** — see above.
- **No per-sequence mixed-adapter batching** — not a fox limitation; llama.cpp's
  C API doesn't support it. A future llama.cpp release exposing a
  punica-style path would be the trigger to revisit this.
- **No hot-reload** — adapters are loaded once at server startup from
  `--lora-modules`; adding/removing an adapter requires a restart, same as
  `--mmproj`/`--draft-model`.
- **Embeddings never apply adapters** — `api/ollama/embed.rs`/`api/v1/embeddings.rs`
  discard the resolved `LoraSelection` (`let (entry, _lora) = ...`); embedding
  requests always use the base model's weights.

## Where to look

| Concern | File |
|---|---|
| CLI flag + parsing | `cli/serve.rs` (`--lora-modules`, `parse_lora_modules`) |
| Adapter loading, `llama_adapter_lora` lifecycle | `engine/model/llama_cpp/mod.rs` (`load()`, `Drop`, `lora_adapter_names`) |
| Alias resolution (`model` field → primary model + selection) | `model_registry/mod.rs` (`resolve_for_request`, `is_lora_alias`) |
| Group-and-switch prefill/decode | `engine/model/llama_cpp/batch.rs` (`group_by_lora`, `apply_lora_group`, `do_prefill`/`do_decode` wrappers) |
| Request-side plumbing | `scheduler/batch.rs` (`LoraSelection`, `with_lora`, `skip_prefix_cache`) |
| Prefix-cache skip | `scheduler/schedule.rs` |
| API-layer wiring | `api/error.rs` (`load_model_or_respond`), `api/v1/chat.rs`, `api/ollama/chat.rs`, `api/ollama/generate.rs` |
| Real end-to-end check (adapter-vs-base, no cross-contamination) | `scripts/e2e_smoke.py` check 15, opt-in via `E2E_LORA=name=/path/to/adapter.gguf[:scale]` |

## Verified against a real model + real adapter (2026-08-01)

Ran the full e2e suite against `bartowski/Qwen2.5-7B-Instruct-GGUF` (Q4_K_M) +
`ggml-org/LoRA-Deepthink-Reasoning-Qwen2.5-7B-Instruct-Q8_0-GGUF` (a real,
independently-trained reasoning-style adapter). **24/24 checks pass**,
including check 15: the adapter measurably changes output on an open-ended
prompt (base gives a plain worked answer; the adapter prepends a structured
`<|thinking|>` chain-of-thought exactly matching its trained style), and every
request across an interleaved base→adapter→base→adapter sequence decodes fully
regardless of which config preceded it.

**A first attempt with a smaller pairing failed instructively**:
`ggml-org/LoRA-Qwen2.5-1.5B-Instruct-abliterated-F16-GGUF` against
`bartowski/Qwen2.5-1.5B-Instruct-GGUF` failed to load —
`llama_adapter_lora_init` correctly rejected it with `LoRA tensor
'output.weight' does not exist in base model`. Root cause: Qwen2.5's small
variants (0.5B/1.5B/3B) tie the output projection to the input embedding, so a
quantized GGUF of the 1.5B base has no separate `output.weight` tensor at all,
while this particular adapter was exported against a variant that does have
one. This is a genuine adapter/base mismatch (pick an adapter trained against
the exact base you're serving, or a base with untied embeddings — 7B+ models
generally aren't tied), not a fox bug — and it confirms `load()`'s error path
surfaces llama.cpp's real failure reason usefully rather than loading silently
wrong.

This run also caught a **check-design bug, not a fox bug**: check 15's first
draft asserted byte-identical output across same-target requests at
temperature 0. A control experiment (two plain base-only requests back to
back, no adapter involved) showed fox's decode is *not* bit-reproducible in
general — prefix-cache hit vs. miss alone takes a different compute path with
different floating-point rounding, which can occasionally flip a near-tied
greedy token. The check was redesigned to assert what the design actually
guarantees (adapter measurably changes output; every request in an interleaved
sequence decodes fully and healthily) instead of an exact-text assumption the
system was never meant to provide.
