# Serving robustness under load (0.13)

fox is correct (0.11) and GPU-capable (0.12). 0.13 makes it hold up as an actual
*server* under concurrent, long-prompt, long-conversation load on the target machine
(AMD Ryzen + Radeon 890M, a few concurrent requests of small models).

Three gaps from the [engine-capabilities checklist](engine-capabilities-checklist.md),
in priority order:

| Gap | Symptom today | Marked |
|-----|---------------|--------|
| **Chunked prefill** | a long prompt is one giant `llama_decode` that head-of-line-blocks every other request's decode until it finishes | 🎯❌ |
| **Context rolling** | when a sequence fills `n_ctx`, `llama_decode` fails and the request is stopped with `Length` instead of continuing | ⚠️ |
| **Template compile cache** | the model's Jinja chat template is re-parsed on every request (`vocab.rs` `TODO(perf)`) | perf |

---

## 1. Chunked prefill (flagship)

### Problem

`do_prefill` (`engine/model/llama_cpp/batch.rs`) submits **all** of a request's prompt
tokens in a single `llama_batch` / `llama_decode`. `build.rs`/`mod.rs` even set
`n_batch ≥ n_ctx` so the whole prompt fits in one pass. The engine loop
(`engine/run.rs`) runs that decode synchronously before it can service anyone else, so
a 4k-token prompt stalls every concurrent request's token generation for the full
prefill. Under load this is the dominant tail-latency source.

### Design

Prefill each request's prompt in **chunks of at most `max_prefill_chunk` tokens per
scheduler step**, interleaved with other requests' decode steps. A request stays in
`Prefilling` across multiple steps until its prompt is fully in the KV cache; only the
step that submits the *final* chunk requests logits and transitions it to `Decoding`.

- New per-request cursor `prefill_pos: usize` — prompt tokens already placed in KV
  (starts at `effective_skip` after a prefix-cache hit). Distinct from the existing
  one-shot `prefilled_tokens` (which records the final total).
- `schedule_step` keeps a `Prefilling` request in `running` and re-emits it in
  `batch.prefill` each step until `prefill_pos == prompt_tokens.len()`. Decodes of
  already-active requests are emitted in the same batch, so generation never stalls
  for more than one chunk.
- `do_prefill` submits only `prompt_tokens[prefill_pos .. prefill_pos + chunk]` with
  absolute positions; `logits = 1` only on the very last prompt token (i.e. only when
  the chunk reaches the end). Non-final chunks return no logits and advance the cursor.
- Config: `--max-prefill-chunk` / `FOX_MAX_PREFILL_CHUNK` / config key, default `512`
  (a good balance for the 890M — small enough to interleave, large enough to keep the
  GPU busy). `0` disables chunking (single-shot, current behavior).

### Invariants (guarded by tests)

- A request's `prefill_pos` advances monotonically and equals `prompt_tokens.len()`
  exactly when it leaves `Prefilling`.
- KV/seq-id conservation continues to hold (reuse the
  `stress_prefix_cache_no_leak` harness pattern; extend with chunked admissions).
- No logits are consumed before the final chunk (no premature sampling).

### Staging

- **S1 — scheduler state machine + config** (stub-testable): `prefill_pos`, the
  `Prefilling`-across-steps loop in `schedule_step`, the flag. Unit tests on the stub
  drive multi-step prefill and assert chunk boundaries + no premature completion.
- **S2 — FFI batch** (real build, Docker/golden-verified): `do_prefill` submits one
  chunk and reports progress; `run.rs` only samples on completion.
- **S3 — validation**: golden/bench that a long prompt no longer stalls a concurrent
  short request (measure inter-token latency of the short request during the long
  prefill).

## 2. Context rolling on full

When `context_len` reaches `n_ctx`, discard the oldest KV window (keep a configurable
head, e.g. the system prompt) via `llama_memory_seq_rm` + position shift, so decode
continues instead of erroring. Recurrent/hybrid caches that can't shift keep today's
"stop with Length" behavior. Config: `--context-shift` (default on for shiftable
caches). Contained to the decode path + KV manager.

## 3. Template compile cache

Build the `minijinja::Environment` (pycompat callback + the model's chat template added
once) at model load and reuse it, instead of re-parsing per request in
`render_chat_jinja`. Store as an owned `Environment<'static>` on the model. Pure perf;
behavior-identical, so the existing golden template assertions cover it.

---

Each item ships independently and is gated by the regression net from 0.12 (golden in
CI + the scheduler conservation stress test).
