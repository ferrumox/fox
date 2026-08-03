# Design — Reactive context-rolling on OOM

Status: **Shipped (0.18)**

> Current per-feature status lives in [`STATUS.md`](../../STATUS.md); the
> comparison vs Ollama/vLLM lives in
> [`vllm-gap-analysis.md`](vllm-gap-analysis.md). This doc records *why* the
> fix was built the way it was, and a more severe, unrelated crash bug real
> concurrent-load testing found along the way.

## Why this shape

Two KV-pressure mechanisms already existed, shipped separately, and had never
been connected:

1. **0.16's decode-time OOM bisection retry** (`bisection_split`,
   `engine/model/llama_cpp/batch.rs`): when `llama_decode` returns `1` ("no
   KV slot for batch"), the batch is recursively halved and retried. When it
   bottoms out at a single request and *still* returns `1`, it gave up with a
   plain error string — the engine layer force-finished every request in that
   step's batch with `StopReason::EngineError`, without ever asking whether
   the one truly-failing request could have freed space by discarding old
   context.
2. **0.13's proactive context-shift** (`roll_full_contexts`, `engine/run.rs`,
   called before every decode step): scans all decoding requests and, for any
   whose `context_len()` has reached `n_ctx - reserve`, discards its oldest
   tokens (`model.roll_context` + `scheduler.record_context_roll`) to keep
   generating past the context window. Purely preventive — never fires in
   reaction to an actual failure.

**Why they'd never been connected, architecturally**: `roll_context` lives on
`LlamaCppModel`, which has no reference to `Scheduler` and no
`--context-shift` config — those live one layer up, on `InferenceEngine`. The
bisection-retry failure, however, was detected deep inside `batch.rs`'s
recursion, which only returned a generic `anyhow::Error` string — the engine
layer had no way to know *which* request was the one that finally,
unrecoverably failed, so it couldn't target a roll at it even if it wanted to.

Confirmed by reading the KV/scheduler bookkeeping directly: fox's block-level
accounting (`PageTable`, `KVCacheManager` ref-counts) reserves a request's
full worst-case lifetime (`prompt + max_new_tokens`) once, at admission, and
never touches it again — rolling a request's context (as
`record_context_roll` already proved) only ever changes one `usize` field
(`rolled_tokens`) consumed by `context_len()`. **There was no block/page_table
staleness risk to solve** — the only real gap was plumbing a specific request
id (and the decision of *whether* to roll) from the model layer up to the
engine layer that already knew how to roll safely.

## What shipped

1. **A typed error carries the failing request id up**, instead of a generic
   string: `KvCacheFullAtMinimum { req_id }` (`engine/model/mod.rs`).
   `do_decode_batch`'s bisection-exhausted branch
   (`engine/model/llama_cpp/batch.rs`) returns it specifically when `ret == 1`
   at batch size 1 — other fatal codes (`2` aborted, `< -1`) stay generic,
   unretryable errors, unchanged.
2. **The retry lives in `run_loop`, not inside `batch.rs`** — the one place
   with both `self.scheduler` and `self.context_shift` in scope, matching
   where `roll_full_contexts` already lives. On a decode `Err`,
   `try_reactive_roll` checks `e.downcast_ref::<KvCacheFullAtMinimum>()`; if
   present *and* context-shift is enabled/supported *and* the specific
   request has enough context left to discard meaningfully
   (`reactive_roll_amounts`, a small pure function separate from
   `roll_full_contexts`'s own inline math — deliberately not shared, to avoid
   risking the already-shipped proactive path's behavior), it performs **one**
   targeted roll and the caller retries `run_decode` **once** for the full
   original batch (not just the freed request — a successful roll may let the
   whole batch succeed, and `run_decode` always re-fetches fresh scheduler
   state, so no manual request-list rebuilding was needed). Capped at exactly
   one reactive attempt per failing step.
3. **Decode-only, prefill unchanged** — mirrors `roll_full_contexts` itself
   only ever running before decode. A request still ingesting its own prompt
   has no generated content to discard, and a too-long prompt is a
   scheduling-level problem chunked prefill already addresses.
4. **Reuses `roll_context`/`record_context_roll` verbatim.**

In practice, this is a narrow safety net, not a routinely-exercised path:
`roll_full_contexts`'s existing per-request threshold (`ctx_len >= n_ctx -
reserve`, where `n_ctx` here is the *per-sequence* configured context length,
not the aggregate llama.cpp capacity) already keeps each sequence's growth
within its fair share, so aggregate exhaustion across concurrently-decoding
sequences is uncommon under normal load — confirmed live: heavy concurrent
load (9 simultaneous requests against a deliberately tiny `--max-context-len
64`) triggered dozens of *proactive* rolls and zero reactive ones. The
reactive path exists for the residual cases the proactive threshold doesn't
perfectly cover (timing, non-uniform per-token KV footprint across
architectures, the `reserve` headroom calculation not accounting for every
possible next-step shape) — a last-resort degrade step, not the common case.

## A more severe, unrelated bug found by the same testing

Stress-testing this feature with real concurrent load (9 simultaneous
requests, `--max-context-len 64`, `--max-batch-size 8`) crashed the **entire
server process** — not a graceful per-request failure:

```
llama-context.cpp:1748: GGML_ASSERT(n_tokens_all <= cparams.n_batch) failed
```

Root cause: `--max-prefill-chunk` (default 512) caps how many tokens of **one
request's own prompt** are submitted per prefill call, but several requests
admitted into the *same* scheduler step each contribute their own chunk to
the *same* shared `llama_decode` call, and their **sum** is what `n_batch`
actually bounds. Nothing capped that sum. Concretely: 9 requests × 36-token
prompts = 288 tokens submitted in one call against `n_batch = 64` (`n_batch =
max(effective_max_ctx, max_batch_size)`, shrunk down here by a small
`--max-context-len`). The same failure mode is also reachable with a
**single** request whenever `max_prefill_chunk` itself exceeds `n_batch` —
concurrency isn't required, just a small enough `--max-context-len`.

Unlike `ret == 1`, `GGML_ASSERT` has **no graceful return code** — it aborts
the process via `abort()`, which bisection retry cannot catch or recover
from. This made it strictly more severe than what reactive context-rolling
addresses: a full-process crash reachable by legitimate concurrent load,
plausible on exactly the small-VRAM hardware fox targets (a small
`--max-context-len` is a normal choice there).

**Fix** (`engine/model/llama_cpp/batch.rs`, `do_prefill_batch`): query
`llama_n_batch(ctx)` (already bindgen-exposed, zero new FFI work) and
allocate its budget across the group's requests in submission order via a new
pure function, `allocate_batch_budget(desired_lens, n_batch)` — once
exhausted, later requests in that call submit zero tokens this step; their
`prefill_pos` doesn't advance, so the scheduler re-offers them next step,
exactly the mechanism a single request's own multi-step chunking already
relied on. This never changes correctness (every request eventually gets
prefilled, just possibly spread across one more step under contention) and
applies uniformly whether the aggregate-vs-`n_batch` mismatch comes from one
request or many.

## v1 scope cuts (deliberate, not oversights)

- **Speculative decoding's draft-verify batch** (`draft_len + 1` tokens for a
  single request) isn't covered by the `n_batch` fix above — it's a
  narrower, pre-existing theoretical risk (would need an unusually small
  `n_batch` combined with a large custom `--spec-draft-len`) not implicated
  in the crash this session reproduced, left as a known residual limitation
  rather than expanding scope further.
- **Reactive rolling doesn't rank which request to roll** when a step's
  bisection narrows to different single requests across retries — it only
  ever sees the one request `batch.rs` already isolated by construction, so
  there's no ranking decision to make in v1.

## Where to look

| Concern | File |
|---|---|
| Typed error | `engine/model/mod.rs` (`KvCacheFullAtMinimum`) |
| Signaling it from bisection-exhausted decode | `engine/model/llama_cpp/batch.rs` (`do_decode_batch`) |
| Reactive roll-and-retry | `engine/run.rs` (`try_reactive_roll`, `reactive_roll_amounts`, `run_loop`'s decode branch) |
| `n_batch` aggregate-overflow fix | `engine/model/llama_cpp/batch.rs` (`do_prefill_batch`, `allocate_batch_budget`) |
| Existing, unmodified proactive mechanism | `engine/run.rs` (`roll_full_contexts`), `engine/model/llama_cpp/mod.rs` (`roll_context`) |
