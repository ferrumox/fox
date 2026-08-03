# Design — `n` / `best_of` (multiple completions per request)

Status: **Shipped (0.18)**

> Current per-feature status lives in [`STATUS.md`](../../STATUS.md); the
> comparison vs Ollama/vLLM lives in
> [`vllm-gap-analysis.md`](vllm-gap-analysis.md). This doc records *why*
> multi-completion support was built the way it was, for whoever extends it
> next (shared-prefill forking). Beam search specifically is **not** a
> natural next step here — see `vllm-gap-analysis.md`'s §2 for why it was
> investigated and closed as a deliberate non-goal (2026-08-01) rather than
> left as a future extension.

## Why this shape

`n>1 / best_of / beam search` was `vllm-gap-analysis.md`'s last remaining
"achievable" gap in the advanced-decoding section — a standard OpenAI API
feature (`n`) plus a legacy-but-still-common one (`best_of`) fox had no way to
serve at all. This is purely an **OpenAI-surface** feature: Ollama's
`/api/chat`/`/api/generate` have no equivalent concept, so `src/api/ollama/*`
is untouched.

fox already had everything this needed as *existing, reusable* primitives —
this shipped without any scheduler or KV-cache changes:

- `ChatCompletionChoice`/`ChatCompletionChunkChoice` already carried an
  `index: u32` field inside a `Vec` — the response shape was already
  multi-choice-ready, just always populated with exactly one element.
- Per-token logprobs (`SamplingParams.logprobs`, `Token.logprob`, shipped
  0.14) made `best_of`'s ranking-by-log-likelihood a matter of summing an
  already-computed field, not new engine work.
- `SamplingParams.seed: None` already samples via `rand::thread_rng()`
  (`engine/model/sampling.rs`), so independent requests naturally diverge
  under `temperature > 0` with no extra plumbing.
- The engine's existing disconnect-preemption (`send().is_err()` → preempt,
  frees GPU memory) is reused as-is for admission-failure safety (see below).

## Key decision: independent fan-out, not KV-level forking

vLLM's `n` shares the prompt's KV blocks across branches via copy-on-write and
only diverges at the first sampled token. fox's ref-counted PagedAttention-style
block manager could theoretically support the same trick, but that needs new
block-manager mechanics that don't exist today: forking a live sequence's
blocks mid-generation (not just donating a *completed* sequence's blocks to a
*later* request, which the prefix cache already does), and copy-on-write on
the last partially-filled block once branches start writing different tokens
into it.

**v1 instead submits `n` (or `best_of`) fully independent `InferenceRequest`s
with identical prompt tokens** — correctness-first, zero scheduler/KV
changes, reuses 100% of the existing request lifecycle. Cost: each branch
reprocesses the prompt independently; there's no guaranteed shared prefill
(only an incidental prefix-cache hit if a same-prefix request happened to
complete and donate first). This is the same scope-cut pattern already used
for vision (no multimodal prefix caching) and LoRA (no per-sequence
mixed-adapter batching) — documented honestly, not silently eaten. True
shared-prefill forking is a natural, separable follow-up if the redundant
prefill cost turns out to matter in practice.

## Other decisions worth recording

- **`n`/`best_of` capped at 8** (`sampling_defaults::openai::MAX_N`) — each
  unit is a fully independent generation competing for
  `--max-queue-depth`/`--max-batch-size`, so an uncapped value would let one
  HTTP call monopolize the scheduler for every other concurrent client.
- **`best_of > n` rejected with `stream: true`** (matches OpenAI's own real
  restriction) — ranking needs every candidate's full completion before
  choosing what to show, which is incompatible with incremental streaming.
  Because of this, whenever `stream: true`, validation guarantees
  `effective_best_of == n`, so the streaming paths never need to rank or
  discard a branch.
- **Per-branch seed perturbation**: branch `i` uses `seed.wrapping_add(i)`
  instead of the caller's raw seed (branch 0 keeps the literal seed, so a
  plain `seed` + `n: 1` request is unaffected) — otherwise multiple branches
  would be byte-identical, since the sampler's RNG is a pure function of seed
  and token position, with no per-request salt of its own. At `temperature: 0`
  all branches are still identical regardless of seed — expected, matches
  real OpenAI behavior under greedy decoding, not a bug to work around.
- **`best_of` ranking**: `SamplingParams.logprobs` is forced to `Some(0)`
  internally when `best_of > n` and the caller didn't already request
  logprobs (cheap — just the sampled token's own logprob, no top-k
  alternatives). Branches are scored by total log-likelihood (sum of
  per-token logprobs) and the top `n` are kept
  (`select_best_of` in `api/v1/chat.rs`). `usage.completion_tokens` in the
  response counts only the **returned** `n` branches, not the discarded
  ones — a documented choice: usage reflects what the client received.
- **Admission-failure safety is free**: if branch *k*'s `submit_request`
  fails (e.g. queue full) after branches `0..k` were already admitted, the
  handler returns the error immediately and the already-submitted branches'
  `tx`/`rx` pairs simply drop out of scope. That's exactly the existing
  client-disconnect preemption path (`send().is_err()` → preempt) — no
  separate cancellation mechanism was built for this.
- **Streaming merge via `tokio_stream::StreamMap`**: each branch's
  `UnboundedReceiver<Token>` is wrapped in `UnboundedReceiverStream` and
  inserted into one `StreamMap<usize, _>` keyed by branch index, driven by a
  single `async_stream::stream!` block yielding `(branch_idx, Token)` pairs in
  arrival order — no manual spawn-and-forward tasks. Each SSE chunk carries
  one `ChatCompletionChunkChoice` tagged with the right `index` (same
  one-choice-per-chunk shape fox already emitted for `n:1`, just no longer
  hardcoded to index `0`). Summed usage attaches once, on the very last chunk
  overall (once every branch has signaled `stop_reason`).
- **Tool calling + `n`**: reuses the same buffer-all-branches-then-parse
  approach non-streaming already needs — extended mechanically to loop the
  existing per-branch 2-chunk (role + finish) SSE pattern once per branch
  instead of once total. No new mechanism beyond looping.

## v1 scope cuts (deliberate, not oversights)

- **No shared-prefill forking** — see the key decision above. Each branch
  independently reprocesses the prompt.
- **Beam search stays unimplemented, now a closed non-goal, not a gap** —
  `n`/`best_of` are independent-sample fan-out, not a beam-search decoding
  algorithm. Investigated and formally closed in `vllm-gap-analysis.md`
  (2026-08-01): llama.cpp removed its beam-search API in 2024 and vLLM itself
  demoted beam search out of its fast serving path, so this isn't tracked as
  future work for this feature to grow into.
- **OpenAI-only** — no Ollama equivalent exists to wire this into.

## Where to look

| Concern | File |
|---|---|
| Request fields + validation (`n`, `best_of`, bounds, `best_of ≥ n`, `best_of > n` + `stream` rejection) | `api/types/v1.rs` (`ChatCompletionRequest::validate`) |
| `MAX_N` cap | `api/shared/sampling_defaults.rs` |
| Branch construction, per-branch seed/logprobs, admission-failure safety | `api/v1/chat.rs` (top of `chat_completions`, before the streaming/non-streaming split) |
| `best_of` ranking | `api/v1/chat.rs` (`select_best_of`) |
| Non-streaming fan-out (incl. tool calling) | `api/v1/chat.rs` (the `else` branch, `join_all(branch_rxs...)`) |
| Streaming fan-out, no tools (`StreamMap` merge) | `api/v1/chat.rs` (the plain-streaming branch) |
| Streaming fan-out, with tools (buffered) | `api/v1/chat.rs` (the `has_tools` streaming branch) |
| Legacy `/v1/completions` delegation | `api/v1/completions.rs` |
| Real end-to-end check | `scripts/e2e_smoke.py` check 16 (no opt-in flag — runs on the model every other check already requires) |
