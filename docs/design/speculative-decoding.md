# Speculative decoding (0.15)

0.15 makes decoding *faster* by verifying several tokens per forward pass instead of
one. From the [vLLM gap analysis](vllm-gap-analysis.md) this is the largest real latency
win still within reach of a llama.cpp wrapper.

## The load-bearing facts

1. **Decode is memory-bandwidth-bound at batch-of-1.** Each generated token is one
   `llama_decode` that streams the whole model through memory to produce a single token.
   The hardware could verify *many* candidate tokens in essentially the same pass — the
   weights are already loaded. When the output echoes the context (code, JSON, RAG,
   repetition), the next few tokens are often guessable for free.

2. **llama.cpp's speculative helpers live in `common/`, which fox does not build** —
   `common/speculative.{h,cpp}`, `ngram-cache`, `ngram-map`. So, exactly like the
   JSON-schema→GBNF converter, fox implements the glue in Rust on top of the **core**
   primitives it already uses: a multi-token `llama_batch` + `llama_decode`, per-position
   `llama_get_logits_ith`, and `llama_memory_seq_rm` for KV cleanup.

3. **A draft-*model* needs a second model loaded**; **n-gram / prompt-lookup needs none.**
   For fox's niche (one static binary, consumer laptop), the first step is the
   model-free **n-gram / prompt-lookup** variant — self-contained, no second model to
   load or manage. A draft model is a later extension (see §5).

---

## 1. N-gram / prompt-lookup speculation (flagship)

### The idea

Guess the next few tokens by finding where the recent output has occurred **before** in
the same sequence, then verify all the guesses in one target-model pass.

- **Propose.** The request's tokens so far are `seq = prompt ++ generated`. Take the
  last `ngram` tokens (the suffix). Find the most recent earlier occurrence of that
  n-gram in `seq`; the up-to-`draft_len` tokens that *followed* that occurrence are the
  draft. (No match → no draft → fall back to an ordinary 1-token decode.)
- **Verify.** Submit `[last_token, draft₁, …, draft_k]` as **one** `llama_decode` batch
  at consecutive positions, requesting logits at every position. Position *i*'s logits
  are the target's true distribution for the token that follows `draft₁..draftᵢ`.
- **Accept.** Sample the target at position 0 → `t₀` (a genuine target sample). If
  `t₀ == draft₁`, accept and move to position 1 (whose logits are valid, since they were
  conditioned on `draft₁ == t₀`); otherwise emit `t₀` and stop. Repeat. One step commits
  `a + 1` tokens where `a` is the number of accepted drafts.
- **Clean up.** The batch wrote `k + 1` KV cells but only `a + 1` are kept; remove the
  rejected tail with `llama_memory_seq_rm(seq, kept_end, -1)` (same mechanism as context
  rolling). Advance `context_len` by `a + 1` and stream the committed tokens.

### Why it's exact (the key invariant)

Sampling the target at each batch position in order and accepting only on a match yields
**exactly** the target model's distribution — greedy *and* stochastic. Every emitted
token is a genuine target sample conditioned on the accepted prefix; a mismatch just
means the extra positions we computed are discarded. With per-position RNG seeded on the
correct `token_count` (`seed ^ (generated_so_far + i)`), a **seeded** speculative run is
byte-identical to the non-speculative run. That equality is the golden test.

The only effect of speculation is *speed*: on a hit the engine advances several tokens in
one `llama_decode`; on a miss it costs one decode plus a little wasted compute for the
rejected drafts.

### Interactions

- **Guided decoding (grammar).** A drafted token that the grammar forbids would be
  rejected anyway. v1 simply **disables speculation while a grammar is active**; a later
  version can intersect the draft with the grammar. (Guided output is usually the case
  that least needs the speed, so this is a fine first cut.)
- **min_tokens.** Speculation must not commit an end-of-generation token below the floor;
  reuse the `eog_tokens` mask during verification. A drafted EOG below the floor is
  treated as a mismatch.
- **logprobs.** Each *committed* token still has the target's per-position logits in the
  same batch, so `logprobs` are reported from those — no change to the numbers.
- **Prefill / prefix cache / chunked prefill.** Speculation is a decode-phase change and
  is orthogonal to all of them.

### KV / scheduler notes

- The verify batch needs `draft_len + 1` positions of headroom beyond `context_len`.
  Block reservation is already `prompt + max_new_tokens`; confirm the peak (pre-cleanup)
  never exceeds the reserved blocks, and cap `draft_len` accordingly.
- A speculating request advances by `a + 1` tokens per scheduler step instead of 1;
  `context_len`, `generated_tokens`, and `generated_token_ids` all move by `a + 1`. Stop
  conditions (EOS, stop strings, max_tokens) are checked against each committed token in
  order, truncating the commit at the first stop.

### Staging

- **S1 — propose + verify + cleanup, single request** ✅ (real build, golden-verified):
  `engine/speculative.rs::propose_ngram` (pure, unit-tested) does the n-gram lookup;
  `LlamaCppModel::do_speculative_decode` builds the multi-token verify batch, samples the
  target at each position with a growing penalty context (so it's exact under penalties),
  accepts on match, and `seq_rm`s the rejected KV tail. Golden
  `golden_speculative_matches_greedy`: on a repetitive prompt the committed tokens are
  byte-identical to a plain decode loop and the acceptance count is > 0. Model-level only
  — the scheduler/engine wiring is S2.
- **S2 — engine / continuous-batching integration** ✅: `run_decode` takes the
  speculative path for a **single** decoding request with no grammar when the engine is
  built with `Some((ngram, draft_len))` (multi-request batches decode normally —
  speculation helps most at low concurrency, and this sidesteps mixed-length batch
  bookkeeping). `speculative_decode_sync` on the `Model` trait (default = plain 1-token
  decode; stub override commits 2 tokens to exercise the path). `handle_logits` now
  processes a per-request **list** of committed tokens, checking stop conditions
  (EOS / stop strings / max_tokens) per token in order and truncating the commit at the
  first stop; `min_tokens` EOG masking is applied inside the verify sampler. Config
  plumbed: `--speculative` / `--spec-ngram` / `--spec-draft-len` on `fox serve` (+ env +
  config file), `--speculative` on `fox run`. Integration test
  `test_speculative_engine_commits_multiple_tokens_and_respects_max` covers the
  multi-token commit path on the stub.
- **S3 — metrics** ✅ (config already landed with S2): `speculative_decode_sync` reports
  how many drafts it proposed; the engine accrues lifetime `spec_proposed`/`spec_accepted`
  atomics (exposed via `spec_stats()`), and the run-loop metrics block propagates deltas
  to Prometheus — `ferrumox_spec_tokens_proposed_total`,
  `ferrumox_spec_tokens_accepted_total`, and the `ferrumox_spec_acceptance_ratio` gauge.
- **S4 — validation bench** ✅: `fox bench-spec` runs the same greedy generation with
  speculation off and on, on a repetitive workload (n-gram lookup's best case) and a
  prose workload (closer to its worst case), reporting tok/s, acceptance ratio, and the
  speedup — and **asserts the off/on outputs are identical** (the exactness invariant,
  checked end-to-end through the engine, not just at the model layer).

### Invariants (test-guarded)

- Seeded speculative output == non-speculative output, byte-for-byte (greedy and
  stochastic).
- `context_len` / `generated_tokens` advance by exactly `accepted + 1` per step.
- No KV leak: rejected draft cells are removed; the conservation stress test still holds.
- Never commit an EOG below `min_tokens`, nor a grammar-illegal token.

---

## 2. Config

- `--speculative` / `FOX_SPECULATIVE` (default `false`) — opt-in for 0.15.
- `--spec-ngram` / `FOX_SPEC_NGRAM` (default `2`) — suffix length matched against history.
- `--spec-draft-len` / `FOX_SPEC_DRAFT_LEN` (default `4`) — max tokens proposed per step.

---

## 3. Metrics

- `fox_spec_tokens_proposed_total`, `fox_spec_tokens_accepted_total` (counters).
- `fox_spec_acceptance_ratio` (gauge) — accepted / proposed, the headline health number.

---

## 4. Why not draft-model first

A draft model (a small model proposing for a big one) generalizes beyond context-echoing
text, but it needs a **second model** resident in memory and a second decode per step,
and it complicates loading/eviction/VRAM budgeting — all against fox's single-binary,
consumer-hardware niche. N-gram lookup captures the most common local wins (code, JSON,
RAG, repetition) with zero extra memory. Draft-model speculation is a clean **later**
extension once the verify/accept/cleanup machinery from §1 exists — it reuses all of it,
swapping the n-gram proposer for a draft-model proposer.

---

Each stage ships independently and is gated by the regression net (the greedy-equality
golden + the scheduler conservation stress test). Speculation is **off by default** in
0.15; turning it on must never change *what* fox produces, only how fast.
