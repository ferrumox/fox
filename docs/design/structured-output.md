# Structured & controllable output (0.14)

fox is correct (0.11), GPU-capable (0.12) and holds up under load (0.13). 0.14 makes
its **output controllable** — the top cluster of "achievable" gaps from the
[vLLM gap analysis](vllm-gap-analysis.md): apps can't get *reliable* JSON, can't read
token probabilities, and can't reach a handful of standard sampling knobs.

Three gaps, in priority order:

| Gap | Symptom today | vLLM |
|-----|---------------|------|
| **Guided decoding (GBNF / JSON schema)** | JSON mode is best-effort prompting — the model can and does emit invalid JSON | ✅ (outlines / xgrammar) |
| **logprobs / top_logprobs** | `logprobs` requests are ignored; no way to score tokens for evals/classification | ✅ |
| **Missing sampling** (min_p, logit_bias, min_tokens) | silently dropped from requests | ✅ |

The enabling fact: **llama.cpp ships a proven GBNF grammar engine in the *core* library**
(`llama_sampler_init_grammar`), so guided decoding is wireable through fox's existing FFI
— no kernel work, no new dependency. The one piece that is *not* in the core is the JSON-
schema→GBNF converter (it lives in `common/`, which fox deliberately does not build), so
that conversion is ours to write in Rust.

---

## 1. Guided decoding via GBNF (flagship)

### Problem

fox samples in **pure Rust** from a raw `&[f32]` logits slice
(`engine/model/sampling.rs::sample_token`): penalties → temperature → top-K → top-P →
weighted draw. There is no way to constrain the token set to a grammar, so a request
that needs JSON gets a prompt-engineered hint and hopes. Under any real load that
produces invalid JSON often enough to break callers.

### Design

Reuse llama.cpp's GBNF engine instead of reimplementing it. A grammar is a **stateful
per-request sampler**: `llama_sampler_apply` masks every token the grammar forbids at
the current position (sets its logit to `-inf`), and `llama_sampler_accept` advances the
grammar state once a token is chosen. fox keeps its Rust sampler for
temperature/top-K/top-P/penalties — **the grammar only constrains the candidate set**;
normal sampling then picks within it.

- New field `grammar: Option<Arc<str>>` on `SamplingParams` (and threaded onto
  `InferenceRequestForModel`). `None` = today's unconstrained path, byte-for-byte.
- The model owns per-request grammar samplers in a `DashMap<u64, GrammarSampler>`, where
  `GrammarSampler` is a `Send + Sync` newtype over the raw `*mut llama_sampler` (access
  is already serialized by the `_ctx` mutex that `do_prefill`/`do_decode` hold).
- **Sampling flow** when a request has a grammar (inside `do_prefill`'s final chunk and
  every `do_decode`, right before `sample_token`):
  1. Lazily create the grammar sampler for `req_id` on first use
     (`llama_sampler_init_grammar(vocab, grammar_str, "root")`); a `NULL` return means
     the grammar failed to parse → the request fails loudly (400 at the API layer, see
     S3), never silently unconstrained.
  2. Build a `llama_token_data_array` from the raw logits and call
     `llama_sampler_apply` → disallowed tokens become `-inf`.
  3. Run fox's existing `sample_token` on the masked logits, so the pick is *both*
     grammar-legal and faithful to the request's temperature/top-p/penalties. (`-inf`
     survives temperature division and is excluded by top-K/top-P — correct by
     construction.)
  4. `llama_sampler_accept(gsmpl, chosen_token)` to advance the grammar.
- **Only generated tokens are `accept`ed**, never prompt tokens — the grammar constrains
  the *output*, and prefill just seeds the KV.

### Lifecycle (the load-bearing correctness concern)

A grammar sampler is a heap object that must be created once and freed exactly once. It
is freed on **every** terminal path a request can take — `Eos`, `Length`,
`StopSequence`, `Preempt`, and client disconnect — reusing the same cleanup points that
already remove `per_request_state` and call `clear_sequence`. A new
`model.free_grammar(req_id)` is called there. Invariant (test-guarded): the grammar map
is empty once no requests are in flight (no leak), and `free_grammar` on an unknown id is
a no-op (no double-free).

### Interaction with thinking models

If a model emits `<think>…</think>` before the answer, a grammar applied from the first
token would forbid the reasoning. v1 applies the grammar to the whole generation (fine
for non-thinking models and the common JSON case). Thinking + guided is deferred to a
follow-up using llama.cpp's **lazy grammar**
(`llama_sampler_init_grammar_lazy_patterns`), which only engages the grammar after a
trigger pattern (e.g. `</think>`). Called out so it isn't a silent gap.

### Staging

- **S1 — grammar plumbing + raw GBNF** ✅ (real build, golden-verified): `grammar` on
  `SamplingParams`/`InferenceRequestForModel`; the per-request `GrammarSampler` map
  (a `Send + Sync` newtype over `*mut llama_sampler`, freed on `Drop`), `sample_constrained`
  masking forbidden tokens via `llama_sampler_apply` before fox's Rust sampler and
  advancing with `llama_sampler_accept`, and reliable `free_grammar` on every terminal
  path (completion, length, stop, disconnect). Accepts a raw GBNF string end-to-end (no
  API surface yet — driven by the golden test). Golden `golden_grammar_constrains_output`:
  a grammar that only admits `yes`/`no` forces output into `{yes, no}` on a real model,
  and asserts the sampler is cached then freed (no leak). Grammar-`None` keeps the
  original sampling path unchanged.
- **S2 — JSON-schema → GBNF (Rust)** ✅: `api/shared/json_schema.rs` converts a schema
  to GBNF — `type` (object/array/string/integer/number/boolean/null), `properties` +
  `required`, `items`, `enum`, nesting; untyped/empty → any JSON; `any_json_gbnf()`
  backs `json_object` mode. Simplification: object rules require exactly the `required`
  set (or all declared props when `required` is absent) in the caller's order; optional
  props are dropped (output stays schema-valid, just not *more* permissive). 10 stub
  unit tests on the conversion + golden `golden_json_schema_constrains_to_valid_json`,
  which feeds a generated grammar to a real model and asserts the output parses as a
  conforming JSON object (proving the emitted GBNF is valid).
- **S3 — API surface** ✅: OpenAI `response_format` — `{ "type": "json_object" }` and
  `{ "type": "json_schema", "json_schema": { "schema": … } }` — wired in `v1/chat`;
  Ollama `format` — the string `"json"` or a schema object — wired in `ollama/chat` and
  `ollama/generate`, reading the raw `format` value so schema objects aren't flattened to
  generic JSON. `grammar_from_response_format` / `grammar_from_ollama_format` build the
  grammar and set `SamplingParams.grammar`; an unconvertible schema returns `400`, never
  a silent fallback. Covered by adapter unit tests and integration tests
  (`test_v1_chat_response_format_json_object_accepted`,
  `test_v1_chat_response_format_bad_schema_rejected`).
- **S4 — validation**: golden that a schema-constrained run parses as valid JSON matching
  the schema across several seeds; plus the S1 raw-GBNF golden. Stub-level: `grammar`
  threads through the scheduler; schema→GBNF unit tests.

### Invariants (test-guarded)

- Disallowed tokens are never sampled (golden).
- `grammar = None` is byte-identical to today's sampler (regression).
- Exactly-once create / exactly-once free per request; grammar map empties when idle.
- `accept` sees only generated tokens, in order.

---

## 2. logprobs / top_logprobs ✅

fox already carries the full logits vector (`Logits::values`) out of every decode step,
so this is pure plumbing, no extra compute — a log-softmax over `values` for the chosen
token plus its top-N alternatives.

- **Engine** (`engine/logits.rs`): `logprob_core` (pure, unit-tested) does the
  numerically-stable log-sum-exp and top-N selection; `compute_token_logprob` adds the
  token pieces. `handle_logits` attaches a `TokenLogprob` to each streamed `Token` when
  `SamplingParams.logprobs` is set. `Logits::values` is already golden-covered (the
  chunked/context goldens read it), so only the transform needed new tests.
- **API** (OpenAI chat): request `logprobs` + `top_logprobs` (capped at 20); response
  `choices[].logprobs.content[]` with `token` / `logprob` / `bytes` / `top_logprobs`,
  on both the streaming chunks and the non-streaming choice.
- Note: logprobs are over the model's **raw** distribution (before any grammar mask), so
  with guided decoding active they report the unconstrained model probabilities.
- Contained to `logits.rs` + response types; no scheduler/KV impact. Not surfaced for
  tool-call responses or the legacy completions endpoint.

## 3. Missing sampling params ✅

Pure additions to the Rust sampler (`engine/model/sampling.rs`) plus the request types.

- **min_p**: after the softmax sort, drop tokens whose probability is below
  `min_p × max_prob` (kept at least one). OpenAI (`min_p`, fox extension) and Ollama
  (`options.min_p`, native).
- **logit_bias**: a `{token_id: bias}` map added to the raw logits before penalties
  (OpenAI semantics; ±100 effectively bans/forces a token). Held as
  `Arc<HashMap<i32, f32>>` so per-step request clones stay cheap; the API parses the
  string token-id keys, dropping non-integer ones. OpenAI only.
- **min_tokens**: suppress every end-of-generation token (mask to `-inf`) until
  `generated_tokens >= min_tokens`. To avoid a per-token vocab scan, the model's EOG id
  set is precomputed once at load (`eog_tokens`); `sample_constrained` masks it. OpenAI
  (fox extension).
- Tests: `logit_bias_forces_a_token` / `logit_bias_bans_a_token` / `min_p_keeps_only_dominant_token`
  (stub) and golden `golden_min_tokens_suppresses_eog` (real model: no EOG below the floor).

---

Each item ships independently and is gated by the 0.12 regression net (golden tests in
CI + the scheduler conservation stress test). Guided decoding (§1) is the flagship and
lands first; logprobs (§2) and the sampling knobs (§3) are small, high-demand follow-ups
that round out "controllable output".
