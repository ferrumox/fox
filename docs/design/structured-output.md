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
- **S2 — JSON-schema → GBNF (Rust)**: since `json_schema_to_grammar` is in `common/`
  (not linked), implement a pragmatic converter in Rust covering the JSON-Schema subset
  that matters: `type` (object/array/string/number/integer/boolean/null), `properties` +
  `required`, `items`, `enum`, and nesting. Unit-tested against known schema→grammar
  pairs. A plain "any JSON" grammar backs `json_object` mode.
- **S3 — API surface**: OpenAI `response_format` — `{ "type": "json_object" }` and
  `{ "type": "json_schema", "json_schema": { "schema": … } }`; Ollama `format` — the
  string `"json"` or a schema object. Build the grammar in the chat/completions handlers
  of *both* families and set `SamplingParams.grammar`. A grammar that fails to
  parse/convert is a `400`, not a silent fallback.
- **S4 — validation**: golden that a schema-constrained run parses as valid JSON matching
  the schema across several seeds; plus the S1 raw-GBNF golden. Stub-level: `grammar`
  threads through the scheduler; schema→GBNF unit tests.

### Invariants (test-guarded)

- Disallowed tokens are never sampled (golden).
- `grammar = None` is byte-identical to today's sampler (regression).
- Exactly-once create / exactly-once free per request; grammar map empties when idle.
- `accept` sees only generated tokens, in order.

---

## 2. logprobs / top_logprobs

fox already carries the full logits vector (`Logits::values`) out of every decode step,
so this is pure plumbing, no extra compute. Compute a log-softmax over `values`, then
return the chosen token's logprob plus the top-N alternatives per position.

- OpenAI: chat `logprobs` + `top_logprobs` (0–20), and the completions `logprobs` field.
- Wire through the streaming and non-streaming response types in `src/api/types/`; the
  numbers come from `engine/logits.rs`, which already has `values` in hand.
- Contained to `logits.rs` + response types; no scheduler/KV impact.

## 3. Missing sampling params

All three are pure additions to the Rust sampler (`engine/model/sampling.rs`) plus the
request types — no llama.cpp involvement.

- **min_p**: after temperature, drop tokens whose probability is below
  `min_p × max_prob`. A cheap nucleus alternative; slots between steps 4 and 5 of
  `sample_token`.
- **logit_bias**: a `{token_id: bias}` map added to the raw logits before sampling
  (OpenAI semantics; `-100`/`+100` effectively ban/force a token). Threaded from the API
  as `HashMap<i32, f32>`.
- **min_tokens**: suppress every EOG token (mask to `-inf`) until
  `generated_tokens >= min_tokens`, so a request can be forced to keep going. Reuses the
  model's EOG set (`is_eog_token`).

---

Each item ships independently and is gated by the 0.12 regression net (golden tests in
CI + the scheduler conservation stress test). Guided decoding (§1) is the flagship and
lands first; logprobs (§2) and the sampling knobs (§3) are small, high-demand follow-ups
that round out "controllable output".
