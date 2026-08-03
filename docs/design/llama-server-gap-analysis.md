# llama-server gap analysis

**Date:** 2026-08-02 · **Vendored llama.cpp:** `6f4f53f2b` ·
**Companion:** [`vllm-gap-analysis.md`](vllm-gap-analysis.md)

A code-level audit of `vendor/llama.cpp/tools/server/` against fox, to decide what is
worth porting. This is the counterpart to the vLLM gap analysis: that one compares fox
against a *different* architecture, this one against the reference server for the exact
engine fox embeds — so every difference here is directly actionable, with no
"llama.cpp can't do that" escape hatch.

Upstream's server has been split into `server-context.cpp` (slots, batching, decode
loop), `server-task.{h,cpp}` (task/result types + `server_prompt_cache`),
`server-queue.{h,cpp}`, `server-common.{h,cpp}` (`server_tokens`, tokenization),
`server-schema.cpp` (per-request JSON parameter schema), `server-http.cpp` and
`server-models.cpp` (router mode). Line references below are against `6f4f53f2b`.

---

## 0. Three findings that invalidate existing assumptions

These came out of the audit and matter more than any individual feature, because
current fox design docs argue *against* work that these findings unblock.

### 0.1 `kv_unified = true` — partial `seq_cp` is legal and cheap

`src/engine/model/llama_cpp/mod.rs:676` (and `:857` for the draft context) sets
`ctx_params.kv_unified = true`. It landed in `1c36faf`, *"unified KV cache closes the
prefix-cache throughput regression"* (2026-08-01). The consequences, each traced in the
vendored source:

- `llama-kv-cache.cpp:98` — `n_stream(unified ? 1 : n_seq_max)` ⇒ `n_stream == 1`.
- `llama-kv-cache.cpp:161-168` — `seq_to_stream` stays all-zeros when `n_stream == 1`.
- `llama-kv-cache.cpp:475-504` — `seq_cp` therefore always sees `s0 == s1` and takes
  the **metadata-only** branch (`cells.seq_add`, cell sharing, no buffer copy),
  returning at `:504` — *before* the `GGML_ASSERT(is_full)` at `:518`.

So an arbitrary `[p0, p1)` sub-range `llama_memory_seq_cp` is now both legal and cheap.

**This retires a documented "ruled out" conclusion.** `vllm-gap-analysis.md` §1
("Known limitation") states that migrating a prefix-cache-hit request to a fresh seq_id
*"crashes the server — `GGML_ASSERT(is_full)` … fox's non-unified KV cache
(`kv_unified = false`) only supports cross-stream `seq_cp` of the entire KV buffer"*.
That was true when written and is not true now; `1c36faf` changed the premise without
the doc being revisited. §2's beam-search analysis rests on the same stale premise.

A second consequence: `llama-kv-cache.cpp:725` selects `split_simple` when
`n_stream == 1`, which has no "consecutive ascending seq_id" requirement. **The
dense-seq_id invariant documented at `src/scheduler/mod.rs:36-47` is no longer
load-bearing.** This matters because LCP-based slot affinity (§1) inherently hands out
non-dense IDs; preserving that invariant would cost cache hits for nothing.

### 0.2 fox's block IDs never reach llama.cpp

`block_ids` / `page_table` appear only in `src/scheduler/`, `src/kv_cache/` and one
bookkeeping loop at `src/engine/run.rs:396`. The pool is sized `llama_n_ctx / block_size`
(`src/model_registry/loader.rs:139-142`). llama.cpp allocates its own cells by
`(seq_id, pos)` and never learns about fox's blocks.

The page table is therefore **a pure admission-control budget, not an address space**.
The tension one would expect when porting llama-server's token-exact prompt reuse onto
fox's 16-token block allocator does not exist: block granularity decides only *how much
budget is charged*, token exactness lives entirely in llama.cpp positions. They are
different layers and never need reconciling.

### 0.3 `prefix_seq_id` is dead code with a live implementation

`InferenceRequestForModel.prefix_seq_id` (`src/engine/model/mod.rs:233`) is only ever
assigned `None` (`src/scheduler/schedule.rs:161,202`, `src/scheduler/batch.rs:257`). But
its handler at `src/engine/model/llama_cpp/batch.rs:250-270` already performs exactly the
`seq_cp` that shared-prefill forking needs. Combined with 0.1, the `n>1`/`best_of` fork is
substantially *turning on existing code* rather than writing new code.

**All required FFI is already bound.** `build.rs:395` uses
`allowlist_function("llama_.*")`, so `llama_memory_seq_rm/cp/add/can_shift` and
`llama_state_seq_{get_size,get_data,set_data}_ext` are all available with no bindgen work.

---

## 1. The structural gap: KV reuse

This is the one difference that is architectural rather than a missing feature.

### What fox does

| | |
|---|---|
| Cache shape | `lru::LruCache<u64, PrefixCacheEntry{seq_id, block_ids}>`, capacity `(max_batch_size/4).max(1)` — **8 entries at defaults**, with no flag to raise it (`src/scheduler/mod.rs:108-118`) |
| Key | Chained hash of the **last complete block** (`src/kv_cache/mod.rs:36-59`); reuse is aligned to `block_size` (16) and the partial trailing block is discarded |
| On hit | The entry is `pop()`ed — **moved, not shared**. The hitting request takes over the cached `seq_id` and blocks; prefill restarts at `cached_tokens - 1` (`src/scheduler/schedule.rs:122-160`) |
| On completion | Only the **prompt's** whole-block prefix is donated back; generation blocks are freed and the llama.cpp sequence trimmed (`src/scheduler/schedule.rs:348-444`, `src/engine/logits.rs:205-222`) |
| Resident-prompt tracking | **None.** A seq_id is a bare integer from a min-heap; a cache miss re-prefills from token 0 |

The consequence that matters most: **the generated response is always discarded**. In a
multi-turn chat the next turn's prompt contains the previous assistant reply, so it can
never match a cached entry beyond the point where the last turn's prompt ended.

### What llama-server does

Each slot permanently keeps `slot.prompt.tokens` — the tokens currently resident in its
KV, prompt *and* generation.

- **Slot selection** (`server-context.cpp:1586-1694`): explicit `id_slot`; else
  longest-common-prefix similarity over idle non-empty slots,
  `sim = LCP(slot.prompt.tokens, task.tokens) / task.tokens.size()`, best above
  `--slot-prompt-similarity` (default `0.1`); else LRU by `t_last_used`. It also computes
  `f_keep = sim·task_len / slot_len` and, when `f_keep < 0.5`, flags the slot for a push
  to the RAM prompt cache before its context is thrown away.
- **Reuse** (`server-context.cpp:3166-3243`):
  `n_past = slot.prompt.tokens.get_common_prefix(input)` — **token-exact**, not
  block-aligned — then `keep_first(n_past)` and
  `common_context_seq_rm(ctx, slot.id, pos_next, -1)` to drop everything past the
  divergence point. Guard `[TAG_PROMPT_LOGITS]` (`:3356-3361`): on a total hit,
  `n_past--` so at least one token still decodes and produces logits.
- **`--cache-reuse N`** (`server-context.cpp:3187-3239`): tolerates *deleted spans in the
  middle* of a prompt by RoPE-shifting surviving runs of ≥N matching tokens with
  `llama_memory_seq_rm` + `llama_memory_seq_add`. Off by default upstream
  (`n_cache_reuse = 0`).
- **`server_prompt_cache` / `--cache-ram MiB`** (`server-task.cpp:1600-1793`): serialises
  a slot's whole KV to host RAM via `llama_state_seq_get_data_ext` and restores it with
  `llama_state_seq_set_data_ext`. `alloc()` skips a prompt that is a prefix of a cached
  entry and erases superseded ones; `load()` accepts a candidate only if **both** `f_keep`
  and `sim` improve, skipping entries with `f_keep < 0.25`; `update()` evicts FIFO by size
  then by token count.
- **`--cache-idle-slots`** (default on): pushes idle slots to the RAM cache on each new
  task and, when `kv_unified`, clears their KV so the cells become reusable.

### Plan

| Stage | What | Status |
|---|---|---|
| **1.A** | `repeat_last_n` window for the penalties | **done** — see §3 |
| **1.B** | Per-seq_id resident-token tracking + LCP slot affinity; finished requests park as `Idle` instead of freeing; LRU reclaim of idle slots under block pressure. New `src/scheduler/slots.rs` replacing `prefix_cache.rs`. Flags `--kv-reuse`, `--slot-prompt-similarity` | planned |
| **1.C** | Host-RAM sequence state cache (`--cache-ram`) — keeps a conversation warm *without* holding GPU blocks | **done** — see below |
| **1.D** | Shared-prefill fork for `n>1`/`best_of` | **done** — 3.4× on an 801-token `n=4` request (6.60 s → 2.01 s); one prefill instead of four |
| **deferred** | `--cache-reuse` chunk shifting | see below |

**Why `--cache-reuse` is deferred rather than skipped.** It is mechanically possible:
`llama_memory_seq_rm` + `seq_add` + `can_shift` are already used *together* in
`Model::roll_context` (`src/engine/model/llama_cpp/mod.rs:1131-1146`), the exact pair
llama-server uses at `server-context.cpp:3216-3220`. But it only pays when a span is
*deleted from the middle* of a prompt; 1.B's token-exact LCP already covers the dominant
case of pure prefix growth, upstream ships it off by default, and it collides with fox's
`rolled_tokens` accounting (`context_len()`, `src/scheduler/batch.rs:295-310`) which
assumes one contiguous window. Revisit after 1.B is measured.

### Measured (2026-08-02, CPU/zen4, llama-3.2-1b-instruct-q8_0)

Two separate questions, two separate experiments. Both alternate arms round by round
with exactly one server up at a time, per `scripts/ab_bench.sh`'s rules.

**1. Does KV reuse work at all?** `--kv-reuse false` vs `true`, same binary, TTFT on a
repeated ~3.5k-token prompt (`scripts/ab_bench.sh --prompt-file`, 3 rounds × 6 samples):

| arm | median TTFT | range |
|---|---|---|
| `--kv-reuse false` | 4760.65 ms | [4698.45, 4795.63] |
| `--kv-reuse true` | 781.69 ms | [777.30, 825.33] |

Disjoint ranges, both arms loaded the same `libggml-cpu-zen4.so` — **6.1×**.

**2. Does the new design beat the old one?** Question 1 does *not* answer this: for a
single repeated prompt the old block-hash cache also hits. So: old binary (HEAD) vs new,
same llama.cpp build, on a **12-conversation working set** — larger than the old cache's
8 entries, smaller than the new table's 32 slots — cycled 3 passes, first discarded:

| arm | median | mean | p90 |
|---|---|---|---|
| old (block-hash cache) | 52.09 ms | 1247.61 ms | 3650.52 ms |
| new (slot table) | 37.60 ms | 36.31 ms | 38.70 ms |
| | **1.4×** | **34×** | **94×** |

**Read the mean and p90, not the median.** The median improvement (28%) is the
token-exact-vs-block-aligned win on requests that *hit* in both designs. The real
finding is the tail: the old 8-entry cache evicts a third of a 12-conversation working
set on every pass, and each eviction costs a full re-prefill — which shows up as a
heavy tail, not as a shifted median. A median-only report would have hidden the very
thing under test.

### 1.C as built, and when it actually earns its keep

`--cache-ram` serialises a reclaimed sequence to host memory
(`llama_state_seq_get_data_ext`) and restores it later
(`llama_state_seq_set_data_ext`) instead of re-prefilling. Ordering in the engine is
load-bearing and documented at the call site: **saves → clears → restores → trims**. A
save must read the sequence before the clear wipes it; a restore must land after the
clears (its destination may itself have just been reclaimed) and before the trims,
which bound the *restored* state at the new request's divergence point. A failed
restore resets the request to prefill from token 0 rather than letting it read cells
that were never written — slower, never wrong.

**Verification, and its limit.** The FFI round-trip is proven against a real model
(`golden_state_seq_round_trip_preserves_decode`): a saved state restored into a
*different* sequence predicts the identical token, with logits matching to <1e-3, and
restoring over a dirty destination is correct because `state_seq_load` clears first.
The scheduler side is proven by unit tests asserting the exact save/restore intents.

What was **not** observed is the full chain firing against a real model under load, and
the reason is worth recording. Reclamation only triggers when the block pool is
exhausted *and* the claiming request needs more blocks than the slot it inherits. Under
sequential single-client traffic neither holds: every request shares the chat-template
prefix, so LCP affinity routes them all onto one slot whose blocks they inherit
unchanged. Nine requests against an 8-slot server left 1 slot idle and 7 free, across
four attempts at staging pressure.

So `--cache-ram` is **not a general speedup**. It earns its keep specifically under
concurrent, distinct conversations that exhaust the block pool — where the slot table
alone would have to throw a conversation away. That is why it defaults to `0`.

### Greedy output is not stable under concurrency — pre-existing, amplified by reuse

An intermittent `make e2e` failure (one run in twelve reported `21 passed, 1 failed`)
prompted a direct investigation rather than waiting for it to recur. Three experiments,
all at `temperature: 0` where any difference is a real difference:

| Experiment | Result |
|---|---|
| **Sequential reuse.** Same prompt 4× in a row; request 1 prefills in full, 2–4 inherit the parked sequence (`cached_tokens` 0 → 396) | **byte-identical every time** |
| **Concurrent, `--kv-reuse true`.** 4 clients, same 4 prompts, 10 rounds, compared against a sequential baseline | **10 / 10 rounds differed** |
| **Concurrent, `--kv-reuse false`** (control — reuse disabled, everything else identical) | **2 / 10 rounds differed** |

Two conclusions, and the order matters:

1. **The nondeterminism is pre-existing, not introduced by this work.** The control arm
   has reuse switched off entirely and still drifts. Concurrent requests are batched
   together according to arrival timing, and llama.cpp does not guarantee bit-identical
   logits across batch compositions — different shapes take different reduction orders
   and kernel paths. At `temperature: 0` a near-tie then flips the argmax. Every batched
   inference server has this property; llama.cpp does not promise otherwise.

2. **Reuse makes it much more frequent** (2/10 → 10/10). The likely mechanism: reuse
   collapses prefill, so requests reach the decode phase sooner and spend far more of
   their life decoding *alongside* each other — which is exactly the condition that
   perturbs the batch. Stated as the likely mechanism, not a measured one.

**This is not incorrect KV.** That alternative is ruled out by the first row: if reuse
reconstructed the wrong state, sequential reuse would drift too — reuse is reuse, and
the sequential case exercises the same code path with the same `cached_tokens`. It is
byte-exact. What changes under concurrency is *who shares a batch*, not what the cache
holds.

Caveat on the absolute rates: both arms were measured with an unrelated `make e2e` loop
running on the same machine, so the CPU contention likely inflates *both* numbers. The
arms shared that load, so the 2/10 vs 10/10 comparison stands; the individual figures
should be read as "this happens often", not as calibrated probabilities.

**The e2e failure is explained, and it was not the drift above.** Checks 1 and 9 send
`max_tokens: 12` with no `temperature` (so fox's stochastic 0.8 default) and no
`min_tokens`, then assert `finish == "length"`. Nothing stopped the model emitting EOS
before the twelfth token, which yields `"stop"` and fails the check.

Measured directly rather than waited for: 600 requests in the concurrent shape check 9
uses produced **2** early stops — 0.33% per request. Across this suite's 7 such requests
that is a 2.31% chance per run, i.e. **~1 failing run in 43**. The observed rate was 1 in
~52. (A first attempt with 250 *sequential* requests found 0, which looked like a refutation
but was not: at 0.33% the expected count there is 0.7, so it had no power to detect it.
Concurrency also raises the rate, for the same batch-composition reason as the drift above.)

Fixed by adding `min_tokens: 12`, which suppresses EOG until the cap so `finish ==
"length"` is a fact about the engine rather than a coin flip. The checks' intent is
untouched — they exist to catch a request dying after its prefill token, and such a
request still reports `n < 12`. Verified under the same concurrent conditions: 600/600
`length`, zero short, against 2/600 before.

Worth keeping in mind generally: this was a **test** defect that survived two rounds of
investigation because the first instinct was to suspect the code under change. The
drift finding above is real and was worth having, but it was not this.

Chasing it did surface a genuine fragility in the suite, since fixed. Checks 2 and 6
both parse grammar-constrained output, and **guided decoding does not guarantee a
*complete* document within a token budget**: JSON Schema's `type: integer` carries no
length bound to translate, so the grammar's `integer` rule allows unbounded digits, and
Ollama's `format: "json"` compiles to the fully permissive any-JSON grammar. A run cut
off mid-document fails `json.loads` — and with the old 60-token cap that was
*indistinguishable* from "the grammar emitted something non-conforming", which is the
bug the check exists to catch. Both checks now get generous headroom (256 tokens) and
test `finish_reason`/`done_reason` for truncation *first*, reporting it as its own
distinct failure with the partial document attached. Verified by injecting a 3-token cap:
the branch fires and reports `partial='{"answer":'` rather than a parse error.

This is a fix to the test, not to fox, and it is **not** a fix for the failure above —
that remains unexplained. Note also that check 2's grammar is unaffected by the
optional-properties fix: both its properties are `required`, so the new code takes the
required-only branch and emits exactly what it did before.

Practical consequence, worth knowing before relying on it: fox has never guaranteed
reproducible greedy output under concurrent load, and now it delivers it noticeably
less often. A caller who needs reproducibility needs a serialised request stream — a
`seed` is not sufficient, because the variation is in the forward pass, not the
sampler. The e2e suite's `temperature: 0` checks are correspondingly a little more
likely to flake; that is the cost of the change, recorded rather than hidden.

**Measurement discipline.** No performance claim from any of this lands without
`scripts/ab_bench.sh`. A single before/after pair is worthless on this hardware — see
`cpu-benchmark-isolation` and the script's own header. `--prompt-file` was added for
this work: the default 2-token `"Hi"` probe has no prefill to save, so any prompt-reuse
change measures as pure noise through it.

---

## 2. API / compatibility parity

Things that broke or misled real clients. **Most of this table is now fixed** — the
Evidence column records where the defect lived, for anyone auditing the change.

| Gap | Status |
|---|---|
| `/v1/completions` dropped nearly every sampling parameter and returned a `chat.completion` object instead of `text_completion` | **fixed** — parameters threaded, response rewritten (streaming too), `echo`/`suffix` now rejected loudly |
| No prefill/decode timing split; `load_duration` and `prompt_eval_duration` hard-coded `0` | **fixed** — all four durations measured on `/api/generate` and `/api/chat`, plus `prefill_ms`/`decode_ms` in the log line |
| `stream_options.include_usage` parsed and ignored | **fixed** — explicit `false` honoured; omitting it keeps the previous behaviour |
| `options.repeat_last_n` silently dropped on the Ollama surface | **fixed** (with the `--repeat-last-n` work, §3) |
| No `cached_tokens` in `usage` | **fixed** — carried on every `Token` from `skip_prefix_tokens`, surfaced as `usage.prompt_tokens_details.cached_tokens`, omitted when zero |
| No `/tokenize`, `/detokenize`, `/apply-template` | **fixed** — routed; `/apply-template` renders through the real request path and detokenizes, so it shows exactly what the model receives |
| No raw GBNF `grammar` request field | **fixed** — on chat and legacy completions; conflicting with `response_format` is a 400 |
| Ollama options still dropped by serde: `num_ctx`, `num_keep`, `typical_p`, `mirostat*`, `penalize_newline`; `keep_alive` parsed but never applied | **open** |

Original evidence, for reference:

| Gap | Evidence in fox | Upstream reference |
|---|---|---|
| **No prefill/decode timing split.** `load_duration` and `prompt_eval_duration` are hard-coded `0` in every Ollama response; `total_duration` and `eval_duration` are the same wall clock | `src/api/ollama/generate.rs:200,206,231,233`, `src/api/ollama/chat.rs:247,253,320,322` | `timings` block, `server-task.cpp:240-261` |
| **No `cached_tokens`** — a client cannot see the prefix-cache benefit | — | `usage.prompt_tokens_details.cached_tokens`, `server-task.cpp:389-396` |
| **`/v1/completions` is broken for compat**: hard-codes `None` for `top_p`, `top_k`, `stop`, `seed`, `logprobs`, `logit_bias` and the penalties, and returns a `chat.completion` object. `CompletionResponse`/`CompletionChoice` are declared and never constructed | `src/api/v1/completions.rs:14-48`, `src/api/types/v1.rs:439-454` | `server-context.cpp:4704-4714` |
| **`stream_options.include_usage` ignored** — usage always rides the last chunk | `src/api/v1/chat.rs:485-489` | `server-schema.cpp:26-29` |
| **Ollama options silently dropped by serde**: `num_ctx`, `num_keep`, `typical_p`, `mirostat*`, `penalize_newline`. `keep_alive` is parsed and never applied despite fox having a per-model TTL | `src/api/types/ollama.rs:116-136,154,226` | — |
| **No `/tokenize`, `/detokenize`, `/apply-template`** — `InferenceEngine::tokenize` exists and is simply unrouted; the Jinja renderer exists too | `src/engine/mod.rs:136-138`, `render_chat_jinja` | `server-context.cpp:4899-4956, 4846-4856` |
| **No raw GBNF `grammar` field** — the engine has full GBNF support; the only door is `response_format` / `format` | `src/engine/model/llama_cpp/batch.rs:659-745` | `server-schema.cpp:262-275` |

Also stale, found while auditing: `src/api/types/v1.rs:236,240` claim
`frequency_penalty`/`presence_penalty` are "accepted, not applied". They *are* applied
(`src/api/v1/chat.rs:200-205` → `src/engine/model/sampling.rs:34-55`).

---

## 3. Samplers

**The correctness bug is fixed.** fox's JSON-Schema→GBNF converter *dropped*
non-`required` properties, so a schema with optional fields produced a grammar that
**forbade** them — the grammar contradicted the schema rather than merely being
stricter than it. Optional properties are now emitted as optional members
(declaration order only; every permutation is exponential, and llama.cpp's own
converter has the same limit). An explicitly empty `"required": []` is now
distinguished from an absent `required`, which is what makes an all-optional object
expressible at all. `anyOf`/`oneOf`/`$ref`/`$defs` are still missing.

Present in fox: temperature, top_k, top_p, min_p, repetition penalty,
frequency/presence penalties, `logit_bias`, seed, GBNF grammar, `min_tokens`,
and — as of this work — `repeat_last_n`, `top_n_sigma` and `min_keep`.

| Missing | Cost | Reference |
|---|---|---|
| `dynatemp_range`/`dynatemp_exponent`, `ignore_eos` | low | `server-schema.cpp:85-190` |
| `typical_p` | medium — see below | `server-schema.cpp:108` |
| `mirostat` 1/2 (`mirostat_tau`, `mirostat_eta`) — needs per-request state | medium | `server-schema.cpp:153-160` |
| XTC (`xtc_probability`, `xtc_threshold`) | medium | `server-schema.cpp:100-104` |
| DRY (`dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`) — needs suffix matching | high | `server-schema.cpp:135-149` |
| Configurable sampler chain order (`samplers`) | low, but only worth it once enough samplers exist to order | `server-schema.cpp:477-486` |

**Why `typical_p` and mirostat are not just "more of the same".** fox's sampler
deliberately avoids materialising the full 128K-entry distribution: it grows a
candidate pool adaptively (64 → 256 → 1024 → …) and stops as soon as the pool
provably contains enough mass for min-p/top-p to give the same answer as a full sort
(`sampling.rs`, step 4/5). Locally-typical sampling ranks tokens by
`|−log p − H|`, which is **not** monotonic in probability, so the tokens it wants are
not necessarily in that head — and mirostat carries per-request feedback state across
steps. Both need either a full-vocabulary pass or a redesign of the pool, so they are
a design decision rather than a missing line of code.

`top_n_sigma` was cheap precisely because it *is* monotonic in the logit, so it
composes with the existing pool as one more leading-run truncation. It also turns out
to be invariant under temperature (scaling every logit by `1/T` scales `max` and `σ`
identically), which is asserted in a test.

### Shipped: `repeat_last_n` (stage 1.A)

Before this change both penalty passes scanned **every** token generated so far on every
step. Two distinct problems: `apply_frequency_presence_penalty` rebuilt a full `HashMap`
over the whole history per token, making the pass `O(generated²)` per request; and the
Ollama surface's `repeat_penalty = 1.1` default kept penalising tokens from thousands of
positions back, degrading long outputs.

`penalty_window` (`src/engine/model/sampling.rs`) now slices the trailing window with
llama.cpp's semantics: `-1` = whole history, `0` = disabled, `n` = last `n`. Exposed as
`--repeat-last-n` / `FOX_REPEAT_LAST_N` / `repeat_last_n` in `config.toml`, overridable
per request via `repeat_last_n` (`/v1/*`, a fox extension) and `options.repeat_last_n`
(`/api/*`, which upstream Ollama supports and fox previously dropped silently).

Two deliberate decisions:

- **Default `-1`, not llama.cpp's `64`.** Adopting 64 would silently change output for
  every existing caller, since the Ollama surface's `repeat_penalty` is 1.1. Shipping the
  bit-identical default and letting users opt in keeps the change safe.
- **fox's window covers only *generated* tokens**, while llama.cpp's spans
  prompt+generated. fox has never penalised prompt tokens; the window narrows an existing
  behaviour rather than redefining it. Documented at the call site.

---

## 4. Endpoints and features fox lacks

| Feature | Upstream | Notes for fox |
|---|---|---|
| `/v1/rerank`, `/rerank` | `server-context.cpp:4962-5040`, prompt `[BOS]query[EOS][SEP]doc[EOS]`, score from `llama_get_embeddings_seq(...)[0]` | needs a `--reranking` flag and an exposed pooling type |
| `/infill` (FIM) | `server-context.cpp:4614-4690` — `input_prefix`, `input_suffix`, `input_extra`, FIM special tokens, `n_indent`, `t_max_predict_ms` | what code-completion plugins consume; pairs naturally with `--cache-reuse` |
| `/props`, `/slots` | `server-context.cpp:4551-4600`, `4475-4516` | fox's `/api/show` returns `parameters` and `template` as empty strings and `parameter_size: "unknown"` (`src/api/ollama/management.rs:140-158`). `/slots` falls out of stage 1.B, which creates the slot table |
| Runtime LoRA (`GET`/`POST /lora-adapters`) | `server-context.cpp:5042-5102`, `1728-1757` | **done** — list adapters and change their scale without a restart. Per-request `lora: [{id, scale}]` (several adapters on one request) is **not** done: fox groups a decode batch by adapter *name*, so multi-adapter requests would need set-based grouping first |
| Slot state save/restore to disk | `server-context.cpp:4518-4549` | pure HTTP plumbing once stage 1.C exposes `state_seq_save`/`state_seq_load` |

Deliberately **not** planned: router/multi-model mode (`server-models.cpp`) — fox's own
`ModelRegistry` with LRU + TTL eviction already covers that ground differently and
better for fox's use case; the WebUI, CORS proxy and built-in `/tools`; the Anthropic
Messages and OpenAI Responses compatibility surfaces; TTS/vocoder.

---

## Where fox is ahead

Worth recording so the comparison stays honest — this is not a one-way deficit list.

- Full Ollama-compatible management surface (`/api/tags`, `/api/ps`, `/api/show`,
  `/api/pull`, `/api/copy`, `/api/create`, `/api/delete`) — upstream has none of it.
- Multi-model registry with LRU + keep-alive eviction and on-demand load/unload;
  llama-server needs router mode and child processes for the equivalent.
- Its own scheduler: chunked prefill (`--max-prefill-chunk`), LIFO preemption on KV
  pressure, `--max-queue-depth` backpressure with HTTP 429, batch-size bisection retry
  on OOM plus reactive context rolling.
- Four tool-call parsers auto-detected from the model's own chat template.
- `min_tokens` and `think` as first-class request extensions; `best_of` with server-side
  log-likelihood ranking.
- Per-request speculative decoding that is golden-verified byte-identical regardless of
  proposer; upstream's per-request `speculative.*` is compiled out entirely
  (`server-schema.cpp:194-222`, `#if 0`).

---

## Upstream API churn to watch

The vendored tree removed `--draft`, `--draft-n`, `--draft-max`, `--draft-min` and
`--draft-n-min`; they now hard-error with a migration message
(`vendor/llama.cpp/common/arg.cpp:3906-3919`). Replacements are `--spec-draft-n-max` /
`--spec-draft-n-min`. fox uses its own flag names (`--spec-draft-len`, `--spec-ngram`) and
its own speculative implementation, so nothing breaks — but any tooling or docs that shell
out to `llama-server` for A/B comparison (`scripts/ab_bench.sh`,
`Dockerfile.llama-server-rocm`) must use the new names.
