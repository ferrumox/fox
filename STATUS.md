# Fox — Feature & Correctness Status

A living inventory of **everything fox does** and an honest assessment of **what works
and what doesn't**. Use it to decide what to fix, in what order, and to track progress
per release.

- **Tracks through:** 0.20.1 (branch `feature/0.20`); `main`/`develop` are at 0.13.0
  as of this writing — 0.14 through 0.20.1 are done and closed but not yet merged up
  (deliberate, releases are being cut gradually). This file describes the code, not
  what's tagged.
- **0.19 in one line:** a llama-server gap audit
  ([`llama-server-gap-analysis.md`](docs/design/llama-server-gap-analysis.md)), the KV
  reuse rework it motivated (34× mean / 94× p90 on a multi-conversation working set),
  shared-prefill forking for `n>1` (3.4×), and the API-parity and correctness fixes the
  audit exposed along the way.
- **Last updated:** 2026-08-02
- **Companion:** [Model-architecture correctness rework](docs/design/model-architecture-rework.md)
  — the design that resolved most of the ❌/⚠️ items below (see "Known issues" for what's
  still open). [`docs/design/vllm-gap-analysis.md`](docs/design/vllm-gap-analysis.md) is the
  maintained source of truth for the feature-gap comparison vs Ollama/vLLM, and
  [`docs/design/llama-server-gap-analysis.md`](docs/design/llama-server-gap-analysis.md)
  the same against llama.cpp's own reference server — this file no longer duplicates
  either (see "Comparison" below).

  Note that the llama-server audit **retired two conclusions the vLLM doc still argued
  for**: `kv_unified = true` (since `1c36faf`) makes partial `seq_cp` legal and cheap,
  and fox's block IDs never reach llama.cpp, so the page table is an admission budget
  rather than an address space. Both are marked superseded in place there.

### Assessment basis

Status is from **code review** (per-subsystem) unless a row says otherwise. "✅" means
*no defect found in review*, not *verified by running*. Items needing a running server or
GPU to confirm are marked ❓.

The 0.19 rows are the exception and say so explicitly: where one quotes a number
(6.1×, 34×/94×, 3.4×, 600/600) it came from a real server and a real model, one server
at a time, arms alternated — per `scripts/ab_bench.sh`'s rule that a single
before/after pair proves nothing on this hardware. Rows that were *not* verified end to
end say that too, in the same sentence as the claim rather than in a footnote — see
`--cache-ram` and the `/rerank` history in the CHANGELOG.

### Legend

| | Meaning |
|---|---|
| ✅ | Correct — no defect found in review |
| ⚠️ | Works with caveats / partial / footgun |
| ❌ | Incorrect for some models or inputs |
| 🚧 | Stub / parsed-but-unused / not wired |
| ❓ | Unconfirmed — needs a running/stress test |

---

## Serving runtime

| | Feature | Notes |
|---|---------|-------|
| ✅ | Axum HTTP server, startup, graceful shutdown, signal handling | |
| ✅ | Continuous batching (fox's own scheduler, not llama.cpp's) | prefill + decode per step |
| ✅ | LIFO preemption on KV pressure | frees blocks, returns seq_id, re-queues |
| ✅ | Request cancellation on client disconnect | `send()` fails → finished + KV freed immediately |
| ✅ | Multi-model registry with LRU + keep-alive eviction | engine loop aborted on `Drop` |
| ⚠️ | `max_models = 1` default | **0.18** — the *silent* part is fixed (a startup log now states the trade-off explicitly, `max_models_default_hint`); the default itself deliberately stays 1 — fox has no cross-model VRAM accounting (the per-load "does this fit" check compares against a static, whole-GPU figure from startup, never subtracting what's already resident), so raising the default without that accounting would trade a churn footgun for a real OOM-crash footgun. Real fix is proper multi-model VRAM budgeting — not done, tracked as its own follow-up, not this item |

## Model loading & architecture handling

> This is where most defects live — architecture facts derived by formula/literal,
> scattered across layers.

| | Feature | Notes |
|---|---------|-------|
| ✅ | GGUF load via FFI with actionable failure diagnosis | magic bytes / memory / GGUF version |
| ✅ | Runtime backend detection (CUDA/ROCm/Vulkan/Metal/CPU) | one binary |
| ✅ | `head_dim` from GGUF metadata (`<arch>.attention.key_length`) | **recently patched**; was `n_embd/n_head` (wrong for Gemma/MLA) |
| ✅ | Flash attention = AUTO | **recently patched**; was forced ENABLED → Gemma softcap garbage on CUDA |
| ✅ | `embedding_dim = n_embd` (read from `llama_model_n_embd`) | **fixed**; was `num_heads * head_dim` (wrong for Gemma/MLA + an out-of-bounds read). Stored on `ModelConfig.n_embd` |
| ✅ | `n_ctx` in `load()` no longer capped by a per-token formula | **0.18** — empirical create-then-shrink-on-failure retry loop (`shrink_n_ctx`) replaces the pre-creation byte-budget cap; the formula survives only as a soft first-guess ceiling under `--gpu-memory-fraction`. See `docs/design/mla-recurrent-kv-sizing.md` |
| ✅ | MLA & recurrent KV sizing correctness | **0.18** — the fix above applies uniformly (no per-arch branching); verified against real DeepSeek-V2-Lite (MLA) and Mamba (recurrent) models. Lightweight `KvMemoryClass` (Standard/Latent/Recurrent) added to `ModelInfo`/`fox probe` for observability |
| ✅ | Recurrent/hybrid detected (`llama_model_is_recurrent`/`llama_model_is_hybrid`); **cross-sequence** prefix copying disabled for them, slot reuse kept | **0.20.0** — the detection was right and the consequence was too broad. Copying a prefix out of *another* live sequence needs `seq_cp` and is illegal here; inheriting the KV a sequence already holds copies nothing and is not. Both hung off one flag, so a hybrid model lost prompt reuse entirely while `llama-server` performed it on the same architecture and the same llama.cpp — measured on Qwen3.5-9B: fox warm TTFT 42923 ms with `cached_tokens: 0`, `llama-server` 13264 ms with 14680. Now split into `supports_seq_copy()` and `supports_slot_reuse()`; needs `--rs-rollback > 0`, and `trim_sequence`'s refusal is checked rather than assumed. Warm TTFT 42923 → 652 ms |
| ⚠️❓ | `n_ctx`/`n_batch`/`n_seq` heuristic | `.max(effective_ctx)` may size the pool for ~1 sequence while `n_seq_max=32` → possible tightness under concurrency; unconfirmed |

## Inference correctness (prefill / decode / sampling / output)

| | Feature | Notes |
|---|---------|-------|
| ✅ | Prefill/decode with stable `seq_id` (not batch slot), boundary-token resubmission | solid |
| ✅ | Sampling: rep-penalty → temp → top_k → stable softmax → min_p → top-nσ → top_p → draw; greedy if temp≤0; seeded | |
| ✅ | `repeat_last_n` — bounded penalty window | **0.19** — the penalties previously scanned the *whole* generated history every step: `O(generated²)` per request (`apply_frequency_presence_penalty` rebuilt a `HashMap` per token), and the Ollama surface's `repeat_penalty = 1.1` default kept penalising tokens thousands of positions back. `-1`/`0`/`n` semantics as llama.cpp. **Defaults to `-1`, so output is bit-identical unless set** — adopting llama.cpp's `64` would have silently changed output for every existing caller. Deliberate divergence, documented at the call site: fox's window covers only *generated* tokens, never the prompt |
| ✅ | `top_n_sigma`, `min_keep` | **0.19** — `top_n_sigma` keeps tokens within N σ of the top logit; invariant under temperature (scaling by 1/T scales max and σ identically), asserted by test. It composes with fox's adaptive candidate pool precisely because it is monotonic in the logit |
| ❌ | `typical_p`, `mirostat`, XTC, DRY | **deliberate, not an oversight** — fox's sampler avoids materialising the full 128K distribution, growing a candidate pool adaptively. `typical_p` ranks by `\|−log p − H\|`, which is *not* monotonic in probability, so the tokens it wants may not be in that head; mirostat carries per-request feedback state. Both need a full-vocabulary pass or a pool redesign — a design decision, not a missing line. See `docs/design/llama-server-gap-analysis.md` §3 |
| ✅ | `frequency_penalty` / `presence_penalty` | **fixed (0.11)** — applied in the sampler with OpenAI semantics (`logit -= presence*seen + frequency*count`); was accepted but silently ignored |
| ✅ | UTF-8 reassembly across tokens (split emoji/CJK) | byte buffer, no `??` artifacts |
| ✅ | Multi-piece control-token holdback (BPE-split `<|im_end|>`) | |
| ✅ | User stop sequences | rolling buffer, cross-token-boundary |
| ⚠️ | Reasoning-block delimiters are per-model via `REASONING_FORMATS` (0.11) | not a hardcoded `<think>` literal anymore (`engine/output_filter.rs`'s `think_open`/`think_close` are configurable); but the registry has only one non-default entry (Gemma/GPT-OSS `<\|channel\|>`) so an unlisted model's real markers still fall through to the `<think>` default |
| ⚠️ | `U+2581 → space` applied unconditionally | SentencePiece assumption; would corrupt a BPE model containing that codepoint |
| ✅ | "supports thinking?" detection | **improved (0.11)** — checks the model's real Jinja template for `enable_thinking` first (`supports_thinking()`, `llama_cpp/mod.rs`); falls back to the old tokenize-`"<think>"`-heuristic only when the model has no template |

## APIs

| | Feature | Notes |
|---|---------|-------|
| ✅ | OpenAI: `/v1/chat/completions` (SSE + non-stream), `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/health`, `/metrics` | |
| ✅ | Ollama: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/delete`, `/api/copy`, `/api/create`, `/api/version`, load/unload | |
| ✅ | `/v1/completions` is a real legacy completions endpoint | **fixed (0.19)** — it hard-coded `None` for `top_p`, `top_k`, `stop`, `seed`, `logprobs`, `logit_bias` and both penalties (accepted, silently ignored), and returned a `chat.completion` object while clients of this endpoint read `choices[].text`. `CompletionResponse`/`CompletionChoice` were declared and never constructed. Now threads every parameter and rewrites the response to `text_completion` (streaming too); `echo`/`suffix` are rejected with 400 rather than silently dropped. The test asserting `object == "chat.completion"` encoded the bug as intent and was replaced |
| ✅ | Real prefill/decode timing split on the Ollama surface | **fixed (0.19)** — `load_duration` and `prompt_eval_duration` were the literal constant `0`, and `total_duration`/`eval_duration` the same wall clock, on generate and chat, streaming and not. All four measured; `prefill_ms`/`decode_ms` added to the done log line. `prompt_eval_duration` includes scheduler queueing as well as prefill compute — real latency the client paid, and there is no cheaper place to separate them |
| ✅ | `usage.prompt_tokens_details.cached_tokens` | **0.19** — how many prompt tokens were served from resident KV. The only way a client can observe the KV-reuse work; omitted entirely when nothing was cached |
| ✅ | `stream_options.include_usage` honoured | **fixed (0.19)** — was parsed and ignored (usage always rode the last chunk). An explicit `false` now suppresses it; omitting `stream_options` keeps the previous behaviour, so no existing caller changes |
| ✅ | Ollama `keep_alive` per request | **fixed (0.19)** — was parsed into a `String` and thrown away, and the test asserted only the `200` it returned either way, so the field being inert was invisible. Accepts durations and bare numbers; negative pins the model, `0` sets a zero TTL. `start_eviction_task` no longer bails when `--keep-alive-secs` is 0 (a request can set a TTL where the server has none), and `unload()` clears the override. Deliberate divergence: `0` unloads on the next eviction tick (≤60s), which is also why it can never kill the request that asked for it |
| ⚠️ | Ollama options fox does not implement | **visible instead of silent (0.19)** — `num_ctx`, `num_keep`, `typical_p`, `mirostat*`, `penalize_newline` were dropped by serde. Now declared, and a warning names whichever a request set. Not rejected: clients send Ollama's defaults on every request and a 400 would break them. Still not *implemented* — `num_ctx` needs a model reload, `num_keep` exists only as the server-wide `--context-keep` |
| ✅ | `POST /tokenize`, `/detokenize`, `/apply-template` | **0.19** — llama-server parity; no inference, just the vocabulary and chat template. Every underlying piece already existed and was simply unrouted. `with_pieces` reports raw bytes for a token holding half a codepoint rather than decoding it lossily; `/apply-template` renders through the *real* request path and detokenizes, so it shows what the model actually receives |
| ✅ | `GET /props`, `GET /slots` | **0.19** — `/props` reports architecture, backend, allocated vs trained context, dimensions and capability flags read from the **loaded model**, never inferred from its filename; never triggers a load (that would evict the serving model under `--max-models 1`), so `model` is `null` when nothing is resident. `/slots` reports each sequence as free/processing/idle with resident tokens, blocks and idle time — which is what makes KV reuse observable. Slot *contents* are deliberately not exposed: a parked sequence is another user's conversation |
| ✅ | `POST /infill` (FIM) | **0.19** — emits the token layout FIM models are trained on (`[SUF] suffix [PRE] prefix [MID]`; suffix first, so the model reads what it must join up to before writing) and accepts `input_extra` for repo context. A model with no FIM tokens is **rejected with an explanation** rather than answered: prompting a chat model for infill yields fluent text that ignores the suffix, and the caller cannot tell. Verified against a real FIM model |
| ✅ | `POST /rerank`, `/v1/rerank` + `--reranking` | **0.19** — accepts both the Jina/Cohere (`documents`) and TEI (`texts`) spellings, plus `top_n`/`return_text`, preserving original indices across sorting. `--reranking` creates the context with `RANK` pooling; **this cannot be auto-detected** — a reranker GGUF does not reliably carry a `<arch>.pooling_type` key (jina-reranker-v1-tiny-en has none), so llama.cpp falls back to `NONE`. llama-server takes a flag for the same reason. A `NULL` from `llama_get_embeddings_seq` is the signal used to reject a non-reranker instead of inventing a score from a mean-pooled vector. Verified end to end: Paris ranked above the Eiffel Tower above bananas |
| ✅ | `GET`/`POST /lora-adapters` — runtime scale changes | **0.19** — list adapters and change strength without a restart. Only the *default* scale is mutable and that suffices: a request's `LoraSelection` already carries its own scale to `llama_set_adapters_lora`. A body naming one valid and one unknown adapter is resolved before anything is applied, so it changes **nothing**. Per-request `lora: [{id, scale}]` (several adapters at once) is out of scope — fox groups a decode batch by adapter *name*, so it would need set-based grouping first |
| ✅ | Raw GBNF `grammar` request field | **0.19** — the engine has had GBNF since 0.14 but the only door was `response_format`/`format`, which can only describe JSON. Setting both is a 400, not a silent precedence rule |
| ✅ | GGUF chat template rendered via real Jinja (`minijinja`) | **fixed (0.11, `cc12851`)** — was llama.cpp's legacy C engine, which doesn't run Jinja (see "Finding" below, kept as historical record); `render_chat_jinja`/`build_prompt_tokens_impl` (`engine/model/llama_cpp/vocab.rs`) render the model's actual embedded template, threading `enable_thinking`; environment compiled once per model, not per request (0.13). Falls back to the legacy built-in format only when a model has no embedded template or rendering fails. **Hardened (0.17)**: some GGUF conversions store a legacy template *name* (e.g. literally `"vicuna"`) in `tokenizer.chat_template` instead of real Jinja source — minijinja happily "renders" a no-tag string as itself, so the entire prompt silently collapsed to that one word (found via real e2e testing against `ggml-org/moondream2-20250414-GGUF` while validating vision support). `render_chat_jinja` now requires the template to contain `{{`/`{%` before trusting it as Jinja, falling through to `apply_chat_template_impl`'s name-based classifier otherwise |
| ⚠️ | Fallback template `"{role}: {content}"` when none present | may not match what the model expects |
| ✅ | Sampling defaults diverge between APIs | **intentional, documented (0.11 P4)** — centralized in `api/shared/sampling_defaults.rs`: `/v1/*` mirrors OpenAI (no `top_k`, no repeat penalty), `/api/*` mirrors Ollama (`top_k=40, repeat_penalty=1.1`); a unit test locks the divergence so it can't be "unified" by accident. Was previously undocumented duplicated literals, not a difference in behavior |
| ✅ | Optional Bearer auth (`FOX_API_KEY`), permissive CORS, OpenAI-style error mapping | |

## Product features

| | Feature | Notes |
|---|---------|-------|
| ✅ | Tool/function calling | **Hermes + Mistral + Llama3 parsers (0.16)** — `tools` is threaded into the Jinja render context, so a model whose real template natively formats tool calls (Hermes/Qwen `<tool_call>{...}</tool_call>`, Mistral `[TOOL_CALLS]`) renders and parses its own format instead of fox's generic listing; auto-detected from the model's own template (`--tool-call-parser auto\|generic\|hermes\|mistral\|llama3` to override). The Mistral parser handles both real-world wire formats (classic JSON array and the newer per-call `name[ARGS]{...}`). Llama3 (`{"name":..,"parameters":..}`, optional `<|python_tag|>`) is explicit-opt-in only — most GGUF chat templates for Llama3 models strip the tool-calling block entirely (verified against a cached `llama-3.2-1b-instruct` GGUF), so there's no reliable template signal to auto-detect it by. Models without a detected/selected native format keep the original generic prompt-based JSON parsing (`{"name","arguments"}` / `{"tool_calls":[…]}`) as the fallback |
| ✅ | JSON mode / structured output | **fixed (0.14)** — GBNF-constrained via `response_format`/`format`, JSON-schema→grammar in Rust, golden-verified; was prompt-instruction-only, no enforcement. Regex/choice-based grammar still absent. **Correctness bug fixed (0.19)**: the converter *dropped* non-`required` properties, so a schema like `{properties:{a,b}, required:[a]}` produced a grammar that **forbade** `b` — the grammar contradicted the schema rather than merely being stricter than it, and the test asserting the old behaviour encoded the bug as intent. Optional properties are now emitted as optional members (declaration order only; every permutation is exponential, and llama.cpp's own converter has the same limit). An explicitly empty `"required": []` is now distinguished from an absent `required`, which is what makes an all-optional object expressible. `anyOf`/`oneOf`/`$ref`/`$defs` still missing |
| ✅ | Thinking / `--show-thinking` | **improved (0.11)** — `enable_thinking` is threaded through the real Jinja render, and detection uses the template's own `enable_thinking` marker before falling back to a literal-`<think>` tokenize check; the reasoning-delimiter registry (`REASONING_FORMATS`) knows Gemma/GPT-OSS's `<\|channel\|>` framing. Still whack-a-mole for any *other* model family whose real marker isn't `<think>` and isn't yet in the registry |
| ⚠️ | Vision / multimodal | **shipped (0.17)** — `--mmproj <file>` loads a paired vision projector via llama.cpp's `mtmd` library; OpenAI `image_url` (base64 `data:` URI only — no remote fetch) and Ollama `images` are encoded and answered. v1 scope: one global mmproj pairing (like `--draft-model`), no fox-level chunked-prefill/prefix-caching for the image turn (atomic `mtmd_helper_eval_chunks` call — a documented tradeoff, not a bug), no OOM bisection-retry on this path. Verified end-to-end on a real model (Gemma 4 E2B, 24/24 e2e checks). See `docs/design/vision-support.md` |
| ✅ | Embeddings | **fixed**: correct length (`n_embd`), mean-pooled + L2-normalized, non-degenerate (was all-zeros due to `pooling_type=NONE`) |

## Scheduler / KV / performance

| | Feature | Notes |
|---|---------|-------|
| ✅ | Paged KV cache (PagedAttention-style): block pool, ref-count, copy-on-write | |
| ✅ | **KV reuse: resident-sequence tracking + LCP slot affinity** | **reworked (0.19)** — replaces the chained-block-hash prefix cache, which had three structural limits: **8 entries** at defaults (`max_batch_size/4`, no flag), reuse aligned to `block_size` so a 31-of-32-token match reused 16, and — worst — it cached only the *prompt*, discarding the generated reply, so multi-turn chat could never hit past the previous prompt's end. Now every sequence records what its KV holds, prompt **and** generation; a finished request parks its slot instead of freeing it; admission picks the sequence sharing the longest common prefix; reuse is token-exact. Idle slots are reclaimed LRU under block pressure — **not preemption**: they belong to requests that already finished, and `Busy` slots are never touched, so `admission_never_preempts_running_requests` is preserved verbatim. `--kv-reuse false` restores the old behaviour. Ordering is load-bearing: `ScheduledBatch::kv_trims` is applied by `run_loop` **before** prefill, or stale cells past the divergence point silently corrupt the next occupant. Measured (CPU/zen4, llama-3.2-1b, alternating arms): reuse off→on 4760→782 ms median TTFT (**6.1×**); old build→new on a 12-conversation working set, median 52.1→37.6 ms but **mean 1247.6→36.3 (34×)** and **p90 3650.5→38.7 (94×)** — the old 8-entry cache evicted a third of the set every pass. See `docs/design/llama-server-gap-analysis.md` |
| ⚠️ | Greedy output is less reproducible under concurrent load | **known consequence of the above (0.19), measured not assumed** — at `temperature: 0` with 4 concurrent clients, 2/10 rounds differed with `--kv-reuse false` and 10/10 with it on. The nondeterminism is **pre-existing**: llama.cpp does not guarantee bit-identical logits across batch compositions, and the control arm drifts too. Reuse amplifies it by collapsing prefill so requests decode alongside each other more. **Not incorrect KV** — *sequential* reuse is byte-identical across repeats, the same code path with the same cache state. A caller needing reproducibility needs a serialised request stream; `seed` does not help, the variation is in the forward pass |
| ✅ | Host-RAM prompt cache (`--cache-ram <MiB>`) | **0.19** — serialises a reclaimed sequence to host memory instead of re-prefilling it later, so a conversation stays reusable without holding a GPU block. Engine ordering is load-bearing and stated at the call site: **saves → clears → restores → trims**. A failed restore resets the request to prefill from token 0 rather than reading cells nobody wrote. Round-trip verified against a real model (`golden_state_seq_round_trip_preserves_decode`): a state restored into a *different* sequence predicts the identical token, logits matching to <1e-3. **Not a general speedup, defaults to `0`** — reclamation needs the pool exhausted *and* the claiming request needing more blocks than the slot it inherits, which sequential single-client traffic never produces (LCP affinity routes it all onto one slot). Earns its keep under concurrent distinct conversations that exhaust the pool |
| ✅ | `n>1` / `best_of` forks the prefill | **0.19** — was N independent full prefills of the identical prompt. Branch 0 prefills, the rest copy its KV via `llama_memory_seq_cp` (cheap here: `kv_unified` makes it the metadata-only path). Measured on an 801-token `n=4` request: **6.60→2.01 s (3.4×)**, one full admission and three branches reporting `cached_tokens: 800` of 801. A branch waits until its parent is *decoding* and is re-queued rather than left at the queue head (it is blocked on a sibling, not on capacity); a parent that never materialises drops it to an ordinary prefill. Multimodal and LoRA branches excluded — a boundary would not line up. Branches allocate their own block budget, so the dormant copy-on-write path stays dormant |
| ✅ | KV quantization: `f16`/`q8_0`/`q4_0`, independent K/V | TurboQuant (`turbo2/3/4`) removed when migrating to upstream llama.cpp — see CHANGELOG |
| ✅ | Chunked prefill (0.13) | `--max-prefill-chunk` (default 512): a long prompt is prefilled in chunks across scheduler steps, interleaved with other requests' decode — closes the head-of-line-blocking gap vs vLLM |
| ✅ | Context rolling on full (0.13) | `--context-shift` (default on): drops the oldest KV window when a conversation fills `n_ctx` so generation continues instead of stopping with `length`; fixed in 0.15.1 to reserve headroom so it fires *before* the boundary, not exactly at it |
| ✅ | Speculative decoding — n-gram (0.15) + draft-model (0.16) | `--speculative`: byte-identical output regardless of proposer (golden-verified); n-gram 1.78× at 98% draft acceptance on repetitive output. `--draft-model <name>` (0.16) generalizes to any text via a second resident model — vocab-fingerprint checked at load time, fails loudly on mismatch; loaded eagerly, no eviction pairing/VRAM budgeting yet (documented limitation, see `docs/design/speculative-roadmap.md`) |
| ✅ | Prefix-cache block/seq_id leak on eviction | **resolved (0.12)** — was a suspected leak, closed by a dedicated stress test (`stress_prefix_cache_no_leak`) proving allocation returns to zero after draining; the original automated flag was a false positive |
| ✅ | Multi-GPU (layer/row split, manual or auto tensor-split) | |
| ✅ | MoE CPU offload (`--moe-cpu`) via expert-tensor regex | |
| 🚧 | `--swap-fraction` | parsed but unused (placeholder — real CPU↔GPU KV swap blocked on a missing llama.cpp API). **0.18**: no longer silently ignored — warns at startup when set to a nonzero value |
| ✅ | Backpressure / fail-fast (0.16) | `--max-queue-depth` rejects a full queue with HTTP 429 instead of queueing forever; a real engine failure gets a distinct `StopReason::EngineError` and an explicit terminal token instead of silently closing the response channel |
| ✅ | OOM recovery — batch-size bisection retry (0.16) + reactive context-roll (0.18) | `do_prefill`/`do_decode` distinguish `llama_decode`'s return codes (per `llama.h`) instead of treating any non-zero as fatal: `1` ("no KV slot for the batch") retries by splitting the batch in half, recursing down to a single request before giving up. **0.18** adds the "further degrade" step once bisection bottoms out: if that one remaining request has old context to discard, `engine/run.rs` performs one reactive context roll (reusing the existing `--context-shift` mechanism) and retries the whole batch once more before falling back to `EngineError`. See `docs/design/reactive-context-rolling.md`. Observable via `ferrumox_decode_bisection_retries_total` + `tracing::warn!`/`tracing::info!` per retry/roll |
| ✅ | Prefill batch-size overflow no longer crashes the process (0.18) | A real, more severe bug found while verifying the above: several requests admitted into the same prefill step each contributed their own chunk to one shared `llama_decode` call, and their **sum** could exceed `n_batch` — llama.cpp aborts via `GGML_ASSERT(n_tokens_all <= cparams.n_batch)` for this, a hard process crash with no graceful return code (unlike `ret==1`), reachable by ordinary concurrent load under a small `--max-context-len`. Fixed by capping the aggregate per-call submission against `llama_n_batch(ctx)` (`allocate_batch_budget`), spreading any excess to the next scheduler step. See `docs/design/reactive-context-rolling.md` |
| ✅ | Copy a shared prefix from a **live** sequence | **0.19** — slot affinity only inherits *idle* sequences, so N requests arriving together behind one system prompt reused nothing from each other and each prefilled it. Under `kv_unified`, `seq_cp` shares llama.cpp's cells rather than duplicating the buffer, so a request can copy from a sibling that is already decoding. Requests behind a still-prefilling donor are deferred and re-queued, not left at the queue head. Measured vs `llama-server` (same vendored llama.cpp, Radeon 890M/Vulkan, 3 rounds, disjoint ranges): **4.0× cold TTFT at 8 concurrent clients behind a 1856-token prompt, 5.75× at 16**; whole-burst wall 3.8 s vs 16.2 s. Doubling the clients costs fox 24% more TTFT and `llama-server` 79%. `llama-server` cannot do this by construction — `get_available_slot()` skips `is_processing()` slots in both its similarity pass and its LRU fallback |
| ✅ | A shared prefix is charged to the block budget **once** | **0.19** — sharing the prefill initially left the accounting duplicated: each sharer skipped the prefill and still reserved blocks for the positions it had copied. Pool occupancy on 6 concurrent clients behind a 673-token prompt: **282 → 72 blocks**. The reservation is sized *before* allocating, so the capacity check stops turning bursts away for capacity they were never going to hold. Only *whole* blocks are shared — the block straddling the divergence point stays private, which is what guarantees a shared block never receives a write, and is why `run_decode` deliberately has no copy-on-write pass |
| ✅ | `kv_blocks_used` / `kv_blocks_total` on `/slots` | **0.19** — summing `slots[].blocks` does not give pool occupancy: a shared block is counted once by *every* slot referencing it, so that sum cannot fall when sharing works. Reading it as memory use is what hid the block-accounting win across two measurements |

## Model management / CLI

| | Feature | Notes |
|---|---------|-------|
| ✅ | Subcommands: `serve, run, pull, list, show, ps, rm, models, search, alias, bench, bench-kv`; implicit `fox <model> "prompt"` → `run` | |
| ✅ | `pull`/`search` from HuggingFace; `registry.json` (~14 curated models + aliases) | |
| ⚠️ | Ambiguous name resolution | two alias systems (registry.json vs `aliases.toml`), `:`→`-` normalization, prefix/substring match → can resolve to an unexpected file or trigger an unwanted `pull`. **One concrete case fixed (0.19)**: a model loaded via `--model-path` from outside `models_dir` was advertised in `/health` and `/props` under a name every request path then answered `404` for — `resolve_model_name` only scans `models_dir`, and `get_or_load` resolved *before* checking what was resident, so a serving model could not vouch for its own name. The registry now records where each model it loads came from and consults that before the directory scan; deliberately not forgotten on unload, or the bug would resurface at the first keep-alive expiry |
| ⚠️ | VRAM estimate `file_size × 1.8` | informational warning only; does not prevent real OOM |
| ✅ | `fox pull` downloads **sharded** GGUFs | **0.19.1** — large models publish split across `name-00001-of-00014.gguf` …; llama.cpp loads the set when handed the first part, so fetching one part left an unusable file. Kimi K3, DeepSeek V4, GLM 5.2 and MiniMax M3 were unreachable through `fox pull` and could not be catalogued. Parts are nested under a per-quant subdirectory and a repo commonly holds several differently-sized sets, so grouping keys on the split count as well as the name — both verified against real repositories, not assumed |
| ✅ | Built-in catalogue refreshed: 18 → 43 entries, 22 model families | **0.19.1** — the catalogue was not broken (all 18 old entries still resolve) but its selection had aged, so `fox pull` and every worked example handed a new user a 2024 model. Added by role rather than download count, across Qwen, Gemma, IBM Granite, AI2 Olmo, TII Falcon-H1R, Apertus, NVIDIA Nemotron, Tencent Hunyuan, Ornith. Two gaps closed where a feature shipped with nothing to run it on: no reranker existed for `/rerank`, and no MoE model for `--moe-cpu`. Every repo, filename, projector and size verified against the HuggingFace API; sizes are real byte counts. The four sharded entries carry their real size (82–594 GB) in the description as well as `size_gb` |

## Config / build / ops

| | Feature | Notes |
|---|---------|-------|
| ✅ | Config: flags + `FOX_*` env + `config.toml`, precedence flag > env > file | |
| ✅ | `build.rs`: builds llama.cpp with `GGML_BACKEND_DL`, auto-enables backends per host; ROCm FP8 patch | |
| ✅ | Prometheus metrics, JSON logs, Docker, systemd, installers | |
| ⚠️ | `vendor/llama.cpp` submodule required | without `--recurse-submodules` it won't build; stub build only via `FOX_SKIP_LLAMA=1` |

---

## Finding (2026-06-29): chat templates are not executed — no Jinja engine

> **RESOLVED in 0.11 (`cc12851`, "execute the model's real Jinja chat template", 2026-07-02).**
> Kept below as a historical record of the investigation — it documents *why* fox went with
> `minijinja` over llama.cpp's `minja`/`common_chat_*` path (bumping llama.cpp's own Jinja
> support would have meant tracking a moving upstream API; rendering in Rust with `minijinja`
> keeps the template-execution logic in fox's own tests/control). The fix shipped exactly as
> described in the "Fix" and "Implication" notes below, all three parts: (1) `minijinja` +
> `enable_thinking` threading — `engine/model/llama_cpp/vocab.rs:97-161`
> (`render_chat_jinja`/`build_prompt_tokens_impl`), template compiled once per model (0.13);
> (2) `tokenize_prompt_impl` (`vocab.rs`) parses the *template's* control tokens
> (`parse_special=true`) while user content still tokenizes literally
> (`tokenize_impl`, `add_special`/no `parse_special`) — the injection risk the finding
> flagged is handled by keeping the two tokenize paths separate; (3) `supports_thinking()`
> and `REASONING_FORMATS` (`llama_cpp/mod.rs`) detect the model's real reasoning markers
> instead of a hardcoded `<think>` literal (currently one non-default entry: Gemma/GPT-OSS's
> `<\|channel\|>` framing — an unlisted family still falls back to `<think>`). Golden test:
> `golden_chat_template_renders`. **Gap this finding did NOT originally cover, closed
> separately**: `tools` is now threaded into the Jinja context too (0.16's Hermes-parser
> work, see Product features above), so native tool-formatting macros are exercised.

fox applies chat templates through llama.cpp's **legacy C template engine**, which does
**not** run Jinja. The model's real template is detected by substring and replaced with a
hardcoded simplified format. Consequence: **thinking mode and native tool-calling are lost**
for any model whose behavior lives in its Jinja template (Gemma 4, Qwen3, …).

Verified on **Gemma 4 E2B** + pinned llama.cpp **`bc05a68`**:

- Gemma 4's GGUF ships a full Jinja template — `enable_thinking` toggle (×4), `<|think|>`
  token, tool-formatting macros.
- `apply_chat_template_impl` (`src/engine/model/llama_cpp/vocab.rs:144`) passes the template
  string to `llama_chat_apply_template`.
- That C API → `llm_chat_apply_template` (`vendor/llama.cpp/src/llama-chat.cpp:237`); **no
  `minja` exists in this commit**.
- It classifies by substring: `<start_of_turn>` → `LLM_CHAT_TEMPLATE_GEMMA`
  (`llama-chat.cpp:153`) → emits a simplified `<start_of_turn>…` format (`:372–392`) with
  **no thinking, no tools**.
- Also: `supports_thinking()` looks for the literal `<think>`, missing Gemma 4's
  `<|think|>` → reports `thinking:false`.
- Empirically: fox loaded gemma-4-E2B and answered coherently, but with `thinking:false`
  and no `<|think|>` ever emitted (the simplified template never enables it).

This is a **single root cause** behind two ⚠️ rows above (tool calling, thinking), and it
degrades fidelity for every model whose real behavior needs Jinja — so it ranks **above**
feature gaps like vision.

**Fix (architectural — belongs in the rework):** adopt a real Jinja engine — either bump
llama.cpp and use its `minja` + `common_chat_*`/`--jinja` path, or render templates in Rust
with `minijinja`, threading `enable_thinking`/tools — and detect the model's actual thinking
token (`<|think|>` vs `<think>`).

### Experiment (2026-06-29): minijinja + `enable_thinking` validates the fix

A standalone test confirmed the fix path end-to-end on the target machine (CPU,
`gemma-4-E2B`):

1. Extracted Gemma 4's real Jinja chat template from the GGUF.
2. Rendered it with **minijinja** (+ `minijinja-contrib` `pycompat`, needed for the template's
   `.get()` calls), passing `enable_thinking=true` → produced the correct
   `<|turn>system\n<|think|>\n…<|turn>model` prompt. With `enable_thinking=false` the
   `<|think|>` block is absent.
3. Temporarily patched fox to tokenize with `parse_special=true` (so `<|think|>` etc. encode as
   single control tokens, not literal text — confirmed: prompt token count dropped, `<|think|>`
   became 1 token) and fed the rendered prompt to `/v1/completions`.

**Result:** on a non-trivial problem (relative-speed word problem), Gemma 4 produced its
**native reasoning trace** in the `<|channel>thought … <channel|>` channel — thinking
activated. On trivial prompts or with `enable_thinking=false`, no thinking. The
`parse_special` patch was an experiment only and has been **reverted**.

**Implication — the thinking fix has three parts, not one:**

1. A real Jinja engine (minijinja, or llama.cpp `minja`) + thread `enable_thinking`/tools.
2. `parse_special` for the **template-added structure** so control tokens encode correctly —
   but *not* for user content (injection risk); the two must be tokenized separately.
3. Output-filter detection of the model's **actual** thinking markers — Gemma 4 uses
   `<|think|>` / `<|channel>thought`, **not** the `<think>` literal fox currently matches (so
   today fox would also leak the reasoning channel into the normal answer).

## Known issues, by severity

Mapped to the fix in the [design doc](docs/design/model-architecture-rework.md).

| # | Severity | Issue | Resolved by |
|---|----------|-------|-------------|
| 1 | ✅ Landed | `embedding_dim`→`n_embd`, embeddings pooling, KV pool follows `llama_n_ctx` | `ModelInfo` §4.1 + `fox probe` + golden tests (feature/0.11) |
| 2 | ✅ Resolved | Positional KV sizing applied to MLA/recurrent → instability in those families | **0.18** — §4.2's "ask llama.cpp, don't predict" applied via an empirical create-then-shrink retry loop at context creation (no per-arch formula, no per-arch branching); lightweight `KvMemoryClass` added for observability. Verified against real DeepSeek-V2-Lite (MLA) and Mamba (recurrent) models — also surfaced and fixed a real, separate bug where recurrent detection (`llama_memory_can_shift`) had been silently wrong since an upstream llama.cpp change. See `docs/design/mla-recurrent-kv-sizing.md` |
| 3 | ⚠️ Partial | Hardcoded control/think literals + thinking heuristic ("whack-a-mole") | Capabilities from model §4.3 — real Jinja detection + `REASONING_FORMATS` landed (0.11), but the registry covers only one non-default family; still whack-a-mole for the rest |
| 4 | ✅ Resolved | Sampling defaults diverge between APIs | **0.11 (P4)** — turned out to be a documentation/duplication problem, not a bug; centralized + the divergence is now intentional and test-locked |
| 5 | ✅ Resolved (simple scope) | Footguns: `max_models=1`, silent multimodal drop, ignored `frequency/presence_penalty`, dead `swap_fraction` | Phase P4 — multimodal drop now warns (0.11), `frequency/presence_penalty` now applied (0.11). **0.18** — `max_models=1`'s trade-off is now stated explicitly at startup instead of silent (default itself intentionally unchanged — no cross-model VRAM accounting exists yet, a separate, larger follow-up); `--swap-fraction` now warns when set to a nonzero value instead of silently doing nothing (real CPU↔GPU swap remains blocked on a missing llama.cpp API, unchanged) |
| 6 | ✅ Resolved | Prefix-cache eviction cleanup | **0.12** — stress test proved no leak; three *other* prefix-cache correctness bugs (unrelated to leak) were later found and fixed in 0.15.1 via real end-to-end testing. **0.19** — the whole block-hash cache was replaced by resident-sequence tracking, and the stress test rewritten as `stress_slot_reuse_no_leak` preserving every invariant it checked, restated over the slot table |
| 7 | ✅ Resolved | Chat templates not executed (no Jinja) → thinking + native tool-calling lost (Gemma 4, Qwen3, …) | **0.11 (`cc12851`)** — real Jinja via `minijinja`, `enable_thinking` threaded, per-model reasoning-marker detection. `tools` threading (needed for native tool-*calling* specifically) landed separately in **0.16**, tracked as item 9 below |
| 8 | ✅ Resolved | No backpressure/OOM recovery — an admission-rejected or engine-crashed request silently closes its response channel (fake 200), and a real `llama_decode` failure always killed every request in the batch even when llama.cpp itself reports it as recoverable | **0.16** — `--max-queue-depth` limit + explicit error signaling + a distinct `StopReason::EngineError`, plus batch-size-bisection retry on `llama_decode` ret==1 ("no KV slot for batch"). **0.18** — added reactive context-rolling as the further "degrade" step once bisection bottoms out (see `docs/design/reactive-context-rolling.md`), and, found by the same real-concurrent-load testing, fixed a more severe *process-crash* bug: aggregate prefill tokens across several same-step requests could exceed `n_batch`, which llama.cpp enforces via a hard `GGML_ASSERT` abort with no graceful return code — now capped before the call (`allocate_batch_budget`) |
| 9 | ✅ Resolved | Tool calling was generic prompt-based only; native per-model formats were never exercised even though real Jinja rendering exists | **0.16** — Hermes, Mistral, and Llama3 parsers, `tools` threaded into the Jinja context. Llama3 is explicit-opt-in only (unreliable template auto-detection, see item above) |
| 10 | ✅ Resolved (simple scope) | Draft-model speculative decoding (generalizes 0.15's n-gram win beyond repetitive/context-echoing output) | **0.16** — `Proposer` trait + `--draft-model`. Deliberately no eviction pairing/VRAM budgeting (operator sizes both models to fit) — see `docs/design/speculative-roadmap.md` Level 2 |
| 11 | ✅ Resolved | Guided decoding **forbade** optional properties — a schema with non-`required` fields produced a grammar that could never emit them | **0.19** — not a stricter grammar but one that contradicted the schema. Optional properties emitted as optional members; absent vs explicitly-empty `required` now distinguished. The test asserting the old behaviour had encoded the bug as intent |
| 12 | ✅ Resolved | `/v1/completions` ignored nearly every sampling parameter and returned the wrong response shape | **0.19** — see the APIs table |
| 13 | ✅ Resolved | Ollama timings were constants: `load_duration`/`prompt_eval_duration` literally `0`, `total_duration` == `eval_duration` | **0.19** — all four measured |
| 14 | ✅ Resolved | An intermittent `make e2e` failure (~1 run in 52), unexplained across two sessions | **0.19** — **it was the test.** Checks 1 and 9 asserted `finish == "length"` while sending `max_tokens: 12` with no `temperature` (stochastic 0.8 default) and no `min_tokens`, so an early EOS failed them. Measured, not waited for: 600 requests in check 9's concurrent shape gave 2 early stops (0.33% each) → ~1 failing run in 43, against the 1-in-~52 observed. Fixed with `min_tokens: 12`; verified 600/600 under identical conditions. Two sessions went into this because the first instinct was to suspect the code under change rather than the test verifying it |

**Bottom line:** the serving skeleton (batching, preemption, paged KV/CoW, prefix
caching, UTF-8/stop handling, multi-GPU, both APIs, CLI, ops) is solid, and the original
rework (items 1-7) closed most of the architecture-facts-scattered-across-layers class of
defect — real Jinja execution, centralized sampling defaults, fixed embeddings, a leak-free
prefix cache. Items 8-10 (backpressure, tool calling, draft-model speculation) are
0.16's feature work, now landed, plus vision/multimodal (0.17, see
`docs/design/vision-support.md`), LoRA adapters (0.18, see
`docs/design/lora-support.md`), multiple completions per request (0.18, see
`docs/design/n-best-of-support.md`), MLA/recurrent KV sizing (0.18, see
`docs/design/mla-recurrent-kv-sizing.md`), and reactive context-rolling plus a
process-crash fix in prefill batching (0.18, see
`docs/design/reactive-context-rolling.md`). What's left is genuine remaining
correctness debt with no work scheduled (the `REASONING_FORMATS` registry's narrow
coverage) — `docs/design/vllm-gap-analysis.md`'s feature-gap list is now fully closed;
its one remaining row (beam search) was investigated and reclassified as a deliberate
non-goal (2026-08-01): llama.cpp removed its beam-search API in 2024, vLLM itself
demoted beam search out of its fast serving path, and no major LLM API exposes
real token-level beam search today.

---

## Comparison & scope vs Ollama / vLLM

This section used to duplicate a full comparison table here; that copy drifted out of date
(it still said speculative decoding and chunked prefill were unshipped after both had
landed). The comparison is now maintained in one place —
**[`docs/design/vllm-gap-analysis.md`](docs/design/vllm-gap-analysis.md)** — and this file
just tracks the current bottom line:

Fox is a **single binary over llama.cpp/GGUF**: it competes *down* with Ollama (ease,
local-first) and looks *up* at vLLM (production throughput), and is **not** trying to become
a smaller vLLM (distributed serving, per-sequence mixed-adapter LoRA batching, non-GGUF
formats, kernel-level tensor parallel are explicit non-goals — see that doc's "What NOT
to chase").

**Already shipped since the gap analysis was last written up:** guided/structured decoding
via GBNF (0.14), logprobs/top_logprobs (0.14), min_p/logit_bias/min_tokens (0.14),
speculative decoding — n-gram (0.15) and draft-model (0.16), chunked prefill (0.13),
context rolling (0.13), backpressure/max-queue + fail-fast (0.16), Hermes/Mistral/Llama3
tool-call parsers (0.16), OOM recovery via batch-size-bisection retry (0.16),
vision/multimodal via `mtmd` (0.17, see `docs/design/vision-support.md`), single-base-model
multi-LoRA via `--lora-modules` (0.18, see `docs/design/lora-support.md`), `n`/`best_of`
multiple completions per request (0.18, see `docs/design/n-best-of-support.md`),
correct MLA/recurrent KV sizing (0.18, see `docs/design/mla-recurrent-kv-sizing.md`), and
reactive context-rolling on OOM (0.18, see `docs/design/reactive-context-rolling.md`).

**Nothing left open** on `vllm-gap-analysis.md`'s "Prioritized shortlist." Its one
remaining row, beam search, was investigated (2026-08-01) and closed as a deliberate
non-goal rather than a backlog item — see that doc for the full reasoning
(llama.cpp removed its beam-search API in 2024, vLLM demoted beam search out of its
own fast serving path, and no major LLM API exposes real token-level beam search
today; a naive fan-out approximation would just be a weaker, more expensive variant
of the `n`/`best_of` already shipped in 0.18).

**Investigated and fixed (2026-08-01):** head-to-head benchmarking against
Ollama on this machine's actual target hardware (AMD Radeon 890M) found
fox's ROCm build beats its own Vulkan build (~15%), matches Ollama at
single-request concurrency, but trailed by ~2x at concurrency 4. An early
pass wrongly concluded this was a structural llama.cpp/ggml-cuda kernel
limit — that didn't survive a direct comparison against vanilla
`llama-server` (same llama.cpp commit, no fox/Ollama), which hit 173 t/s on
the identical benchmark with no trouble, proving the compute path itself is
fine. Two real causes were found in **fox's own code**:

1. The OpenAI surface's `top_k = 0` default (deliberately matching real
   OpenAI's API — no `top_k` param) forced every sampled token through a
   full sort + doubled `exp()` pass over the ~128K-token vocab
   (~4.35ms/request); Ollama/`llama-server` never pay this since they
   always default `top_k = 40` regardless of API surface. Fixed with an
   adaptive candidate-selection algorithm in
   `src/engine/model/sampling.rs` (provably identical output distribution,
   no default changed).
2. The bigger one: `src/scheduler/mod.rs`'s `seq_id_pool` was a LIFO stack
   handing out IDs in essentially arbitrary order, and
   `do_decode_batch` emitted the `llama_batch` in scheduler-admission
   order — but llama.cpp's `split_equal` splitter only groups sequences
   into one ubatch when their seq_ids are strictly consecutive and
   increasing in emission order. Non-monotonic IDs silently collapsed a
   4-wide decode batch into four separate 1-token GPU calls — real
   serialization at the kernel level, invisible from fox's own
   scheduler-level metrics. Fixed by making the pool a min-heap (dense,
   ascending IDs) and emitting the batch in ascending-`seq_id` order.

**Net result: ~46-52 t/s → ~122-146 t/s** on the standard benchmark — fox now
matches or beats Ollama's ~110-148 t/s on this exact benchmark, and reaches
~70-85% of vanilla `llama-server`'s 173 t/s, up from ~30-35% before these
fixes.

3. **The prefix-cache follow-on, now fixed too (`kv_unified`).** The seq_id
   fix above only guaranteed dense IDs for requests getting a fresh pool
   allocation; a block-level prefix-cache *hit* inherits the donated
   request's old seq_id as-is, which drifts non-consecutive under heavy
   cache reuse — realistically common (any deployment sharing a system
   prompt across conversations) — degrading throughput back toward the
   pre-fix baseline. Sorting cannot repair it: `{0, 1, 29, 31}` still
   splits. The fix is to stop depending on ID density at all — setting
   `ctx_params.kv_unified = true` makes `llama_kv_cache::init_batch` select
   `split_simple` instead of `split_equal`, and `split_simple` has no
   consecutive-ID requirement, so the batch folds into one full-width
   ubatch whatever IDs the scheduler holds. Measured directly (via the new
   `FOX_LLAMA_LOG=1` + `LLAMA_BATCH_DEBUG=1` path, no vendor patch needed):
   average decode ubatch width under sustained load went **1.74 → 3.90 of a
   possible 4**, with zero ubatches taking the `split_equal` path.
   **ROCm throughput: median 110.9 t/s (range [72.3, 154.6]) → 158.2 t/s
   (range [155.0, 158.9])** — now above Ollama's 144-155 and at ~91% of
   vanilla `llama-server`'s 173. The collapsing range is the real tell: the
   earlier run-to-run swings were this fragmentation, not thermal noise.
   Costs no extra memory (KV total is identical, 4224 MiB either way — only
   `4096 cells × 33 streams` vs `135168 shared cells`), and the shared pool
   is strictly more flexible, since one long conversation can exceed the
   per-stream ceiling when others are idle. This also retires the crashing
   `llama_memory_seq_cp` migration approach as unnecessary — and, incidentally,
   would have unblocked it, since same-stream `seq_cp` supports partial
   ranges where cross-stream does not. Architecture coverage verified
   2026-08-02 across the shapes a KV-layout change could plausibly break —
   dense (llama-3.2), recurrent/SSM (mamba-130m), SWA (Gemma-3 `n_swa=512`,
   Gemma-4) and MLA (DeepSeek-V2-Lite) — single and 4-way concurrent on
   each.
Full evidence chain, including the abandoned "structural" theory, this
limitation, and the new `scripts/repeat_bench.sh` (multi-repetition,
warmup, alternating-order, error-discarding — built after single ad-hoc
runs proved too noisy to trust), in `docs/design/rocm-benchmarking-2026-08.md`;
the throughput-gap row in `docs/design/vllm-gap-analysis.md` §1 reflects the
corrected status.
