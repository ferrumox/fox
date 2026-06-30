# Fox — Feature & Correctness Status

A living inventory of **everything fox does** and an honest assessment of **what works
and what doesn't**. Use it to decide what to fix, in what order, and to track progress
per release.

- **Version:** 1.0.0
- **Last updated:** 2026-06-29
- **Companion:** [Model-architecture correctness rework](docs/design/model-architecture-rework.md)
  — the design that resolves most ❌/⚠️ items below.

### Assessment basis

Status is from **code review** (per-subsystem), not live runtime testing. "✅" means *no
defect found in review*, not *verified by running*. Items needing a running server or GPU
to confirm are marked ❓.

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
| ⚠️ | `max_models = 1` default | footgun: silent evict/reload churn if multiple models expected (`cli/serve.rs`) |

## Model loading & architecture handling

> This is where most defects live — architecture facts derived by formula/literal,
> scattered across layers.

| | Feature | Notes |
|---|---------|-------|
| ✅ | GGUF load via FFI with actionable failure diagnosis | magic bytes / memory / GGUF version |
| ✅ | Runtime backend detection (CUDA/ROCm/Vulkan/Metal/CPU) | one binary |
| ✅ | `head_dim` from GGUF metadata (`<arch>.attention.key_length`) | **recently patched**; was `n_embd/n_head` (wrong for Gemma/MLA) |
| ✅ | Flash attention = AUTO | **recently patched**; was forced ENABLED → Gemma softcap garbage on CUDA |
| ❌ | `embedding_dim = num_heads * head_dim` | wrong for GQA; the head_dim fix makes it worse — should be `n_embd` (`engine/model/llama_cpp/batch.rs`, `mod.rs`) |
| ❌ | KV `bytes_per_token` assumes f16 in `load()` | inconsistent with `kv_cache/mod.rs` + `new_context()` which use real KV type → wrong memory budget with quantized (`q8_0`/`q4_0`) KV |
| ❌ | Positional KV sizing applied to MLA & recurrent | MLA (DeepSeek latent KV) over-reserves; Mamba/RWKV have no per-token KV → risk of mismatch with llama.cpp's real `n_ctx` → `llama_decode failed`/hangs |
| ✅ | Recurrent/hybrid detected (`llama_memory_can_shift`); prefix caching disabled for them | historic fix (v0.3.1) |
| ⚠️❓ | `n_ctx`/`n_batch`/`n_seq` heuristic | `.max(effective_ctx)` may size the pool for ~1 sequence while `n_seq_max=32` → possible tightness under concurrency; unconfirmed |

## Inference correctness (prefill / decode / sampling / output)

| | Feature | Notes |
|---|---------|-------|
| ✅ | Prefill/decode with stable `seq_id` (not batch slot), boundary-token resubmission | solid |
| ✅ | Sampling: rep-penalty → temp → top_k → stable softmax → top_p → draw; greedy if temp≤0; seeded | |
| ⚠️ | `frequency_penalty` / `presence_penalty` accepted but ignored | silent; confuses OpenAI clients (`api/types/v1.rs`) |
| ✅ | UTF-8 reassembly across tokens (split emoji/CJK) | byte buffer, no `??` artifacts |
| ✅ | Multi-piece control-token holdback (BPE-split `<|im_end|>`) | |
| ✅ | User stop sequences | rolling buffer, cross-token-boundary |
| ❌ | Hardcoded control-token literals + `<think>` substring matching | `engine/output_filter.rs` — other conventions slip through |
| ⚠️ | `U+2581 → space` applied unconditionally | SentencePiece assumption; would corrupt a BPE model containing that codepoint |
| ⚠️ | "supports thinking?" heuristic (tokenize `"<think>"`, ≤2 tokens) | fragile; false results inevitable |

## APIs

| | Feature | Notes |
|---|---------|-------|
| ✅ | OpenAI: `/v1/chat/completions` (SSE + non-stream), `/v1/completions`, `/v1/models`, `/v1/embeddings`, `/health`, `/metrics` | |
| ✅ | Ollama: `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/delete`, `/api/copy`, `/api/create`, `/api/version`, load/unload | |
| ⚠️ | GGUF chat template via `apply_chat_template` | **legacy llama.cpp engine — does NOT run Jinja** (see finding below); the model's real template is replaced by a simplified built-in format |
| ⚠️ | Fallback template `"{role}: {content}"` when none present | may not match what the model expects |
| ❌ | Sampling defaults diverge between APIs | OpenAI `top_k=0, rep=1.0` vs Ollama `top_k=40, rep=1.1` → same request, different output by endpoint |
| ✅ | Optional Bearer auth (`FOX_API_KEY`), permissive CORS, OpenAI-style error mapping | |

## Product features

| | Feature | Notes |
|---|---------|-------|
| ⚠️ | Tool/function calling | prompt-based; parses `{"name","arguments"}` / `{"tool_calls":[…]}`; no enforcement; unknown tool → treated as text; own `[tool_call: …]` wire format — **root cause: the model's native tool format (in its Jinja template) is discarded; see finding below** |
| ⚠️ | JSON mode / structured output | prompt instruction only; no validation/grammar — best-effort |
| ⚠️ | Thinking / `--show-thinking` | hides `<think>` block, `max_thinking_chars` budget; fragile detection (literal `<think>`) **misses Gemma 4's `<|think|>`**, and the Jinja `enable_thinking` toggle is never executed — see finding below |
| ❌ | Vision / multimodal | image blocks silently dropped, no warning (`api/types/v1.rs`) |
| ⚠️ | Embeddings | implemented, but the `embedding_dim` bug → wrong dimension/values on GQA |

## Scheduler / KV / performance

| | Feature | Notes |
|---|---------|-------|
| ✅ | Paged KV cache (PagedAttention-style): block pool, ref-count, copy-on-write | |
| ✅ | Prefix caching by chained block hash; correct boundary resubmission | |
| ✅ | KV quantization: `f16`/`q8_0`/`q4_0`, independent K/V | TurboQuant (`turbo2/3/4`) removed when migrating to upstream llama.cpp — see CHANGELOG |
| ❓ | Suspected prefix-cache block/seq_id leak on eviction | review shows `pop`-before-`put` + `len>=max` guard keep it balanced (likely NO leak); needs stress test to close |
| ✅ | Multi-GPU (layer/row split, manual or auto tensor-split) | |
| ✅ | MoE CPU offload (`--moe-cpu`) via expert-tensor regex | |
| 🚧 | `--swap-fraction` | parsed but unused (placeholder) |

## Model management / CLI

| | Feature | Notes |
|---|---------|-------|
| ✅ | Subcommands: `serve, run, pull, list, show, ps, rm, models, search, alias, bench, bench-kv`; implicit `fox <model> "prompt"` → `run` | |
| ✅ | `pull`/`search` from HuggingFace; `registry.json` (~14 curated models + aliases) | |
| ⚠️ | Ambiguous name resolution | two alias systems (registry.json vs `aliases.toml`), `:`→`-` normalization, prefix/substring match → can resolve to an unexpected file or trigger an unwanted `pull` |
| ⚠️ | VRAM estimate `file_size × 1.8` | informational warning only; does not prevent real OOM |

## Config / build / ops

| | Feature | Notes |
|---|---------|-------|
| ✅ | Config: flags + `FOX_*` env + `config.toml`, precedence flag > env > file | |
| ✅ | `build.rs`: builds llama.cpp with `GGML_BACKEND_DL`, auto-enables backends per host; ROCm FP8 patch | |
| ✅ | Prometheus metrics, JSON logs, Docker, systemd, installers | |
| ⚠️ | `vendor/llama.cpp` submodule required | without `--recurse-submodules` it won't build; stub build only via `FOX_SKIP_LLAMA=1` |

---

## Finding (2026-06-29): chat templates are not executed — no Jinja engine

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
| 1 | High | `embedding_dim` and `bytes_per_token` (f16) wrong → bad embeddings & KV budget | `ModelInfo` (single source of truth) §4.1 / `KvModel` §4.2 |
| 2 | High | Positional KV sizing applied to MLA/recurrent → instability in those families | `KvModel` per architecture §4.2 |
| 3 | High | Hardcoded control/think literals + thinking heuristic ("whack-a-mole") | Capabilities from model §4.3 |
| 4 | Medium | Sampling defaults diverge between APIs | API consistency §4.4 |
| 5 | Medium | Footguns: `max_models=1`, silent multimodal drop, ignored `frequency/presence_penalty`, dead `swap_fraction` | Phase P4 |
| 6 | Low/❓ | Prefix-cache eviction cleanup | P0 stress test (open question) |
| 7 | High | Chat templates not executed (no Jinja) → thinking + native tool-calling lost (Gemma 4, Qwen3, …) | real Jinja engine + `parse_special` for template + real thinking-token detection — **fix path validated 2026-06-29 (3 parts), see finding + experiment above** |

**Bottom line:** the serving skeleton (batching, preemption, paged KV/CoW, prefix
caching, UTF-8/stop handling, multi-GPU, both APIs, CLI, ops) is solid. The defects
cluster around a **single root cause** — architecture facts derived by formula/literal and
scattered across layers — which the rework centralizes behind a regression net.

---

## Comparison & scope vs Ollama / vLLM

Fox is a **single binary over llama.cpp/GGUF**. It competes *down* with Ollama (ease,
local-first) and looks *up* at vLLM (production throughput). Gaps differ by comparison.
The **Scope** column is a deliberate decision, not just a backlog:

- 🎯 **Roadmap** — a gap fox should close to win its positioning.
- ⛔ **Out of scope** — intentionally not pursued; chasing it would change the product.
- ✅ **Have** — already at parity or better.

### vs Ollama (fox's drop-in target — these matter most)

| Status | Capability | Ollama | Fox | Scope |
|--------|------------|--------|-----|-------|
| ❌ | Vision / multimodal input | llama3.2-vision, llava, qwen-vl | image blocks silently dropped | 🎯 Roadmap |
| ❌ | Structured output with enforcement | `format: json` + JSON-schema grammar | prompt instruction only | 🎯 Roadmap |
| ⚠️ | Modelfile customization | full (SYSTEM/PARAMETER/TEMPLATE/ADAPTER) | `/api/create` parses only `FROM` + copies file | 🎯 Roadmap |
| ❌ | LoRA adapters | supported | none | 🎯 Roadmap (lower) |
| ⚠️ | Curated model catalog + push | ollama.com registry, consistent tags | HF + ~14-model `registry.json`, no push | 🎯 Roadmap (lower) |
| ⚠️ | Native per-model reasoning (`think`) | per-model handling | fragile heuristic | 🎯 Roadmap (covered by rework) |

### vs vLLM (production serving engine)

| Status | Capability | vLLM | Fox | Scope |
|--------|------------|------|-----|-------|
| ❌ | Distributed serving | tensor + pipeline parallel, multi-node | single node, llama.cpp layer split | ⛔ Out of scope |
| ❌ | Speculative decoding | draft model, n-gram, EAGLE/Medusa | none | ⛔ Out of scope (revisit later) |
| ❌ | Multi-LoRA dynamic serving | yes | none | ⛔ Out of scope |
| ❌ | Guided decoding | xgrammar/outlines (schema, regex, choice) | prompt-only | 🎯 Roadmap (shared with Ollama gap) |
| ❌ | Chunked prefill | yes (interleaves long prompts with decode) | full prompt in one batch → head-of-line blocking | 🎯 Roadmap (stability under load) |
| ❌ | Non-GGUF formats | safetensors HF, AWQ/GPTQ/FP8/Marlin | GGUF only | ⛔ Out of scope (GGUF is the niche) |
| ❌ | Logprobs / prompt_logprobs / echo | yes | not exposed | 🎯 Roadmap (lower) |
| ❌ | `n>1` / `best_of` / beam search | yes | single response only | 🎯 Roadmap (lower) |

### Common API / sampling gaps (both)

| Status | Capability | Notes | Scope |
|--------|------------|-------|-------|
| ❌ | `min_p`, `logit_bias`, `min_tokens`, `typical_p`, mirostat | absent from request types | 🎯 Roadmap (cheap wins) |
| ⚠️ | `frequency_penalty` / `presence_penalty` | accepted but ignored (silent) | 🎯 Roadmap |
| ⚠️ | Tool calling | generic prompt-based, no enforcement | vs native per-model parsers (hermes/mistral/llama3) | 🎯 Roadmap |

### Where fox already matches or beats them

| | Capability | Edge |
|---|-----------|------|
| ✅ | Single static binary, no Python/deps, runtime backend detection | far lighter to deploy than vLLM |
| ✅ | Dual API (OpenAI + Ollama) native | |
| ✅ | Continuous batching + paged KV + prefix caching + CoW | vLLM's core, present |
| ✅ | MoE CPU offload, multi-GPU | |

### Scope verdict

Fox's positioning is **"a faster, drop-in Ollama"**, not "a smaller vLLM". The gaps worth
closing, in priority order:

1. **Vision / multimodal** — biggest pure-functional gap.
2. **Structured output with grammar** — Ollama already has it; prompt-only is noticeably weaker.
3. **Chunked prefill** — stability under load (long prompts blocking others).
4. **Robust tool calling** — per-model parsers.

The vLLM-tier items (distributed, speculative decoding, multi-LoRA, non-GGUF) are a
**different product** and are deliberately out of scope; they belong in the support
contract as explicit non-goals so they don't read as silent debt.
