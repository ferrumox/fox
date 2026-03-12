# Changelog

All notable changes to ferrumox are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-03-10

### Added

- **`fox search <query>`** — search HuggingFace Hub for GGUF models in real time.
  Results are sorted by downloads (default) or likes (`--sort likes`), show repo name,
  download count, and like count, and mark already-downloaded models with `✓`.
  Supports multi-word queries (`fox search qwen coder`) and `--limit N`.

- **`fox pull` friendly name resolution** — `fox pull` now searches HuggingFace
  automatically when given a friendly name instead of a raw `owner/repo` path.
  The top result by downloads is selected.

- **`fox pull` model spec syntax** — extended pull syntax for specifying size and
  quantization without knowing the exact filename:
  - `fox pull gemma3` — top HF result, balanced quant auto-selected
  - `fox pull gemma3:12b` — top result for "gemma3 12b"
  - `fox pull gemma3:12b-q4` — top result for "gemma3 12b", Q4 variant (picks `Q4_K_M`)
  - `fox pull owner/repo:q5` — exact repo + quant prefix
  - When a quant prefix matches multiple files (e.g. `q4` → `Q4_K_M`, `Q4_K_S`, `Q4_0`),
    the most balanced variant is selected automatically (`Q4_K_M` > `Q4_K_S` > `Q4_0`).

- **`fox-bench --compare-url <URL>`** — run the same workload against two servers in
  parallel and display a side-by-side comparison table with improvement percentages for
  TTFT P50/P95, latency P50/P95/P99, and throughput. Designed for benchmarking
  ferrumox vs Ollama.

- **`fox-bench --output json`** — emit the full benchmark report as structured JSON
  (including `primary`, `comparison`, and `improvement` keys). Suitable for embedding
  in CI pipelines or README generation.

- **`fox-bench --label` / `--compare-label`** — custom labels for the two servers in
  the comparison table (defaults: `ferrumox` / `ollama`).

- **`scripts/benchmark.sh`** — reproducible benchmark script. Starts ferrumox if not
  already running, detects Ollama, runs fox-bench, and appends results to
  `benches/results.md` with timestamp, hardware, model, and all metrics.

- **`examples/curl.sh`** — curl examples for all API routes (health, models, chat,
  completions, embeddings, Ollama endpoints, JSON mode, stop sequences, metrics).

- **`examples/openwebui.md`** — step-by-step guide to connect Open WebUI to ferrumox,
  including Docker setup, multi-model config, and aliases.

- **`examples/langchain.py`** — five LangChain integration examples: basic chat,
  streaming, prompt templates + LLMChain, embeddings with cosine similarity, and
  structured JSON output.

- **README rewrite** — new developer-facing README with benchmark table, 3-command
  quick start, client compatibility table, explanation of prefix caching and continuous
  batching, full API reference, and project structure.

### Changed

- `Cargo.toml`: version bumped to `1.0.0`.
- `src/api/routes.rs`: `GET /api/version` now returns `{"version":"1.0.0"}`.

---

## [0.10.0] - 2026-03-10

### Added

- **`GET /api/version`** — returns `{"version":"0.10.0"}`. Ollama clients (Open WebUI,
  Continue.dev, etc.) call this endpoint on startup to detect the server.

- **`POST /api/generate`** (native Ollama) — generation with NDJSON streaming. Accepts `model`,
  `prompt`, `system`, `stream`, and `options` (`temperature`, `top_p`, `top_k`, `repeat_penalty`,
  `seed`, `num_predict`, `stop`). Compatible with `ollama run` and native Ollama clients.

- **`POST /api/chat`** (native Ollama) — chat with NDJSON streaming. Format `{"message":{"role","content"}}`.
  Compatible with Open WebUI using the Ollama backend.

- **Keep-alive / time-based eviction** — `--keep-alive-secs` (`FOX_KEEP_ALIVE_SECS`, default 300).
  Models idle for longer than this duration are automatically unloaded from memory.
  Background task uses `Arc::downgrade` for clean lifecycle management. Set to 0 to never evict by time.

- **`fox serve` without mandatory `--model-path`** — the flag is now optional. Without it, the
  server starts in lazy mode: models are loaded on the first request that needs them. The primary
  model for `/health` is automatically detected from the first `.gguf` in `models_dir`.

- **Request cancellation** — when a client disconnects mid-stream, the engine detects the closed
  channel (`send()` returns `Err`) and cancels the request immediately, freeing the KV cache block
  and the scheduler slot.

- **Function calling / tool use** — `tools: [{type:"function", function:{name, description, parameters}}]`
  and `tool_choice` in `POST /v1/chat/completions`. Tools are injected as a system message in JSON
  format; the response is parsed to detect `{"name":..., "arguments":{...}}` and returned as
  `tool_calls` in the response. Streaming is automatically disabled when tools are present
  (required to parse the full response).

- **Structured output** — `response_format: {"type": "json_object"}` injects the instruction
  "Respond ONLY with valid JSON" into the system prompt for compatible models.

- **Config file** — `~/.config/ferrumox/config.toml` (or `$FOX_CONFIG`). Values are applied
  before clap parses CLI arguments (via env vars), so explicit flags always take precedence.
  Supported fields: `model_path`, `host`, `port`, `max_models`, `keep_alive_secs`,
  `system_prompt`, `gpu_memory_fraction`, `max_batch_size`, `max_context_len`,
  `block_size`, `hf_token`, `alias_file`, `json_logs`.

### Changed

- `model_path` in `fox serve` is now optional (previously required).
- `--keep-alive-secs` added to `ServeArgs` (default 300 s).
- `OllamaOptions` new shared type used by both `/api/generate` and `/api/chat`.
- `ChatCompletionRequest` extended with `tools`, `tool_choice`, `response_format`.
- `ChatMessageResponse` extended with `tool_calls: Option<Vec<ToolCall>>`.

---

## [0.9.0] - 2026-03-10

### Added

- **Multi-Model support** — `ModelRegistry` loads and serves multiple models simultaneously
  with LRU eviction.
  - New `src/model_registry.rs`: `ModelRegistry`, `EngineEntry`, `RegistryConfig`.
  - `GET /api/ps` now lists **all** currently-loaded models (previously only the one model).
  - `GET /v1/models` now lists **all** `.gguf` files in `models_dir` (not just the loaded one).
  - Each inference/embedding request is routed to the correct engine based on the `model` field;
    unknown models return HTTP 404.
  - `DELETE /api/delete` now also unloads the model from the registry if it was loaded.

- **`--max-models` flag** (`FOX_MAX_MODELS` env var, default `1`) — maximum number of models
  kept in memory simultaneously; excess models are evicted LRU-first.

- **`--alias-file` flag** (`FOX_ALIAS_FILE` env var) — optional TOML file mapping short names
  to model stems (e.g. `"llama3" = "Llama-3.2-3B-Instruct-f16"`).
  Default path: `~/.config/ferrumox/aliases.toml`.

### Changed

- `AppState` replaces `engine: Arc<InferenceEngine>` with `registry: Arc<ModelRegistry>` +
  `primary_model: String`. Backward-compatible: `fox serve --model-path X.gguf` works unchanged.
- `router()` signature updated accordingly.
- Engine run-loop is now started inside `ModelRegistry::get_or_load` and aborted automatically
  on LRU eviction via `Drop` on `EngineEntry`.

---

## [0.8.0] - 2026-03-10

### Added

- **Embeddings API** — unlocks RAG pipelines (LangChain, LlamaIndex, Open WebUI RAG, etc.)
  - `POST /v1/embeddings` — OpenAI-compatible endpoint; accepts `input` as a string or array
    of strings, returns `data[].embedding` vectors.
  - `POST /api/embed` — Ollama-compatible endpoint; returns `embeddings: [[f32]]`.
  - `InferenceEngine::embed()` async method; `Model::get_embeddings()` + `Model::embedding_dim()`
    trait methods with full `LlamaCppModel` implementation via `llama_set_embeddings` /
    `llama_get_embeddings_seq` FFI and stub fallback.
  - New types: `EmbeddingInput` (untagged enum for String/Vec<String>), `EmbeddingRequest`,
    `EmbeddingObject`, `EmbeddingUsage`, `EmbeddingResponse`, `OllamaEmbedRequest`,
    `OllamaEmbedResponse`.

- **`POST /api/pull` with SSE streaming** — download models from HuggingFace Hub via the
  server API, identical to Ollama's pull flow.
  - Emits newline-delimited JSON events: `pulling manifest` → `downloading` (with `digest`,
    `total`, `completed` bytes) → `verifying sha256 digest` → `success`.
  - Automatically selects Q4_K_M quantization when available, otherwise picks the first GGUF.
  - New `--hf-token` flag on `fox serve` (also `HF_TOKEN` env var) forwarded to pulls.
  - New `AppState.hf_token` field; new file `src/api/pull_handler.rs`.
  - New types: `PullRequest`, `PullStatus`.

- **Release binaries + `install.sh`** — one-command installation.
  - `.github/workflows/release.yml` — triggered on `v*` tags; builds for four targets:
    `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`,
    `aarch64-apple-darwin`. Uploads tarballs as GitHub Release assets.
  - `install.sh` — detects OS + arch, downloads the correct tarball, installs to
    `/usr/local/bin/fox` (configurable via `--prefix`).
  - `fox.service` — systemd unit for running `fox serve` as a daemon.

### Changed

- `Cargo.toml`: version bumped to `0.8.0`.
- `src/api/routes.rs`: `router()` now takes an extra `hf_token: Option<String>` parameter.
- `src/cli/serve.rs`: `ServeArgs` gains `--hf-token` / `HF_TOKEN`.

---

## [0.7.0] - 2026-03-10

### Added

- **Ollama-compatible API layer** (`src/api/routes.rs`, `src/api/types.rs`)
  - `GET /api/tags` — lists all `.gguf` models in `~/.cache/ferrumox/models/` with name,
    size, SHA256 digest, architecture family, quantization level, and `modified_at` timestamp.
    Open WebUI and Continue.dev use this endpoint to discover available models.
  - `GET /api/ps` — returns the currently loaded model with real file size (bytes) and
    SHA256 digest looked up from disk.
  - `POST /api/show` — returns detailed metadata for a named model: architecture family,
    quantization, human-readable size, digest, modification date, and file path.
  - `DELETE /api/delete` — removes a `.gguf` file from the models directory by model name
    or filename. Returns `404` if the model is not found.
  - New response types: `OllamaModel`, `OllamaDetails`, `TagsResponse`, `PsEntry`,
    `PsResponse`, `ShowRequest`, `ShowResponse`, `DeleteRequest`.
  - SHA256 digest computed once per file via `sha2` + `hex` and cached in `AppState`
    (`Arc<Mutex<HashMap<PathBuf, String>>>`). Subsequent requests for the same file return
    instantly.
  - New dependencies: `sha2 = "0.10"`, `hex = "0.4"`.

- **`models_dir` added to `AppState`** (`src/api/routes.rs`, `src/cli/serve.rs`)
  - `router()` now accepts a `models_dir: PathBuf` parameter (default:
    `~/.cache/ferrumox/models`) used by the Ollama-compat handlers.
  - `src/cli/show::parse_architecture` and `parse_quantization` promoted to `pub(crate)`
    so they can be reused by the API layer without duplication.

### Compatibility

With v0.7.0, **Open WebUI** and **Continue.dev** work out of the box by pointing their
Ollama URL to `http://localhost:8080`. No other configuration change is required.

---

## [0.6.0] - 2026-03-10

### Added

- **CLI visual overhaul — minimalista con color** (`src/cli/theme.rs`, all CLI modules)
  - New `src/cli/theme.rs` module centralises all ANSI styling. Respects `NO_COLOR` and
    non-TTY contexts (pipes, CI) — every helper silently falls back to plain text.
  - New direct dependency: `crossterm = "0.28"`.
  - **`fox run` loading spinner** — replaces the static `"Loading model… done."` line with a
    cyan Braille spinner (`indicatif`) that clears itself and prints `  ✓  Model loaded.`
    (bold green) on success.
  - **REPL banner** — after load, prints `🦊  <model name>` (bold white), a dim separator
    and a dim hint line (`/bye o Ctrl+D para salir · N tokens`).
  - **Prompt glyph** — `  ❯ ` (bold cyan) replaces `"You: "`.
  - **Thinking spinner** — a dim Braille spinner labelled `"Thinking…"` runs while the model
    generates; cleared on the first emitted token.
  - **Role label** — `  Fox  ` (bold yellow) is printed once to stderr immediately before the
    first token, producing `  Fox  <streamed response>` inline.
  - **Per-turn timing** — dim `  N tokens · X.Xs` line printed after each assistant turn.
  - **`fox list`** — table header bold, separator dim, SIZE column blue, MODIFIED dim.
  - **`fox ps`** — table header bold, separator dim; STATUS `ok` → bold green; KV cache
    usage colour-coded (green < 50 %, yellow < 80 %, red ≥ 80 %).
  - **`fox show`** — all key/value rows use `theme::print_kv_pair` (key bold+dim, padded).
  - **`fox pull`** — post-download success line uses `  ✓  Saved to …` (bold green); hint
    lines for `fox run` / `fox serve` are dimmed.
  - **`fox serve`** — prints `  🦊  <model>  ·  listening on <addr>` (green) to stderr when
    the server is ready.

- **Interactive REPL mode for `fox run`** (`src/cli/run.rs`)
  - Running `fox run --model-path model.gguf` without a prompt now opens a conversational chat session.
  - Full message history is maintained across turns: each new turn sends the complete history through `apply_chat_template`, giving the model proper context.
  - Exit commands: `/bye`, `/exit`, `exit`, `quit`, or Ctrl+D (EOF).
  - Existing one-shot behavior (`fox run --model-path model.gguf "prompt"`) is fully preserved.
  - The engine loop stays alive across turns; no model reload between messages.

### Changed

- **Project renamed from `ferrum-engine` to `ferrumox`** — the CLI binary is now `fox`, the benchmark binary is `fox-bench`.
  - All environment variables renamed from `FERRUM_*` to `FOX_*` (e.g. `FOX_MODEL_PATH`, `FOX_PORT`).
  - Model cache directory changed from `~/.cache/ferrum/models` to `~/.cache/ferrumox/models`.
  - Prometheus metric names updated from `ferrum_*` to `ferrumox_*`.
  - Build stub flag renamed from `FERRUM_SKIP_LLAMA` to `FOX_SKIP_LLAMA`.
  - Docker image tag changed from `ferrum-engine:latest` to `ferrumox:latest`.

---

## [0.5.1] - 2026-03-09

### Added

- **`--show-thinking` flag for `ferrum run`** (`src/cli/run.rs`, `src/scheduler/batch.rs`, `src/engine/mod.rs`)
  - New `SamplingParams::show_thinking: bool` field (default `false`).
  - When `--show-thinking` is passed, the model's `<think>…</think>` reasoning block is
    forwarded to stdout instead of being silently discarded. The `<think>` and `</think>`
    tags themselves are also emitted so the user sees the complete block.
  - Thinking tokens are still excluded from API responses (`show_thinking = false` in
    `src/api/routes.rs`).
  - `PerRequestState` initialised with `show_thinking` taken from the request's
    `SamplingParams` on first token arrival (`or_insert_with` instead of `or_default`).

### Fixed

- **EOG token detection for multi-token-EOS models** (`src/engine/model.rs`, `src/engine/mod.rs`)
  - `is_eos` was computed as `token_id == self.model.eos_token_id()`, which only matched the
    *primary* EOS token.  Models like Qwen3.5 declare five EOG tokens
    (`<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`).
  - New `Model::is_eog_token(token_id) -> bool` method added to the trait and both
    implementations (`LlamaCppModel` delegates to `ffi::llama_vocab_is_eog(vocab, token)`;
    stub returns `token_id == 2`).
  - `handle_logits` now uses `self.model.is_eog_token(token_id)` so any EOG token
    correctly stops generation and produces empty output text.

- **Multi-token `<|im_end|>` leaking into user output** (`src/engine/mod.rs`)
  - Qwen3.5 (and other ChatML models running without forced-greedy sampling) may generate
    `<|im_end|>` as six individual BPE tokens: `<` (27), `|` (91), `im` (316), `_end` (6018),
    `|` (91), `>` (29) instead of the single special token 248046.  The previous per-token
    `raw.contains("<|")` check missed this because no individual fragment matched.
  - **New two-stage output pipeline**:
    1. `apply_output_filter` now returns `(String, bool)` — the emittable text plus a
       `control_stop` flag.  Text is buffered in `state.pending_output` before being
       released; `flush_pending_output` scans for complete control-token patterns
       (`CONTROL_TOKEN_PATTERNS`) and calls `find_holdback_start` to hold back any suffix
       that could still be the beginning of such a pattern.
    2. `find_holdback_start(text)` — returns the index of the first `<` from which *some*
       control-token pattern could start (i.e. the pattern `starts_with` the suffix).
       Everything before that index is safe to emit immediately; the rest stays in
       `pending_output` for the next token.
  - `handle_logits` combines the two stop signals:
    `is_stop_hit = control_stop || user_stop` (where `user_stop` comes from
    `check_stop_sequences` as before).
  - `SPECIAL_TOKEN_PATTERNS` renamed to `CONTROL_TOKEN_PATTERNS` to better reflect their
    role (end-of-turn markers that must never reach the user and must stop generation).
  - `check_stop_sequences` reverted to handle only *user-supplied* stop strings; control
    patterns are fully owned by `apply_output_filter` / `flush_pending_output`.
  - **New unit tests** covering the two-stage pipeline:
    - `test_filter_control_single_token_stopped` — single-token `<|im_end|>` triggers stop.
    - `test_filter_control_multi_token_im_end` — 5-token sequence triggers stop only at `>`.
    - `test_filter_holdback_released_on_non_pattern` — `<x` releases `<` when `x`
      confirms the sequence cannot be a control pattern.
    - `test_filter_text_before_control_token_emitted` — normal text before `<|im_end|>`
      is emitted correctly and the pattern itself stops generation.

---

## [0.5.0] - 2026-03-09

### Added

- **Block-level chain-hash prefix caching** (`src/kv_cache/mod.rs`, `src/scheduler/mod.rs`, `src/engine/mod.rs`)
  - Added `compute_block_hash(parent_hash, tokens) -> u64` and
    `prompt_block_hashes(tokens, block_size) -> Vec<u64>` to `kv_cache/mod.rs`.
    Each block's hash chains the previous block's hash with the block's token IDs
    (same design as vLLM).  Two prompts that share their first N complete blocks
    therefore produce the same chain hash at each of the first N boundaries.
  - `schedule_step` now computes block hashes on admission and searches the cache
    from the longest matching block prefix down to 1 block, enabling partial prefix
    matches: a request whose prompt starts with the same system prompt as a previous
    request reuses those cached blocks even if the rest of the prompt differs.
  - `try_insert_prefix` no longer accepts an external `token_hash` parameter; it
    computes the chain hash internally.  Only the *complete* block prefix of the
    prompt is stored — partial trailing blocks and all generation blocks are freed
    immediately, reducing memory pressure.
  - `PrefixCacheEntry.token_count` removed (derivable as `block_ids.len() × block_size`).
  - New test: `test_prefix_cache_block_level_partial_match` — verifies that request B
    (prompt = shared 16-token prefix + 4 different tokens) gets a prefix hit against
    request A (prompt = the same 16 tokens) and has `skip_prefix_tokens = 16`.

- **True copy-on-write before decode** (`src/kv_cache/mod.rs`, `src/scheduler/mod.rs`, `src/engine/mod.rs`)
  - `KVCacheManager::is_shared(block_id) -> bool` — returns `true` when `ref_count > 1`.
  - `Scheduler::cow_update_page_table(req_id, logical_idx, new_block_id)` — replaces a
    single page-table entry for a running request (called by the engine's CoW path).
  - `InferenceEngine::run_decode` now inspects every block in each request's page table
    before issuing `decode_sync`.  Any block with `ref_count > 1` is privatised via
    `KVCacheManager::copy_on_write`; the new exclusive block ID is written back via
    `cow_update_page_table`.  This guarantees a decoding request never writes into a
    block shared with the prefix cache or another future request.

- **`RequestState::Swapped` + CPU↔GPU swap scaffold** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`, `src/cli/serve.rs`, `src/cli/run.rs`)
  - New `RequestState::Swapped` variant with full documentation on the intended
    semantics and current API limitation (byte-level KV tensor transfer requires
    low-level buffer access not yet exposed by llama.cpp's public API).
  - `Scheduler::swap_out(req_id) -> bool` — transitions a `Decoding` request to
    `Swapped`; caller is responsible for the GPU→CPU KV copy before calling.
  - `Scheduler::swap_in(req_id) -> bool` — transitions a `Swapped` request back to
    `Decoding`; caller is responsible for the CPU→GPU KV copy before calling.
  - `--swap-fraction` flag added to both `ferrum serve` and `ferrum run` (env:
    `FERRUM_SWAP_FRACTION`, default `0.0`).  Accepted but no-op until the llama.cpp
    transfer API is available; enables future configuration files to specify the flag
    without breaking.

### Changed

- `engine/mod.rs` local `hash_tokens` function removed (was using `DefaultHasher`);
  replaced by `kv_cache::compute_block_hash` / `prompt_block_hashes`.
- `DefaultHasher` and `std::hash::{Hash, Hasher}` imports removed from `engine/mod.rs`.

---

## [0.4.0] - 2026-03-09

### Added

- **Unified `ferrum` CLI with subcommands** (`src/cli/`, `src/main.rs`, `Cargo.toml`)
  - The project now ships a single `ferrum` binary (renamed from `ferrum-engine`) with three
    subcommands dispatched via `clap`:
  - `ferrum serve` — start the OpenAI-compatible HTTP server. Accepts all previous flags plus
    `--system-prompt <text>` (injected as the first system message if not already present) and
    `--json-logs`. Logic extracted from `main.rs` into `src/cli/serve.rs`.
  - `ferrum run <prompt>` — single-shot terminal inference: loads the model, runs prefill +
    decode, streams tokens to stdout, then exits. Useful for quick one-off queries without
    running a server. Flags: `--model-path`, `--temperature`, `--top-p`, `--top-k`,
    `--repetition-penalty`, `--seed`, `--max-new-tokens`, `--system-prompt`,
    `--no-system-prompt`, `--ctx-len`, `--gpu-memory-fraction`, `--verbose`.
    Implemented in `src/cli/run.rs`.
  - `ferrum pull <model-id>` — download a GGUF model from HuggingFace Hub. Fetches the model
    file list from the Hub API, presents an interactive selector when multiple GGUF files are
    found (`dialoguer`), downloads with a live progress bar (`indicatif`), and saves to
    `--output-dir` (default: `./models/`). Supports `--hf-token` for private repositories.
    Implemented in `src/cli/pull.rs`.
  - `src/config.rs` deleted; all configuration is now owned by the CLI arg structs.
  - New dependencies: `indicatif = "0.17"`, `dialoguer = "0.11"`.

- **Configurable system prompt for the HTTP server** (`src/api/routes.rs`)
  - `AppState` struct introduced to hold `Arc<InferenceEngine>` + `Option<String>` system
    prompt. `router()` now takes the prompt as a parameter and injects it into every
    `chat/completions` request that doesn't already have a system message.

- **13 sampler unit tests** (`src/engine/model.rs`)
  - `sample_greedy`: argmax correctness, single-token input, tie-breaking behaviour.
  - `apply_repetition_penalty`: positive/negative logit cases, no-op on empty history,
    out-of-range token IDs.
  - `sample_token`: greedy path at temperature ≤ 0, seeded reproducibility, top-K candidate
    restriction (50-sample Monte Carlo), top-P nucleus restriction (dominant token always
    sampled), repetition penalty overrides raw logit ranking.

### Fixed

- **KV cache positional gap on hybrid/recurrent models** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`, `src/engine/model.rs`, `src/engine/mod.rs`)
  - When a prefix-cache hit was used, the decode step was starting at position
    `prompt_tokens.len() + generated_tokens` instead of the number of tokens actually
    submitted to llama.cpp. For models with a recurrent memory backend (Qwen3.5, Mamba)
    this caused `find_slot: non-consecutive token position` warnings and incoherent output.
  - Fix: added `prefilled_tokens: usize` to `InferenceRequest` (initialised to 0). After
    prefill, `InferenceEngine::run_prefill` calls `Scheduler::set_prefilled_tokens` with the
    actual count. `context_len()` returns `prefilled_tokens + generated_tokens` once set,
    falling back to `prompt_tokens.len()` only before prefill completes.
  - `Model::prefill_sync` / `LlamaCppModel::do_prefill` return type extended from
    `Vec<(u64, Logits)>` to `Vec<(u64, Logits, usize)>` (third element is `tokens_submitted`).

- **Graceful recovery on KV cache exhaustion** (`src/engine/mod.rs`)
  - `llama_decode` returning a non-zero error code (e.g. `init_batch: failed to prepare
    attention ubatches` / `decode: failed to find a memory slot`) previously propagated as a
    hard `anyhow::Error` and crashed the engine loop.
  - `run_prefill` and `run_decode` errors are now caught with `match`; affected requests are
    marked `StopReason::Length` and the engine loop continues. Subsequent requests are
    unaffected.

- **Stale `stop_reason` on preempted-request re-admission** (`src/scheduler/mod.rs`)
  - When a request was LIFO-preempted its `stop_reason` was set to `Some(Preempt)`. On
    re-admission to `Prefilling` the field was never cleared, so the engine could see a
    non-`None` stop reason on a still-active request.
  - Fix: `schedule_step` now sets `req.stop_reason = None` in both the prefix-cache-hit and
    normal admission paths before transitioning to `Prefilling`.

- **CUDA build with non-standard CUDA installations** (`build.rs`)
  - Removed the hard `CUDA_PATH` env-var requirement. `build.rs` now locates `nvcc` via
    `which nvcc`, falling back to `$CUDACXX` and then `/usr/local/cuda/bin/nvcc`. The
    resolved path is passed to CMake as `CMAKE_CUDA_COMPILER`; the parent directory is used
    to derive the CUDA library search paths.

### Changed

- **`ahash` replaces `DefaultHasher` for token hashing** (`src/scheduler/mod.rs`)
  - `hash_tokens` now uses `ahash::AHasher` (initialised from a process-stable
    `OnceLock<ahash::RandomState>`). Faster with better avalanche properties; still
    deterministic within a single run. Dependency added: `ahash = "0.8"`.

- **Prefix cache backed by `lru::LruCache`** (`src/scheduler/mod.rs`)
  - `HashMap<u64, PrefixCacheEntry>` replaced with `lru::LruCache<u64, PrefixCacheEntry>`.
    The LRU ordering is preserved in preparation for future automatic eviction (currently the
    manual capacity check is kept to avoid silent block leaks until the eviction path is
    wired through properly). Dependency added: `lru = "0.12"`.

- `src/config.rs` deleted; server configuration now lives in `src/cli/serve.rs::ServeArgs`.
- `src/api/mod.rs` now re-exports `AppState` alongside `router`.

---

## [0.3.1] - 2026-03-09

### Fixed

- **Crash on hybrid/recurrent models** (`src/engine/model.rs`, `src/engine/mod.rs`)
  - Qwen3.5, Mamba, and other hybrid architectures use `llama_memory_recurrent` instead of
    the standard attention KV cache. Calling `llama_memory_seq_cp` on those models triggered
    `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")` inside
    llama.cpp, terminating the process with `SIGABRT` on the second request with an identical
    prompt.
  - Added `Model::supports_seq_copy()` backed by `llama_memory_can_shift()`: returns `true`
    only for full KV cache backends (standard attention-only transformers).
  - `InferenceEngine::new()` stores the result as `supports_prefix_cache` and logs it at
    startup.
  - `do_prefill` now guards `llama_memory_seq_cp` with a `can_shift` check as a second safety
    net.
  - Prefix caching is automatically disabled for incompatible models; all other features
    (stop sequences, metrics, streaming usage) remain fully functional.

- **CUDA build** (`build.rs`, `Cargo.toml`)
  - Removed the optional `cudarc` dependency (only used to query GPU memory, but its
    `build.rs` requires a CUDA-version feature flag that caused `--features cuda` to fail with
    a compile error). Replaced with a `nvidia-smi` subprocess call — no extra dependencies.
  - `build.rs` now links `ggml-cuda`, `libcuda` (driver API), `libcudart`, `libcublas`, and
    `libcublasLt`, searching both `/cuda/lib64` and `/cuda/targets/x86_64-linux/lib` to
    support different CUDA installation layouts.

- **Prefix-cache boundary token position** (`src/engine/model.rs`)
  - `do_prefill` was copying positions `0..skip_prefix_tokens` via `seq_cp` and then
    submitting the last prompt token at the wrong position (`context_len` instead of
    `skip_prefix_tokens - 1`). Changed to copy `0..skip_prefix_tokens-1` and always
    re-submit the boundary token in the batch at the correct position, ensuring valid
    positional encodings and correct logits.

---

## [0.3.0] - 2026-03-09

### Added

- **PageTable — explicit logical→physical block mapping** (`src/kv_cache/mod.rs`)
  - Replaced the flat `kv_block_ids: Vec<BlockId>` field in `InferenceRequest` with a named
    `PageTable` struct. The struct encapsulates the `entries` vector
    (`logical_block_index → physical_block_id`) and exposes `block_ids()`, `len()`,
    `is_empty()`, `clear()`, and `extend()`.
  - Added `ref_count: Vec<AtomicUsize>` to `KVCacheManager` (one entry per physical block).
    `allocate` sets `ref_count = 1`; `free_blocks` decrements and only returns the block to the
    free list when the count reaches zero.
  - `retain_block(id)` — increments ref_count (used when a block is shared for prefix caching).
  - `copy_on_write(id) -> Option<BlockId>` — allocates a new exclusive block and decrements the
    shared one's ref_count; foundational for future true memory-sharing CoW.

- **Prefix caching — skip re-prefill for identical prompts** (`src/scheduler/mod.rs`, `src/engine/mod.rs`, `src/engine/model.rs`)
  - `Scheduler` now embeds a `PrefixCache: HashMap<u64, PrefixCacheEntry>` keyed by
    `hash(prompt_tokens)`. Max capacity = `max_batch_size / 4` entries.
  - When a request finishes (EOS, Length, or StopSequence), `InferenceEngine` calls
    `Scheduler::try_insert_prefix` which atomically transfers the request's `kv_seq_id` and
    `page_table` blocks into the cache (the KV data in llama.cpp is preserved — `clear_sequence`
    is skipped for cached entries).
  - On the next admission of a request with the same prompt hash, `schedule_step` detects the
    hit, transfers the cached blocks to the new request's `PageTable`, allocates only the
    generation blocks, sets `skip_prefix_tokens` and `prefix_seq_id` on the request.
  - `do_prefill` calls `llama_memory_seq_cp(mem, prefix_seq_id, new_seq_id, 0, skip_tokens)`
    inside the blocking task before building the batch, then submits only
    `prompt_tokens[skip_prefix_tokens..]` starting at the correct absolute position.
    After prefill, the engine clears the now-redundant prefix sequence and returns its ID to
    the pool via `Scheduler::return_prefix_seq_id`.
  - New `Model` trait method: `copy_sequence_range(src, dst, token_count)`, backed by
    `llama_memory_seq_cp`; no-op in the stub.
  - Counters `prefix_hits` / `prefix_misses` (atomic) on `Scheduler`; exposed on `InferenceEngine`.

- **Stop sequences** (`src/scheduler/batch.rs`, `src/engine/mod.rs`, `src/api/types.rs`)
  - `SamplingParams` gains `stop: Option<Vec<String>>`.
  - `ChatCompletionRequest.stop` accepts both a JSON string and an array (OpenAI spec). Uses a
    custom `deserialize_stop` Serde helper.
  - `StopReason::StopSequence` variant added.
  - `handle_logits` now runs a rolling-buffer stop-sequence check (last `2 × max_stop_len` chars)
    per request. The stop string is **not** emitted in the output; only the prefix before the
    match is sent to the client (OpenAI behaviour). Detection works across token boundaries.
  - Output filtering (`<think>` suppression, special token stripping) and stop sequence detection
    now share a single lock acquisition on `per_request_state` to prevent deadlocks.

- **Prometheus `/metrics` endpoint** (`src/metrics.rs`, `src/api/routes.rs`, `src/main.rs`)
  - New `GET /metrics` route returning Prometheus text exposition format (version 0.0.4).
  - Metrics registered at startup via `Metrics::new()`:
    - `ferrum_requests_total{finish_reason}` — counter
    - `ferrum_tokens_generated_total` — counter
    - `ferrum_request_latency_seconds` — histogram (10 buckets: 0.05 s … 60 s)
    - `ferrum_kv_cache_usage_ratio` — gauge
    - `ferrum_queue_depth` — gauge
    - `ferrum_active_requests` — gauge
    - `ferrum_prefix_cache_hits_total` — counter
    - `ferrum_prefix_cache_misses_total` — counter
  - Dependency added: `prometheus = "0.14"`.
  - Gauges and counter deltas are refreshed on every engine scheduling step.

- **Streaming `usage` in the final chunk** (`src/api/routes.rs`, `src/api/types.rs`)
  - The last SSE chunk (the one that carries `finish_reason`) now includes a `usage` object
    with `prompt_tokens`, `completion_tokens`, and `total_tokens`, matching the OpenAI
    streaming spec. Intermediate chunks omit the field (`skip_serializing_if = "Option::is_none"`).

### Changed

- `InferenceRequest` fields: `kv_block_ids: Vec<BlockId>` → `page_table: PageTable`;
  new fields `skip_prefix_tokens: usize`, `prefix_seq_id: Option<i32>`,
  `submitted_at: Instant` (for latency metrics).
- `InferenceEngine::new` now accepts `metrics: Option<Arc<Metrics>>`.
- `OutputFilterState` renamed to `PerRequestState`; its `text_buffer` field drives stop
  sequence detection.
- `PLAN.md` updated: Phase 1 marked completed with v0.1.0/v0.2.0 summaries; Phase 2
  progress tracked.

---

## [0.2.0] - 2026-03-09

### Added

- **Real stochastic sampling** (`src/engine/model.rs`)
  - Replaced the deterministic `sample_top_p` (which always returned the same
    token for identical logits) with a weighted random draw from the nucleus.
  - New pipeline per token: repetition penalty → temperature scaling →
    top-K masking → softmax → top-P nucleus truncation → weighted random sample.
  - Added `top_k: u32` (0 = disabled) and `repetition_penalty: f32` (1.0 =
    disabled) parameters — surfaced in `SamplingParams`, `InferenceRequest`,
    `InferenceRequestForModel`, and the OpenAI request types.
  - Added `seed: Option<u64>` for reproducible output: when set, each token
    position uses `StdRng::seed_from_u64(seed ^ token_count)`.
  - Scheduler now tracks `generated_token_ids` per request for repetition
    penalty without storing the full history in the model.
  - Dependency added: `rand = "0.8"`.

- **Docker support**
  - `Dockerfile` — multi-stage build (Rust 1.84 builder + `debian:bookworm-slim`
    runtime); ships both `ferrum-engine` and `ferrum-bench`.
  - `docker-compose.yml` — mounts a local `./models` volume; optional NVIDIA
    GPU passthrough via commented `deploy.resources` section.
  - `.dockerignore` — excludes `target/`, `models/`, `.git/`, editors.
  - `README.md` updated with a new **Docker** section.
  - `Makefile` new targets: `docker` and `docker-run`.

- **Integrated benchmark binary** (`src/bin/bench.rs` → `ferrum-bench`)
  - Launches N concurrent workers, each sending `--requests` SSE chat
    completions to the target server.
  - Reports **TTFT** (P50/P95), **total latency** (P50/P95/P99), aggregate
    **tokens/second**, and total elapsed time.
  - CLI: `--url`, `--model`, `--concurrency`, `--requests`, `--prompt`,
    `--max-tokens`.
  - Dependency added: `reqwest = "0.12"` with `json` + `stream` features.
  - `Makefile` new target: `bench` (configurable via `BENCH_CONCURRENCY`,
    `BENCH_REQUESTS`, `BENCH_PROMPT`, `BENCH_MAX_TOKENS`).
  - `README.md` updated with a new **Benchmark** section.

### Fixed

- **`llama_decode` crash: "non-consecutive token position" / "inconsistent sequence positions"**
  — Two related bugs caused `llama_decode` to return `-1` and kill the engine loop
  when more than one request was ever processed:
  1. **Unstable `seq_id`** — each request was assigned `seq_id = its_index_in_the_current_batch`.
     A request prefilled as `seq_id=0` would end up as `seq_id=1` in the next decode batch
     (if another request joined), so llama.cpp looked up the wrong KV/recurrent-memory slot.
     Fix: each request is now assigned a stable `kv_seq_id` from a pool
     (`0..max_batch_size`) at admission time; the ID does not change until the request
     finishes or is preempted.
  2. **Stale KV state on seq_id reuse** — when a request ended or was LIFO-preempted its
     seq_id was returned to the pool but the llama.cpp memory module still held all its
     cached positions. The next request that received that ID would submit tokens at
     position 0 while the context still recorded the previous occupant's last position
     (e.g. 55), causing an M-RoPE position assertion failure.
     Fix: `Model::clear_sequence(seq_id)` — backed by `llama_memory_seq_rm(mem, seq_id, 0, -1)`
     — is now called (a) in `handle_logits` when a request finishes and (b) in `run_loop`
     for every `seq_id` in `ScheduledBatch::preempted_seq_ids`, before those IDs can be
     handed to a new request.

### Changed

- `SamplingParams` struct introduced in `scheduler/batch.rs` to group all
  sampling hyper-parameters; `InferenceRequest::new` now accepts it instead of
  individual `temperature`/`top_p` arguments.
- `ChatCompletionRequest` in `api/types.rs` exposes `top_k`, `repetition_penalty`,
  and `seed` as optional JSON fields, fully forward-compatible with the
  OpenAI spec.
- `ScheduledBatch` now carries a `preempted_seq_ids: Vec<i32>` field so the engine
  can clear stale KV state immediately after preemption.
- `Scheduler::new` now accepts `max_batch_size: usize` to size the seq_id pool.
- `Model` trait gains a new required method: `clear_sequence(&self, seq_id: i32)`.

---

## [0.1.0] - 2026-03-08

Initial release.

### Added

- **OpenAI-compatible HTTP API**
  - `POST /v1/chat/completions` — streaming (SSE) and non-streaming chat
  - `POST /v1/completions` — text completion (delegates to chat endpoint)
  - `GET /v1/models` — returns the name of the loaded model derived from the file path
  - `GET /health` — KV cache usage, queue depth, active requests
- **Inference engine**
  - llama.cpp FFI backend with GGUF model support
  - Continuous batching scheduler with LIFO preemption
  - Block-based KV-cache memory manager
  - `temperature` and `top_p` sampling wired through the full pipeline (API → scheduler → model)
  - Output filtering: `<think>...</think>` blocks, `<|...|>` special tokens, SentencePiece `▁` word-boundary character
- **Configuration** (CLI flags and environment variables)
  - `--model-path` / `FERRUM_MODEL_PATH`
  - `--max-context-len` / `FERRUM_MAX_CONTEXT_LEN` (default: 4096)
  - `--gpu-memory-fraction` / `FERRUM_GPU_MEMORY_FRACTION` (default: 0.85)
  - `--max-batch-size` / `FERRUM_MAX_BATCH_SIZE` (default: 32)
  - `--block-size` / `FERRUM_BLOCK_SIZE` (default: 16)
  - `--host` / `FERRUM_HOST` (default: 0.0.0.0)
  - `--port` / `FERRUM_PORT` (default: 8080)
  - `--json-logs` / `FERRUM_JSON_LOGS`
- **Operability**
  - Graceful shutdown on SIGTERM and SIGINT
  - `tokio::sync::Notify` wakes the engine loop on new requests (replaces 100 µs polling)
  - Unified per-token scheduler update (`update_after_token`) — single lock acquisition per token
- **MIT + Apache 2.0 dual license**

### Fixed

- SSE stream now correctly emits `finish_reason: "length"` when `max_tokens` is reached (previously the client would hang waiting for more data)
- Panic on `Event::json_data(...).unwrap()` in the SSE path replaced with a safe fallback
- `CString::new(...).unwrap()` panics when role or content strings contain null bytes in `apply_chat_template`
- Partial special tokens (`<|`) no longer leak into the output stream
- `gpu_memory_fraction` out-of-range values now produce a clear error at startup instead of silently misbehaving

### Changed

- Context length is no longer hardcoded to 2048; controlled via `--max-context-len`
- Removed unused runtime dependencies: `thiserror`, `tokenizers` (HuggingFace)
- `KVCacheManager` debug set (`_allocated_set: Mutex<HashSet<BlockId>>`) removed — eliminated a second mutex acquisition on every allocate/free call
- bindgen warnings from llama.cpp FFI bindings suppressed in `ffi.rs`
