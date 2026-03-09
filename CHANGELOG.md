# Changelog

All notable changes to ferrum-engine are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
