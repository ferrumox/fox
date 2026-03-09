# Changelog

All notable changes to ferrum-engine are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
