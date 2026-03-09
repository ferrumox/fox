# Plan — Ferrum Engine

## Vision
LLM inference engine in Rust, open source, with higher throughput than vLLM and simpler distribution than Ollama. Key differentiator: **a single binary, no Python, no runtime dependencies**.

---

## Released versions

### v0.1.0 (2026-03-08) — Functional MVP
- OpenAI-compatible API: `POST /v1/chat/completions` (SSE streaming + non-streaming), `POST /v1/completions`, `GET /v1/models`, `GET /health`
- Inference engine: llama.cpp FFI, GGUF, continuous batching with LIFO preemption
- KV cache manager: block pool with free_list, automatic calculation from GPU memory
- Sampling: temperature, top-p
- Full config: CLI flags + env vars
- Graceful shutdown, structured logging (tracing, JSON mode)

### v0.2.0 (2026-03-09) — Performance and observability
- Full stochastic sampling: top-K, repetition penalty, seed for reproducibility
- Critical fix: stable `kv_seq_id` per request (prevents llama_decode crashes with >1 request)
- Docker support: multi-stage Dockerfile + docker-compose
- `ferrum-bench`: integrated benchmark with TTFT P50/P95, throughput, latency P50/P95/P99
- `SamplingParams` unified struct for grouping sampling hyper-parameters

### v0.3.0 (2026-03-09) — Memory optimisation and complete API
- PageTable: explicit logical→physical block mapping per request; ref_count + CoW infrastructure per block
- Hash-based prefix caching: reuse KV cache across requests with identical prompts (skip re-prefill)
- Stop sequences: `stop: string | string[]` with rolling-buffer cross-token-boundary detection
- `GET /metrics` Prometheus: 8 metrics (request rate, latency histogram, KV usage, prefix hit ratio)
- Streaming SSE: `usage` included in the final chunk (OpenAI spec)
- Repo quality: GitHub Actions CI, `cargo fmt`, `clippy -D warnings` clean, 22 unit tests

### v0.3.1 (2026-03-09) — Patch release
- Fix: SIGABRT crash on hybrid/recurrent models (Qwen3.5, Mamba) — `llama_memory_seq_cp` is
  now only called when `llama_memory_can_shift()` returns true; prefix caching is automatically
  disabled for unsupported backends
- Fix: CUDA build — removed `cudarc` dep, corrected linker flags for `ggml-cuda` + driver API
- Fix: prefix-cache boundary token submitted at wrong position (pos n instead of n-1)

### v0.4.0 (2026-03-09) — CLI, robustness and internal improvements
- `ferrum` binary: unified CLI with `serve`, `run` (single-shot terminal inference) and `pull` (HuggingFace Hub download) subcommands
- `--system-prompt` flag for both `serve` and `run`; `--json-logs` on `serve`
- CUDA build: dynamic `nvcc` detection in `build.rs`; works with non-standard CUDA installations
- Fix: KV cache positional gap — `prefilled_tokens` field + corrected `context_len()` prevents `find_slot: non-consecutive token position` on hybrid/recurrent models
- Fix: graceful recovery on KV cache exhaustion (`llama_decode` failure now marks requests `StopReason::Length` instead of crashing the engine)
- Fix: stale `stop_reason` cleared when a preempted request is re-admitted to `Prefilling`
- Internal: `ahash` replaces `DefaultHasher` for `hash_tokens` (faster, less collision-prone)
- Internal: prefix cache uses `lru::LruCache` instead of `HashMap` (correct LRU ordering for future eviction)
- 13 unit tests for sampler pipeline (`sample_greedy`, `apply_repetition_penalty`, `sample_token`)

### v0.5.0 (next) — Block-level prefix caching and CPU↔GPU swap
- Block-level prefix caching: share KV cache for prompts with a common prefix (system prompt + different users)
- CPU↔GPU swap: persist KV cache of preempted requests in RAM instead of discarding it
- True CoW: use ref_count + copy_on_write to share blocks between active requests

---

## Phases

### Phase 1 — Functional MVP `(Month 1-3)` ✅ COMPLETED
**Goal: something that works and can be demonstrated**

```
Week 1-2: Setup and FFI
  [x] Base Rust project with Cargo workspace
  [x] build.rs with bindgen → llama.cpp
  [x] Load GGUF model and generate tokens
  [x] Functional llama.cpp backend

Week 3-4: Basic server
  [x] axum with POST /v1/chat/completions and /v1/completions
  [x] OpenAI-compatible API with SSE streaming
  [x] CLI config with clap + env vars
  [x] curl working against the server

Week 5-6: KV-Cache Manager
  [x] PhysicalBlock pool with free_list
  [x] Allocate / free / can_allocate
  [x] Automatic block calculation from available GPU memory
  [x] Manager unit tests

Week 7-8: Scheduler + Continuous Batching
  [x] InferenceRequest with states (Waiting→Prefilling→Decoding→Finished)
  [x] schedule_step() with admit / evict / LIFO preempt
  [x] Main engine loop with tokio + spawn_blocking for FFI
  [x] SSE streaming in the API
  [x] Stable kv_seq_id (multi-request crash fix)

Week 9-10: MVP polish
  [x] /health endpoint with basic metrics
  [x] GET /v1/models
  [x] Structured logging with tracing (pretty + JSON)
  [x] README with installation and examples
  [x] Official Docker image + docker-compose
  [x] ferrum-bench: integrated benchmark (TTFT, throughput, latency percentiles)
  [ ] Comparative benchmark published vs Ollama (pending execution)
```

**Deliverable:** functional server, OpenAI-compatible, with basic continuous batching

---

### Phase 2 — Performance `(Month 3-6)`
**Goal: outperform vLLM in throughput on equivalent hardware**

```
Month 3-4: Memory optimisations (v0.3.0) ✅
  [x] PageTable: explicit logical→physical mapping per request
  [x] ref_count per block (infrastructure for true CoW)
  [x] Prefix caching: reuse KV cache across requests with identical prompts (exact-match)
  [x] copy_on_write: infrastructure for shared blocks

Month 3-4: API extensions (v0.3.0) ✅
  [x] Stop sequences (stop: string[] parameter)
  [x] GET /metrics endpoint (Prometheus scrape format)
  [x] Streaming: include usage in the final chunk

Month 4-5: CLI, robustness and internals (v0.4.0) ✅
  [x] `ferrum` unified CLI: serve / run / pull subcommands
  [x] KV cache positional fix and graceful exhaustion recovery
  [x] ahash + LruCache for prefix cache
  [x] 13 sampler unit tests

Month 5-6: Advanced memory optimisations (v0.5.0)
  [ ] Block-level prefix caching (shared KV prefix, not exact-match) — the real vLLM approach
  [ ] CPU↔GPU swap for preempted requests (persist KV in RAM instead of discarding)
  [ ] True CoW: use ref_count + copy_on_write to share blocks between active requests

Month 4-5: Compute optimisations
  [ ] Flash Attention 2 via FFI (Tri Dao C++ kernels)
  [ ] AWQ and GPTQ quantisation (in addition to GGUF)
  [ ] CUDA Graphs for repetitive decode steps
  [ ] Prefill/decode overlap with CUDA streams

Month 5-6: Scaling
  [ ] Tensor parallelism (multi-GPU, one model across multiple GPUs)
  [ ] Speculative decoding (small draft model + large model)
  [ ] Prefill/decode disaggregation (separate machines)
  [ ] Full benchmark: latency P50/P95/P99, throughput, memory usage
```

**Deliverable:** published benchmarks, technical blog post, first GitHub stars

---

### Phase 3 — Complete product `(Month 6-12)`
**Goal: real alternative with a community**

```
Month 6-7: User experience
  [x] Ollama-style CLI (ferrum pull, ferrum run) — moved forward to v0.4.0
  [x] Automatic model download from HuggingFace Hub — implemented in `ferrum pull`
  [ ] Basic web UI for testing models (optional)

Month 7-8: Model ecosystem
  [ ] Native safetensors support (without llama.cpp for HF models)
  [ ] candle as a second backend (pure Rust)
  [ ] Vision model support (LLaVA, Qwen-VL)
  [ ] Embedding model support

Month 8-9: Observability and production
  [ ] Full Prometheus metrics (started in v0.3.0)
  [ ] Distributed tracing with OpenTelemetry
  [ ] Rate limiting per API key
  [ ] Basic authentication

Month 9-12: Community
  [ ] Full documentation (mdBook)
  [ ] Plugin system for custom samplers
  [ ] Apple Silicon support (Metal backend)
  [ ] CI/CD with automatic benchmarks on every PR
```

---

## Current architecture (v0.4.0)

```
┌─────────────────────────────────────────────────────┐
│                    Client                           │
│         (curl, OpenAI SDK, LangChain...)            │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────┐
│              API Layer (axum)                       │
│  /v1/chat/completions  /v1/completions              │
│  /v1/models  /health  /metrics                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Scheduler (tokio async)                  │
│  Continuous batching · LIFO preemption              │
│  PrefixCache (hash-based, exact prompt match)       │
└───────────┬──────────────────────┬──────────────────┘
            │                      │
┌───────────▼──────────┐ ┌────────▼─────────────────┐
│   KV Cache Manager   │ │     Inference Engine      │
│   PageTable per req  │ │   prefill() + decode()    │
│   ref_count / block  │ │   stop sequences          │
│   copy_on_write infra│ │   prefix KV copy          │
└──────────────────────┘ └────────┬─────────────────┘
                                  │
                       ┌──────────▼──────────────────┐
                       │       Model Backend          │
                       │  llama.cpp FFI (GGUF)        │
                       └──────────┬──────────────────┘
                                  │
                       ┌──────────▼──────────────────┐
                       │      GPU / CPU              │
                       │  CUDA · CPU-only            │
                       └─────────────────────────────┘
```

---

## Success metrics per phase

| Metric | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Tokens/second (7B, A100) | >500 | >3000 | >5000 |
| TTFT latency | <500ms | <100ms | <50ms |
| Concurrent requests | 8 | 64 | 256 |
| Supported models | GGUF | GGUF + AWQ | GGUF + safetensors |
| GitHub stars | — | 500+ | 2000+ |

---

## CLI commands

```
ferrum serve       # start the HTTP server
ferrum run         # single-shot terminal inference (available since v0.4.0)
ferrum pull        # download GGUF model from HuggingFace Hub (available since v0.4.0)
ferrum-bench       # integrated benchmark (available since v0.2.0)
```
