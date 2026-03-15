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

### v0.5.0 (2026-03-09) — Block-level prefix caching, True CoW, and CPU↔GPU swap scaffold
- Block-level chain-hash prefix caching: `compute_block_hash` + `prompt_block_hashes`; partial prefix hits (e.g. shared system prompt)
- True CoW before decode: `is_shared` + `copy_on_write` guard in `run_decode`
- `RequestState::Swapped` + `swap_out` / `swap_in` scaffold; `--swap-fraction` flag (placeholder pending llama.cpp API)

### v0.5.1 (2026-03-09) — Output pipeline fixes and `--show-thinking`
- Fix: EOG token detection uses `llama_vocab_is_eog` — covers all end-of-generation tokens, not just the primary EOS (critical for Qwen3.5 which has 5 EOG tokens)
- Fix: `<|im_end|>` generated as 6 separate BPE tokens no longer leaks into user output — two-stage `pending_output` buffer detects multi-token control patterns and stops generation correctly
- `--show-thinking` flag for `ferrum run`: stream the `<think>…</think>` reasoning block to stdout for debugging or transparency

### v0.7.0 — Ollama Drop-in
- Full Ollama-compatible API: `POST /api/generate`, `POST /api/chat` (NDJSON streaming)
- `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `DELETE /api/delete`
- `POST /api/pull` with SSE progress events; HuggingFace Hub download backend
- `GET /api/version` for Ollama client auto-detection

### v0.8.0 — Ecosystem Ready
- Function calling: tools → system message injection + `try_parse_tool_call` post-generation
- Structured output: `response_format.type == "json_object"` → system message injection
- Embeddings: `POST /v1/embeddings`, `POST /api/embed`
- `fox show` CLI: `parse_architecture()`, `parse_quantization()`, QUANT_TAGS / ARCH_TAGS
- KV cache type configuration; enhanced model loading with GPU support

### v0.9.0 — Multi-Model
- `ModelRegistry` with DashMap + LRU eviction (`EngineEntry`, `RegistryConfig`)
- `AppState` with `registry: Arc<ModelRegistry>` replacing single engine
- Engine loops started inside `ModelRegistry::get_or_load`; aborted on eviction via `Drop`
- Keep-alive: `last_used: DashMap<String, Instant>` + background task (`start_eviction_task`)
- `--max-models`, `--keep-alive-secs` flags; `GET /api/ps` lists all loaded models
- Model name resolution: alias → exact stem → starts-with → contains

### v0.10.0 — Ollama Compat + Tools
- Aliases file: `~/.config/ferrumox/aliases.toml`; `--alias-file` flag
- Config file: `~/.config/ferrumox/config.toml` (env `$FOX_CONFIG`) loaded before CLI parse
- Request cancellation: `handle_logits` detects `send().is_err()` → `clear_sequence + mark_finished(Preempt)`
- `--model-path` now optional (lazy loading when omitted)
- Comprehensive tests for configuration and pull handler

### v1.0.0 — Production
- Full authentication and middleware support (API key validation, middleware stack)
- System metrics tracking in benchmark tool
- Binary renamed to `fox`; default models dir `~/.cache/ferrumox/models`
- All endpoints stable; production-grade error handling

---

## Phases (completed)

### Phase 1 — Functional MVP ✅
**Goal: something that works and can be demonstrated**

- OpenAI-compatible server with SSE streaming
- Continuous batching with LIFO preemption
- KV cache manager with block pool
- ferrum-bench integrated benchmark

**Deliverable:** functional server, OpenAI-compatible, with basic continuous batching

### Phase 2 — Performance ✅
**Goal: outperform vLLM in throughput on equivalent hardware**

- PageTable + true CoW + block-level prefix caching
- CPU↔GPU swap scaffold
- Stop sequences, Prometheus metrics

**Deliverable:** published benchmarks, technical blog post, first GitHub stars

### Phase 3 — Complete product ✅
**Goal: real alternative with a community**

- Multi-model registry (v0.9.0)
- Ollama drop-in compatibility (v0.7.0)
- Function calling + structured output (v0.8.0)
- Production-grade auth + middleware (v1.0.0)

---

## Future Roadmap (post-v1.0.0)

### v1.1.0 — "Benchmarks & Visibility"
**Theme:** Demonstrate publicly that ferrumox outperforms Ollama, and make it trivially easy to install on any platform. Without published data and simple installation, the performance differentiator has no credibility.

#### Observability

| Feature | Details |
|---|---|
| CI Benchmark Gate | GitHub Actions: `fox-bench --output json` on each PR; fails if TTFT P95 regresses >5% or throughput drops >3% |
| TTFT Histogram (Prometheus) | `ferrumox_ttft_seconds` histogram + `model_name` label on all existing metrics |
| OpenTelemetry (feature flag) | Spans on `run_prefill` / `run_decode` / `handle_logits`; OTLP export; `--features telemetry` |
| Grafana Dashboard Template | `docs/grafana.json` with pre-built dashboard (queue depth, TTFT, KV usage, prefix hit ratio) |
| Published Benchmarks | Results vs Ollama on RTX 3090 / A10G / A100 in `benches/results.md` with README badge |

#### Distribution

| Platform | Feature | Details |
|---|---|---|
| All | Shell install script | `curl -fsSL https://get.ferrumox.dev \| sh`; detects OS/arch, downloads correct binary from GitHub Releases |
| All | GitHub Releases matrix | CI cross-compiles and uploads binaries on every tag: `fox-linux-x86_64`, `fox-linux-aarch64`, `fox-macos-arm64`, `fox-macos-x86_64`, `fox-windows-x86_64.exe` |
| macOS | Homebrew formula | `brew install ferrumox/tap/fox`; universal binary (x86_64 + arm64 via `lipo`); code-signed + notarized for Gatekeeper |
| macOS | Metal backend | `--features metal` in build; `lparams.n_gpu_layers` routed to Metal on Apple Silicon; no CUDA required |
| Windows | WinGet package | `winget install ferrumox.fox`; MSVC-compiled binary; ships with bundled CUDA runtime DLLs (optional) |
| Windows | CPU-only build | Default Windows binary uses CPU; `--features cuda` variant available separately for users with NVIDIA GPUs |
| Linux | `.deb` / `.rpm` packages | Built in CI via `cargo-deb` and `cargo-generate-rpm`; systemd unit file included |
| Linux | AppImage | Single-file portable binary; no root, no package manager required |

**Key files:** `.github/workflows/release.yml`, `build.rs`, `Cargo.toml` (features), `Formula/fox.rb`
**Prerequisite:** none
**Success criterion:** ferrumox >2x throughput vs Ollama on 7B (RTX 3090), visible in CI; `brew install fox` works on a fresh M-series Mac; Windows binary runs without admin rights

---

### v1.2.0 — "Compute Acceleration"
**Theme:** Maximize tokens/sec on existing hardware. Closes Phase 2 pending items.

| Feature | Details |
|---|---|
| Flash Attention | `lparams.flash_attn = true` in `LlamaCppModel::load`; `--flash-attn` flag in `ServeArgs` |
| CUDA Graphs | `lparams.use_cuda_graphs = true`; expose in `RegistryConfig`; only effective on CUDA |
| Full CPU↔GPU Swap | C glue in `vendor/llama.cpp/src/ferrum_kv_transfer.cpp` with `ggml_backend_tensor_get/set`; called from `InferenceEngine::run_decode` |
| Apple Silicon — Unified Memory | Detect `ggml_backend_metal_*`; skip CPU↔GPU swap path entirely (memory is already shared); report unified memory pool in `GET /metrics` |
| Additional Samplers | min-p, mirostat v2 in `SamplingParams` and `ChatCompletionRequest` (map to existing llama.cpp APIs) |
| `fox convert` CLI | Wraps `llama-convert` for AWQ/GPTQ → GGUF; access to quantized HF models without a GGUF |

**Key files:** `src/engine/model.rs`, `src/cli/serve.rs`, `src/scheduler/mod.rs`, `vendor/llama.cpp/`
**Prerequisite:** CI benchmark gate from v1.1.0 (to measure impact of each change)
**Success criterion:** >3000 tok/s on 7B (A100), TTFT P95 <100ms

---

### v1.3.0 — "MCP + Ecosystem"
**Theme:** Make ferrumox the easiest LLM server to integrate. MCP is the highest-adoption-impact feature in the roadmap.

| Feature | Details |
|---|---|
| MCP Server | `fox mcp` subcommand; stdio transport (IDEs) + SSE on `/mcp` (axum); tools: `generate`, `chat`, `embed`; resource: model list |
| Web UI | SPA embedded at `/ui`; model selector, streaming chat, system prompt editor, sampling sliders |
| Rate Limiting per API Key | Token bucket per key in `AppState`; `--rate-limit-rpm N`, `--rate-limit-tpm N`; 429 with `Retry-After` |
| Multi-key Auth | `[[api_keys]]` in config.toml; `ApiKeyConfig` with rate_limit, allowed_models, label |
| Webhook Notifications | `--webhook-url`; POST JSON on pull complete, request finish, error |

**Key files:** `src/api/routes.rs` (AppState), `src/api/auth.rs`, new `src/mcp/`
**Prerequisite:** stable v1.2.0 baseline
**Success criterion:** ferrumox listed as compatible MCP server in Cursor/Continue.dev; Web UI functional without extra configuration

---

### v1.4.0 — "Vision & Structured Output"
**Theme:** Expand supported model types. Vision models are the most-requested feature after text LLMs.

| Feature | Details |
|---|---|
| Vision Models (LLaVA / Qwen-VL) | Extend `Model` trait with `encode_image`; preprocessing with `image` crate; `content: Vec<ContentPart>` in `ChatMessage` for OpenAI vision format |
| JSON Schema Constrained Output | `response_format: {"type": "json_schema", "json_schema": {...}}`; JSON Schema → GBNF grammar → llama.cpp sampler |
| Reranker API | `POST /v1/rerank` (Cohere-compatible); cross-encoder via embeddings path |
| RoPE Scaling | `--rope-freq-base`, `--rope-freq-scale` in `ServeArgs`; enables 32K–128K context on compatible models |

**Key files:** `src/engine/model.rs` (trait), `src/api/types.rs` (ChatMessage), `src/api/routes.rs`
**Prerequisite:** llama.cpp multimodal API audit (clip model loading)
**Success criterion:** LLaVA-1.6 answering image questions; JSON Schema output passing test suite

---

### v2.0.0 — "Scale-Out"
**Theme:** Multi-GPU and distributed inference. Major version bump due to invasive changes in `EngineEntry` and the scheduler.

| Feature | Details |
|---|---|
| Tensor Parallelism | `EngineEntry.gpu_devices: Vec<u32>`; `--tensor-parallel-size N`; sharding via llama.cpp multi-device |
| Speculative Decoding | `InferenceEngine.draft_engine: Option<Arc<...>>`; `--speculative-model`; K-token draft + verify in one forward pass |
| Prefill/Decode Disaggregation (experimental) | `--prefill-only` / `--decode-only` modes; KV transfer protocol via gRPC |
| Persistent KV Cache | Serialize prefix cache entries to disk (`~/.cache/ferrumox/kv/`); reload on restart |
| Request Prioritization | `priority: u8` field in requests; `X-Fox-Priority` header; weighted admission in scheduler |

**Key files:** `src/model_registry.rs`, `src/scheduler/mod.rs`, `src/engine/mod.rs`
**Prerequisite:** full CPU-GPU swap from v1.2.0; multi-GPU hardware for testing
**Success criterion:** >5000 tok/s on 7B (A100), TTFT P95 <50ms, 256 concurrent requests

---

### v2.1.0 — "Alternative Backends"
**Theme:** Reduce dependency on llama.cpp; new deployment targets.

| Feature | Details |
|---|---|
| candle Backend (Pure Rust) | Implement `Model` trait with `candle`; `--backend candle`; `--features candle-backend`; Llama 3 + Mistral first |
| Safetensors Loading | `fox pull` downloads safetensors when `--backend candle`; eliminates GGUF conversion step |
| WebGPU Backend (experimental) | `wgpu` compute shaders; works on DirectX 12 / Vulkan / Metal without CUDA drivers; primary path for Windows CPU-GPU users |
| CoreML Backend (Apple) | `--backend coreml`; `--features coreml`; model converted via `coremltools`; runs on ANE (Neural Engine) on M-series chips; dramatically lower power draw for small models |
| ONNX Runtime | `ort` bindings; serve fine-tuned models with PEFT/LoRA exported to ONNX |

**Key files:** `src/engine/model.rs` (stable trait boundary), new `src/engine/candle_model.rs`
**Prerequisite:** stable `Model` trait (no breaking changes since v1.x)
**Success criterion:** Llama 3.1 8B via candle within 20% of llama.cpp throughput

---

### v2.2.0 — "Platform & Plugin System"
**Theme:** Extensibility and enterprise features.

| Feature | Details |
|---|---|
| Plugin System (Samplers) | `SamplerPlugin` trait; dynamic `.so/.dll` loading; C ABI for cross-language compatibility |
| LoRA Adapter Hot-Loading | `llama_lora_adapter_init`; `--lora-path`; `lora_adapter` field in requests; multiple simultaneous adapters |
| Batch Inference API | `POST /v1/batch`; array of requests → array of responses; concurrent scheduler dispatch |
| Admin API | `GET/POST /admin/models`, `GET /admin/config`; separate `--admin-key` |
| Helm Chart | Official chart; HPA based on `ferrumox_queue_depth`; graceful shutdown with drain timeout |

**Key files:** `src/api/routes.rs`, `src/engine/model.rs`, new `charts/ferrumox/`
**Success criterion:** watermarking plugin example documented; LoRA with 5+ simultaneous adapters; Helm chart in public repo

---

## Dependency Graph

```
v1.1.0 (Benchmarks & Visibility)
    │
    └── CI gate required before v1.2.0
v1.2.0 (Compute Acceleration)
    │
    ├── v1.3.0 (MCP + Ecosystem)   ─── can advance in parallel
    └── v1.4.0 (Vision)            ─── can advance in parallel
            │
        v2.0.0 (Scale-Out)
            │
        v2.1.0 (Alt Backends)
            │
        v2.2.0 (Platform)
```

---

## Current architecture (v1.0.0)

```
┌─────────────────────────────────────────────────────┐
│                    Client                           │
│     (curl, OpenAI SDK, LangChain, Ollama CLI...)    │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE / NDJSON
┌──────────────────────▼──────────────────────────────┐
│              API Layer (axum)                       │
│  OpenAI:  /v1/chat/completions  /v1/completions     │
│           /v1/embeddings  /v1/models                │
│  Ollama:  /api/generate  /api/chat  /api/pull       │
│           /api/tags  /api/ps  /api/show             │
│  System:  /health  /metrics  /api/version           │
│  Auth:    API key middleware                        │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│           Model Registry (DashMap + LRU)            │
│  ModelRegistry · RegistryConfig                     │
│  get_or_load · LRU eviction · keep-alive task       │
│  Alias resolution: alias → stem → prefix → contains │
└───────────┬──────────────────────┬──────────────────┘
            │                      │
┌───────────▼──────────┐ ┌────────▼─────────────────┐
│   KV Cache Manager   │ │     Inference Engine      │
│   PageTable per req  │ │   prefill() + decode()    │
│   ref_count / block  │ │   stop sequences          │
│   copy_on_write      │ │   prefix KV copy          │
│   block-level hash   │ │   function calling        │
└──────────────────────┘ │   structured output       │
                         └────────┬─────────────────┘
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

## Success metrics

| Metric | v1.0.0 (current) | v1.2.0 | v2.0.0 |
|---|---|---|---|
| Tokens/sec (7B, A100) | >500 | >3 000 | >5 000 |
| TTFT P95 (512 tokens, A100) | <500ms | <100ms | <50ms |
| Concurrent requests | 64 | 64 | 256 |
| Supported model formats | GGUF | GGUF | GGUF + safetensors |
| GitHub stars | — | 500+ | 2 000+ |

---

## CLI commands

```
fox serve          # start the HTTP server
fox run            # single-shot terminal inference
                   #   --show-thinking   stream the <think>…</think> block
fox pull           # download GGUF model from HuggingFace Hub
fox show           # show model info (architecture, quantization, size)
fox mcp            # start MCP server (planned v1.3.0)
fox convert        # convert AWQ/GPTQ → GGUF (planned v1.2.0)
fox-bench          # integrated benchmark
```
