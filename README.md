# fox

Run local LLMs. Drop-in replacement for Ollama — same API, faster responses.

[![CI](https://github.com/ferrumox/fox/actions/workflows/ci.yml/badge.svg)](https://github.com/ferrumox/fox/actions/workflows/ci.yml)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](CHANGELOG.md)

---

## Quick start

```bash
# 1. Install
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh

# 2. Pull a model and start the server
fox pull llama3.2          # searches HuggingFace, picks best result
fox serve

# 3. Query it (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Llama-3.2-3B-Instruct-Q4_K_M","messages":[{"role":"user","content":"Hello!"}],"stream":true}'

# 4. Interactive REPL (no model-path needed)
fox run
```

That's it. If you're already using Ollama, just change the port from `11434` to `8080`.

---

## Client compatibility

| Client / Tool | Protocol | Status |
|---------------|----------|--------|
| Open WebUI | Ollama | ✓ Works out of the box |
| Continue.dev | Ollama | ✓ Works out of the box |
| LangChain | OpenAI | ✓ Works out of the box |
| LlamaIndex | OpenAI | ✓ Works out of the box |
| Cursor / Copilot Chat | OpenAI | ✓ Works out of the box |
| `ollama` CLI | Ollama | ✓ Works out of the box |
| `openai` Python SDK | OpenAI | ✓ Works out of the box |

See [`examples/`](examples/) for integration guides.

---

## Performance vs Ollama

RTX 4060 · Llama-3.2-3B-Instruct-Q4_K_M · 4 concurrent clients · 50 requests:

<!-- BENCH_TABLE_START -->
| Metric | fox | Ollama | Improvement |
|--------|-----|--------|-------------|
| First token (P50) | 87ms | 310ms | **+72%** |
| First token (P95) | 134ms | 480ms | **+72%** |
| Response time (P50) | 412ms | 890ms | **+54%** |
| Response time (P95) | 823ms | 1740ms | **+53%** |
| Throughput | 312 t/s | 148 t/s | **+111%** |
<!-- BENCH_TABLE_END -->

> Reproduce: `./scripts/benchmark.sh gemma3 4 50`

---

## Why is fox faster?

**Conversations get faster over time.** Fox remembers the context it already processed — system prompts and previous messages aren't re-read from scratch on every turn. Ollama does. In a long conversation, fox skips up to 75% of that work from the second message onward, which is why the first token arrives much sooner.

**Multiple users don't block each other.** Fox processes several requests at the same time instead of waiting for one to finish before starting the next. A long generation for one user doesn't delay a quick question from another.

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions — streaming + non-streaming (OpenAI) |
| POST | `/v1/completions` | Text completions (OpenAI) |
| POST | `/v1/embeddings` | Embeddings (OpenAI) |
| GET | `/v1/models` | List all models on disk |
| POST | `/api/chat` | Chat — NDJSON streaming (Ollama) |
| POST | `/api/generate` | Generate — NDJSON streaming (Ollama) |
| POST | `/api/embed` | Embeddings (Ollama) |
| GET | `/api/tags` | List models (Ollama) |
| GET | `/api/ps` | List loaded models (Ollama) |
| POST | `/api/show` | Model metadata (Ollama) |
| DELETE | `/api/delete` | Remove a model (Ollama) |
| POST | `/api/pull` | Pull a model from HuggingFace (SSE) |
| GET | `/api/version` | Server version — for Ollama client detection |
| GET | `/health` | Health + KV cache metrics |
| GET | `/metrics` | Prometheus scrape endpoint |

---

## Requirements

| Backend | Requirement |
|---------|-------------|
| CPU | x86_64 or arm64, AVX2 |
| CUDA | CUDA 12.x + cuDNN |
| Metal | macOS 13+, Apple Silicon |

No runtime dependencies — single static binary.

---

## Installation

### Linux / macOS

```bash
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
```

Supports `x86_64` and `arm64` (Apple Silicon with Metal).

### Windows

```powershell
irm https://raw.githubusercontent.com/ferrumox/fox/main/install.ps1 | iex
```

Installs `fox.exe` to `%LOCALAPPDATA%\ferrumox\bin` and offers to add it to your PATH.

### Build from source

```bash
git clone --recurse-submodules https://github.com/ferrumox/fox
cd fox

# CPU backend
cargo build --release

# CUDA (requires CUDA toolkit)
cargo build --release --features cuda

# Apple Silicon (Metal)
cargo build --release --features metal
```

Binaries: `target/release/fox` and `target/release/fox-bench`.

### Docker

```bash
# Docker Hub (no build required)
docker run -p 8080:8080 \
  -v ~/.cache/ferrumox/models:/root/.cache/ferrumox/models \
  ferrumox/fox serve

# Or build locally with docker compose
# 1. Put your GGUF model in ./models/
mkdir -p models

# 2. Start the server
docker compose up

# 3. Or pull via the API
docker compose up -d
curl -X POST http://localhost:8080/api/pull \
  -d '{"name":"llama3.2"}'
```

---

## Usage

```bash
# Search HuggingFace for GGUF models
fox search gemma
fox search qwen coder --limit 5
fox search gemma --sort likes

# Pull a model — searches HuggingFace automatically
fox pull gemma3                    # top result for "gemma3", balanced quant
fox pull gemma3:12b                # top result for "gemma3 12b"
fox pull gemma3:12b-q4             # top result for "gemma3 12b", Q4 variant
fox pull gemma3:12b-q8             # top result for "gemma3 12b", Q8 variant

# Pull a specific HuggingFace repo directly
fox pull bartowski/gemma-3-12b-it-GGUF
fox pull bartowski/gemma-3-12b-it-GGUF:q5   # repo + quant prefix

# Start server (model is optional — lazy loading if omitted)
fox serve
fox serve --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Serve multiple models simultaneously (LRU eviction)
fox serve --max-models 3

# Interactive REPL (lazy loading — no model-path needed)
fox run
fox run "Explain ownership in Rust"   # single-shot

# With a specific model
fox run --model-path ~/.cache/ferrumox/models/model.gguf

# List downloaded models
fox list

# Show model info
fox show Llama-3.2-3B-Instruct-Q4_K_M
```

---

## Configuration

All flags can also be set via environment variable or `~/.config/ferrumox/config.toml`.

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--model-path` | `FOX_MODEL_PATH` | — | GGUF model to pre-load (optional) |
| `--port` | `FOX_PORT` | 8080 | Bind port |
| `--host` | `FOX_HOST` | 0.0.0.0 | Bind host |
| `--max-models` | `FOX_MAX_MODELS` | 1 | Max models in memory simultaneously |
| `--keep-alive-secs` | `FOX_KEEP_ALIVE_SECS` | 300 | Evict idle models after N seconds |
| `--max-context-len` | `FOX_MAX_CONTEXT_LEN` | 4096 | Context window size |
| `--gpu-memory-fraction` | `FOX_GPU_MEMORY_FRACTION` | 0.85 | Fraction of GPU RAM for KV cache |
| `--max-batch-size` | `FOX_MAX_BATCH_SIZE` | 32 | Continuous batch size |
| `--system-prompt` | `FOX_SYSTEM_PROMPT` | — | System prompt injected in every request |
| `--hf-token` | `HF_TOKEN` | — | HuggingFace token for private repos |
| `--alias-file` | `FOX_ALIAS_FILE` | `~/.config/ferrumox/aliases.toml` | Short name → model stem mapping |
| `--json-logs` | `FOX_JSON_LOGS` | false | Structured JSON logs |

### Config file (`~/.config/ferrumox/config.toml`)

```toml
port = 8080
max_models = 3
keep_alive_secs = 300
system_prompt = "You are a helpful assistant."
```

### Aliases (`~/.config/ferrumox/aliases.toml`)

```toml
[aliases]
"llama3" = "Llama-3.2-3B-Instruct-Q4_K_M"
"mistral" = "Mistral-7B-Instruct-v0.3-Q4_K_M"
```

---

## Benchmark

```bash
# Build first
cargo build --release

# Single server
./target/release/fox-bench \
  --url http://localhost:8080 \
  --model llama3.2 \
  --concurrency 8 \
  --requests 100

# Compare vs Ollama (side-by-side table)
./target/release/fox-bench \
  --url http://localhost:8080 \
  --compare-url http://localhost:11434 \
  --model llama3.2

# JSON output for CI / embedding in docs
./target/release/fox-bench \
  --url http://localhost:8080 \
  --compare-url http://localhost:11434 \
  --model llama3.2 \
  --output json

# Reproducible benchmark script (saves results to benches/results.md)
./scripts/benchmark.sh llama3.2 4 50
```

Sample comparison output:

```
┌─────────────────┬──────────────┬──────────────┬──────────┐
│ Metric          │   ferrumox   │    ollama    │ Δ        │
├─────────────────┼──────────────┼──────────────┼──────────┤
│ TTFT P50        │          87ms│         310ms│ +72%     │
│ TTFT P95        │         134ms│         480ms│ +72%     │
│ Latency P50     │         412ms│         890ms│ +54%     │
│ Latency P95     │         823ms│        1740ms│ +53%     │
│ Latency P99     │        1204ms│        2600ms│ +54%     │
│ Throughput      │    312.4 t/s │    148.1 t/s │ +111%    │
└─────────────────┴──────────────┴──────────────┴──────────┘
```

---

## Features

- Runs any GGUF model (Llama, Mistral, Gemma, Qwen, and more)
- **OpenAI-compatible API** — works with any tool that supports OpenAI
- **Ollama-compatible API** — works with any tool that supports Ollama
- **Multi-model serving** — keep multiple models loaded, switch between them instantly
- **Lazy loading** — no need to specify a model upfront; fox loads it on first request
- **Function calling** and **structured JSON output** (OpenAI spec)
- **Request cancellation** — closing the connection immediately frees GPU memory
- **Prometheus metrics** — latency, throughput, memory usage out of the box
- **Config file** at `~/.config/ferrumox/config.toml`
- **Aliases** — use short names instead of full model filenames
- **Docker** and **systemd** support included

---

## Project structure

```
fox/
├── src/
│   ├── main.rs              # Entry point, config, signal handling
│   ├── metrics.rs           # Prometheus metrics registry
│   ├── model_registry.rs    # Multi-model registry with LRU eviction
│   ├── api/                 # REST API (OpenAI + Ollama compat)
│   │   ├── router.rs        # Axum router setup
│   │   ├── routes.rs        # AppState + handler wiring
│   │   ├── types.rs         # Request/response types
│   │   ├── auth.rs          # API key middleware
│   │   ├── error.rs         # Unified error types
│   │   ├── pull_handler.rs  # POST /api/pull SSE streaming
│   │   ├── mod.rs           # Re-exports
│   │   ├── v1/              # OpenAI-compat handlers
│   │   │   ├── chat.rs
│   │   │   ├── completions.rs
│   │   │   ├── embeddings.rs
│   │   │   └── models.rs
│   │   ├── ollama/          # Ollama-compat handlers
│   │   │   ├── chat.rs
│   │   │   ├── generate.rs
│   │   │   ├── embed.rs
│   │   │   └── management.rs
│   │   └── shared/          # Shared inference + streaming helpers
│   │       ├── inference.rs
│   │       ├── streaming.rs
│   │       └── digest.rs
│   ├── scheduler/           # Continuous batching + prefix cache
│   ├── kv_cache/            # PageTable, ref-counted block manager
│   ├── engine/              # Inference engine, sampling, output filtering
│   ├── cli/                 # Subcommands: serve, run, pull, list, show, ps
│   └── bin/
│       └── bench.rs         # fox-bench standalone benchmark binary
├── examples/
│   ├── curl.sh              # curl examples for all API routes
│   ├── langchain.py         # LangChain integration
│   └── openwebui.md         # Open WebUI setup guide
├── scripts/
│   └── benchmark.sh         # Reproducible benchmark vs Ollama
├── benches/
│   └── results.md           # Benchmark results (generated)
├── vendor/llama.cpp/        # Git submodule
├── Dockerfile
├── docker-compose.yml
├── fox.service              # systemd unit
├── install.sh               # One-liner installer
├── Makefile
├── CHANGELOG.md
└── Cargo.toml
```

---

## Make targets

```
make build           Compile release binaries (fox + fox-bench)
make run             Build and start the server
make dev             Start with RUST_LOG=debug
make test            Run unit tests
make check           Fast type-check (cargo check)
make bench           Run fox-bench against a running server
make docker          Build Docker image
make docker-run      Start via docker compose
make install-rust    Install Rust toolchain
make download-model  Download default model (Llama-3.2-3B Q4_K_M)
```

---

## Contributing

Issues and PRs welcome at [github.com/ferrumox/fox](https://github.com/ferrumox/fox).

To run tests:

```bash
FOX_SKIP_LLAMA=1 cargo test --all
```

---

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache 2.0](LICENSE-APACHE) — your choice.
