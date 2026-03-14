# Ferrumox

High-performance LLM inference engine in Rust — an alternative to Ollama and vLLM.

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

> **Ferrum** (iron in Latin) **+ ox** (oxidation) = rust — a meta-reference to the language it's written in.

## Features

- **GGUF support** via llama.cpp FFI
- **OpenAI-compatible API** (chat completions, completions, models, health)
- **Continuous batching** with LIFO preemption
- **PagedAttention** — logical→physical KV block mapping with ref-counted CoW infrastructure
- **Prefix caching** — block-level chain-hash prefix sharing (same design as vLLM)
- **Stop sequences** — `stop: string | string[]` halts generation at any user-defined string
- **Prometheus metrics** — scrape `/metrics` for request rates, latency histogram, KV usage, prefix hit ratio
- **Real stochastic sampling** — temperature, top_p, top_k, repetition_penalty, seed
- **Output filtering** — `<think>` blocks, special tokens, SentencePiece word boundaries
- **Graceful shutdown** on SIGTERM / SIGINT
- **Docker support** — multi-stage build + `docker compose up`
- **Integrated benchmark** — TTFT, throughput, P50/P95/P99 latency

## Prerequisites

- Rust toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- CMake 3.14+
- C++ compiler with C++17 support
- (Optional) CUDA toolkit for GPU inference
- (Optional) libclang for bindgen

## Build

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/your-org/rabbit-engine
cd rabbit-engine

# Install Rust if needed
make install-rust

# Download a model
make download-model

# Build and run
make run
```

Manual build options:

```bash
# CPU backend
cargo build --release

# CUDA
cargo build --release --features cuda

# Stub only (no llama.cpp, for CI/testing)
FOX_SKIP_LLAMA=1 cargo build --release
```

## Usage

```bash
# Pull a model from HuggingFace
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF

# Start server
fox serve --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# With env vars
FOX_MODEL_PATH=~/.cache/ferrumox/models/model.gguf FOX_PORT=8080 fox serve

# Single-shot inference
fox run --model-path ~/.cache/ferrumox/models/model.gguf "Explain what Rust is"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (OpenAI compatible, streaming + non-streaming) |
| POST | `/v1/completions` | Text completions |
| GET | `/v1/models` | List loaded model |
| GET | `/health` | Health check with KV cache metrics |
| GET | `/metrics` | Prometheus scrape endpoint |

### Example

```bash
# Streaming chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": true
  }'

# With stop sequences (generation stops before emitting the stop string)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "List 3 items:"}],
    "stop": ["\n4.", "User:"],
    "max_tokens": 200
  }'

# Prometheus metrics
curl http://localhost:8080/metrics
```

## Docker

The fastest way to get started without a Rust toolchain:

```bash
# 1. Put your GGUF model in ./models/
mkdir -p models
# cp /path/to/my-model.gguf models/model.gguf

# 2. Start the server
docker compose up          # or: make docker-run

# 3. (optional) build only
make docker
# docker run -v ./models:/models -e FOX_MODEL_PATH=/models/model.gguf \
#   -p 8080:8080 ferrumox
```

Edit `docker-compose.yml` to change the model path or environment variables.
Uncomment the `deploy.resources` section to pass an NVIDIA GPU into the container.

## Benchmark

Run the built-in benchmark tool against a running server:

```bash
# Quick smoke test (server must be running)
make bench

# Custom run
./target/release/fox-bench \
  --url http://localhost:8080 \
  --model my-model \
  --concurrency 8 \
  --requests 100 \
  --max-tokens 256
```

Sample output:

```
fox-bench
  URL         : http://localhost:8080
  Model       : my-model
  Concurrency : 8
  Requests    : 100
  Max tokens  : 256

Results (100 ok, 0 errors)
─────────────────────────────────────────
  TTFT        P50:     87ms   P95:    134ms
  Latency     P50:    412ms   P95:    823ms   P99:   1204ms
  Throughput  : 312.4 tokens/sec
  Total time  : 14.2s
  Tokens out  : 4438
```

## Configuration

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--model-path` | `FOX_MODEL_PATH` | required | Path to GGUF model file |
| `--max-context-len` | `FOX_MAX_CONTEXT_LEN` | 4096 | Maximum context length in tokens |
| `--gpu-memory-fraction` | `FOX_GPU_MEMORY_FRACTION` | 0.85 | Fraction of GPU memory for KV cache |
| `--max-batch-size` | `FOX_MAX_BATCH_SIZE` | 32 | Maximum batch size for inference |
| `--block-size` | `FOX_BLOCK_SIZE` | 16 | Tokens per KV cache block |
| `--host` | `FOX_HOST` | 0.0.0.0 | Bind host |
| `--port` | `FOX_PORT` | 8080 | Bind port |
| `--json-logs` | `FOX_JSON_LOGS` | false | JSON log format (for production) |

## Make Targets

```
make install-rust    Install Rust toolchain
make download-model  Download default model (Qwen3.5 0.8B Q4_K_M)
make build           Compile release binaries (fox + fox-bench)
make run             Build and start the server
make dev             Start with RUST_LOG=debug
make test            Run unit tests
make check           Fast type-check
make bench           Run benchmark against a running server
make docker          Build Docker image
make docker-run      Start via docker compose
```

## Project Structure

```
ferrumox/
├── src/
│   ├── main.rs          # Entry point, config validation, signal handling
│   ├── metrics.rs       # Prometheus metrics registry
│   ├── api/             # REST API (OpenAI compatible) + /metrics endpoint
│   ├── scheduler/       # Continuous batching scheduler + prefix cache
│   ├── kv_cache/        # PageTable, ref-counted block manager
│   ├── engine/          # Inference engine, stop sequences, output filtering
│   └── bin/
│       └── bench.rs     # Standalone benchmark binary (fox-bench)
├── vendor/llama.cpp/    # Git submodule
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── CHANGELOG.md
└── Cargo.toml
```

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
