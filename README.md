# Ferrum Engine

High-performance LLM inference engine in Rust — an alternative to Ollama and vLLM.

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

## Features

- **GGUF support** via llama.cpp FFI
- **OpenAI-compatible API** (chat completions, completions, models, health)
- **Continuous batching** with LIFO preemption
- **KV-cache management** with block-based allocation
- **Temperature + top_p sampling** wired through the full pipeline
- **Output filtering** — `<think>` blocks, special tokens, SentencePiece word boundaries
- **Graceful shutdown** on SIGTERM / SIGINT

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

# Download a model (Qwen3.5 0.8B by default)
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
FERRUM_SKIP_LLAMA=1 cargo build --release
```

## Usage

```bash
# Start server
ferrum-engine --model-path /path/to/model.gguf

# With env vars
FERRUM_MODEL_PATH=/path/to/model.gguf FERRUM_PORT=8080 ferrum-engine

# Override context length and port
make run MAX_CONTEXT_LEN=8192 PORT=9000
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | Chat completions (OpenAI compatible, streaming + non-streaming) |
| POST | `/v1/completions` | Text completions |
| GET | `/v1/models` | List loaded model |
| GET | `/health` | Health check with KV cache metrics |

### Example

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 0.7,
    "stream": true
  }'
```

## Configuration

| Flag | Env | Default | Description |
|------|-----|---------|-------------|
| `--model-path` | `FERRUM_MODEL_PATH` | required | Path to GGUF model file |
| `--max-context-len` | `FERRUM_MAX_CONTEXT_LEN` | 4096 | Maximum context length in tokens |
| `--gpu-memory-fraction` | `FERRUM_GPU_MEMORY_FRACTION` | 0.85 | Fraction of GPU memory for KV cache |
| `--max-batch-size` | `FERRUM_MAX_BATCH_SIZE` | 32 | Maximum batch size for inference |
| `--block-size` | `FERRUM_BLOCK_SIZE` | 16 | Tokens per KV cache block |
| `--host` | `FERRUM_HOST` | 0.0.0.0 | Bind host |
| `--port` | `FERRUM_PORT` | 8080 | Bind port |
| `--json-logs` | `FERRUM_JSON_LOGS` | false | JSON log format (for production) |

## Make Targets

```
make install-rust    Install Rust toolchain
make download-model  Download default model (Qwen3.5 0.8B Q4_K_M)
make build           Compile release binary
make run             Build and start the server
make dev             Start with RUST_LOG=debug
make test            Run unit tests
make check           Fast type-check
```

## Project Structure

```
ferrum-engine/
├── src/
│   ├── main.rs          # Entry point, config validation, signal handling
│   ├── config.rs        # CLI/env configuration
│   ├── api/             # REST API (OpenAI compatible)
│   ├── scheduler/       # Continuous batching scheduler
│   ├── kv_cache/        # KV-cache block manager
│   └── engine/          # Inference engine + llama.cpp FFI
├── vendor/llama.cpp/    # Git submodule
├── Makefile
├── CHANGELOG.md
└── Cargo.toml
```

## License

Licensed under either of:

- [MIT License](LICENSE-MIT)
- [Apache License, Version 2.0](LICENSE-APACHE)

at your option.
