# Contributing to fox

Thank you for your interest in contributing! This guide covers how to build the project, run tests, understand the architecture, and open pull requests.

## Building and running tests

```bash
# Clone with submodules (llama.cpp is a Git submodule — REQUIRED)
git clone --recurse-submodules https://github.com/ferrumox/fox
cd fox

# Standard build — GPU backend (CUDA / ROCm / Vulkan / Metal) detected at runtime.
# build.rs compiles llama.cpp via CMake, so this needs the submodule + a C/C++ toolchain.
cargo build --release

# Stub build — sets cfg(fox_stub) and swaps in a no-op model, so the crate builds and
# tests run without llama.cpp, the submodule, or GPU drivers. CI runs entirely like this.
FOX_SKIP_LLAMA=1 cargo build

# Fast type-check (no codegen)
cargo check
```

`FOX_SKIP_LLAMA=1` is the key escape hatch for any work that doesn't touch real inference.

### GPU backends

`build.rs` picks the GPU backend **at build time** from the toolchains it finds on the
host (CUDA via `nvcc`, ROCm via `hipcc`, Vulkan via `glslc`, Metal on macOS), and the
backend is then `dlopen`-ed at runtime — one binary runs on GPU or CPU, falling back to
CPU automatically when no GPU is present.

**Key split:** *building* a GPU backend needs its toolchain; *running* it only needs the
GPU driver. So you can build in a container and run the binary on the host.

**Vulkan** (AMD / Intel iGPUs and any Vulkan GPU — validated on an AMD Radeon 890M
`gfx1150`). Building the Vulkan backend needs, on top of the standard `cmake clang
libclang-dev ninja-build`:

```
glslc  glslang-tools  libvulkan-dev  spirv-headers
```

(llama.cpp's ggml-vulkan CMake wants both the Vulkan/`glslc` tooling **and**
`SPIRV-Headers`.) These are all packaged on Ubuntu 24.04; 22.04 does not ship `glslc`
easily. The reproducible way to build + run is [`Dockerfile.vulkan`](Dockerfile.vulkan):

```bash
# Build the image (installs the toolchain above and compiles the Vulkan backend)
docker build -f Dockerfile.vulkan -t fox:vulkan .

# Run with the host GPU passed through (the image ships the Mesa driver)
docker run --rm --device /dev/dri --group-add video \
  -p 8080:8080 -v ~/.cache/ferrumox/models:/root/.cache/ferrumox/models \
  fox:vulkan serve

# …or extract the self-contained bundle and run it natively on a host that already
# has a Vulkan driver (Mesa/RADV, etc.):
id=$(docker create fox:vulkan) \
  && docker cp "$id:/usr/local/lib/fox" ./fox-vulkan && docker rm "$id" \
  && ./fox-vulkan/fox serve --model-path <model.gguf>
```

`make vulkan` wraps the build-and-extract step: it produces `./fox-vulkan/` ready to
run (`./fox-vulkan/fox serve --model-path <model.gguf>`).

## Code style and CI

`make ci` runs exactly what CI runs — do this before pushing:

```bash
FOX_SKIP_LLAMA=1 cargo fmt --all -- --check
FOX_SKIP_LLAMA=1 cargo clippy --all-targets --features test-helpers -- -D warnings
FOX_SKIP_LLAMA=1 cargo test --all --features test-helpers
```

- **Format**: run `cargo fmt` before committing — CI rejects unformatted code.
- **Lints**: the project must compile clean under `clippy -- -D warnings`.
- `make setup` installs a pre-push git hook that runs these automatically.

## Architecture overview

Understanding how the pieces fit together makes it much easier to know where to make changes.

```
CLI (src/cli/)
    │
    ├─ fox serve  →  builds AppState (ModelRegistry + config)
    │               starts Axum HTTP server
    │
    └─ fox run    →  loads one model directly, runs inference loop

API layer (src/api/)
    ├─ router.rs       Axum router + AppState; routes.rs lists the routes
    ├─ v1/             OpenAI-compatible handlers (chat, completions, embeddings, models)
    ├─ ollama/         Ollama-compatible handlers (chat, generate, embed, management)
    ├─ shared/         Inference + streaming helpers reused by both API families
    ├─ types/          Request/response types split by surface (v1, ollama, embeddings, …)
    ├─ auth.rs         Optional FOX_API_KEY bearer-token middleware
    └─ pull_handler.rs POST /api/pull — SSE download from HuggingFace

Model registry (src/model_registry/)
    ModelRegistry — DashMap<String, EngineEntry> + LRU eviction
    get_or_load()  — loads a model on first request, starts its engine loop
    EngineEntry    — holds Arc<InferenceEngine>, aborted on Drop (eviction)
    loader.rs      — resolves model names/aliases to GGUF files

Engine (src/engine/)
    InferenceEngine        — receives InferenceRequest, runs the decode loop
    model/llama_cpp/       — wraps the llama.cpp C bindings (FFI in engine/ffi.rs)
    model/stub.rs          — the cfg(fox_stub) no-op model for FOX_SKIP_LLAMA builds

Scheduler (src/scheduler/)
    Scheduler       — priority queue, assigns KV cache blocks to requests
    prefix_cache.rs — reuses already-processed shared prefixes (continuous batching)

KV cache (src/kv_cache/)
    KVCacheManager  — manages free/used blocks, implements PagedAttention-style allocation
```

**Request lifecycle** (HTTP → token stream):

1. `POST /v1/chat/completions` arrives at the router (`src/api/router.rs`) → handler `v1/chat.rs`
2. Handler resolves model name → calls `registry.get_or_load(model_name)`
3. Registry returns `Arc<InferenceEngine>` (loading model if not cached)
4. Handler formats prompt, builds `InferenceRequest`, submits to engine
5. Engine's run loop picks up the request, allocates KV blocks via scheduler
6. Tokens are sent back through an `mpsc` token channel
7. Handler streams tokens as SSE (OpenAI) or NDJSON (Ollama) to the client
8. When the client disconnects, `send().is_err()` signals the engine to preempt

## Adding a new API endpoint

1. Add the request/response types under `src/api/types/`
2. Write the handler in the matching `src/api/v1/` (OpenAI) or `src/api/ollama/` module, reusing `src/api/shared/` helpers
3. Register the route in `src/api/router.rs` / `src/api/routes.rs`
4. Add documentation under `docs/api/`

## Adding a new CLI command

1. Create `src/cli/<command>.rs` with a `<Command>Args` struct (clap `Parser`) and a `run_<command>()` async function
2. Add the variant to the `Command` enum in `src/cli/mod.rs` and wire its match arm in `run()`
3. Add documentation to `docs/cli/<command>.md`
4. Place the page under `docs/` (readable directly on GitHub)

## Integration tests

Integration tests live in `tests/`. The HTTP-layer tests need the `test-helpers` feature
(which exposes `StubModel`, `EngineEntry::for_test`, `ModelRegistry::preload_for_test`, and
`src/api/test_helpers.rs`); the live tests in `tests/integration.rs` need a running server:

```bash
# Full suite as CI runs it (stub model, no server needed)
FOX_SKIP_LLAMA=1 cargo test --all --features test-helpers

# Live integration tests — start a server first (in a separate terminal)
fox serve --model-path models/<model>.gguf --port 8081
cargo test --test integration -- --test-threads=1
```

Unit tests inside `src/` (marked `#[cfg(test)]`) can run without a server:

```bash
cargo test --lib
```

## Adding models to the registry

The built-in model registry lives in `registry.json` at the repo root. Each entry maps a short
alias to a HuggingFace repo and the preferred GGUF filename pattern:

```json
{
  "llama3.2": {
    "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
    "pattern": "Q4_K_M"
  }
}
```

To add a new model:
1. Add an entry to `registry.json` following the format above.
2. Verify `fox models` displays it and `fox pull <alias>` resolves correctly.
3. Include the change in your PR with a brief description of the model.

## Design decisions

- **Rust** — memory safety without GC, zero-cost async, small binary. The inference hot path runs with minimal allocations.
- **llama.cpp as a submodule** — avoids a separate install step and guarantees version parity between the Rust bindings and the C library. This is why `--recurse-submodules` is required when cloning.
- **DashMap for the model registry** — lock-free concurrent reads; models are loaded infrequently but looked up on every request.
- **PagedAttention-style KV cache** — prevents memory fragmentation under concurrent load. Enables accurate eviction without OOM.
- **mpsc channel for token streaming** — decouples the engine loop from the HTTP handler. The handler detects client disconnects via `send().is_err()` without polling.

## Pull request process

`develop` is the active trunk; `main` is the release branch. Branch from and target `develop`.

1. **Fork** the repository and create a feature branch from `develop`:
   ```bash
   git checkout develop
   git checkout -b feat/my-feature
   ```
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run `make ci` locally (fmt + clippy + tests) before pushing.
4. **Open a PR** targeting `develop`. Fill in the PR description explaining what changed and why.
5. A maintainer will review and may request changes. Once approved, the PR is squash-merged.

## Reporting bugs

Open a [GitHub issue](https://github.com/ferrumox/fox/issues) with:
- fox version (`fox --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behaviour
