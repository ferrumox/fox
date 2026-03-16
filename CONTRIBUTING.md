# Contributing to fox

Thank you for your interest in contributing! This guide covers how to build the project, run tests, understand the architecture, and open pull requests.

## Building and running tests

```bash
# Clone with submodules (llama.cpp is a Git submodule)
git clone --recurse-submodules https://github.com/ferrumox/fox
cd fox

# Standard build — GPU backend (CUDA / Metal / Vulkan) detected at runtime
cargo build --release

# Stub build for CI environments without GPU drivers
FOX_SKIP_LLAMA=1 cargo build --release

# Run the test suite
cargo test

# Fast type-check (no codegen)
cargo check
```

## Code style

- **Format**: run `cargo fmt` before committing. CI will reject unformatted code.
- **Lints**: the project compiles with `cargo clippy -- -D warnings`. Fix all warnings before opening a PR.

```bash
cargo fmt
cargo clippy -- -D warnings
```

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
    ├─ routes.rs       Axum router, AppState, all HTTP handlers
    ├─ types.rs        Request/response types (OpenAI + Ollama + Embeddings)
    ├─ mod.rs          Re-exports
    └─ pull_handler.rs POST /api/pull  — SSE download from HuggingFace

Model registry (src/model_registry.rs)
    ModelRegistry — DashMap<String, EngineEntry> + LRU eviction
    get_or_load()  — loads a model on first request, starts its engine loop
    EngineEntry    — holds Arc<InferenceEngine>, aborted on Drop (eviction)

Engine (src/engine/)
    InferenceEngine — receives InferenceRequest, runs decode loop
    LlamaCppModel   — wraps the llama.cpp C bindings

Scheduler (src/scheduler/)
    Scheduler       — priority queue, assigns KV cache blocks to requests
    InferenceRequest — prompt tokens + sampling params + response channel

KV cache (src/kv_cache/)
    KVCacheManager  — manages free/used blocks, implements PagedAttention-style allocation
```

**Request lifecycle** (HTTP → token stream):

1. `POST /v1/chat/completions` arrives at `routes.rs`
2. Handler resolves model name → calls `registry.get_or_load(model_name)`
3. Registry returns `Arc<InferenceEngine>` (loading model if not cached)
4. Handler formats prompt, builds `InferenceRequest`, submits to engine
5. Engine's `run_loop` picks up the request, allocates KV blocks via scheduler
6. Tokens are sent back through an `mpsc::UnboundedSender<Token>` channel
7. Handler streams tokens as SSE (OpenAI) or NDJSON (Ollama) to the client
8. When the client disconnects, `send().is_err()` signals the engine to preempt

## Adding a new API endpoint

1. Add the request/response types to `src/api/types.rs`
2. Write the handler function in `src/api/routes.rs` (or a new module for complex handlers)
3. Register the route in the router in `src/api/routes.rs` → `router()`
4. Add documentation to `docs/api/openai.md` or `docs/api/ollama.md`

## Adding a new CLI command

1. Create `src/cli/<command>.rs` with a `<Command>Args` struct (clap `Parser`) and a `run_<command>()` async function
2. Add the variant to the `Commands` enum in `src/cli/mod.rs` or `src/main.rs`
3. Match the variant and call `run_<command>()` in the main dispatch block
4. Add documentation to `docs/cli/<command>.md`
5. Add the new page to `mkdocs.yml` under `nav:`

## Integration tests

Integration tests live in `tests/`. They require a running `fox serve` instance on localhost. To run them:

```bash
# Start the server (in a separate terminal)
fox serve --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf --port 8081

# Run only integration tests
cargo test --test '*' -- --test-threads=1
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

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` locally.
4. **Open a PR** targeting `main`. Fill in the PR description explaining what changed and why.
5. A maintainer will review and may request changes. Once approved, the PR is squash-merged.

## Reporting bugs

Open a [GitHub issue](https://github.com/ferrumox/fox/issues) with:
- fox version (`fox --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behaviour
