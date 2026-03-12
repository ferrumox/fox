# Contributing to ferrumox

Thank you for your interest in contributing! Below are the guidelines to get started.

## Building and running tests

```bash
# Clone with submodules (llama.cpp is a Git submodule)
git clone --recurse-submodules https://github.com/ManuelSLemos/ferrum-engine
cd ferrum-engine

# CPU-only build (no GPU required)
cargo build --release

# CUDA build (requires CUDA toolkit installed)
cargo build --release --features cuda

# Apple Silicon / Metal build
cargo build --release --features metal

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

## Pull request process

1. **Fork** the repository and create a feature branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes. Keep commits focused — one logical change per commit.
3. Run `cargo fmt`, `cargo clippy -- -D warnings`, and `cargo test` locally.
4. **Open a PR** targeting `main`. Fill in the PR description explaining what changed and why.
5. A maintainer will review and may request changes. Once approved, the PR is squash-merged.

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

## Reporting bugs

Open a [GitHub issue](https://github.com/ManuelSLemos/ferrum-engine/issues) with:
- ferrumox version (`fox --version`)
- OS and architecture
- Steps to reproduce
- Expected vs actual behaviour
