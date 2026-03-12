# Installation

fox ships as a single self-contained binary. There is no runtime to install, no Python environment to manage, and no system libraries required beyond your GPU drivers.

---

## Pre-built binaries

The fastest way to get started is to download the latest release binary for your platform.

### Linux (x86_64)

```bash
curl -L https://github.com/ferrumox/fox/releases/latest/download/fox-linux-x86_64.tar.gz \
  | tar xz
sudo mv fox /usr/local/bin/
fox --version
```

### macOS (Apple Silicon)

```bash
curl -L https://github.com/ferrumox/fox/releases/latest/download/fox-macos-arm64.tar.gz \
  | tar xz
sudo mv fox /usr/local/bin/
fox --version
```

### macOS (Intel)

```bash
curl -L https://github.com/ferrumox/fox/releases/latest/download/fox-macos-x86_64.tar.gz \
  | tar xz
sudo mv fox /usr/local/bin/
fox --version
```

### Windows (x86_64)

Download `fox-windows-x86_64.zip` from the [releases page](https://github.com/ferrumox/fox/releases/latest), extract it, and place `fox.exe` somewhere on your `PATH`.

---

## Docker

The official Docker image includes the binary, CUDA libraries, and a minimal base. It is the recommended way to run fox in production on Linux with GPU acceleration.

```bash
docker pull ferrumox/fox:latest

docker run -d \
  --name fox \
  --gpus all \
  -p 8080:8080 \
  -v ~/.cache/ferrumox:/root/.cache/ferrumox \
  ferrumox/fox:latest
```

For CPU-only environments:

```bash
docker run -d \
  --name fox \
  -p 8080:8080 \
  -v ~/.cache/ferrumox:/root/.cache/ferrumox \
  ferrumox/fox:cpu
```

See the [Deployment guide](./deployment.md) for Docker Compose examples and production configurations.

---

## Build from source

Building from source gives you full control over compile-time flags and lets you enable optional backends (CUDA, Metal, CPU-only).

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| Rust | 1.80 |
| CMake | 3.14 |
| C++17 compiler (GCC/Clang/MSVC) | — |
| CUDA Toolkit (optional, for NVIDIA GPU) | 11.8 |
| Xcode Command Line Tools (optional, for Metal) | — |

### Clone and build

```bash
git clone --recurse-submodules https://github.com/ferrumox/fox
cd fox
cargo build --release
```

The `--recurse-submodules` flag is required because llama.cpp is included as a git submodule in `vendor/llama.cpp`.

The compiled binaries will be at:

```
target/release/fox          # main server binary
target/release/fox-bench    # standalone benchmark tool
```

### Build with CUDA support (NVIDIA GPU)

```bash
cargo build --release --features cuda
```

CMake will detect your CUDA installation and compile llama.cpp with CUDA acceleration. Make sure `nvcc` is on your `PATH` before building.

### Build with Metal support (Apple Silicon)

```bash
cargo build --release --features metal
```

### Build for CPU only (no GPU)

```bash
cargo build --release --features cpu-only
```

### Install to PATH

```bash
sudo cp target/release/fox /usr/local/bin/
sudo cp target/release/fox-bench /usr/local/bin/
```

Or use `cargo install` to install directly into `~/.cargo/bin`:

```bash
cargo install --path . --features cuda
```

---

## Verifying the installation

```bash
fox --version
# fox 1.0.0

fox --help
# Usage: fox <COMMAND>
# Commands:
#   serve    Start the inference server
#   run      Run inference directly (no HTTP server)
#   pull     Download a model from HuggingFace
#   list     List downloaded models
#   show     Show model details
#   rm       Remove a downloaded model
#   ps       Show running model servers
#   search   Search HuggingFace for models
#   models   List curated models
```

---

## Model storage

By default, downloaded models are stored in:

| Platform | Path |
|----------|------|
| Linux | `~/.cache/ferrumox/models` |
| macOS | `~/Library/Caches/ferrumox/models` |
| Windows | `%LOCALAPPDATA%\ferrumox\models` |

You can override this with the `--output-dir` flag on `fox pull` or the `--path` flag on `fox list` / `fox show`.

---

## Updating

To update to the latest release, repeat the installation steps. The new binary replaces the old one. Model files are not affected.

If you built from source:

```bash
cd fox
git pull --recurse-submodules
cargo build --release
sudo cp target/release/fox /usr/local/bin/
```
