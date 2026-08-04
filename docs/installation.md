# Installation

fox ships as a single self-contained binary. There is no runtime to install, no Python environment to manage, and no system libraries required beyond your GPU drivers.

---

## Pre-built binaries

Releases publish **Linux x86_64** only, in two variants: a CPU build and a **Vulkan**
build that runs on AMD/Intel iGPUs and any Vulkan-capable GPU (and falls back to CPU on
its own when there is no device).

### Installer (recommended)

```bash
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
```

It detects `/dev/dri` and picks the Vulkan build when a GPU is present, verifies the
published SHA-256, refuses to start if the target directory is not writable, and prints
what to do next. Options:

| flag | effect |
|---|---|
| `--vulkan` / `--cpu` | force a variant instead of detecting |
| `--version vX.Y.Z` | install a specific release |
| `--prefix ~/.local` | install somewhere you own, no `sudo` |

### By hand

```bash
V=0.20.2
curl -LO https://github.com/ferrumox/fox/releases/download/v$V/fox-$V-x86_64-unknown-linux-gnu-vulkan.tar.gz
curl -LO https://github.com/ferrumox/fox/releases/download/v$V/fox-$V-x86_64-unknown-linux-gnu-vulkan.tar.gz.sha256
sha256sum -c fox-$V-x86_64-unknown-linux-gnu-vulkan.tar.gz.sha256
tar xzf fox-$V-x86_64-unknown-linux-gnu-vulkan.tar.gz
```

Drop `-vulkan` from the name for the CPU build. Keep the `.so` files next to the binary:
`fox` is linked with `RPATH=$ORIGIN` and finds its backends nowhere else.

### macOS and Windows

Not published as binaries. Use WSL2 with the Linux installer, or build from source
(below). `install.ps1` exists and will tell you the same thing rather than failing with a
404 — it checks the release's assets before promising anything.

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

Building from source gives you full control over compile-time flags and lets you enable optional backends (CUDA, ROCm, Metal, Vulkan, CPU-only).

### Prerequisites

| Tool | Minimum version | Required for |
|------|----------------|-------------|
| Rust | 1.80 | Always |
| CMake | 3.14 | Always |
| C++17 compiler (GCC/Clang/MSVC) | — | Always |
| CUDA Toolkit | 11.8 | NVIDIA GPU on Linux/Windows |
| ROCm / HIP SDK | 6.2+ | AMD GPU on Linux |
| Vulkan SDK (`libvulkan-dev` + `glslc`) | 1.3 | Vulkan on Linux |
| Xcode Command Line Tools | — | Metal on macOS |

Install Vulkan toolchain on Linux (Debian/Ubuntu):

```bash
sudo apt install libvulkan-dev glslc
```

Install ROCm on Linux (Debian/Ubuntu):

```bash
wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
  | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
  https://repo.radeon.com/rocm/apt/6.2 jammy main" \
  | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt-get update && sudo apt-get install -y rocm-hip-sdk
```

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

### GPU backend detection

fox detects GPU backends **at runtime** — no compile-time feature flags are required. A single `cargo build --release` produces a binary that automatically uses the best available backend:

```bash
cargo build --release   # runs on CPU, CUDA, Metal, or Vulkan — same binary
```

The backend is loaded via `llama_backend_load` when the first model is initialised. fox prints the detected backend at startup:

```
INFO fox::engine: GPU backend: CUDA 12.5 (device 0: RTX 4060, 8192 MB)
```

If no GPU is found, fox falls back to CPU automatically.

### Build for CPU only (skip GPU detection)

If you are building in a CI environment without GPU drivers and want to skip GPU detection:

```bash
FOX_SKIP_LLAMA=1 cargo build --release   # stub build, CPU only
```

### Install to PATH

```bash
sudo cp target/release/fox /usr/local/bin/
sudo cp target/release/fox-bench /usr/local/bin/
```

Or use `cargo install` to install directly into `~/.cargo/bin`:

```bash
cargo install --path .
```

---

## Verifying the installation

```bash
fox --version
# fox 0.19.1

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
