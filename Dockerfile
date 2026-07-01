# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# Uses the CUDA devel image so nvcc is available and libggml-cuda.so is built.
# Rust is installed manually since it is not included in the CUDA base image.
# ──────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl cmake clang libclang-dev ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Fetch Rust dependencies before copying source so this layer is cached
# as long as Cargo.toml / Cargo.lock do not change.
COPY Cargo.toml Cargo.lock ./
COPY vendor/ vendor/
RUN --mount=type=cache,target=/root/.cargo/registry \
    cargo fetch

# Build.  BuildKit cache mounts keep the incremental build cache across
# image rebuilds so only changed crates (and llama.cpp when vendor/ changes)
# are recompiled.
COPY . .
RUN --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/app/target \
    cargo build --release --bin fox --bin fox-bench && \
    # Collect everything the runtime stage needs into /bundle so the COPY
    # below does not require glob expansion across build cache mounts.
    mkdir -p /bundle && \
    cp target/release/fox target/release/fox-bench /bundle/ && \
    find target/release -maxdepth 1 \( -name 'lib*.so*' -o -name 'lib*.dylib' \) \
         -exec cp {} /bundle/ \;

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: minimal runtime image
#
# libggml-cuda.so is present but loaded lazily at runtime.
# When a GPU is available (docker run --gpus all), nvidia-container-toolkit
# injects the host CUDA driver libraries and the CUDA backend activates.
# Without a GPU the engine falls back to CPU automatically — no env-var
# toggle required, no separate image needed.
# ──────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# All binaries and shared backends go into the same directory so that the
# RPATH=$ORIGIN embedded in the fox binary resolves the backends at runtime.
COPY --from=builder /bundle/ /usr/local/lib/fox/

# Register the backend libraries and create versioned SONAME symlinks
# (e.g. libllama.so.0 → libllama.so) so the dynamic linker finds them.
RUN echo "/usr/local/lib/fox" > /etc/ld.so.conf.d/fox.conf && ldconfig

# Wrapper so fox is on PATH while backends stay in /usr/local/lib/fox/.
RUN ln -s /usr/local/lib/fox/fox      /usr/local/bin/fox && \
    ln -s /usr/local/lib/fox/fox-bench /usr/local/bin/fox-bench

ENV FOX_HOST=0.0.0.0
ENV FOX_PORT=8080

EXPOSE 8080

ENTRYPOINT ["fox"]
CMD ["serve"]
