# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: builder
# Full Rust + CMake + clang toolchain needed to compile llama.cpp via the
# build.rs script.
# ──────────────────────────────────────────────────────────────────────────────
FROM rust:1.84-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache dependency compilation separately from source changes.
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src && echo 'fn main() {}' > src/main.rs && \
    cargo build --release 2>/dev/null || true
RUN rm -rf src

# Copy the full source tree and build for real.
COPY . .
RUN cargo build --release

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: minimal runtime image
# Only the compiled binaries and their shared-library dependencies are kept.
# ──────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/ferrum-engine /usr/local/bin/ferrum-engine
COPY --from=builder /app/target/release/ferrum-bench  /usr/local/bin/ferrum-bench

# Default port
EXPOSE 8080

ENTRYPOINT ["ferrum-engine"]
