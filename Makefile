# Makefile for ferrumox
# Usage: make build && make run

PATH := $(HOME)/.cargo/bin:$(PATH)
export PATH

# --- Model ---
MODELS_DIR    ?= models
MODEL_REPO    ?= unsloth/Qwen3.5-0.8B-GGUF
MODEL_FILE    ?= Qwen3.5-0.8B-Q4_K_M.gguf
MODEL_PATH    ?= $(MODELS_DIR)/$(MODEL_FILE)
DOCKER_IMAGE  ?= python:3.11-slim

# --- Server ---
HOST              ?= 0.0.0.0
PORT              ?= 8080
MAX_CONTEXT_LEN   ?= 4096
GPU_MEM_FRACTION  ?= 0.85
MAX_BATCH_SIZE    ?= 32
BLOCK_SIZE        ?= 16

# --- Bench ---
BENCH_CONCURRENCY ?= 4
BENCH_REQUESTS    ?= 50
BENCH_PROMPT      ?= Write a short paragraph about the Rust programming language.
BENCH_MAX_TOKENS  ?= 128

.PHONY: help install-rust build run dev test bench download-model check docker docker-run

help:
	@echo "Targets:"
	@echo "  make install-rust    Install Rust toolchain (run once if not installed)"
	@echo "  make download-model  Download $(MODEL_FILE) from HuggingFace to $(MODELS_DIR)/"
	@echo "  make build           Compile release binaries"
	@echo "  make run             Build and start the server"
	@echo "  make dev             Start with verbose logging (RUST_LOG=debug)"
	@echo "  make test            Run unit tests"
	@echo "  make check           Fast type-check without producing a binary"
	@echo "  make bench           Run the integrated benchmark against a running server"
	@echo "  make docker          Build the Docker image"
	@echo "  make docker-run      Start the server via docker compose"
	@echo ""
	@echo "Variables (override with make run VAR=value):"
	@echo "  MODEL_PATH=$(MODEL_PATH)"
	@echo "  HOST=$(HOST)  PORT=$(PORT)"
	@echo "  MAX_CONTEXT_LEN=$(MAX_CONTEXT_LEN)"
	@echo "  GPU_MEM_FRACTION=$(GPU_MEM_FRACTION)"
	@echo "  MAX_BATCH_SIZE=$(MAX_BATCH_SIZE)"
	@echo "  BENCH_CONCURRENCY=$(BENCH_CONCURRENCY)  BENCH_REQUESTS=$(BENCH_REQUESTS)"

install-rust:
	@command -v cargo >/dev/null 2>&1 && \
		(echo "Rust already installed:"; cargo --version) || \
		(curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
		. $(HOME)/.cargo/env && cargo --version && \
		echo "Rust installed. Run: source ~/.cargo/env && make build")

download-model:
	@mkdir -p $(MODELS_DIR)
	@echo "Downloading $(MODEL_FILE) from $(MODEL_REPO)..."
	docker run --rm \
		-e PIP_ROOT_USER_ACTION=ignore \
		-v "$(PWD)/$(MODELS_DIR):/data" \
		-w /data \
		$(DOCKER_IMAGE) \
		sh -c "pip install --quiet huggingface_hub && \
			python -c \"from huggingface_hub import hf_hub_download; \
			hf_hub_download(repo_id='$(MODEL_REPO)', filename='$(MODEL_FILE)', local_dir='.')\""
	@echo "Model saved to $(MODELS_DIR)/$(MODEL_FILE)"

check:
	cargo check

build:
	@command -v cargo >/dev/null 2>&1 || \
		(echo "Rust not found. Run: make install-rust" && exit 1)
	cargo build --release

run: build
	@test -f "$(MODEL_PATH)" || \
		(echo "Model not found at $(MODEL_PATH). Run: make download-model" && exit 1)
	./target/release/fox \
		--model-path $(MODEL_PATH) \
		--host $(HOST) \
		--port $(PORT) \
		--max-context-len $(MAX_CONTEXT_LEN) \
		--gpu-memory-fraction $(GPU_MEM_FRACTION) \
		--max-batch-size $(MAX_BATCH_SIZE)

dev: build
	@test -f "$(MODEL_PATH)" || \
		(echo "Model not found at $(MODEL_PATH). Run: make download-model" && exit 1)
	RUST_LOG=debug ./target/release/fox \
		--model-path $(MODEL_PATH) \
		--host $(HOST) \
		--port $(PORT) \
		--max-context-len $(MAX_CONTEXT_LEN) \
		--gpu-memory-fraction $(GPU_MEM_FRACTION) \
		--max-batch-size $(MAX_BATCH_SIZE)

test:
	cargo test

bench: build
	@echo "Running benchmark against $(HOST):$(PORT)..."
	./target/release/fox-bench \
		--url http://$(HOST):$(PORT) \
		--model $(MODEL_FILE) \
		--concurrency $(BENCH_CONCURRENCY) \
		--requests $(BENCH_REQUESTS) \
		--max-tokens $(BENCH_MAX_TOKENS) \
		--prompt "$(BENCH_PROMPT)"

docker:
	docker build -t ferrumox:latest .

docker-run:
	docker compose up
