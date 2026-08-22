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

.PHONY: help install-rust build run dev test budgets bench download-model check ci setup docker docker-run

help:
	@echo "Targets:"
	@echo "  make install-rust    Install Rust toolchain (run once if not installed)"
	@echo "  make download-model  Download $(MODEL_FILE) from HuggingFace to $(MODELS_DIR)/"
	@echo "  make build           Compile release binaries"
	@echo "  make run             Build and start the server"
	@echo "  make dev             Start with verbose logging (RUST_LOG=debug)"
	@echo "  make test            Run unit tests"
	@echo "  make budgets         Re-record perf-budgets.json (only for intended changes)"
	@echo "  make check           Fast type-check without producing a binary"
	@echo "  make ci              Run the full CI suite locally (fmt + clippy + tests)"
	@echo "  make e2e             E2E smoke: real server + real model over HTTP (E2E_MODEL=...)"
	@echo "  make setup           Install git pre-push hook (run once after cloning)"
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

# Run exactly what CI runs (use this before pushing to avoid surprises).
ci:
	FOX_SKIP_LLAMA=1 cargo fmt --all -- --check
	FOX_SKIP_LLAMA=1 cargo clippy --all-targets --features test-helpers -- -D warnings
	FOX_SKIP_LLAMA=1 cargo test --all --features test-helpers
	@FOX_SKIP_LLAMA=1 cargo build --bin fox -q
	python3 scripts/check_docs_flags.py
	python3 scripts/check_prompt_tokenization.py
	@echo ""
	@echo "==> everything above ran with FOX_SKIP_LLAMA=1 and never compiled the"
	@echo "    llama.cpp module. Checking it for real now (slow the first time,"
	@echo "    incremental afterwards):"
	@$(MAKE) --no-print-directory check-real

# Type-check against a REAL llama.cpp build. Slow the first time (CMake compiles
# llama.cpp) and cached afterwards.
#
# Exists because `make ci` cannot see this class of error at all: adding a parameter
# to LlamaCppModel::load() left eight call sites broken while fmt, clippy and the
# whole test suite stayed green, because none of them compile that module. The stub
# in llama_cpp/stub.rs mirrors those signatures by hand, so it breaks from the other
# side just as silently.
#
# --all-targets on purpose: the binaries and integration tests are outside the lib,
# and `cargo test --lib` (what the golden job runs) does not reach them.
check-real:
	cargo check --all-targets --features test-helpers

# Golden regression tests — assert model-facing invariants against a REAL model
# (ModelInfo numbers, non-degenerate embeddings, tokenize round-trip). Requires a
# real build (not the stub) and a GGUF. The backend .so files are copied next to
# the working dir so llama.cpp's dlopen search (executable dir + cwd) finds them.
#   Usage: make golden GOLDEN_MODEL=~/.cache/ferrumox/models/gemma-4-E2B-it-Q4_K_M.gguf
GOLDEN_MODEL ?=
golden:
	@test -n "$(GOLDEN_MODEL)" || \
		(echo "Set GOLDEN_MODEL=/path/to/model.gguf" && exit 1)
	cargo test --lib --no-run
	find target \( -name 'libggml*.so*' -o -name 'libllama*.so*' \) -exec cp {} . \;
	# Unfiltered: this is the only CI job that builds real llama.cpp, so it must also
	# run the plain (non-"golden") unit tests that only compile in a real build
	# (llama_cpp::batch, llama_cpp::mod, sampling, vocab) — a "golden"-substring
	# filter here silently skipped them even though they built and passed locally.
	FOX_GOLDEN_MODEL="$(GOLDEN_MODEL)" cargo test --lib -- --nocapture

# End-to-end smoke — a REAL server with a REAL model over HTTP, across multiple
# requests. Covers the layer no unit/golden/stub test reaches (cross-request
# prefix-cache lifecycle, guided decoding, logprobs, sampling controls, Ollama
# surface, speculation). This is the release-closing gate.
#   Usage: make e2e E2E_MODEL=~/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf
#   Add E2E_MMPROJ=/path/to/mmproj.gguf to also run check 14 (vision/image input).
#   Add E2E_LORA=name=/path/to/adapter.gguf[:scale] to also run check 15 (LoRA).
E2E_MODEL ?=
E2E_MMPROJ ?=
E2E_LORA ?=
e2e:
	@test -n "$(E2E_MODEL)" || \
		(echo "Set E2E_MODEL=/path/to/model.gguf" && exit 1)
	cargo build --bin fox
	./scripts/e2e_smoke.sh --bin target/debug/fox --model-path "$(E2E_MODEL)" \
		$(if $(E2E_MMPROJ),--mmproj-path "$(E2E_MMPROJ)",) \
		$(if $(E2E_LORA),--lora-modules "$(E2E_LORA)",)

# Build a Vulkan-enabled fox bundle in Docker and extract it to ./fox-vulkan/ so you
# can run it natively on any host with a Vulkan driver (AMD/Intel iGPU, etc.) — no
# build toolchain needed on the host. Building needs glslc/spirv-headers, which the
# Dockerfile.vulkan image provides; running only needs the Mesa/Vulkan driver.
#   make vulkan && ./fox-vulkan/fox serve --model-path <model.gguf>
VULKAN_OUT ?= fox-vulkan
vulkan:
	@command -v docker >/dev/null 2>&1 || \
		(echo "Docker not found — needed to build the Vulkan bundle." && exit 1)
	docker build -f Dockerfile.vulkan -t fox:vulkan .
	@id=$$(docker create fox:vulkan); \
		rm -rf $(VULKAN_OUT) && mkdir -p $(VULKAN_OUT); \
		docker cp "$$id:/usr/local/lib/fox/." "$(VULKAN_OUT)/"; \
		docker rm "$$id" >/dev/null; \
		echo ""; \
		echo "Vulkan bundle -> $(VULKAN_OUT)/  (fox, fox-bench, libggml-vulkan.so)"; \
		echo "Run: ./$(VULKAN_OUT)/fox serve --model-path <model.gguf>"

# Prepare a release: refuses to continue if the tree is dirty, the CHANGELOG has no
# entry, ci or e2e fail, or the version does not end up matching in every file. Stops at
# the release commit — tagging is `make publish`, deliberately a separate step.
#   make release VERSION=0.21.0
release:
	@test -n "$(VERSION)" || (echo "uso: make release VERSION=X.Y.Z" && exit 1)
	./scripts/release.sh $(VERSION)

# Tag ONE release from main and verify a Release run actually started. Never
# `git push --tags`: GitHub fires no workflow when more than three tags arrive at once.
#   make publish VERSION=0.21.0
publish:
	@test -n "$(VERSION)" || (echo "uso: make publish VERSION=X.Y.Z" && exit 1)
	./scripts/publish.sh $(VERSION)

# Soak test — sustained mixed traffic against a REAL server, then a verdict.
#
# Covers what nothing else does: `make e2e` is 23 checks over two minutes and every
# other test starts from a fresh process, so a leak, a KV pool that never returns or
# latency drift are all invisible. Traffic mixes conversations, one-off prompts and
# clients that hang up mid-stream, because each shape has broken something before.
#
#   make soak SOAK_MODEL=~/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf
#   make soak SOAK_MODEL=... SOAK_MINUTES=60 SOAK_CONC=8
SOAK_MODEL ?=
SOAK_MINUTES ?= 10
SOAK_CONC ?= 4
SOAK_PORT ?= 8410
soak:
	@test -n "$(SOAK_MODEL)" || (echo "Set SOAK_MODEL=/path/to/model.gguf" && exit 1)
	cargo build --release --bin fox
	@echo "arrancando fox en :$(SOAK_PORT)…"
	@./target/release/fox serve --model-path "$(SOAK_MODEL)" --host 127.0.0.1 \
		--port $(SOAK_PORT) --max-context-len 4096 --max-batch-size $(SOAK_CONC) \
		> /tmp/fox-soak.log 2>&1 & \
	for i in $$(seq 1 90); do \
		curl -sf -m 2 http://127.0.0.1:$(SOAK_PORT)/health >/dev/null && break; sleep 2; \
	done; \
	python3 scripts/soak.py http://127.0.0.1:$(SOAK_PORT) \
		"$$(basename "$(SOAK_MODEL)" .gguf)" $(SOAK_MINUTES) $(SOAK_CONC); \
	rc=$$?; \
	pid=$$(ss -lptn "sport = :$(SOAK_PORT)" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1); \
	[ -n "$$pid" ] && kill $$pid; \
	exit $$rc

# Install the pre-push git hook so CI checks run automatically on every push.
setup:
	bash scripts/install-hooks.sh

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

# Rewrite perf-budgets.json from the current scheduler. Only ever run this when a
# change was *meant* to move the numbers — the diff is the performance review.
budgets:
	FOX_SKIP_LLAMA=1 FOX_UPDATE_BUDGETS=1 cargo test --lib scheduler::budgets
	@git --no-pager diff --stat perf-budgets.json

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
