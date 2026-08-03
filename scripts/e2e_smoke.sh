#!/usr/bin/env bash
# e2e_smoke.sh — start a real fox server and run the end-to-end smoke suite.
#
# This is the release/CI gate for the layer no other test covers: a REAL model
# behind the REAL HTTP server, across MULTIPLE requests (prefix-cache lifecycle,
# guided decoding, logprobs, sampling controls, Ollama surface, speculation).
#
# Usage:
#   ./scripts/e2e_smoke.sh --bin <path/to/fox> --model-path <model.gguf> [--port N]
#
#   make e2e E2E_MODEL=/path/to/model.gguf     # builds target/debug/fox first
#
# The server is started with --speculative true (check 7 needs it) and a small
# context; it is killed on exit regardless of outcome.

set -euo pipefail

FOX_BIN=""
MODEL_PATH=""
PORT="8199"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bin)        FOX_BIN="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -n "$FOX_BIN" && -n "$MODEL_PATH" ]] || {
    echo "Usage: $0 --bin <fox binary> --model-path <model.gguf> [--port N]"
    exit 1
}
[[ -x "$FOX_BIN" ]] || { echo "fox binary not executable: $FOX_BIN"; exit 1; }
[[ -f "$MODEL_PATH" ]] || { echo "model not found: $MODEL_PATH"; exit 1; }

# The ggml/llama shared libs live next to the binary (build.rs copies them there).
export LD_LIBRARY_PATH="$(cd "$(dirname "$FOX_BIN")" && pwd)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

FOX_PID=""
cleanup() {
    if [[ -n "$FOX_PID" ]] && kill -0 "$FOX_PID" 2>/dev/null; then
        kill "$FOX_PID" 2>/dev/null || true
        wait "$FOX_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "── starting fox serve (port $PORT, speculative on) ──"
"$FOX_BIN" serve \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 --port "$PORT" \
    --max-context-len 2048 \
    --speculative true &
FOX_PID=$!

# Wait for /health (model load can take a while on CI CPU).
for _ in $(seq 1 120); do
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$FOX_PID" 2>/dev/null; then
        echo "fox serve exited before becoming healthy"
        exit 1
    fi
    sleep 1
done
curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null || {
    echo "server never became healthy"
    exit 1
}

echo "── running smoke checks ──"
python3 "$(dirname "$0")/e2e_smoke.py" "http://127.0.0.1:$PORT"
