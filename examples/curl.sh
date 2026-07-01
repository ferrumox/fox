#!/usr/bin/env bash
# curl.sh — examples for all major ferrumox API routes
# Usage: ./examples/curl.sh [HOST]
# Default HOST: http://localhost:8080

HOST="${1:-http://localhost:8080}"
MODEL="${2:-default}"

echo "=== ferrumox API examples ==="
echo "  Server : $HOST"
echo "  Model  : $MODEL"
echo

# ── Health ────────────────────────────────────────────────────────────────────
echo "--- GET /health ---"
curl -s "$HOST/health" | python3 -m json.tool 2>/dev/null || curl -s "$HOST/health"
echo; echo

# ── List models ───────────────────────────────────────────────────────────────
echo "--- GET /v1/models ---"
curl -s "$HOST/v1/models" | python3 -m json.tool 2>/dev/null || curl -s "$HOST/v1/models"
echo; echo

# ── Chat completion (non-streaming) ──────────────────────────────────────────
echo "--- POST /v1/chat/completions (non-streaming) ---"
curl -s "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2? Answer in one word.\"}],
    \"max_tokens\": 16,
    \"stream\": false
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Chat completion (streaming) ───────────────────────────────────────────────
echo "--- POST /v1/chat/completions (streaming) ---"
curl -s "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in 10 words.\"}],
    \"max_tokens\": 32,
    \"stream\": true
  }"
echo; echo

# ── Text completion ───────────────────────────────────────────────────────────
echo "--- POST /v1/completions ---"
curl -s "$HOST/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Rust is a systems programming language that\",
    \"max_tokens\": 32,
    \"stream\": false
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Embeddings ────────────────────────────────────────────────────────────────
echo "--- POST /v1/embeddings ---"
curl -s "$HOST/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"input\": \"Hello, world!\"
  }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'embedding dim: {len(d[\"data\"][0][\"embedding\"])}')" 2>/dev/null
echo; echo

# ── Ollama: tags ──────────────────────────────────────────────────────────────
echo "--- GET /api/tags (Ollama compat) ---"
curl -s "$HOST/api/tags" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Ollama: generate ─────────────────────────────────────────────────────────
echo "--- POST /api/generate (Ollama compat) ---"
curl -s "$HOST/api/generate" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"Why is Rust memory safe?\",
    \"stream\": false,
    \"options\": {\"num_predict\": 32}
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Ollama: chat ─────────────────────────────────────────────────────────────
echo "--- POST /api/chat (Ollama compat) ---"
curl -s "$HOST/api/chat" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Name one benefit of Rust.\"}],
    \"stream\": false
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Structured output (JSON mode) ────────────────────────────────────────────
echo "--- POST /v1/chat/completions (JSON mode) ---"
curl -s "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Return a JSON object with fields: name (string), year (number).\"}],
    \"response_format\": {\"type\": \"json_object\"},
    \"max_tokens\": 64,
    \"stream\": false
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Stop sequences ────────────────────────────────────────────────────────────
echo "--- POST /v1/chat/completions (stop sequences) ---"
curl -s "$HOST/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"List 5 items:\"}],
    \"stop\": [\"3.\", \"\\n3\"],
    \"max_tokens\": 100,
    \"stream\": false
  }" | python3 -m json.tool 2>/dev/null
echo; echo

# ── Prometheus metrics ────────────────────────────────────────────────────────
echo "--- GET /metrics ---"
curl -s "$HOST/metrics" | head -20
echo "  (truncated)"
echo
