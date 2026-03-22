#!/usr/bin/env bash
# e2e_thinking.sh — Verifies thinking suppression across all API endpoints
# using a real running fox server with a reasoning model (Qwen3, DeepSeek-R1, etc.)
#
# Usage:
#   ./scripts/e2e_thinking.sh [--host HOST] [--port PORT] [--model MODEL]
#
# Defaults:
#   HOST  = localhost
#   PORT  = 11434
#   MODEL = auto-detected from /api/tags (first model)
#
# The script starts fox if it's not already running, runs the checks, then
# stops it (unless --no-start is passed, in which case a running server is assumed).
#
# Examples:
#   # Use a running server
#   ./scripts/e2e_thinking.sh --no-start --model Qwen3-2B-Q4_K_M
#
#   # Start server with a specific model
#   ./scripts/e2e_thinking.sh --model-path ~/.cache/ferrumox/models/Qwen3-2B-Q4_K_M.gguf

set -euo pipefail

HOST="localhost"
PORT="11434"
MODEL=""
MODEL_PATH=""
NO_START=false
FOX_PID=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)     HOST="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --no-start) NO_START=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BASE="http://${HOST}:${PORT}"
PASS=0
FAIL=0

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; ((PASS++)); }
fail() { echo -e "  ${RED}✗${NC} $1"; ((FAIL++)); }
info() { echo -e "  ${YELLOW}→${NC} $1"; }

# ── Server startup ────────────────────────────────────────────────────────────
cleanup() {
    if [[ -n "$FOX_PID" ]]; then
        kill "$FOX_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "$NO_START" == false ]]; then
    if [[ -z "$MODEL_PATH" ]]; then
        echo "Error: --model-path required when starting fox (or use --no-start)"
        exit 1
    fi
    info "Starting fox on port $PORT with $MODEL_PATH..."
    ./target/release/fox serve --port "$PORT" --model-path "$MODEL_PATH" &>/tmp/fox_e2e.log &
    FOX_PID=$!

    # Wait for server to be ready (up to 60s)
    for i in $(seq 1 60); do
        if curl -sf "${BASE}/health" &>/dev/null; then
            info "Server ready (${i}s)"
            break
        fi
        sleep 1
        if [[ $i -eq 60 ]]; then
            echo "Server did not start in time. Log:"
            cat /tmp/fox_e2e.log
            exit 1
        fi
    done
fi

# ── Model detection ───────────────────────────────────────────────────────────
if [[ -z "$MODEL" ]]; then
    MODEL=$(curl -sf "${BASE}/api/tags" | python3 -c "
import sys, json
tags = json.load(sys.stdin)
models = tags.get('models', [])
if not models:
    print('', end='')
else:
    print(models[0]['name'], end='')
")
    if [[ -z "$MODEL" ]]; then
        echo "No models found on server. Is a model loaded?"
        exit 1
    fi
    info "Using model: $MODEL"
fi

# ── Check model supports thinking ─────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Fox Thinking E2E — model: $MODEL  host: $BASE"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Helper: assert string NOT present ────────────────────────────────────────
assert_no_tags() {
    local label="$1"
    local text="$2"
    if echo "$text" | grep -q '<think>\|</think>'; then
        fail "$label: response contains <think> tags"
        info "Got: $(echo "$text" | head -c 200)"
    else
        ok "$label: no <think> tags in content"
    fi
}

assert_contains() {
    local label="$1"
    local text="$2"
    local needle="$3"
    if echo "$text" | grep -q "$needle"; then
        ok "$label: found '$needle'"
    else
        fail "$label: expected '$needle' not found"
        info "Got: $(echo "$text" | head -c 200)"
    fi
}

PROMPT='Say only "hello" and nothing else.'

# ── 1. OpenAI /v1/chat/completions — non-streaming ───────────────────────────
echo "1. OpenAI /v1/chat/completions (non-streaming)"
RESP=$(curl -sf -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"stream\":false,\"max_tokens\":512}")
CONTENT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
assert_no_tags "OpenAI non-stream content" "$CONTENT"

# ── 2. OpenAI /v1/chat/completions — streaming ───────────────────────────────
echo "2. OpenAI /v1/chat/completions (streaming)"
STREAM_CONTENT=$(curl -sf -N -X POST "${BASE}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"stream\":true,\"max_tokens\":512}" \
    | grep '^data: ' | grep -v '\[DONE\]' \
    | python3 -c "
import sys, json
parts = []
for line in sys.stdin:
    line = line.strip()
    if line.startswith('data: '):
        try:
            d = json.loads(line[6:])
            c = d.get('choices',[{}])[0].get('delta',{}).get('content','')
            if c:
                parts.append(c)
        except:
            pass
print(''.join(parts), end='')
" 2>/dev/null || echo "PARSE_ERROR")
assert_no_tags "OpenAI stream content" "$STREAM_CONTENT"

# ── 3. Ollama /api/chat — non-streaming ──────────────────────────────────────
echo "3. Ollama /api/chat (non-streaming)"
RESP=$(curl -sf -X POST "${BASE}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"stream\":false}")
CHAT_CONTENT=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['message']['content'])" 2>/dev/null || echo "PARSE_ERROR")
CHAT_THINKING=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['message'].get('thinking',''))" 2>/dev/null || echo "")
assert_no_tags "Ollama chat non-stream content" "$CHAT_CONTENT"
if [[ -n "$CHAT_THINKING" ]]; then
    ok "Ollama chat non-stream: thinking field populated (${#CHAT_THINKING} chars)"
else
    info "Ollama chat non-stream: thinking field empty (model may not support it)"
fi

# ── 4. Ollama /api/chat — streaming ──────────────────────────────────────────
echo "4. Ollama /api/chat (streaming)"
NDJSON_CONTENT=$(curl -sf -N -X POST "${BASE}/api/chat" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"stream\":true}" \
    | python3 -c "
import sys, json
parts = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        c = d.get('message',{}).get('content','')
        if c:
            parts.append(c)
    except:
        pass
print(''.join(parts), end='')
" 2>/dev/null || echo "PARSE_ERROR")
assert_no_tags "Ollama chat stream content" "$NDJSON_CONTENT"

# ── 5. Ollama /api/generate — non-streaming ───────────────────────────────────
echo "5. Ollama /api/generate (non-streaming)"
RESP=$(curl -sf -X POST "${BASE}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":false}")
GEN_RESPONSE=$(echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['response'])" 2>/dev/null || echo "PARSE_ERROR")
assert_no_tags "Ollama generate non-stream response" "$GEN_RESPONSE"

# ── 6. Ollama /api/generate — streaming ──────────────────────────────────────
echo "6. Ollama /api/generate (streaming)"
GEN_STREAM=$(curl -sf -N -X POST "${BASE}/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"${PROMPT}\",\"stream\":true}" \
    | python3 -c "
import sys, json
parts = []
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
        r = d.get('response','')
        if r:
            parts.append(r)
    except:
        pass
print(''.join(parts), end='')
" 2>/dev/null || echo "PARSE_ERROR")
assert_no_tags "Ollama generate stream response" "$GEN_STREAM"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo -e "  Results: ${GREEN}${PASS} passed${NC}  ${RED}${FAIL} failed${NC}"
echo "═══════════════════════════════════════════════════════"
echo ""

[[ $FAIL -eq 0 ]]
