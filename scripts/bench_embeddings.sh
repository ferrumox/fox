#!/usr/bin/env bash
# ============================================================================
# Fox vs Ollama — CPU-only embedding benchmark
#
# Compares latency and throughput of embedding endpoints using
# nomic-embed-text-v1.5 in CPU-only Docker containers.
#
# Prerequisites:
#   - Docker installed
#   - curl, jq, bc, python3 installed
#   - ~2 GB disk for the model (downloaded into each container)
#
# Usage:
#   chmod +x scripts/bench_embeddings.sh
#   ./scripts/bench_embeddings.sh
# ============================================================================

set -euo pipefail

MODEL_NAME="nomic-embed-text"
MODEL_HF_REPO="nomic-ai/nomic-embed-text-v1.5-GGUF"
MODEL_FILE="nomic-embed-text-v1.5.Q8_0.gguf"
FOX_MODEL="${MODEL_FILE%.gguf}"
MODEL_CACHE_DIR="${HOME}/.cache/ferrumox/models"

FOX_PORT=11500
OLLAMA_PORT=11501
REQUESTS=50
WARMUP=5

SHORT_TEXT="The quick brown fox jumps over the lazy dog."
MEDIUM_TEXT="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance on a specific task over time without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."
LONG_TEXT="${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT}"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cleanup() {
    echo -e "\n${BLUE}Cleaning up...${NC}"
    docker rm -f fox-embed-bench ollama-embed-bench 2>/dev/null || true
}
trap cleanup EXIT

# ── Helper: time a single curl request (ms) ─────────────────────────────────
time_request_ms() {
    local url="$1"
    local data="$2"
    curl -s -o /dev/null -w '%{time_total}' \
        -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data" | awk '{printf "%.2f", $1 * 1000}'
}

# ── Helper: run N requests, report stats ─────────────────────────────────────
run_bench() {
    local label="$1"
    local url="$2"
    local data="$3"
    local n="$4"
    local warmup="$5"

    echo -e "  ${BOLD}${label}${NC} (${n} requests, ${warmup} warmup)"

    for ((i=0; i<warmup; i++)); do
        curl -s -o /dev/null -X POST "$url" -H "Content-Type: application/json" -d "$data"
    done

    local times=()
    local start_all
    start_all=$(date +%s%N)

    for ((i=0; i<n; i++)); do
        local t
        t=$(time_request_ms "$url" "$data")
        times+=("$t")
    done

    local end_all
    end_all=$(date +%s%N)
    local wall_ms=$(( (end_all - start_all) / 1000000 ))

    IFS=$'\n' sorted=($(sort -g <<<"${times[*]}")); unset IFS

    local sum=0
    for t in "${times[@]}"; do
        sum=$(echo "$sum + $t" | bc)
    done
    local avg=$(echo "scale=2; $sum / $n" | bc)
    local p50=${sorted[$(( n / 2 ))]}
    local p95=${sorted[$(( n * 95 / 100 ))]}
    local p99=${sorted[$(( n * 99 / 100 ))]}
    local min=${sorted[0]}
    local max=${sorted[$(( n - 1 ))]}
    local rps=$(echo "scale=2; $n / ($wall_ms / 1000)" | bc)

    printf "    avg: %8s ms   p50: %8s ms   p95: %8s ms   p99: %8s ms\n" "$avg" "$p50" "$p95" "$p99"
    printf "    min: %8s ms   max: %8s ms   rps: %8s\n" "$min" "$max" "$rps"
}

# ── Helper: validate embedding response ──────────────────────────────────────
validate_embedding() {
    local label="$1"
    local url="$2"
    local data="$3"

    local resp
    resp=$(curl -s -X POST "$url" -H "Content-Type: application/json" -d "$data")

    local dim
    dim=$(echo "$resp" | jq -r '
        if .data then .data[0].embedding | length
        elif .embeddings then .embeddings[0] | length
        else 0 end
    ' 2>/dev/null || echo "0")

    local non_zero
    non_zero=$(echo "$resp" | jq -r '
        if .data then [.data[0].embedding[] | select(. != 0)] | length
        elif .embeddings then [.embeddings[0][] | select(. != 0)] | length
        else 0 end
    ' 2>/dev/null || echo "0")

    if [ "$dim" -gt 0 ] && [ "$non_zero" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} ${label}: dim=${dim}, non-zero=${non_zero}"
        return 0
    else
        echo -e "  ${RED}✗${NC} ${label}: dim=${dim}, non-zero=${non_zero} — embeddings may be broken"
        echo "    Response: $(echo "$resp" | head -c 500)"
        return 1
    fi
}

# ── Helper: cosine similarity ────────────────────────────────────────────────
cosine_similarity() {
    local resp1="$1"
    local resp2="$2"

    python3 - "$resp1" "$resp2" <<'PYEOF'
import json, math, sys
r1 = json.loads(sys.argv[1])
r2 = json.loads(sys.argv[2])
def extract(r):
    if 'data' in r: return r['data'][0]['embedding']
    if 'embeddings' in r: return r['embeddings'][0]
    return []
a, b = extract(r1), extract(r2)
dot = sum(x*y for x,y in zip(a,b))
na = math.sqrt(sum(x*x for x in a))
nb = math.sqrt(sum(x*x for x in b))
print(f'{dot/(na*nb):.6f}' if na > 0 and nb > 0 else '0.000000')
PYEOF
}

# ── Helper: wait for endpoint ────────────────────────────────────────────────
wait_for() {
    local label="$1"
    local url="$2"
    local max_wait="${3:-60}"
    echo -ne "  Waiting for ${label}..."
    for ((i=0; i<max_wait; i++)); do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e " ${GREEN}ready${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e " ${RED}timeout after ${max_wait}s${NC}"
    return 1
}

# ============================================================================
# 1. Download model if needed
# ============================================================================

echo -e "${BOLD}═══ Fox vs Ollama Embedding Benchmark (CPU-only) ═══${NC}\n"

mkdir -p "${MODEL_CACHE_DIR}"
MODEL_PATH="${MODEL_CACHE_DIR}/${MODEL_FILE}"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${BLUE}Downloading ${MODEL_FILE} from HuggingFace...${NC}"
    curl -L --progress-bar \
        "https://huggingface.co/${MODEL_HF_REPO}/resolve/main/${MODEL_FILE}" \
        -o "$MODEL_PATH"
else
    echo -e "${GREEN}Model already cached: ${MODEL_PATH}${NC}"
fi

# ============================================================================
# 2. Start containers
# ============================================================================

echo -e "${BLUE}Starting Ollama container...${NC}"
docker rm -f ollama-embed-bench 2>/dev/null || true
docker run -d --name ollama-embed-bench \
    --cpus=4 \
    -p ${OLLAMA_PORT}:11434 \
    ollama/ollama:latest >/dev/null

echo -e "${BLUE}Building Fox container (CPU-only)...${NC}"
docker rm -f fox-embed-bench 2>/dev/null || true
docker build -t fox-embed-bench -f "${REPO_DIR}/Dockerfile" "${REPO_DIR}" 2>&1 | tail -1
docker run -d --name fox-embed-bench \
    --cpus=4 \
    -p ${FOX_PORT}:8080 \
    -e FOX_HOST=0.0.0.0 \
    -e FOX_PORT=8080 \
    -v "${MODEL_CACHE_DIR}:/root/.cache/ferrumox/models:ro" \
    fox-embed-bench serve --model-path "/root/.cache/ferrumox/models/${MODEL_FILE}" >/dev/null

# ============================================================================
# 3. Wait for readiness and pull model into Ollama
# ============================================================================

echo ""
wait_for "Ollama" "http://localhost:${OLLAMA_PORT}/api/tags" 60
wait_for "Fox"    "http://localhost:${FOX_PORT}/health"      60

echo -e "\n${BLUE}Pulling model in Ollama...${NC}"
curl -s -X POST "http://localhost:${OLLAMA_PORT}/api/pull" \
    -d "{\"name\":\"${MODEL_NAME}\"}" | jq -r 'select(.status) | .status' | tail -1

echo -e "${BLUE}Warming up models (first inference loads weights)...${NC}"
curl -s -o /dev/null -X POST "http://localhost:${OLLAMA_PORT}/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"input\":\"warmup\"}" || true
sleep 2
# Fox model is already loaded via --model-path; warm it up
curl -s -o /dev/null -X POST "http://localhost:${FOX_PORT}/api/embed" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${FOX_MODEL}\",\"input\":\"warmup\"}" || true
sleep 2

# ============================================================================
# 3. Validate embeddings
# ============================================================================

echo -e "\n${BOLD}── Validation ──${NC}"

FOX_OPENAI_URL="http://localhost:${FOX_PORT}/v1/embeddings"
FOX_OLLAMA_URL="http://localhost:${FOX_PORT}/api/embed"
OLLAMA_URL="http://localhost:${OLLAMA_PORT}/api/embed"

# Fox resolves by filename stem; Ollama by registry name
FOX_OPENAI_DATA="{\"model\":\"${FOX_MODEL}\",\"input\":\"${SHORT_TEXT}\"}"
FOX_OLLAMA_DATA="{\"model\":\"${FOX_MODEL}\",\"input\":\"${SHORT_TEXT}\"}"
OLLAMA_DATA="{\"model\":\"${MODEL_NAME}\",\"input\":\"${SHORT_TEXT}\"}"

validate_embedding "Fox (OpenAI API)" "$FOX_OPENAI_URL" "$FOX_OPENAI_DATA" || true
validate_embedding "Fox (Ollama API)" "$FOX_OLLAMA_URL" "$FOX_OLLAMA_DATA" || true
validate_embedding "Ollama"           "$OLLAMA_URL"     "$OLLAMA_DATA"     || true

echo -e "\n  Cross-checking embedding similarity (same input, same model):"
RESP_FOX=$(curl -s -X POST "$FOX_OLLAMA_URL" -H "Content-Type: application/json" -d "$FOX_OLLAMA_DATA")
RESP_OLLAMA=$(curl -s -X POST "$OLLAMA_URL" -H "Content-Type: application/json" -d "$OLLAMA_DATA")
SIM=$(cosine_similarity "$RESP_FOX" "$RESP_OLLAMA")
echo -e "  Fox↔Ollama cosine similarity: ${BOLD}${SIM}${NC} (expect >0.99 for identical model)"

DIFF_FOX="{\"model\":\"${FOX_MODEL}\",\"input\":\"Quantum physics describes subatomic particles.\"}"
RESP_DIFF=$(curl -s -X POST "$FOX_OLLAMA_URL" -H "Content-Type: application/json" -d "$DIFF_FOX")
SIM_DIFF=$(cosine_similarity "$RESP_FOX" "$RESP_DIFF")
echo -e "  Same↔Different text similarity: ${BOLD}${SIM_DIFF}${NC} (expect <0.9 — different semantics)"

# ============================================================================
# 4. Latency benchmarks
# ============================================================================

echo -e "\n${BOLD}── Single-request latency ──${NC}\n"

echo -e "${GREEN}▸ Short text (${#SHORT_TEXT} chars)${NC}"
run_bench "Fox  (OpenAI)" "$FOX_OPENAI_URL" "$FOX_OPENAI_DATA" "$REQUESTS" "$WARMUP"
run_bench "Fox  (Ollama)" "$FOX_OLLAMA_URL" "$FOX_OLLAMA_DATA" "$REQUESTS" "$WARMUP"
run_bench "Ollama       " "$OLLAMA_URL"     "$OLLAMA_DATA"     "$REQUESTS" "$WARMUP"

FOX_MED_OPENAI="{\"model\":\"${FOX_MODEL}\",\"input\":\"${MEDIUM_TEXT}\"}"
FOX_MED_OLLAMA="{\"model\":\"${FOX_MODEL}\",\"input\":\"${MEDIUM_TEXT}\"}"
OLL_MED="{\"model\":\"${MODEL_NAME}\",\"input\":\"${MEDIUM_TEXT}\"}"

echo ""
echo -e "${GREEN}▸ Medium text (${#MEDIUM_TEXT} chars)${NC}"
run_bench "Fox  (OpenAI)" "$FOX_OPENAI_URL" "$FOX_MED_OPENAI" "$REQUESTS" "$WARMUP"
run_bench "Fox  (Ollama)" "$FOX_OLLAMA_URL" "$FOX_MED_OLLAMA" "$REQUESTS" "$WARMUP"
run_bench "Ollama       " "$OLLAMA_URL"     "$OLL_MED"        "$REQUESTS" "$WARMUP"

FOX_LONG_OPENAI="{\"model\":\"${FOX_MODEL}\",\"input\":\"${LONG_TEXT}\"}"
FOX_LONG_OLLAMA="{\"model\":\"${FOX_MODEL}\",\"input\":\"${LONG_TEXT}\"}"
OLL_LONG="{\"model\":\"${MODEL_NAME}\",\"input\":\"${LONG_TEXT}\"}"

echo ""
echo -e "${GREEN}▸ Long text (${#LONG_TEXT} chars)${NC}"
run_bench "Fox  (OpenAI)" "$FOX_OPENAI_URL" "$FOX_LONG_OPENAI" "$REQUESTS" "$WARMUP"
run_bench "Fox  (Ollama)" "$FOX_OLLAMA_URL" "$FOX_LONG_OLLAMA" "$REQUESTS" "$WARMUP"
run_bench "Ollama       " "$OLLAMA_URL"     "$OLL_LONG"        "$REQUESTS" "$WARMUP"

# ============================================================================
# 5. Batch embedding benchmark
# ============================================================================

echo -e "\n${BOLD}── Batch embedding (10 texts per request) ──${NC}\n"

BATCH_TEXTS='[
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning enables computers to learn from data.",
    "Rust is a systems programming language focused on safety.",
    "Docker containers package applications with their dependencies.",
    "Neural networks are inspired by biological brain structures.",
    "Kubernetes orchestrates containerized workloads at scale.",
    "Natural language processing handles human language computationally.",
    "PostgreSQL is a powerful open-source relational database.",
    "WebAssembly enables near-native performance in web browsers.",
    "Transformers revolutionized natural language understanding."
]'

FOX_BATCH_OPENAI="{\"model\":\"${FOX_MODEL}\",\"input\":${BATCH_TEXTS}}"
FOX_BATCH_OLLAMA="{\"model\":\"${FOX_MODEL}\",\"input\":${BATCH_TEXTS}}"
OLL_BATCH="{\"model\":\"${MODEL_NAME}\",\"input\":${BATCH_TEXTS}}"

BATCH_N=$((REQUESTS / 2))
run_bench "Fox  (OpenAI batch)" "$FOX_OPENAI_URL" "$FOX_BATCH_OPENAI" "$BATCH_N" "$WARMUP"
run_bench "Fox  (Ollama batch)" "$FOX_OLLAMA_URL"  "$FOX_BATCH_OLLAMA" "$BATCH_N" "$WARMUP"
run_bench "Ollama (batch)     " "$OLLAMA_URL"      "$OLL_BATCH"        "$BATCH_N" "$WARMUP"

# ============================================================================
# 6. Summary
# ============================================================================

echo -e "\n${BOLD}═══ Done ═══${NC}"
echo "Containers will be removed on exit."
