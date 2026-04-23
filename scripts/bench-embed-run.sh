#!/usr/bin/env bash
# bench-embed-run.sh — Run embedding latency benchmark against a Fox server.
#
# Usage: bench-embed-run.sh [--url URL] [--model MODEL] [--requests N] [--warmup N]
#
# Outputs JSON to stdout:
#   { "short": { "avg_ms": N, "p50_ms": N, "p95_ms": N, "p99_ms": N, "rps": N },
#     "medium": { ... }, "long": { ... }, "batch": { ... } }

set -euo pipefail

URL="http://localhost:8080"
MODEL=""
REQUESTS=30
WARMUP=5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url)    URL="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

EMBED_URL="${URL}/v1/embeddings"

SHORT_TEXT="The quick brown fox jumps over the lazy dog."
MEDIUM_TEXT="Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. These systems improve their performance on a specific task over time without being explicitly programmed. Common approaches include supervised learning, unsupervised learning, and reinforcement learning."
LONG_TEXT="${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT} ${MEDIUM_TEXT}"
BATCH_INPUT='["The quick brown fox.","Machine learning enables computers to learn from data.","Rust is a systems programming language.","Docker containers package applications.","Neural networks are inspired by biology.","Kubernetes orchestrates workloads.","NLP handles human language.","PostgreSQL is a relational database.","WebAssembly enables near-native perf.","Transformers revolutionized NLU."]'

time_request_ms() {
    curl -s -o /dev/null -w '%{time_total}' \
        -X POST "$EMBED_URL" \
        -H "Content-Type: application/json" \
        -d "$1" | awk '{printf "%.2f", $1 * 1000}'
}

run_bench() {
    local data="$1"
    local n="$2"
    local warmup="$3"

    for ((i=0; i<warmup; i++)); do
        curl -s -o /dev/null -X POST "$EMBED_URL" -H "Content-Type: application/json" -d "$data"
    done

    local times=()
    local start_ns
    start_ns=$(date +%s%N)

    for ((i=0; i<n; i++)); do
        local t
        t=$(time_request_ms "$data")
        times+=("$t")
    done

    local end_ns
    end_ns=$(date +%s%N)
    local wall_ms=$(( (end_ns - start_ns) / 1000000 ))

    IFS=$'\n' sorted=($(sort -g <<<"${times[*]}")); unset IFS

    local sum=0
    for t in "${times[@]}"; do
        sum=$(echo "$sum + $t" | bc)
    done
    local avg=$(echo "scale=2; $sum / $n" | bc)
    local p50=${sorted[$(( n / 2 ))]}
    local p95=${sorted[$(( n * 95 / 100 ))]}
    local p99=${sorted[$(( n * 99 / 100 ))]}
    local rps=$(echo "scale=2; $n / ($wall_ms / 1000)" | bc)

    printf '{"avg_ms":%s,"p50_ms":%s,"p95_ms":%s,"p99_ms":%s,"rps":%s}' \
        "$avg" "$p50" "$p95" "$p99" "$rps"
}

SHORT_DATA="{\"model\":\"${MODEL}\",\"input\":\"${SHORT_TEXT}\"}"
MEDIUM_DATA="{\"model\":\"${MODEL}\",\"input\":\"${MEDIUM_TEXT}\"}"
LONG_DATA="{\"model\":\"${MODEL}\",\"input\":\"${LONG_TEXT}\"}"
BATCH_DATA="{\"model\":\"${MODEL}\",\"input\":${BATCH_INPUT}}"

echo "Running embedding benchmark (${REQUESTS} requests, ${WARMUP} warmup)..." >&2

echo -n "  short..." >&2
SHORT_RESULT=$(run_bench "$SHORT_DATA" "$REQUESTS" "$WARMUP")
echo " done" >&2

echo -n "  medium..." >&2
MEDIUM_RESULT=$(run_bench "$MEDIUM_DATA" "$REQUESTS" "$WARMUP")
echo " done" >&2

echo -n "  long..." >&2
LONG_RESULT=$(run_bench "$LONG_DATA" "$REQUESTS" "$WARMUP")
echo " done" >&2

echo -n "  batch..." >&2
BATCH_RESULT=$(run_bench "$BATCH_DATA" "$REQUESTS" "$WARMUP")
echo " done" >&2

printf '{"short":%s,"medium":%s,"long":%s,"batch":%s}\n' \
    "$SHORT_RESULT" "$MEDIUM_RESULT" "$LONG_RESULT" "$BATCH_RESULT"
