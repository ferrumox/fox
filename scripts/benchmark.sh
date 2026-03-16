#!/usr/bin/env bash
# benchmark.sh — reproducible benchmark: ferrumox vs Ollama
#
# Usage:
#   ./scripts/benchmark.sh [MODEL] [CONCURRENCY] [REQUESTS] [--docker]
#
# Examples:
#   ./scripts/benchmark.sh
#   ./scripts/benchmark.sh llama3.2 8 100
#   ./scripts/benchmark.sh gemma3 4 50 --docker
#
# Requirements (local mode):
#   - ferrumox binary at ./target/release/fox (or in PATH as "fox")
#   - fox-bench binary at ./target/release/fox-bench
#   - Ollama running at http://localhost:11434
#   - The requested model pulled in both ferrumox and Ollama
#
# Requirements (--docker mode):
#   - docker compose v2
#   - fox-bench binary at ./target/release/fox-bench
#
# Output is written to benches/results.md

set -euo pipefail

# ── Arg parsing ───────────────────────────────────────────────────────────────
MODEL="llama3.2"
CONCURRENCY="4"
REQUESTS="50"
DOCKER_MODE=0

POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --docker) DOCKER_MODE=1 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done

[[ ${#POSITIONAL[@]} -ge 1 ]] && MODEL="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && CONCURRENCY="${POSITIONAL[1]}"
[[ ${#POSITIONAL[@]} -ge 3 ]] && REQUESTS="${POSITIONAL[2]}"

MAX_TOKENS=128
PROMPT="Write a short paragraph about the Rust programming language."

FOX_URL="http://localhost:8080"
OLLAMA_URL="http://localhost:11434"
FOX_BIN="${FOX_BIN:-./target/release/fox}"
BENCH_BIN="${BENCH_BIN:-./target/release/fox-bench}"
RESULTS_FILE="benches/results.md"
COMPOSE_FILE="docker-compose.bench.yml"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

wait_http() {
    local url="$1" label="$2" retries="${3:-30}"
    for i in $(seq 1 "$retries"); do
        curl -sf "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    die "$label did not become ready at $url"
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
[[ -f "$BENCH_BIN" ]] || die "fox-bench not found at $BENCH_BIN — run: cargo build --release"
require jq

echo "=== ferrumox benchmark ==="
echo "  Model       : $MODEL"
echo "  Concurrency : $CONCURRENCY"
echo "  Requests    : $REQUESTS"
echo "  Max tokens  : $MAX_TOKENS"
[[ $DOCKER_MODE -eq 1 ]] && echo "  Mode        : Docker"
echo

# ── Docker mode ───────────────────────────────────────────────────────────────
FOX_STARTED=0
if [[ $DOCKER_MODE -eq 1 ]]; then
    require docker
    [[ -f "$COMPOSE_FILE" ]] || die "$COMPOSE_FILE not found — run from repo root"

    echo "Starting Docker services (ferrumox + Ollama)..."
    docker compose -f "$COMPOSE_FILE" up -d --build

    echo "Waiting for ferrumox..."
    wait_http "$FOX_URL/health" "ferrumox" 60

    echo "Waiting for Ollama..."
    wait_http "$OLLAMA_URL/api/tags" "Ollama" 60

    echo "Pulling model in ferrumox..."
    docker compose -f "$COMPOSE_FILE" exec ferrumox fox pull "$MODEL"

    echo "Pulling model in Ollama..."
    docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$MODEL"

    HAVE_OLLAMA=1
    echo "Both services ready."
    echo
else
    # ── Local mode: start ferrumox if needed ──────────────────────────────────
    if ! curl -sf "$FOX_URL/health" >/dev/null 2>&1; then
        echo "Starting ferrumox server..."
        [[ -f "$FOX_BIN" ]] || die "fox binary not found at $FOX_BIN"
        "$FOX_BIN" serve &
        FOX_PID=$!
        FOX_STARTED=1
        wait_http "$FOX_URL/health" "ferrumox" 30
        echo "  ferrumox ready (pid $FOX_PID)"
    fi

    # ── Check Ollama ──────────────────────────────────────────────────────────
    HAVE_OLLAMA=0
    if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
        HAVE_OLLAMA=1
        echo "Ollama detected at $OLLAMA_URL"
    else
        echo "WARNING: Ollama not found at $OLLAMA_URL — running single-server benchmark only"
    fi
    echo
fi

cleanup() {
    [[ $FOX_STARTED -eq 1 && -n "${FOX_PID:-}" ]] && kill "$FOX_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Run benchmark ─────────────────────────────────────────────────────────────
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
HARDWARE="$(uname -m) $(uname -s)"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo 'unknown')"
    HARDWARE="$HARDWARE / GPU: $GPU"
fi

if [[ $HAVE_OLLAMA -eq 1 ]]; then
    echo "Running comparison benchmark (ferrumox vs Ollama)..."
    JSON_OUTPUT="$("$BENCH_BIN" \
        --url "$FOX_URL" \
        --compare-url "$OLLAMA_URL" \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --output json)"

    # Also print text table
    "$BENCH_BIN" \
        --url "$FOX_URL" \
        --compare-url "$OLLAMA_URL" \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --output text
else
    echo "Running single-server benchmark (ferrumox)..."
    JSON_OUTPUT="$("$BENCH_BIN" \
        --url "$FOX_URL" \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --output json)"

    "$BENCH_BIN" \
        --url "$FOX_URL" \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --output text
fi

# ── Write results.md ──────────────────────────────────────────────────────────
mkdir -p benches

# Parse JSON with jq
FOX_TTFT_P50=$(echo "$JSON_OUTPUT" | jq '.primary.ttft_p50_ms')
FOX_TTFT_P95=$(echo "$JSON_OUTPUT" | jq '.primary.ttft_p95_ms')
FOX_LAT_P50=$(echo "$JSON_OUTPUT" | jq '.primary.latency_p50_ms')
FOX_LAT_P95=$(echo "$JSON_OUTPUT" | jq '.primary.latency_p95_ms')
FOX_LAT_P99=$(echo "$JSON_OUTPUT" | jq '.primary.latency_p99_ms')
FOX_THRPT=$(echo "$JSON_OUTPUT" | jq '.primary.throughput_tokens_per_sec')

if [[ $HAVE_OLLAMA -eq 1 ]]; then
    OLL_TTFT_P50=$(echo "$JSON_OUTPUT" | jq '.comparison.ttft_p50_ms')
    OLL_TTFT_P95=$(echo "$JSON_OUTPUT" | jq '.comparison.ttft_p95_ms')
    OLL_LAT_P50=$(echo "$JSON_OUTPUT" | jq '.comparison.latency_p50_ms')
    OLL_LAT_P95=$(echo "$JSON_OUTPUT" | jq '.comparison.latency_p95_ms')
    OLL_LAT_P99=$(echo "$JSON_OUTPUT" | jq '.comparison.latency_p99_ms')
    OLL_THRPT=$(echo "$JSON_OUTPUT" | jq '.comparison.throughput_tokens_per_sec')
    IMP_TTFT_P50=$(echo "$JSON_OUTPUT" | jq '.improvement.ttft_p50_pct | round')
    IMP_THRPT=$(echo "$JSON_OUTPUT" | jq '.improvement.throughput_pct | round')

    cat >> "$RESULTS_FILE" <<EOF

---

## Benchmark run: $TIMESTAMP

**Hardware**: $HARDWARE
**Model**: $MODEL
**Concurrency**: $CONCURRENCY workers / **Requests**: $REQUESTS / **Max tokens**: $MAX_TOKENS

| Metric | ferrumox | Ollama | Δ |
|--------|----------|--------|---|
| TTFT P50 | ${FOX_TTFT_P50}ms | ${OLL_TTFT_P50}ms | +${IMP_TTFT_P50}% |
| TTFT P95 | ${FOX_TTFT_P95}ms | ${OLL_TTFT_P95}ms | — |
| Latency P50 | ${FOX_LAT_P50}ms | ${OLL_LAT_P50}ms | — |
| Latency P95 | ${FOX_LAT_P95}ms | ${OLL_LAT_P95}ms | — |
| Latency P99 | ${FOX_LAT_P99}ms | ${OLL_LAT_P99}ms | — |
| Throughput | ${FOX_THRPT} t/s | ${OLL_THRPT} t/s | +${IMP_THRPT}% |

<details>
<summary>Raw JSON</summary>

\`\`\`json
$JSON_OUTPUT
\`\`\`

</details>
EOF
else
    cat >> "$RESULTS_FILE" <<EOF

---

## Benchmark run: $TIMESTAMP

**Hardware**: $HARDWARE
**Model**: $MODEL
**Concurrency**: $CONCURRENCY workers / **Requests**: $REQUESTS / **Max tokens**: $MAX_TOKENS

| Metric | ferrumox |
|--------|----------|
| TTFT P50 | ${FOX_TTFT_P50}ms |
| TTFT P95 | ${FOX_TTFT_P95}ms |
| Latency P50 | ${FOX_LAT_P50}ms |
| Latency P95 | ${FOX_LAT_P95}ms |
| Latency P99 | ${FOX_LAT_P99}ms |
| Throughput | ${FOX_THRPT} t/s |

<details>
<summary>Raw JSON</summary>

\`\`\`json
$JSON_OUTPUT
\`\`\`

</details>
EOF
fi

echo
echo "Results appended to $RESULTS_FILE"

# ── Update README.md benchmark table (comparison runs only) ──────────────────
if [[ $HAVE_OLLAMA -eq 1 && -f "README.md" ]]; then
    README_TMP="$(mktemp)"
    awk -v fox_ttft_p50="${FOX_TTFT_P50}" \
        -v fox_ttft_p95="${FOX_TTFT_P95}" \
        -v fox_lat_p50="${FOX_LAT_P50}" \
        -v fox_lat_p95="${FOX_LAT_P95}" \
        -v fox_thrpt="${FOX_THRPT}" \
        -v oll_ttft_p50="${OLL_TTFT_P50}" \
        -v oll_ttft_p95="${OLL_TTFT_P95}" \
        -v oll_lat_p50="${OLL_LAT_P50}" \
        -v oll_lat_p95="${OLL_LAT_P95}" \
        -v oll_thrpt="${OLL_THRPT}" \
        -v imp_ttft="${IMP_TTFT_P50}" \
        -v imp_thrpt="${IMP_THRPT}" \
        '
        /<!-- BENCH_TABLE_START -->/ { print; in_table=1; next }
        /<!-- BENCH_TABLE_END -->/  { in_table=0 }
        in_table { next }
        !in_table {
            if (/<!-- BENCH_TABLE_END -->/) {
                print "| Metric | ferrumox | Ollama | Improvement |"
                print "|--------|----------|--------|-------------|"
                printf "| TTFT P50 | %sms | %sms | **+%s%%** |\n", fox_ttft_p50, oll_ttft_p50, imp_ttft
                printf "| TTFT P95 | %sms | %sms | — |\n", fox_ttft_p95, oll_ttft_p95
                printf "| Latency P50 | %sms | %sms | — |\n", fox_lat_p50, oll_lat_p50
                printf "| Latency P95 | %sms | %sms | — |\n", fox_lat_p95, oll_lat_p95
                printf "| Throughput | %s t/s | %s t/s | **+%s%%** |\n", fox_thrpt, oll_thrpt, imp_thrpt
            }
            print
        }
        ' README.md > "$README_TMP" && mv "$README_TMP" README.md
    echo "README.md benchmark table updated."
fi
