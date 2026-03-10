#!/usr/bin/env bash
# benchmark.sh — reproducible benchmark: ferrumox vs Ollama
#
# Usage:
#   ./scripts/benchmark.sh [MODEL] [CONCURRENCY] [REQUESTS]
#
# Examples:
#   ./scripts/benchmark.sh
#   ./scripts/benchmark.sh llama3.2 8 100
#
# Requirements:
#   - ferrumox binary at ./target/release/fox (or in PATH as "fox")
#   - fox-bench binary at ./target/release/fox-bench
#   - Ollama running at http://localhost:11434
#   - The requested model pulled in both ferrumox and Ollama
#
# Output is written to benches/results.md

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="${1:-llama3.2}"
CONCURRENCY="${2:-4}"
REQUESTS="${3:-50}"
MAX_TOKENS=128
PROMPT="Write a short paragraph about the Rust programming language."

FOX_URL="http://localhost:8080"
OLLAMA_URL="http://localhost:11434"
FOX_BIN="${FOX_BIN:-./target/release/fox}"
BENCH_BIN="${BENCH_BIN:-./target/release/fox-bench}"
RESULTS_FILE="benches/results.md"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

# ── Pre-flight ────────────────────────────────────────────────────────────────
[[ -f "$BENCH_BIN" ]] || die "fox-bench not found at $BENCH_BIN — run: cargo build --release"
require jq

echo "=== ferrumox benchmark ==="
echo "  Model       : $MODEL"
echo "  Concurrency : $CONCURRENCY"
echo "  Requests    : $REQUESTS"
echo "  Max tokens  : $MAX_TOKENS"
echo

# ── Check if ferrumox is already running; if not, start it ───────────────────
FOX_STARTED=0
if ! curl -sf "$FOX_URL/health" >/dev/null 2>&1; then
    echo "Starting ferrumox server..."
    [[ -f "$FOX_BIN" ]] || die "fox binary not found at $FOX_BIN"
    "$FOX_BIN" serve &
    FOX_PID=$!
    FOX_STARTED=1
    # Wait for server to be ready (up to 30s)
    for i in $(seq 1 30); do
        curl -sf "$FOX_URL/health" >/dev/null 2>&1 && break
        sleep 1
    done
    curl -sf "$FOX_URL/health" >/dev/null 2>&1 || die "ferrumox failed to start"
    echo "  ferrumox ready (pid $FOX_PID)"
fi

cleanup() {
    [[ $FOX_STARTED -eq 1 && -n "${FOX_PID:-}" ]] && kill "$FOX_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ── Check Ollama ──────────────────────────────────────────────────────────────
HAVE_OLLAMA=0
if curl -sf "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    HAVE_OLLAMA=1
    echo "Ollama detected at $OLLAMA_URL"
else
    echo "WARNING: Ollama not found at $OLLAMA_URL — running single-server benchmark only"
fi
echo

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
