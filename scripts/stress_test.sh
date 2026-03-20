#!/usr/bin/env bash
# stress_test.sh — fox v1.0 full stress test suite
#
# Runs 6 scenarios against fox (via Docker) and compares with Ollama when available.
# Saves one JSON file and one Markdown report per run.
#
# Usage:
#   ./scripts/stress_test.sh
#   ./scripts/stress_test.sh --fox-model Llama-3.2 --ollama-model llama32
#   ./scripts/stress_test.sh --no-ollama           # skip Ollama comparison
#   ./scripts/stress_test.sh --no-docker           # fox is already running

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
BENCH="${BENCH:-./target/release/fox-bench}"
FOX_URL="http://localhost:8080"
OLLAMA_URL="http://localhost:11434"
FOX_MODEL="Llama-3.2"
OLLAMA_MODEL="llama32"
SECOND_MODEL="qwen"          # second model for multi-model scenario
DOCKER_CONTAINER="fox-stress"
NO_OLLAMA=0
NO_DOCKER=0

PROMPT_SHORT="What is Rust in one sentence?"
PROMPT_LONG="Explain Rust's ownership and borrowing system in detail with practical examples."

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fox-model)    FOX_MODEL="$2";    shift 2 ;;
        --ollama-model) OLLAMA_MODEL="$2"; shift 2 ;;
        --no-ollama)    NO_OLLAMA=1;       shift ;;
        --no-docker)    NO_DOCKER=1;       shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RESULTS_DIR="benches"
OUT_JSON="${RESULTS_DIR}/stress-${TIMESTAMP}.json"
OUT_MD="${RESULTS_DIR}/stress-${TIMESTAMP}.md"

# ── Helpers ───────────────────────────────────────────────────────────────────
die()     { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not found"; }
hr()      { printf '%0.s─' {1..60}; echo; }

wait_http() {
    local url="$1" label="$2" retries="${3:-90}"
    printf "  Waiting for %s " "$label"
    for _ in $(seq 1 "$retries"); do
        if curl -sf "$url" >/dev/null 2>&1; then
            echo "✓"
            return 0
        fi
        printf "."
        sleep 1
    done
    echo ""
    die "$label not ready at $url after ${retries}s"
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
[[ -f "$BENCH" ]] || die "fox-bench not found at $BENCH\n  Build it with: cargo build --release --bin fox-bench"
require jq
require curl

# ── Start fox-test container (unless --no-docker or already running) ──────────
FOX_STARTED=0
if [[ $NO_DOCKER -eq 0 ]]; then
    if curl -sf "${FOX_URL}/health" >/dev/null 2>&1; then
        echo "  fox already running at ${FOX_URL}"
    else
        echo "  Starting fox-test container..."
        docker run -d --rm \
            --name "$DOCKER_CONTAINER" \
            -p 8080:8080 \
            -v "${HOME}/.cache/ferrumox/models:/root/.cache/ferrumox/models:ro" \
            -e FOX_HOST=0.0.0.0 \
            -e FOX_PORT=8080 \
            -e FOX_MAX_CONTEXT_LEN=4096 \
            -e FOX_GPU_MEMORY_FRACTION=0.9 \
            -e FOX_MAX_BATCH_SIZE=8 \
            fox-test:latest >/dev/null
        FOX_STARTED=1
        wait_http "${FOX_URL}/health" "fox" 90
    fi
else
    curl -sf "${FOX_URL}/health" >/dev/null 2>&1 || die "fox not running at $FOX_URL (start it or remove --no-docker)"
    echo "  Using fox at ${FOX_URL}"
fi

cleanup() {
    [[ $FOX_STARTED -eq 1 ]] && docker stop "$DOCKER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ── Check Ollama ──────────────────────────────────────────────────────────────
HAVE_OLLAMA=0
if [[ $NO_OLLAMA -eq 0 ]] && curl -sf "${OLLAMA_URL}/api/tags" >/dev/null 2>&1; then
    HAVE_OLLAMA=1
    echo "  Ollama available at ${OLLAMA_URL} (model: ${OLLAMA_MODEL})"
else
    echo "  Ollama not available — comparison scenarios will be skipped"
fi

# ── GPU / hardware info ───────────────────────────────────────────────────────
GPU_INFO="(no GPU detected)"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo unknown)"
fi
CPU_INFO="$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || uname -p)"

echo ""
hr
printf "  fox v1.0 Stress Test — %s\n" "$TIMESTAMP"
hr
echo "  Fox URL    : $FOX_URL"
echo "  Fox model  : $FOX_MODEL"
echo "  2nd model  : $SECOND_MODEL"
echo "  CPU        : $CPU_INFO"
echo "  GPU        : $GPU_INFO"
echo "  Results    : $OUT_MD"
hr
echo ""

mkdir -p "$RESULTS_DIR"

# ── Scenario runner ───────────────────────────────────────────────────────────
# run_scenario NAME CONC REQS MAX_TOKENS PROMPT [compare=1]
declare -A SCENARIO_RESULTS
run_scenario() {
    local name="$1" conc="$2" reqs="$3" tokens="$4" prompt="$5" compare="${6:-0}"
    printf "  %-32s (c=%-2d n=%-3d tok=%-4d) " "$name" "$conc" "$reqs" "$tokens"

    local -a args=(
        --url "$FOX_URL"
        --model "$FOX_MODEL"
        --concurrency "$conc"
        --requests "$reqs"
        --max-tokens "$tokens"
        --prompt "$prompt"
        --label fox
        --output json
    )

    if [[ $compare -eq 1 && $HAVE_OLLAMA -eq 1 ]]; then
        args+=(
            --compare-url "$OLLAMA_URL"
            --compare-model "$OLLAMA_MODEL"
            --compare-label ollama
        )
    fi

    local json
    if json="$("$BENCH" "${args[@]}" 2>/dev/null)"; then
        echo "✓"
        SCENARIO_RESULTS["$name"]="$json"
    else
        echo "✗  (failed)"
        SCENARIO_RESULTS["$name"]='{"error":"benchmark failed"}'
    fi
}

# ── Warmup ────────────────────────────────────────────────────────────────────
echo "Warmup (5 requests to pre-load model into KV cache)..."
"$BENCH" --url "$FOX_URL" --model "$FOX_MODEL" \
    --concurrency 1 --requests 5 --max-tokens 32 \
    --prompt "Hello" --output json >/dev/null 2>&1 || true
echo "  done"
echo ""

# ── Scenarios ─────────────────────────────────────────────────────────────────
echo "Running scenarios..."
echo ""

# S1: Baseline — single worker, short responses
run_scenario "S1: baseline (1 worker)" 1 20 128 "$PROMPT_SHORT"

# S2: Medium concurrency
run_scenario "S2: medium concurrency (4 workers)" 4 50 128 "$PROMPT_SHORT"

# S3: High load
run_scenario "S3: high load (8 workers)" 8 100 256 "$PROMPT_SHORT"

# S4: Long responses (measures sustained throughput)
run_scenario "S4: long responses (4 workers)" 4 20 512 "$PROMPT_LONG"

# S5: Multi-model — run against second model (tests lazy-load + LRU)
echo ""
echo "  S5: multi-model — switching between models (tests LRU eviction)"
printf "    %-28s (c=%-2d n=%-3d tok=%-4d) " "${FOX_MODEL}" 2 10 128 "$PROMPT_SHORT"
if json_a="$("$BENCH" --url "$FOX_URL" --model "$FOX_MODEL" \
    --concurrency 2 --requests 10 --max-tokens 128 \
    --prompt "$PROMPT_SHORT" --label fox --output json 2>/dev/null)"; then
    echo "✓"
else
    json_a='{"error":"failed"}'; echo "✗"
fi
printf "    %-28s (c=%-2d n=%-3d tok=%-4d) " "${SECOND_MODEL}" 2 10 128 "$PROMPT_SHORT"
if json_b="$("$BENCH" --url "$FOX_URL" --model "$SECOND_MODEL" \
    --concurrency 2 --requests 10 --max-tokens 128 \
    --prompt "$PROMPT_SHORT" --label fox --output json 2>/dev/null)"; then
    echo "✓"
else
    json_b='{"error":"failed"}'; echo "✗"
fi
printf "    %-28s (c=%-2d n=%-3d tok=%-4d) " "${FOX_MODEL} (back)" 2 10 128 "$PROMPT_SHORT"
if json_c="$("$BENCH" --url "$FOX_URL" --model "$FOX_MODEL" \
    --concurrency 2 --requests 10 --max-tokens 128 \
    --prompt "$PROMPT_SHORT" --label fox --output json 2>/dev/null)"; then
    echo "✓"
else
    json_c='{"error":"failed"}'; echo "✗"
fi
SCENARIO_RESULTS["S5: multi-model"]="$(jq -n \
    --argjson a "$json_a" --argjson b "$json_b" --argjson c "$json_c" \
    '{model_a_first: $a.primary, model_b: $b.primary, model_a_reload: $c.primary}')"

# S6: Fox vs Ollama comparison
echo ""
if [[ $HAVE_OLLAMA -eq 1 ]]; then
    run_scenario "S6: fox vs ollama (4 workers)" 4 50 128 "$PROMPT_SHORT" 1
else
    echo "  S6: fox vs ollama — SKIPPED (Ollama not available)"
    SCENARIO_RESULTS["S6: fox vs ollama"]='{"skipped":true}'
fi

echo ""
hr

# ── Build JSON output ─────────────────────────────────────────────────────────
{
    echo "{"
    echo "  \"timestamp\": \"$TIMESTAMP\","
    echo "  \"fox_url\": \"$FOX_URL\","
    echo "  \"fox_model\": \"$FOX_MODEL\","
    echo "  \"gpu\": $(echo "$GPU_INFO" | jq -Rs .),"
    echo "  \"cpu\": $(echo "$CPU_INFO" | jq -Rs .),"
    echo "  \"scenarios\": {"
    first=1
    for name in "S1: baseline (1 worker)" \
                "S2: medium concurrency (4 workers)" \
                "S3: high load (8 workers)" \
                "S4: long responses (4 workers)" \
                "S5: multi-model" \
                "S6: fox vs ollama"; do
        [[ $first -eq 0 ]] && echo ","
        local_json="${SCENARIO_RESULTS[$name]:-{}}"
        printf '    %s: %s' "$(echo "$name" | jq -Rs .)" "$local_json"
        first=0
    done
    echo ""
    echo "  }"
    echo "}"
} > "$OUT_JSON"

# ── Build Markdown report ─────────────────────────────────────────────────────
extract() {
    local json="$1" field="$2" default="${3:--}"
    echo "$json" | jq -r "$field // \"$default\"" 2>/dev/null || echo "$default"
}

{
cat <<EOF
# Fox v1.0 Stress Test — $TIMESTAMP

**Hardware**: $CPU_INFO / $GPU_INFO
**Fox model**: \`$FOX_MODEL\`
**Fox URL**: $FOX_URL

---

## S1: Baseline (1 worker, 20 requests, 128 tok)

| Metric | Value |
|--------|-------|
| TTFT P50 | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.ttft_p50_ms')ms |
| TTFT P95 | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.ttft_p95_ms')ms |
| Latency P50 | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.latency_p50_ms')ms |
| Latency P95 | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.latency_p95_ms')ms |
| Throughput | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.throughput_tokens_per_sec | round') tok/s |
| Errors | $(extract "${SCENARIO_RESULTS["S1: baseline (1 worker)"]}" '.primary.requests_err') |

---

## S2: Medium Concurrency (4 workers, 50 requests, 128 tok)

| Metric | Value |
|--------|-------|
| TTFT P50 | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.ttft_p50_ms')ms |
| TTFT P95 | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.ttft_p95_ms')ms |
| Latency P50 | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.latency_p50_ms')ms |
| Latency P95 | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.latency_p95_ms')ms |
| Throughput | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.throughput_tokens_per_sec | round') tok/s |
| Errors | $(extract "${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}" '.primary.requests_err') |

---

## S3: High Load (8 workers, 100 requests, 256 tok)

| Metric | Value |
|--------|-------|
| TTFT P50 | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.ttft_p50_ms')ms |
| TTFT P95 | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.ttft_p95_ms')ms |
| Latency P50 | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.latency_p50_ms')ms |
| Latency P95 | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.latency_p95_ms')ms |
| Throughput | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.throughput_tokens_per_sec | round') tok/s |
| Errors | $(extract "${SCENARIO_RESULTS["S3: high load (8 workers)"]}" '.primary.requests_err') |

---

## S4: Long Responses (4 workers, 20 requests, 512 tok)

| Metric | Value |
|--------|-------|
| TTFT P50 | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.ttft_p50_ms')ms |
| TTFT P95 | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.ttft_p95_ms')ms |
| Latency P50 | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.latency_p50_ms')ms |
| Latency P95 | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.latency_p95_ms')ms |
| Throughput | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.throughput_tokens_per_sec | round') tok/s |
| Errors | $(extract "${SCENARIO_RESULTS["S4: long responses (4 workers)"]}" '.primary.requests_err') |

---

## S5: Multi-Model (LRU eviction test)

Three sequential runs: model A → model B → model A (reload from eviction).

| Phase | Throughput (tok/s) | Latency P50 |
|-------|-------------------|-------------|
| ${FOX_MODEL} (first load) | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_a_first.throughput_tokens_per_sec | round') | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_a_first.latency_p50_ms')ms |
| ${SECOND_MODEL} (evicts A) | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_b.throughput_tokens_per_sec | round') | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_b.latency_p50_ms')ms |
| ${FOX_MODEL} (reload after eviction) | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_a_reload.throughput_tokens_per_sec | round') | $(extract "${SCENARIO_RESULTS["S5: multi-model"]}" '.model_a_reload.latency_p50_ms')ms |

---

## S6: Fox vs Ollama (4 workers, 50 requests, 128 tok)

EOF

if [[ $HAVE_OLLAMA -eq 1 ]]; then
    S6="${SCENARIO_RESULTS["S6: fox vs ollama"]}"
cat <<EOF
| Metric | fox | ollama | Δ |
|--------|-----|--------|---|
| TTFT P50 | $(extract "$S6" '.primary.ttft_p50_ms')ms | $(extract "$S6" '.comparison.ttft_p50_ms')ms | $(extract "$S6" '.improvement.ttft_p50_pct | round')% |
| TTFT P95 | $(extract "$S6" '.primary.ttft_p95_ms')ms | $(extract "$S6" '.comparison.ttft_p95_ms')ms | — |
| Latency P50 | $(extract "$S6" '.primary.latency_p50_ms')ms | $(extract "$S6" '.comparison.latency_p50_ms')ms | $(extract "$S6" '.improvement.latency_p50_pct | round')% |
| Latency P95 | $(extract "$S6" '.primary.latency_p95_ms')ms | $(extract "$S6" '.comparison.latency_p95_ms')ms | — |
| Latency P99 | $(extract "$S6" '.primary.latency_p99_ms')ms | $(extract "$S6" '.comparison.latency_p99_ms')ms | — |
| Throughput | $(extract "$S6" '.primary.throughput_tokens_per_sec | round') tok/s | $(extract "$S6" '.comparison.throughput_tokens_per_sec | round') tok/s | $(extract "$S6" '.improvement.throughput_pct | round')% |
| Errors (fox) | $(extract "$S6" '.primary.requests_err') | — | — |

EOF
else
    echo "_Ollama not available — skipped._"
    echo ""
fi

cat <<EOF
---

## Pass/fail checklist

EOF

# Evaluate pass/fail criteria against the scenarios
check_threshold() {
    local label="$1" value="$2" threshold="$3" lower_is_better="${4:-1}"
    local status
    if [[ "$value" == "-" || "$value" == "null" ]]; then
        status="⚠  (no data)"
    elif [[ $lower_is_better -eq 1 ]]; then
        [[ "$value" -le "$threshold" ]] 2>/dev/null && status="✅ ${value} ≤ ${threshold}" || status="❌ ${value} > ${threshold}"
    else
        [[ "$value" -ge "$threshold" ]] 2>/dev/null && status="✅ ${value} ≥ ${threshold}" || status="❌ ${value} < ${threshold}"
    fi
    echo "- **$label**: $status"
}

S2="${SCENARIO_RESULTS["S2: medium concurrency (4 workers)"]}"
S3="${SCENARIO_RESULTS["S3: high load (8 workers)"]}"

TTFT_P50=$(extract "$S2" '.primary.ttft_p50_ms')
TTFT_P95=$(extract "$S2" '.primary.ttft_p95_ms')
S3_ERRORS=$(extract "$S3" '.primary.requests_err')

check_threshold "TTFT P50 < 500ms (c=4, n=50)" "$TTFT_P50" 500
check_threshold "TTFT P95 < 1000ms (c=4, n=50)" "$TTFT_P95" 1000
check_threshold "Zero 5xx errors (c=8, n=100)" "$S3_ERRORS" 0

if [[ $HAVE_OLLAMA -eq 1 ]]; then
    S6="${SCENARIO_RESULTS["S6: fox vs ollama"]}"
    FOX_THRPT=$(extract "$S6" '.primary.throughput_tokens_per_sec | round')
    OLL_THRPT=$(extract "$S6" '.comparison.throughput_tokens_per_sec | round')
    check_threshold "fox throughput ≥ ollama (tok/s)" "$FOX_THRPT" "$OLL_THRPT" 0
fi

cat <<EOF

---

*Raw JSON saved to: \`$OUT_JSON\`*
EOF
} > "$OUT_MD"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "Results"
hr
echo ""
cat "$OUT_MD"
echo ""
hr
echo "  JSON : $OUT_JSON"
echo "  MD   : $OUT_MD"
