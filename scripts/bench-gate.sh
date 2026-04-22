#!/usr/bin/env bash
# bench-gate.sh — compare two fox-bench JSON outputs, fail on regression.
#
# Usage: bench-gate.sh <baseline.json> <candidate.json>
#
# Thresholds (from PLAN.md v1.1.0 spec):
#   - TTFT P95 regression > 5%   → FAIL
#   - Throughput drop     > 3%   → FAIL
#
# Outputs:
#   - Terminal summary
#   - /tmp/bench-comment.md  (for posting to PR)
#   - Exit code 0 (pass) or 1 (regression detected)

set -euo pipefail

BASELINE="${1:?Usage: bench-gate.sh <baseline.json> <candidate.json>}"
CANDIDATE="${2:?Usage: bench-gate.sh <baseline.json> <candidate.json>}"

TTFT_THRESHOLD=5
THRPT_THRESHOLD=3

for f in "$BASELINE" "$CANDIDATE"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
    jq '.primary' "$f" >/dev/null 2>&1 || { echo "ERROR: $f is not valid bench JSON"; exit 1; }
done

# Extract metrics
BASE_TTFT_P50=$(jq '.primary.ttft_p50_ms' "$BASELINE")
BASE_TTFT_P95=$(jq '.primary.ttft_p95_ms' "$BASELINE")
BASE_THRPT=$(jq '.primary.throughput_tokens_per_sec' "$BASELINE")
BASE_LAT_P50=$(jq '.primary.latency_p50_ms' "$BASELINE")
BASE_LAT_P95=$(jq '.primary.latency_p95_ms' "$BASELINE")

CAND_TTFT_P50=$(jq '.primary.ttft_p50_ms' "$CANDIDATE")
CAND_TTFT_P95=$(jq '.primary.ttft_p95_ms' "$CANDIDATE")
CAND_THRPT=$(jq '.primary.throughput_tokens_per_sec' "$CANDIDATE")
CAND_LAT_P50=$(jq '.primary.latency_p50_ms' "$CANDIDATE")
CAND_LAT_P95=$(jq '.primary.latency_p95_ms' "$CANDIDATE")

# Calculate percentage changes (positive = regression for latency, negative = regression for throughput)
pct_change() {
    local base="$1" cand="$2" higher_is_worse="${3:-true}"
    if [ "$base" = "0" ] || [ "$base" = "0.0" ]; then
        echo "0.0"
        return
    fi
    if [ "$higher_is_worse" = "true" ]; then
        jq -n "($cand - $base) / $base * 100 | . * 100 | round / 100"
    else
        jq -n "($cand - $base) / $base * 100 | . * 100 | round / 100"
    fi
}

TTFT_P50_CHG=$(pct_change "$BASE_TTFT_P50" "$CAND_TTFT_P50")
TTFT_P95_CHG=$(pct_change "$BASE_TTFT_P95" "$CAND_TTFT_P95")
THRPT_CHG=$(pct_change "$BASE_THRPT" "$CAND_THRPT" false)
LAT_P50_CHG=$(pct_change "$BASE_LAT_P50" "$CAND_LAT_P50")
LAT_P95_CHG=$(pct_change "$BASE_LAT_P95" "$CAND_LAT_P95")

# Format: positive change on latency = bad (↑), negative = good (↓)
# Format: positive change on throughput = good (↑), negative = bad (↓)
fmt_latency() {
    local val="$1"
    local is_bad
    is_bad=$(jq -n "$val > 0" 2>/dev/null || echo "false")
    if [ "$is_bad" = "true" ]; then
        echo "+${val}%"
    else
        echo "${val}%"
    fi
}

fmt_throughput() {
    local val="$1"
    local is_good
    is_good=$(jq -n "$val >= 0" 2>/dev/null || echo "false")
    if [ "$is_good" = "true" ]; then
        echo "+${val}%"
    else
        echo "${val}%"
    fi
}

# Gate checks
FAILED=0
FAIL_REASONS=""

TTFT_REGRESSED=$(jq -n "$TTFT_P95_CHG > $TTFT_THRESHOLD")
if [ "$TTFT_REGRESSED" = "true" ]; then
    FAILED=1
    FAIL_REASONS="${FAIL_REASONS}TTFT P95 regressed by ${TTFT_P95_CHG}% (threshold: ${TTFT_THRESHOLD}%)\n"
fi

THRPT_REGRESSED=$(jq -n "(-1 * $THRPT_CHG) > $THRPT_THRESHOLD")
if [ "$THRPT_REGRESSED" = "true" ]; then
    FAILED=1
    FAIL_REASONS="${FAIL_REASONS}Throughput dropped by ${THRPT_CHG}% (threshold: -${THRPT_THRESHOLD}%)\n"
fi

# Terminal output
echo "════════════════════════════════════════════"
echo "  Benchmark Gate"
echo "════════════════════════════════════════════"
printf "  %-16s %10s %10s %10s\n" "Metric" "Base" "PR" "Change"
echo "  ────────────────────────────────────────"
printf "  %-16s %8sms %8sms %9s\n" "TTFT P50"  "$BASE_TTFT_P50" "$CAND_TTFT_P50" "$(fmt_latency "$TTFT_P50_CHG")"
printf "  %-16s %8sms %8sms %9s\n" "TTFT P95"  "$BASE_TTFT_P95" "$CAND_TTFT_P95" "$(fmt_latency "$TTFT_P95_CHG")"
printf "  %-16s %7s t/s %7s t/s %9s\n" "Throughput" "$BASE_THRPT" "$CAND_THRPT" "$(fmt_throughput "$THRPT_CHG")"
printf "  %-16s %8sms %8sms %9s\n" "Latency P50" "$BASE_LAT_P50" "$CAND_LAT_P50" "$(fmt_latency "$LAT_P50_CHG")"
printf "  %-16s %8sms %8sms %9s\n" "Latency P95" "$BASE_LAT_P95" "$CAND_LAT_P95" "$(fmt_latency "$LAT_P95_CHG")"
echo "════════════════════════════════════════════"

# Emoji helpers for markdown
gate_icon() {
    local val="$1" threshold="$2" higher_is_worse="${3:-true}"
    if [ "$higher_is_worse" = "true" ]; then
        if [ "$(jq -n "$val > $threshold")" = "true" ]; then echo "🔴"; else echo "✅"; fi
    else
        if [ "$(jq -n "(-1 * $val) > $threshold")" = "true" ]; then echo "🔴"; else echo "✅"; fi
    fi
}

# Generate PR comment
if [ "$FAILED" -eq 1 ]; then
    RESULT_LINE="**Result: 🔴 FAIL** — Performance regression detected."
else
    RESULT_LINE="**Result: ✅ PASS** — No regressions detected."
fi

cat > /tmp/bench-comment.md <<EOF
## Benchmark Results

> CPU-only, \`ubuntu-latest\` runner. Thresholds: TTFT P95 < +${TTFT_THRESHOLD}%, throughput > -${THRPT_THRESHOLD}%.

| Metric | Base | PR | Change | Gate |
|--------|------|----|--------|------|
| TTFT P50 | ${BASE_TTFT_P50}ms | ${CAND_TTFT_P50}ms | $(fmt_latency "$TTFT_P50_CHG") | — |
| TTFT P95 | ${BASE_TTFT_P95}ms | ${CAND_TTFT_P95}ms | $(fmt_latency "$TTFT_P95_CHG") | $(gate_icon "$TTFT_P95_CHG" "$TTFT_THRESHOLD" true) |
| Throughput | ${BASE_THRPT} t/s | ${CAND_THRPT} t/s | $(fmt_throughput "$THRPT_CHG") | $(gate_icon "$THRPT_CHG" "$THRPT_THRESHOLD" false) |
| Latency P50 | ${BASE_LAT_P50}ms | ${CAND_LAT_P50}ms | $(fmt_latency "$LAT_P50_CHG") | — |
| Latency P95 | ${BASE_LAT_P95}ms | ${CAND_LAT_P95}ms | $(fmt_latency "$LAT_P95_CHG") | — |

${RESULT_LINE}

<details>
<summary>Raw data</summary>

**Base:**
\`\`\`json
$(jq '.primary | {ttft_p50_ms, ttft_p95_ms, latency_p50_ms, latency_p95_ms, throughput_tokens_per_sec, requests_ok, requests_err}' "$BASELINE")
\`\`\`

**PR:**
\`\`\`json
$(jq '.primary | {ttft_p50_ms, ttft_p95_ms, latency_p50_ms, latency_p95_ms, throughput_tokens_per_sec, requests_ok, requests_err}' "$CANDIDATE")
\`\`\`

</details>
EOF

echo
if [ "$FAILED" -eq 1 ]; then
    echo "FAIL: Regression detected"
    printf "  %b" "$FAIL_REASONS"
    exit 1
else
    echo "PASS: No performance regressions detected."
fi
