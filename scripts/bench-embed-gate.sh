#!/usr/bin/env bash
# bench-embed-gate.sh — compare two embedding benchmark JSON outputs, fail on regression.
#
# Usage: bench-embed-gate.sh <baseline.json> <candidate.json>
#
# The input JSON has the structure produced by bench-embed-run.sh:
#   { "short": { "avg_ms": N, "p50_ms": N, "p95_ms": N, "p99_ms": N, "rps": N },
#     "medium": { ... }, "long": { ... }, "batch": { ... } }
#
# Thresholds:
#   - Latency P95 regression > 10%  → FAIL  (any text size)
#   - RPS drop               > 10%  → FAIL  (any text size)

set -euo pipefail

BASELINE="${1:?Usage: bench-embed-gate.sh <baseline.json> <candidate.json>}"
CANDIDATE="${2:?Usage: bench-embed-gate.sh <baseline.json> <candidate.json>}"

THRESHOLD=10

for f in "$BASELINE" "$CANDIDATE"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
    jq '.short' "$f" >/dev/null 2>&1 || { echo "ERROR: $f is not valid embed bench JSON"; exit 1; }
done

pct_change() {
    local base="$1" cand="$2"
    if [ "$base" = "0" ] || [ "$base" = "0.0" ] || [ "$base" = "null" ]; then
        echo "0.0"
        return
    fi
    jq -n "($cand - $base) / $base * 100 | . * 100 | round / 100"
}

FAILED=0
FAIL_REASONS=""

declare -a SIZES=("short" "medium" "long" "batch")

echo "════════════════════════════════════════════════════════"
echo "  Embedding Benchmark Gate"
echo "════════════════════════════════════════════════════════"
printf "  %-10s %10s %10s %10s %10s\n" "Size" "Base P95" "PR P95" "Change" "Gate"
echo "  ──────────────────────────────────────────────────────"

for size in "${SIZES[@]}"; do
    BASE_P95=$(jq -r ".${size}.p95_ms" "$BASELINE")
    CAND_P95=$(jq -r ".${size}.p95_ms" "$CANDIDATE")
    BASE_RPS=$(jq -r ".${size}.rps" "$BASELINE")
    CAND_RPS=$(jq -r ".${size}.rps" "$CANDIDATE")

    P95_CHG=$(pct_change "$BASE_P95" "$CAND_P95")
    RPS_CHG=$(pct_change "$BASE_RPS" "$CAND_RPS")

    P95_REGRESSED=$(jq -n "$P95_CHG > $THRESHOLD")
    RPS_REGRESSED=$(jq -n "(-1 * $RPS_CHG) > $THRESHOLD")

    GATE="PASS"
    if [ "$P95_REGRESSED" = "true" ]; then
        GATE="FAIL"
        FAILED=1
        FAIL_REASONS="${FAIL_REASONS}${size} P95 latency regressed by ${P95_CHG}% (threshold: ${THRESHOLD}%)\n"
    fi
    if [ "$RPS_REGRESSED" = "true" ]; then
        GATE="FAIL"
        FAILED=1
        FAIL_REASONS="${FAIL_REASONS}${size} RPS dropped by ${RPS_CHG}% (threshold: -${THRESHOLD}%)\n"
    fi

    printf "  %-10s %8sms %8sms %9s%% %10s\n" "$size" "$BASE_P95" "$CAND_P95" "$P95_CHG" "$GATE"
done

echo "════════════════════════════════════════════════════════"

# Generate PR comment
if [ "$FAILED" -eq 1 ]; then
    RESULT_LINE="**Result: 🔴 FAIL** — Embedding performance regression detected."
else
    RESULT_LINE="**Result: ✅ PASS** — No embedding regressions detected."
fi

{
    echo "## Embedding Benchmark Results"
    echo ""
    echo "> CPU-only, \`ubuntu-latest\` runner. Threshold: P95 latency and RPS < ±${THRESHOLD}%."
    echo ""
    echo "| Size | Base P95 | PR P95 | Change | Base RPS | PR RPS | Change |"
    echo "|------|----------|--------|--------|----------|--------|--------|"
    for size in "${SIZES[@]}"; do
        BASE_P95=$(jq -r ".${size}.p95_ms" "$BASELINE")
        CAND_P95=$(jq -r ".${size}.p95_ms" "$CANDIDATE")
        BASE_RPS=$(jq -r ".${size}.rps" "$BASELINE")
        CAND_RPS=$(jq -r ".${size}.rps" "$CANDIDATE")
        P95_CHG=$(pct_change "$BASE_P95" "$CAND_P95")
        RPS_CHG=$(pct_change "$BASE_RPS" "$CAND_RPS")
        echo "| ${size} | ${BASE_P95}ms | ${CAND_P95}ms | ${P95_CHG}% | ${BASE_RPS} | ${CAND_RPS} | ${RPS_CHG}% |"
    done
    echo ""
    echo "${RESULT_LINE}"
} > /tmp/bench-embed-comment.md

echo
if [ "$FAILED" -eq 1 ]; then
    echo "FAIL: Embedding regression detected"
    printf "  %b" "$FAIL_REASONS"
    exit 1
else
    echo "PASS: No embedding performance regressions detected."
fi
