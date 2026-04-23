#!/usr/bin/env bash
set -euo pipefail

# Vision batch benchmark: sends N identical-prompt requests with a test image.
# Usage: bench-vision.sh [URL] [REQUESTS] [CONCURRENCY] [IMAGE_B64_FILE]
#
# If IMAGE_B64_FILE is provided, its contents are used as the base64 image payload.
# Otherwise a 32x32 solid red PNG is used (tiny — for smoke tests only).

URL="${1:-http://localhost:8080}"
REQUESTS="${2:-50}"
CONCURRENCY="${3:-4}"
IMAGE_B64_FILE="${4:-}"
MODEL="nanollava-text-model-f16"
PROMPT="What color is this image? Answer in one word."

if [ -n "$IMAGE_B64_FILE" ] && [ -f "$IMAGE_B64_FILE" ]; then
    RED_PNG_B64=$(cat "$IMAGE_B64_FILE")
    IMG_SIZE=$(echo -n "$RED_PNG_B64" | base64 -d 2>/dev/null | wc -c)
    IMG_DESC="custom ($(echo "scale=2; $IMG_SIZE / 1024" | bc) KB)"
else
    # 32x32 solid red PNG, base64-encoded
    RED_PNG_B64="iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg=="
    IMG_DESC="32x32 red PNG (~150 B)"
fi

echo "=== Vision Batch Benchmark ==="
echo "URL:         $URL"
echo "Requests:    $REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Image:       $IMG_DESC"
echo "Model:       $MODEL"
echo ""

# Wait for server readiness
echo -n "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s "$URL/health" >/dev/null 2>&1; then
        echo " ready"
        break
    fi
    if [ "$i" -eq 60 ]; then
        echo " TIMEOUT"
        exit 1
    fi
    sleep 1
done

# Write request body to a temp file (avoids 2MB+ command-line args per curl)
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cat > "$TMPDIR/request.json" <<ENDJSON
{
    "model": "$MODEL",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "$PROMPT"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,$RED_PNG_B64"}}
        ]
    }],
    "max_tokens": 16
}
ENDJSON

# Warm up: single request to load model
echo -n "Warming up (loading model)..."
curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TMPDIR/request.json" -o /dev/null
echo " done"

echo ""
echo "Running $REQUESTS requests at concurrency $CONCURRENCY..."
echo ""

START=$(date +%s%N)

# Fire requests in parallel using background jobs with concurrency limit
ACTIVE=0
for i in $(seq 1 "$REQUESTS"); do
    curl -s -o "$TMPDIR/resp_$i.json" \
        -w "%{time_total}\n" \
        "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"$TMPDIR/request.json" \
        > "$TMPDIR/time_$i.txt" 2>&1 &

    ACTIVE=$((ACTIVE + 1))
    if [ "$ACTIVE" -ge "$CONCURRENCY" ]; then
        wait -n 2>/dev/null || true
        ACTIVE=$((ACTIVE - 1))
    fi
done
wait

END=$(date +%s%N)
ELAPSED_MS=$(( (END - START) / 1000000 ))

# Collect timing data
TIMES=()
ERRORS=0
TOTAL_TOKENS=0
for i in $(seq 1 "$REQUESTS"); do
    if [ -f "$TMPDIR/time_$i.txt" ]; then
        T=$(cat "$TMPDIR/time_$i.txt" | head -1)
        TIMES+=("$T")
    fi
    if [ -f "$TMPDIR/resp_$i.json" ]; then
        # Check for errors
        if grep -q '"error"' "$TMPDIR/resp_$i.json" 2>/dev/null; then
            ERRORS=$((ERRORS + 1))
        fi
        # Sum completion tokens
        TOK=$(jq -r '.usage.completion_tokens // 0' "$TMPDIR/resp_$i.json" 2>/dev/null || echo 0)
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOK))
    fi
done

# Sort times and compute percentiles
printf '%s\n' "${TIMES[@]}" | sort -n > "$TMPDIR/sorted_times.txt"
COUNT=${#TIMES[@]}

p50_idx=$(( COUNT * 50 / 100 ))
p95_idx=$(( COUNT * 95 / 100 ))
p99_idx=$(( COUNT * 99 / 100 ))
[ "$p50_idx" -ge "$COUNT" ] && p50_idx=$((COUNT - 1))
[ "$p95_idx" -ge "$COUNT" ] && p95_idx=$((COUNT - 1))
[ "$p99_idx" -ge "$COUNT" ] && p99_idx=$((COUNT - 1))

P50=$(sed -n "$((p50_idx + 1))p" "$TMPDIR/sorted_times.txt")
P95=$(sed -n "$((p95_idx + 1))p" "$TMPDIR/sorted_times.txt")
P99=$(sed -n "$((p99_idx + 1))p" "$TMPDIR/sorted_times.txt")
MIN=$(head -1 "$TMPDIR/sorted_times.txt")
MAX=$(tail -1 "$TMPDIR/sorted_times.txt")

ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
RPS=$(echo "scale=2; $REQUESTS / ($ELAPSED_MS / 1000)" | bc)
TPS=$(echo "scale=2; $TOTAL_TOKENS / ($ELAPSED_MS / 1000)" | bc)

echo "=== Results ==="
echo "Total time:       ${ELAPSED_S}s"
echo "Requests:         $REQUESTS ($ERRORS errors)"
echo "Tokens generated: $TOTAL_TOKENS"
echo ""
echo "Latency:"
echo "  Min:  ${MIN}s"
echo "  P50:  ${P50}s"
echo "  P95:  ${P95}s"
echo "  P99:  ${P99}s"
echo "  Max:  ${MAX}s"
echo ""
echo "Throughput:"
echo "  Requests/sec:   $RPS"
echo "  Tokens/sec:     $TPS"
echo ""

# Save first response for sanity check
echo "Sample response:"
jq -r '.choices[0].message.content // "N/A"' "$TMPDIR/resp_1.json" 2>/dev/null || echo "N/A"
