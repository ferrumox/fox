#!/usr/bin/env bash
set -euo pipefail

# Vision benchmark with unique images per request (no CLIP cache hits).
# Generates a unique 32x32 PNG for each request by varying pixel color.
# Usage: bench-vision-unique.sh [URL] [REQUESTS] [CONCURRENCY]

URL="${1:-http://localhost:8080}"
REQUESTS="${2:-100}"
CONCURRENCY="${3:-4}"
MODEL="nanollava-text-model-f16"
PROMPT="What color is this image? Answer in one word."

echo "=== Vision Benchmark (unique images) ==="
echo "URL:         $URL"
echo "Requests:    $REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Model:       $MODEL"
echo ""

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

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Generate unique 32x32 PNGs via python — each has a different solid color.
echo -n "Generating $REQUESTS unique images..."
python3 - "$REQUESTS" "$TMPDIR" "$MODEL" "$PROMPT" << 'PYEOF'
import struct, zlib, base64, os, sys

def make_png(r, g, b):
    width, height = 32, 32
    raw = b''
    for _ in range(height):
        raw += b'\x00' + bytes([r, g, b]) * width
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')

n = int(sys.argv[1])
outdir = sys.argv[2]
model = sys.argv[3]
prompt = sys.argv[4]

for i in range(n):
    r = (i * 7 + 31) % 256
    g = (i * 13 + 97) % 256
    b = (i * 19 + 173) % 256
    png = make_png(r, g, b)
    b64 = base64.b64encode(png).decode()
    with open(os.path.join(outdir, f'req_{i+1}.json'), 'w') as f:
        f.write('{"model":"' + model + '","messages":[{"role":"user","content":[{"type":"text","text":"' + prompt + '"},{"type":"image_url","image_url":{"url":"data:image/png;base64,' + b64 + '"}}]}],"max_tokens":16}')
PYEOF
echo " done"

# Warm up with a single request
echo -n "Warming up (loading model)..."
curl -s --connect-timeout 5 --max-time 30 "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$TMPDIR/req_1.json" -o /dev/null
echo " done"

echo ""
echo "Running $REQUESTS requests at concurrency $CONCURRENCY..."
echo ""

START=$(date +%s%N)

ACTIVE=0
for i in $(seq 1 "$REQUESTS"); do
    curl -s --connect-timeout 5 --max-time 30 -o "$TMPDIR/resp_$i.json" \
        -w "%{time_total}\n" \
        "$URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"$TMPDIR/req_$i.json" \
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

TIMES=()
ERRORS=0
TOTAL_TOKENS=0
for i in $(seq 1 "$REQUESTS"); do
    if [ -f "$TMPDIR/time_$i.txt" ]; then
        T=$(cat "$TMPDIR/time_$i.txt" | head -1)
        TIMES+=("$T")
    fi
    if [ -f "$TMPDIR/resp_$i.json" ]; then
        if grep -q '"error"' "$TMPDIR/resp_$i.json" 2>/dev/null; then
            ERRORS=$((ERRORS + 1))
        fi
        TOK=$(jq -r '.usage.completion_tokens // 0' "$TMPDIR/resp_$i.json" 2>/dev/null || echo 0)
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOK))
    fi
done

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

echo "Sample response:"
jq -r '.choices[0].message.content // "N/A"' "$TMPDIR/resp_1.json" 2>/dev/null || echo "N/A"
