#!/usr/bin/env bash
set -euo pipefail

# Vision benchmark comparison: fox vs llama.cpp vs Ollama
# Runs each server sequentially (one at a time on GPU) with unique images.
#
# Prerequisites:
#   - fox binary at ./target/release/fox
#   - docker with ghcr.io/ggml-org/llama.cpp:server-cuda image pulled
#   - docker with ollama/ollama:latest image pulled + qnguyen3/nanollava pulled
#   - nanollava GGUF files in ~/.cache/ferrumox/models/
#   - bc, jq, python3, curl installed
#
# Usage: bench-vision-compare.sh [REQUESTS] [CONCURRENCY]

REQUESTS="${1:-100}"
CONCURRENCY="${2:-4}"
FOX_PORT=8080
LLAMACPP_PORT=8082
OLLAMA_PORT=11434
MODEL_DIR="${HOME}/.cache/ferrumox/models"
TEXT_MODEL="nanollava-text-model-f16.gguf"
MMPROJ="nanollava-mmproj-f16.gguf"

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}=== Vision Benchmark Comparison ===${NC}"
echo "Requests:    $REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Model:       nanollava (text + mmproj)"
echo ""

for f in "$MODEL_DIR/$TEXT_MODEL" "$MODEL_DIR/$MMPROJ"; do
    [ -f "$f" ] || { echo "ERROR: $f not found"; exit 1; }
done

TMPDIR=$(mktemp -d)
trap 'cleanup' EXIT

cleanup() {
    echo -e "\n${BLUE}Cleaning up...${NC}"
    docker rm -f llamacpp-vision-bench ollama-vision-bench 2>/dev/null || true
    rm -rf "$TMPDIR"
}

# Generate unique 32x32 PNGs
echo -n "Generating $REQUESTS unique images..."
python3 - "$REQUESTS" "$TMPDIR" << 'PYEOF'
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

for i in range(n):
    r = (i * 7 + 31) % 256
    g = (i * 13 + 97) % 256
    b = (i * 19 + 173) % 256
    png = make_png(r, g, b)
    b64 = base64.b64encode(png).decode()
    # Write two variants: one for fox/llama.cpp, one for Ollama (different model name)
    for model in ["nanollava-text-model-f16", "qnguyen3/nanollava"]:
        suffix = "ollama" if "/" in model else "default"
        with open(os.path.join(outdir, f'req_{suffix}_{i+1}.json'), 'w') as f:
            f.write('{"model":"' + model + '","messages":[{"role":"user","content":[{"type":"text","text":"What color is this image? Answer in one word."},{"type":"image_url","image_url":{"url":"data:image/png;base64,' + b64 + '"}}]}],"max_tokens":16}')
PYEOF
echo " done"

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------
run_bench() {
    local label="$1" url="$2" req_prefix="$3"
    local outdir="$TMPDIR/$label"
    mkdir -p "$outdir"

    # Wait for server
    echo -n "  Waiting for server..."
    for i in $(seq 1 60); do
        if curl -s "$url/health" >/dev/null 2>&1 || curl -s "$url/v1/models" >/dev/null 2>&1; then
            echo " ready"
            break
        fi
        if [ "$i" -eq 60 ]; then echo " TIMEOUT"; return 1; fi
        sleep 1
    done

    # Warmup
    echo -n "  Warming up..."
    curl -s --connect-timeout 5 --max-time 30 "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d @"$TMPDIR/req_${req_prefix}_1.json" -o /dev/null
    echo " done"

    echo "  Running $REQUESTS requests at concurrency $CONCURRENCY..."
    local START END ELAPSED_MS
    START=$(date +%s%N)

    local ACTIVE=0
    for i in $(seq 1 "$REQUESTS"); do
        curl -s --connect-timeout 5 --max-time 30 -o "$outdir/resp_$i.json" \
            -w "%{time_total}\n" \
            "$url/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d @"$TMPDIR/req_${req_prefix}_$i.json" \
            > "$outdir/time_$i.txt" 2>&1 &
        ACTIVE=$((ACTIVE + 1))
        if [ "$ACTIVE" -ge "$CONCURRENCY" ]; then
            wait -n 2>/dev/null || true
            ACTIVE=$((ACTIVE - 1))
        fi
    done
    wait

    # Check if server is still alive after benchmark
    if ! curl -s --connect-timeout 2 --max-time 5 "$url/health" >/dev/null 2>&1 \
       && ! curl -s --connect-timeout 2 --max-time 5 "$url/v1/models" >/dev/null 2>&1; then
        echo -e "  ${RED}WARNING: server appears to have crashed during benchmark${NC}"
    fi

    END=$(date +%s%N)
    ELAPSED_MS=$(( (END - START) / 1000000 ))

    # Collect results
    local TIMES=() ERRORS=0 TOTAL_TOKENS=0
    for i in $(seq 1 "$REQUESTS"); do
        if [ -f "$outdir/time_$i.txt" ]; then
            TIMES+=("$(head -1 "$outdir/time_$i.txt")")
        fi
        if [ -f "$outdir/resp_$i.json" ]; then
            grep -q '"error"' "$outdir/resp_$i.json" 2>/dev/null && ERRORS=$((ERRORS + 1))
            local TOK
            TOK=$(jq -r '.usage.completion_tokens // 0' "$outdir/resp_$i.json" 2>/dev/null || echo 0)
            TOTAL_TOKENS=$((TOTAL_TOKENS + TOK))
        fi
    done

    printf '%s\n' "${TIMES[@]}" | sort -n > "$outdir/sorted.txt"
    local COUNT=${#TIMES[@]}
    local p50_idx=$(( COUNT * 50 / 100 ))
    local p95_idx=$(( COUNT * 95 / 100 ))
    local p99_idx=$(( COUNT * 99 / 100 ))
    [ "$p50_idx" -ge "$COUNT" ] && p50_idx=$((COUNT - 1))
    [ "$p95_idx" -ge "$COUNT" ] && p95_idx=$((COUNT - 1))
    [ "$p99_idx" -ge "$COUNT" ] && p99_idx=$((COUNT - 1))

    local P50 P95 P99 MIN MAX RPS ELAPSED_S
    P50=$(sed -n "$((p50_idx + 1))p" "$outdir/sorted.txt")
    P95=$(sed -n "$((p95_idx + 1))p" "$outdir/sorted.txt")
    P99=$(sed -n "$((p99_idx + 1))p" "$outdir/sorted.txt")
    MIN=$(head -1 "$outdir/sorted.txt")
    MAX=$(tail -1 "$outdir/sorted.txt")
    ELAPSED_S=$(echo "scale=2; $ELAPSED_MS / 1000" | bc)
    RPS=$(echo "scale=2; $REQUESTS / ($ELAPSED_MS / 1000)" | bc)

    # Save summary
    echo "${ELAPSED_S}|${RPS}|${P50}|${P95}|${P99}|${MIN}|${MAX}|${ERRORS}|${TOTAL_TOKENS}" > "$outdir/summary.txt"
    echo "  Done: ${ELAPSED_S}s, ${RPS} req/s, P50=${P50}s, ${ERRORS} errors"
}

# ---------------------------------------------------------------------------
# 1. Fox
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[1/3] Fox${NC}"
FOX_BIN="./target/release/fox"
if [ ! -f "$FOX_BIN" ]; then
    echo "  ERROR: $FOX_BIN not found — build with: cargo build --release"
    FOX_SKIP=1
else
    FOX_SKIP=0
    $FOX_BIN serve --model-path "$MODEL_DIR/$TEXT_MODEL" \
        --port "$FOX_PORT" --max-context-len 2048 \
        --vision-contexts "$CONCURRENCY" > /dev/null 2>&1 &
    FOX_PID=$!
    sleep 5
    if run_bench "fox" "http://localhost:$FOX_PORT" "default"; then
        true
    fi
    kill "$FOX_PID" 2>/dev/null; wait "$FOX_PID" 2>/dev/null || true
    sleep 2
fi

# ---------------------------------------------------------------------------
# 2. llama.cpp server
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[2/3] llama.cpp server${NC}"
if docker image inspect ghcr.io/ggml-org/llama.cpp:server-cuda >/dev/null 2>&1; then
    LLAMACPP_SKIP=0
    docker run -d --gpus all --name llamacpp-vision-bench \
        -p "$LLAMACPP_PORT":8080 \
        -v "$MODEL_DIR":/models \
        ghcr.io/ggml-org/llama.cpp:server-cuda \
        --model "/models/$TEXT_MODEL" \
        --mmproj "/models/$MMPROJ" \
        --host 0.0.0.0 --port 8080 \
        --ctx-size 8192 --flash-attn on --n-gpu-layers 99 \
        --parallel "$CONCURRENCY" > /dev/null 2>&1
    sleep 8
    if run_bench "llamacpp" "http://localhost:$LLAMACPP_PORT" "default"; then
        true
    fi
    docker rm -f llamacpp-vision-bench > /dev/null 2>&1
    sleep 2
else
    echo "  SKIP: ghcr.io/ggml-org/llama.cpp:server-cuda not pulled"
    LLAMACPP_SKIP=1
fi

# ---------------------------------------------------------------------------
# 3. Ollama
# ---------------------------------------------------------------------------
echo -e "\n${BOLD}[3/3] Ollama${NC}"
if docker image inspect ollama/ollama:latest >/dev/null 2>&1; then
    OLLAMA_SKIP=0
    docker run -d --gpus all --name ollama-vision-bench \
        -p "$OLLAMA_PORT":11434 \
        -v "${HOME}/.ollama:/root/.ollama" \
        ollama/ollama:latest > /dev/null 2>&1
    sleep 5
    # Ensure model is available
    if ! docker exec ollama-vision-bench ollama list 2>/dev/null | grep -q nanollava; then
        echo "  Pulling qnguyen3/nanollava..."
        docker exec ollama-vision-bench ollama pull qnguyen3/nanollava > /dev/null 2>&1
    fi
    if run_bench "ollama" "http://localhost:$OLLAMA_PORT" "ollama"; then
        true
    fi
    docker rm -f ollama-vision-bench > /dev/null 2>&1
    sleep 2
else
    echo "  SKIP: ollama/ollama:latest not pulled"
    OLLAMA_SKIP=1
fi

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}=== Results ($REQUESTS requests, concurrency $CONCURRENCY) ===${NC}"
printf "%-16s %8s %8s %8s %8s %6s %6s\n" "Server" "Time(s)" "Req/s" "P50(s)" "P95(s)" "P99(s)" "Errs"
printf "%-16s %8s %8s %8s %8s %6s %6s\n" "────────────────" "────────" "────────" "────────" "────────" "──────" "──────"

for label in fox llamacpp ollama; do
    summary="$TMPDIR/$label/summary.txt"
    if [ -f "$summary" ]; then
        IFS='|' read -r elapsed rps p50 p95 p99 _min _max errs tokens < "$summary"
        printf "%-16s %8s %8s %8s %8s %6s %6s\n" "$label" "$elapsed" "$rps" "$p50" "$p95" "$errs" "$tokens"
    else
        printf "%-16s %8s\n" "$label" "SKIPPED"
    fi
done
echo ""
