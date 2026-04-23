#!/usr/bin/env bash
# bench-publish.sh — reproducible benchmark via Docker; accumulates results
# into benches/results.json keyed by hardware configuration.
#
# Usage:
#   ./scripts/bench-publish.sh [MODEL] [CONCURRENCY] [REQUESTS] [--vllm] [--vllm-model HF_ID]
#
# Examples:
#   ./scripts/bench-publish.sh
#   ./scripts/bench-publish.sh llama3.2 8 100
#   ./scripts/bench-publish.sh llama3.2 4 50 --vllm --vllm-model meta-llama/Llama-3.2-3B-Instruct
#
# This script is Docker-only for reproducible published results.
# Results are accumulated in benches/results.json — new hardware/model
# combinations are appended; repeat runs are merged via running average.
#
# Requirements:
#   - docker compose v2
#   - fox-bench binary at ./target/release/fox-bench (or $BENCH_BIN)
#   - jq
#   - nvidia-smi (optional, for GPU detection)

set -euo pipefail

# ── Arg parsing ───────────────────────────────────────────────────────────────
MODEL="llama3.2"
CONCURRENCY="4"
REQUESTS="50"
VLLM_MODE=0
VLLM_MODEL=""

POSITIONAL=()
i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --vllm)       VLLM_MODE=1 ;;
        --vllm-model) i=$((i + 1)); VLLM_MODEL="${args[$i]}" ;;
        *)            POSITIONAL+=("$arg") ;;
    esac
    i=$((i + 1))
done

[[ ${#POSITIONAL[@]} -ge 1 ]] && MODEL="${POSITIONAL[0]}"
[[ ${#POSITIONAL[@]} -ge 2 ]] && CONCURRENCY="${POSITIONAL[1]}"
[[ ${#POSITIONAL[@]} -ge 3 ]] && REQUESTS="${POSITIONAL[2]}"

MAX_TOKENS=128
PROMPT="Write a short paragraph about the Rust programming language."

# ── Constants ─────────────────────────────────────────────────────────────────
FOX_URL="http://localhost:8080"
OLLAMA_URL="http://localhost:11434"
VLLM_URL="http://localhost:8000"
BENCH_BIN="${BENCH_BIN:-./target/release/fox-bench}"
RESULTS_FILE="benches/results.json"
COMPOSE_FILE="docker-compose.bench.yml"
VLLM_HF_MODEL="${VLLM_MODEL:-${VLLM_MODEL_ENV:-meta-llama/Llama-3.2-3B-Instruct}}"

# ── Helpers ───────────────────────────────────────────────────────────────────
die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

wait_http() {
    local url="$1" label="$2" retries="${3:-30}"
    for _ in $(seq 1 "$retries"); do
        curl -sf "$url" >/dev/null 2>&1 && return 0
        sleep 1
    done
    die "$label did not become ready at $url"
}

# ── Hardware detection ────────────────────────────────────────────────────────
detect_hardware() {
    HW_ARCH="$(uname -m)"
    HW_OS="$(uname -s)"
    HW_CPU="$(grep 'model name' /proc/cpuinfo 2>/dev/null | head -1 | cut -d: -f2 | xargs || uname -p)"

    HW_GPU=""
    HW_GPU_VRAM_MB=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        HW_GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        local vram_raw
        vram_raw="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
        HW_GPU_VRAM_MB="${vram_raw:-0}"
    fi

    if [[ -n "$HW_GPU" ]]; then
        HARDWARE_KEY="${HW_ARCH}/${HW_GPU}"
    else
        HARDWARE_KEY="${HW_ARCH}/${HW_CPU}"
    fi
}

# ── Pre-flight ────────────────────────────────────────────────────────────────
[[ -f "$BENCH_BIN" ]] || die "fox-bench not found at $BENCH_BIN — run: cargo build --release"
require jq
require docker
require curl
[[ -f "$COMPOSE_FILE" ]] || die "$COMPOSE_FILE not found — run from repo root"

mkdir -p benches
if [[ ! -f "$RESULTS_FILE" ]]; then
    echo '{"version":1,"entries":[]}' > "$RESULTS_FILE"
fi

detect_hardware

echo "=== bench-publish ==="
echo "  Model       : $MODEL"
echo "  Concurrency : $CONCURRENCY"
echo "  Requests    : $REQUESTS"
echo "  Max tokens  : $MAX_TOKENS"
echo "  Hardware    : $HARDWARE_KEY"
echo "  GPU VRAM    : ${HW_GPU_VRAM_MB} MB"
echo "  CPU         : $HW_CPU"
[[ $VLLM_MODE -eq 1 ]] && echo "  vLLM model  : $VLLM_HF_MODEL"
echo

# ── Helpers (Docker) ─────────────────────────────────────────────────────────
cleanup() {
    echo "Stopping Docker services..."
    docker compose -f "$COMPOSE_FILE" down 2>/dev/null || true
}
trap cleanup EXIT

run_bench_single() {
    local url="$1" model="$2"
    shift 2
    "$BENCH_BIN" \
        --url "$url" \
        --model "$model" \
        --concurrency "$CONCURRENCY" \
        --requests "$REQUESTS" \
        --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" \
        --output json \
        "$@"
}

# ── Phase 1: ferrumox CPU ────────────────────────────────────────────────────
echo "Starting ferrumox (CPU)..."
docker compose -f "$COMPOSE_FILE" up -d --build ferrumox

echo "Waiting for ferrumox..."
wait_http "$FOX_URL/health" "ferrumox" 120

echo "Pulling model in ferrumox..."
docker compose -f "$COMPOSE_FILE" exec ferrumox fox pull "$MODEL"

echo "Benchmarking ferrumox (CPU)..."
FERRUMOX_CPU_JSON="$(run_bench_single "$FOX_URL" "$MODEL")"

echo "Stopping ferrumox (CPU)..."
docker compose -f "$COMPOSE_FILE" down

# ── Phase 2: ferrumox GPU (if available) ─────────────────────────────────────
FERRUMOX_GPU_JSON=""
if [[ -n "$HW_GPU" ]]; then
    echo "Starting ferrumox (GPU)..."
    docker compose -f "$COMPOSE_FILE" up -d --build ferrumox-gpu

    echo "Waiting for ferrumox..."
    wait_http "$FOX_URL/health" "ferrumox-gpu" 120

    echo "Pulling model in ferrumox..."
    docker compose -f "$COMPOSE_FILE" exec ferrumox-gpu fox pull "$MODEL"

    echo "Benchmarking ferrumox (GPU)..."
    FERRUMOX_GPU_JSON="$(run_bench_single "$FOX_URL" "$MODEL")"

    echo "Stopping ferrumox (GPU)..."
    docker compose -f "$COMPOSE_FILE" down
fi

# ── Phase 3: ollama CPU ──────────────────────────────────────────────────────
OLLAMA_CPU_JSON=""
if [[ $VLLM_MODE -eq 0 ]]; then
    echo "Starting Ollama (CPU)..."
    docker compose -f "$COMPOSE_FILE" up -d ollama

    echo "Waiting for Ollama..."
    wait_http "$OLLAMA_URL/api/tags" "ollama" 60

    echo "Pulling model in Ollama..."
    docker compose -f "$COMPOSE_FILE" exec ollama ollama pull "$MODEL"

    echo "Benchmarking Ollama (CPU)..."
    OLLAMA_CPU_JSON="$(run_bench_single "$OLLAMA_URL" "$MODEL")"

    echo "Stopping Ollama (CPU)..."
    docker compose -f "$COMPOSE_FILE" down
fi

# ── Phase 4: ollama GPU (if available) ───────────────────────────────────────
OLLAMA_GPU_JSON=""
if [[ $VLLM_MODE -eq 0 && -n "$HW_GPU" ]]; then
    echo "Starting Ollama (GPU)..."
    docker compose -f "$COMPOSE_FILE" up -d ollama-gpu

    echo "Waiting for Ollama..."
    wait_http "$OLLAMA_URL/api/tags" "ollama-gpu" 60

    echo "Pulling model in Ollama..."
    docker compose -f "$COMPOSE_FILE" exec ollama-gpu ollama pull "$MODEL"

    echo "Benchmarking Ollama (GPU)..."
    OLLAMA_GPU_JSON="$(run_bench_single "$OLLAMA_URL" "$MODEL")"

    echo "Stopping Ollama (GPU)..."
    docker compose -f "$COMPOSE_FILE" down
fi

# ── Phase 5: vLLM (GPU only, if requested) ───────────────────────────────────
VLLM_JSON=""
if [[ $VLLM_MODE -eq 1 ]]; then
    echo "Starting vLLM..."
    VLLM_MODEL="$VLLM_HF_MODEL" docker compose -f "$COMPOSE_FILE" up -d vllm

    echo "Waiting for vLLM (model download may take a while)..."
    wait_http "$VLLM_URL/health" "vLLM" 300

    echo "Benchmarking vLLM..."
    VLLM_JSON="$(run_bench_single "$VLLM_URL" "$VLLM_HF_MODEL")"

    echo "Stopping vLLM..."
    docker compose -f "$COMPOSE_FILE" down
fi

# ── Build new entry ───────────────────────────────────────────────────────────
TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

HARDWARE_JSON="$(jq -n \
    --arg arch "$HW_ARCH" \
    --arg os "$HW_OS" \
    --arg gpu "$HW_GPU" \
    --argjson gpu_vram_mb "$HW_GPU_VRAM_MB" \
    --arg cpu "$HW_CPU" \
    '{arch: $arch, os: $os, gpu: $gpu, gpu_vram_mb: $gpu_vram_mb, cpu: $cpu}')"

CONFIG_JSON="$(jq -n \
    --argjson concurrency "$CONCURRENCY" \
    --argjson requests "$REQUESTS" \
    --argjson max_tokens "$MAX_TOKENS" \
    '{concurrency: $concurrency, requests: $requests, max_tokens: $max_tokens}')"

extract_metrics() {
    echo "$1" | jq '{
        ttft_p50_ms:    .primary.ttft_p50_ms,
        ttft_p95_ms:    .primary.ttft_p95_ms,
        latency_p50_ms: .primary.latency_p50_ms,
        latency_p95_ms: .primary.latency_p95_ms,
        latency_p99_ms: .primary.latency_p99_ms,
        throughput_tps:  .primary.throughput_tokens_per_sec
    }'
}

maybe_metrics() {
    if [[ -n "$1" ]]; then extract_metrics "$1"; else echo "null"; fi
}

FERRUMOX_CPU_METRICS="$(extract_metrics "$FERRUMOX_CPU_JSON")"
FERRUMOX_GPU_METRICS="$(maybe_metrics "$FERRUMOX_GPU_JSON")"
OLLAMA_CPU_METRICS="$(maybe_metrics "$OLLAMA_CPU_JSON")"
OLLAMA_GPU_METRICS="$(maybe_metrics "$OLLAMA_GPU_JSON")"
VLLM_METRICS="$(maybe_metrics "$VLLM_JSON")"

NEW_ENTRY="$(jq -n \
    --arg hardware_key "$HARDWARE_KEY" \
    --argjson hardware "$HARDWARE_JSON" \
    --arg model "$MODEL" \
    --argjson config "$CONFIG_JSON" \
    --argjson ferrumox_cpu "$FERRUMOX_CPU_METRICS" \
    --argjson ferrumox_gpu "$FERRUMOX_GPU_METRICS" \
    --argjson ollama_cpu "$OLLAMA_CPU_METRICS" \
    --argjson ollama_gpu "$OLLAMA_GPU_METRICS" \
    --argjson vllm "$VLLM_METRICS" \
    --arg last_updated "$TIMESTAMP" \
    '{
        hardware_key: $hardware_key,
        hardware: $hardware,
        model: $model,
        config: $config,
        ferrumox_cpu: $ferrumox_cpu,
        runs: 1,
        last_updated: $last_updated
    }
    + (if $ferrumox_gpu != null then {ferrumox_gpu: $ferrumox_gpu} else {} end)
    + (if $ollama_cpu    != null then {ollama_cpu:   $ollama_cpu}   else {} end)
    + (if $ollama_gpu    != null then {ollama_gpu:   $ollama_gpu}   else {} end)
    + (if $vllm          != null then {vllm:         $vllm}         else {} end)')"

# ── Merge into results.json ──────────────────────────────────────────────────
TMP_FILE="${RESULTS_FILE}.tmp"

jq --argjson new "$NEW_ENTRY" '
def running_avg(old_val; old_runs; new_val):
    ((old_val * old_runs) + new_val) / (old_runs + 1);

def merge_metrics(old_m; new_m; old_runs):
    if old_m == null then new_m
    elif new_m == null then old_m
    else {
        ttft_p50_ms:    running_avg(old_m.ttft_p50_ms;    old_runs; new_m.ttft_p50_ms),
        ttft_p95_ms:    running_avg(old_m.ttft_p95_ms;    old_runs; new_m.ttft_p95_ms),
        latency_p50_ms: running_avg(old_m.latency_p50_ms; old_runs; new_m.latency_p50_ms),
        latency_p95_ms: running_avg(old_m.latency_p95_ms; old_runs; new_m.latency_p95_ms),
        latency_p99_ms: running_avg(old_m.latency_p99_ms; old_runs; new_m.latency_p99_ms),
        throughput_tps:  running_avg(old_m.throughput_tps;  old_runs; new_m.throughput_tps)
    } end;

(.entries | map(.hardware_key == $new.hardware_key and .model == $new.model) | index(true)) as $idx |

def merge_or_keep(old_field; new_field; old_runs):
    if new_field != null then merge_metrics(old_field; new_field; old_runs)
    elif old_field != null then old_field
    else null end;

if $idx != null then
    .entries[$idx] as $old |
    .entries[$idx] = ($old | {
        hardware_key: .hardware_key,
        hardware:     .hardware,
        model:        .model,
        config:       $new.config,
        ferrumox_cpu: merge_metrics(.ferrumox_cpu; $new.ferrumox_cpu; .runs),
        runs:         (.runs + 1),
        last_updated: $new.last_updated
    }
    + (merge_or_keep($old.ferrumox_gpu; $new.ferrumox_gpu; $old.runs) as $v |
       if $v != null then {ferrumox_gpu: $v} else {} end)
    + (merge_or_keep($old.ollama_cpu; $new.ollama_cpu; $old.runs) as $v |
       if $v != null then {ollama_cpu: $v} else {} end)
    + (merge_or_keep($old.ollama_gpu; $new.ollama_gpu; $old.runs) as $v |
       if $v != null then {ollama_gpu: $v} else {} end)
    + (merge_or_keep($old.vllm; $new.vllm; $old.runs) as $v |
       if $v != null then {vllm: $v} else {} end)
    )
else
    .entries += [$new]
end
' "$RESULTS_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$RESULTS_FILE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=== Results merged into $RESULTS_FILE ==="
echo "  Hardware key : $HARDWARE_KEY"
echo "  Model        : $MODEL"
ENTRY="$(jq --arg hw "$HARDWARE_KEY" --arg m "$MODEL" \
    '.entries[] | select(.hardware_key == $hw and .model == $m)' "$RESULTS_FILE")"
RUNS="$(echo "$ENTRY" | jq '.runs')"
print_tps() {
    local label="$1" field="$2"
    local val
    val="$(echo "$ENTRY" | jq -e ".${field}.throughput_tps | round" 2>/dev/null)" && \
        echo "  ${label} throughput (avg) : ${val} t/s"
}

echo "  Total runs   : $RUNS"
print_tps "ferrumox CPU" "ferrumox_cpu"
print_tps "ferrumox GPU" "ferrumox_gpu"
print_tps "ollama   CPU" "ollama_cpu"
print_tps "ollama   GPU" "ollama_gpu"
print_tps "vllm        " "vllm"
echo
echo "Done."
