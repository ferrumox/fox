#!/usr/bin/env bash
# repeat_bench.sh — statistically sound repeated benchmark of two ALREADY-RUNNING
# OpenAI-compatible servers, addressing the noise seen when hand-running fox-bench
# once: single runs on this hardware varied ~50% run-to-run (throttling, shared
# iGPU memory contention, page-cache warmth), and a run with partial request
# failures still produced a throughput number computed on the smaller surviving
# sample, silently understating how unreliable that number was.
#
# What this adds over a bare `fox-bench --compare-url ...` invocation:
#   - N repetitions per engine, not one
#   - a discarded warmup request against each engine before timing starts
#   - each engine benchmarked in ISOLATION (one at a time), alternating which
#     engine goes first each round, to cancel thermal/cache ordering bias
#   - any repetition with requests_err > 0 is retried once, then dropped (not
#     averaged in) if it fails again, with a loud warning either way
#   - reports median + [min, max] across valid repetitions, not a single number
#
# This script does NOT start/stop your servers — point it at two URLs you
# already have serving (Docker containers, native processes, whatever). That
# keeps it usable regardless of which of fox's several benchmarking setups
# (native, Dockerfile.vulkan, Dockerfile.rocm, docker-compose.bench.yml) you're
# currently running.
#
# Usage:
#   ./scripts/repeat_bench.sh \
#     --url1 http://localhost:8080 --label1 fox --model1 llama-3.2-1b-instruct-q8_0 \
#     --url2 http://localhost:11434 --label2 ollama --model2 bench-llama32-1b \
#     [--repeats 5] [--concurrency 4] [--requests 40] [--max-tokens 256] \
#     [--prompt "..."] [--warmup-requests 2]
#
# Single-engine mode (just --url1/--label1/--model1) reports repeated stats for
# one engine, no comparison.
#
# Requires: fox-bench built at ./target/release/fox-bench (or $BENCH_BIN), jq.

set -euo pipefail

# Force the C locale for numeric parsing/formatting: bash's printf %f is
# locale-aware, and a comma-decimal locale (e.g. es_AR) makes it choke on the
# period-decimal numbers jq emits from JSON — independent of what language
# the surrounding messages are in.
export LC_NUMERIC=C

BENCH_BIN="${BENCH_BIN:-./target/release/fox-bench}"

URL1="" LABEL1="" MODEL1=""
URL2="" LABEL2="" MODEL2=""
REPEATS=5
CONCURRENCY=4
REQUESTS=40
MAX_TOKENS=256
PROMPT="Write a short paragraph about the Rust programming language."
WARMUP_REQUESTS=2

die() { echo "ERROR: $*" >&2; exit 1; }
require() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found in PATH"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --url1) URL1="$2"; shift 2 ;;
        --label1) LABEL1="$2"; shift 2 ;;
        --model1) MODEL1="$2"; shift 2 ;;
        --url2) URL2="$2"; shift 2 ;;
        --label2) LABEL2="$2"; shift 2 ;;
        --model2) MODEL2="$2"; shift 2 ;;
        --repeats) REPEATS="$2"; shift 2 ;;
        --concurrency) CONCURRENCY="$2"; shift 2 ;;
        --requests) REQUESTS="$2"; shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --warmup-requests) WARMUP_REQUESTS="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

[[ -n "$URL1" && -n "$LABEL1" && -n "$MODEL1" ]] || die "--url1/--label1/--model1 are required"
[[ -f "$BENCH_BIN" ]] || die "fox-bench not found at $BENCH_BIN — run: cargo build --release --bin fox-bench"
require jq
require curl

TWO_ENGINES=0
if [[ -n "$URL2" ]]; then
    [[ -n "$LABEL2" && -n "$MODEL2" ]] || die "--url2 given but --label2/--model2 missing"
    TWO_ENGINES=1
fi

echo "=== repeat_bench.sh ==="
echo "  Engine 1    : $LABEL1 ($URL1, model=$MODEL1)"
[[ $TWO_ENGINES -eq 1 ]] && echo "  Engine 2    : $LABEL2 ($URL2, model=$MODEL2)"
echo "  Repeats     : $REPEATS (alternating which engine runs first each round)"
echo "  Concurrency : $CONCURRENCY  Requests: $REQUESTS  Max tokens: $MAX_TOKENS"
echo "  Warmup      : $WARMUP_REQUESTS discarded request(s) per engine before timing"
echo

# ── Warmup: a small discarded request per engine, so the first *measured* run
# isn't paying for cold model/page-cache/clock-ramp effects the others don't.
warmup() {
    local url="$1" model="$2"
    curl -s -X POST "$url/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$WARMUP_REQUESTS}" \
        >/dev/null 2>&1 || echo "  WARNING: warmup request to $url failed — is it up?" >&2
}
echo "Warming up..."
warmup "$URL1" "$MODEL1"
[[ $TWO_ENGINES -eq 1 ]] && warmup "$URL2" "$MODEL2"
echo

# ── One measured repetition against one engine. Echoes throughput on success
# (via stdout), returns non-zero if the run itself failed outright or came
# back with any request errors after one retry.
run_once() {
    local url="$1" label="$2" model="$3"
    local json err ok thrpt
    json="$("$BENCH_BIN" --url "$url" --label "$label" --model "$model" \
        --concurrency "$CONCURRENCY" --requests "$REQUESTS" --max-tokens "$MAX_TOKENS" \
        --prompt "$PROMPT" --output json 2>/dev/null)" || return 1
    err="$(echo "$json" | jq '.primary.requests_err')"
    ok="$(echo "$json" | jq '.primary.requests_ok')"
    thrpt="$(echo "$json" | jq '.primary.throughput_tokens_per_sec')"
    if [[ "$err" -gt 0 ]]; then
        echo "  [$label] $ok ok / $err err — degraded run, discarding" >&2
        return 1
    fi
    echo "$thrpt"
}

# Runs once; on error/err>0, retries exactly once before giving up on this
# repetition entirely (loud either way, never silently averaged in).
run_with_retry() {
    local url="$1" label="$2" model="$3" thrpt
    if thrpt="$(run_once "$url" "$label" "$model")"; then
        echo "$thrpt"
        return 0
    fi
    echo "  [$label] retrying once..." >&2
    if thrpt="$(run_once "$url" "$label" "$model")"; then
        echo "$thrpt"
        return 0
    fi
    echo "  [$label] failed twice — dropping this repetition" >&2
    return 1
}

RESULTS_1=()
RESULTS_2=()

# Runs one engine's repetition, appends to the given array (by name) on
# success. Never lets a dropped repetition abort the script under `set -e` —
# the run/retry failure path is fully handled here, not left to a bare `&&`.
bench_one() {
    local url="$1" label="$2" model="$3" array_name="$4"
    local thrpt=""
    if thrpt="$(run_with_retry "$url" "$label" "$model")"; then
        eval "$array_name+=(\"\$thrpt\")"
        echo "  [$label] $thrpt t/s"
    else
        echo "  [$label] <dropped>"
    fi
}

for i in $(seq 1 "$REPEATS"); do
    echo "--- Round $i/$REPEATS ---"
    if [[ $TWO_ENGINES -eq 1 && $((i % 2)) -eq 0 ]]; then
        # Even rounds: engine 2 first, to cancel ordering bias.
        bench_one "$URL2" "$LABEL2" "$MODEL2" RESULTS_2
        bench_one "$URL1" "$LABEL1" "$MODEL1" RESULTS_1
    else
        bench_one "$URL1" "$LABEL1" "$MODEL1" RESULTS_1
        if [[ $TWO_ENGINES -eq 1 ]]; then
            bench_one "$URL2" "$LABEL2" "$MODEL2" RESULTS_2
        fi
    fi
done
echo

# ── Aggregate: median + [min, max] over whatever repetitions survived.
summarize() {
    local label="$1"; shift
    local values=("$@")
    local n=${#values[@]}
    if [[ $n -eq 0 ]]; then
        echo "  $label: NO VALID RUNS (every repetition errored — investigate before trusting anything)"
        return
    fi
    local sorted
    sorted=$(printf '%s\n' "${values[@]}" | sort -n)
    local median min max
    min=$(echo "$sorted" | head -1)
    max=$(echo "$sorted" | tail -1)
    median=$(echo "$sorted" | awk '{a[NR]=$1} END {if (NR%2==1) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}')
    local flag=""
    [[ $n -lt $((REPEATS / 2 + 1)) ]] && flag="  ⚠ only $n/$REPEATS repetitions were valid"
    printf "  %-10s median=%.1f t/s  range=[%.1f, %.1f]  n=%d%s\n" "$label" "$median" "$min" "$max" "$n" "$flag"
}

echo "=== Summary ==="
summarize "$LABEL1" "${RESULTS_1[@]:-}"
[[ $TWO_ENGINES -eq 1 ]] && summarize "$LABEL2" "${RESULTS_2[@]:-}"
