#!/usr/bin/env bash
# vLLM on this iGPU — the fourth engine, measured on its own terms.
#
# WHY THIS IS A SEPARATE SCRIPT AND NOT AN ARM IN bench_engines.sh. Two variables move
# at once against the Vulkan trio and neither can be held still:
#
#   backend — vLLM has no Vulkan path. It runs on ROCm, the trio runs on Vulkan.
#   weights — vLLM does not consume the Q8_0 GGUF the trio shares. It gets the
#             safetensors repo at BF16, which is a different amount of arithmetic per
#             token and a different memory footprint.
#
# Putting its numbers in the trio's table would publish a backend-and-quantisation
# difference as if it were a serving-layer difference. What this run *can* answer is the
# question a user actually asks — what does the best-known serving stack do on this
# hardware — and that is worth its own section, clearly labelled.
#
# The workloads and the clients are identical to the trio's (bench_burst.py,
# bench_decode.py), so the shapes are comparable even where the absolute numbers are not.
#
# Equal tuning effort, applied here: vLLM's automatic prefix caching is its answer to
# the thing this benchmark is about, so it is enabled explicitly rather than left to a
# default that might change between versions. Its context and concurrency are set to
# the same 4096-per-sequence and 8 slots the trio got.
#
#   MODEL_DIR=~/.cache/ferrumox/vllm-models/Llama-3.2-1B-Instruct scripts/bench_vllm.sh
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$(mktemp -d -t bench-vllm-XXXX)}"
mkdir -p "$OUT"

IMAGE="${IMAGE:-rocm/vllm:latest}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to a local safetensors model directory}"
TAG="${TAG:-vllm-bench}"
# gfx1150 is not among the targets vLLM's ROCm build compiles kernels for; the override
# presents the iGPU as a discrete RDNA3 part. Verified working in the feasibility gate
# (scripts/try_vllm_rocm.sh). This belongs in the published configuration, not a
# footnote — a reader with this hardware needs it too.
OVERRIDE="${OVERRIDE:-11.0.0}"
PORT="${PORT:-8361}"
URL="http://127.0.0.1:$PORT"
CONT="fox-bench-vllm"
ROUNDS="${ROUNDS:-3}"
CONC="${CONC:-8}"
REPEATS="${REPEATS:-30}"
MAXTOK="${MAXTOK:-64}"
CTX_PER_SEQ="${CTX_PER_SEQ:-4096}"
DEC_CONC="${DEC_CONC:-4}"
DEC_MAXTOK="${DEC_MAXTOK:-128}"
# vLLM loads weights, compiles and captures graphs on every start, so it is minutes not
# seconds. The server is still restarted per round rather than reusing one: the cold
# burst measures a server with nothing cached, and /reset_prefix_cache would clear the
# block cache while leaving the compilation and allocator state warm — a different,
# friendlier "cold" than the one the other three engines were measured under.
READY_SECS="${READY_SECS:-900}"

[ -d "$MODEL_DIR" ] || { echo "no existe el directorio de modelo: $MODEL_DIR"; exit 1; }

stop_all() { docker rm -f "$CONT" >/dev/null 2>&1; sleep 3; }
trap 'stop_all; exit 130' INT TERM

start_vllm() {
  docker rm -f "$CONT" >/dev/null 2>&1
  docker run -d --name "$CONT" \
    --device=/dev/dri --device=/dev/kfd --group-add video \
    --ipc=host --shm-size=8g \
    -v "$MODEL_DIR:/model:ro" \
    -p "127.0.0.1:$PORT:8000" \
    -e HSA_OVERRIDE_GFX_VERSION="$OVERRIDE" \
    --entrypoint vllm "$IMAGE" serve /model \
      --served-model-name "$TAG" \
      --max-model-len "$CTX_PER_SEQ" \
      --max-num-seqs "$CONC" \
      --gpu-memory-utilization 0.55 \
      --enable-prefix-caching \
      --disable-log-requests \
      >/dev/null 2>&1 || return 1
  local i=0
  while [ "$i" -lt "$READY_SECS" ]; do
    curl -sf -m 2 "$URL/health" >/dev/null 2>&1 && return 0
    docker ps -q -f "name=$CONT" | grep -q . || { echo "  el contenedor murió"; return 1; }
    sleep 5; i=$((i + 5))
  done
  return 1
}

echo "=== vLLM, aparte del trío Vulkan ==="
echo "    imagen   $IMAGE"
echo "    modelo   $MODEL_DIR (safetensors, BF16 — NO es el Q8_0 del trío)"
echo "    override HSA_OVERRIDE_GFX_VERSION=$OVERRIDE"
echo "    $CONC clientes · ${CTX_PER_SEQ} ctx/secuencia · $ROUNDS rondas"
echo

rm -f "$OUT"/*.dat
for r in $(seq 1 "$ROUNDS"); do
  echo "ronda $r/$ROUNDS:"
  t0=$(date +%s)
  if ! start_vllm; then
    echo "  no arrancó — log completo:"
    docker logs "$CONT" 2>&1 | tail -25 | sed 's/^/    /'
    docker logs "$CONT" > "$OUT/server_vllm.log" 2>&1
    stop_all
    exit 1
  fi
  echo "  arranque: $(( $(date +%s) - t0 ))s"

  out=$(python3 "$S/bench_burst.py" "$URL" "$TAG" "$CONC" "$REPEATS" "$MAXTOK" 2>&1) || {
    echo "  el cliente de ráfaga falló"; echo "$out" | tail -5; }
  while read -r phase p50 p90 wall cached ptok; do
    [ -z "${p50:-}" ] && continue
    echo "$p50 $wall $cached" >> "$OUT/${phase}_vllm.dat"
    printf "  vllm %-5s TTFT p50 %6s ms  p90 %6s ms  wall %5ss  cached %6s  prompt %s tok\n" \
           "$phase" "$p50" "$p90" "$wall" "$cached" "$ptok"
  done <<< "$out"

  # The decode control runs against the same live server: after the burst its prefix
  # cache holds the shared preamble, but these prompts share nothing with it, so there
  # is nothing for it to hit. Restarting in between would only buy a longer run.
  dec=$(python3 "$S/bench_decode.py" "$URL" "$TAG" "$DEC_CONC" "$DEC_MAXTOK" 2>&1) || {
    echo "  el cliente de decode falló"; echo "$dec" | tail -5; }
  while read -r phase tps agg ctok; do
    [ -z "${tps:-}" ] && continue
    echo "$tps $agg $ctok" >> "$OUT/${phase}_vllm.dat"
    printf "  vllm %-5s decode p50 %6s tok/s  agregado %6s tok/s  salida %s tok\n" \
           "$phase" "$tps" "$agg" "$ctok"
  done <<< "$dec"

  # One probe per round: vLLM reports cached_tokens 0 in the burst while its warm TTFT
  # drops fourfold, which is the signature of "reused it, did not report it". Left
  # unchecked, that 0 would go into the table meaning the opposite.
  if [ "$r" = 1 ]; then
    python3 "$S/probe_cached_tokens.py" "$URL" "$TAG" "$REPEATS" 2>&1 | sed 's/^/  /'
  fi

  docker logs "$CONT" > "$OUT/server_vllm_r$r.log" 2>&1
  stop_all
done

echo
python3 - "$OUT" <<'PY'
import statistics, sys, os
d = sys.argv[1]
for phase, label, unit in (("cold", "ráfaga fría TTFT", "ms"),
                           ("warm", "ráfaga caliente TTFT", "ms"),
                           ("decode", "decode por petición", "tok/s")):
    p = f"{d}/{phase}_vllm.dat"
    if not os.path.exists(p):
        continue
    v = [[float(x) for x in l.split()] for l in open(p) if l.strip()]
    main = [r[0] for r in v]
    second = [r[1] for r in v]
    print(f"  {label:<22} {statistics.median(main):8.0f} {unit:<6} "
          f"rango [{min(main):.0f}, {max(main):.0f}]   "
          + (f"agregado {statistics.median(second):.0f} tok/s" if phase == "decode"
             else f"wall {statistics.median(second):.2f}s"))
PY
echo
echo "Estas cifras NO van en la misma columna que el trío Vulkan: distinto backend y"
echo "distinta cuantización. Van en su propia sección, con esta configuración impresa."
echo "datos y logs en $OUT"
