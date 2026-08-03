#!/usr/bin/env bash
# fox vs llama-server on a concurrent burst behind a shared system prompt.
#
# WHAT THIS IS AND IS NOT: this benchmark is deliberately chosen to sit on fox's
# strength, and saying so up front is the only thing that makes the number worth
# anything. The neutral decode-bound workload (scripts/ab_bench.sh, short unrelated
# prompts) is run here too as a control, because a benchmark that only reports the
# favourable workload is marketing, not measurement.
#
# The hypothesis under test, from reading both servers:
#
#   llama-server's get_available_slot() skips slots where is_processing() is true, in
#   BOTH the prompt-similarity pass (server-context.cpp:1609) and the LRU fallback
#   (:1652). Its parent/child fork path asserts the same (:2303). So when N requests
#   sharing a system prompt arrive together, none of them can reuse a prefix from the
#   others — there is no idle sequence holding it yet — and the shared prompt is
#   prefilled N times.
#
#   fox copies from a *live* sequence: under kv_unified, seq_cp shares cells rather
#   than duplicating the buffer, so a request may copy from a sibling that is already
#   decoding. The shared prompt is prefilled once.
#
# That predicts a large cold-burst TTFT gap and near-parity once warm. Measured, 3
# rounds, 8 clients, 1856-token shared prompt, Radeon 890M / Vulkan, both servers from
# the same vendored llama.cpp:
#
#   COLD  TTFT p50   fox  1129 ms   llama-server  4550 ms   fox 4.0x   ranges disjoint
#         wall       fox  2.65 s    llama-server  8.82 s
#         cached_tokens  fox 12908 (= 7 x 1844: seven of eight copied)   ls 0
#   WARM  TTFT p50   fox    50 ms   llama-server   190 ms   fox 3.8x   ranges disjoint
#         cached_tokens  fox 14840   ls 14840  (both reuse; this is the fair floor)
#
# A NOTE ON HOW THIS WAS NEARLY GOT WRONG. The first run of this benchmark reported the
# opposite cold result — llama-server 1.40x ahead, fox cached_tokens 0 — and was written
# up as refuting the hypothesis. The fox binary under test was a prebuilt bundle from 31
# minutes before the feature was committed. The benchmark was correct; it was measuring a
# build that did not contain what was being measured. If an arm shows no effect at all,
# check the binary's timestamp against the commit before believing the result.
#
# Discipline is inherited from scripts/ab_bench.sh and is not optional: ggml's thread
# pool spin-waits, so an idle second server still burns cores and skews the arm under
# test. Exactly one server runs at a time; arms alternate each round so thermal drift
# cannot systematically favour one.
#
#   FOX_BIN=/path/to/fox LLAMA_SERVER_BIN=/path/to/llama-server \
#     MODEL=/path/to/model.gguf scripts/ab_shared_prefix.sh
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$(mktemp -d)}"
mkdir -p "$OUT"
MODEL="${MODEL:?set MODEL to a .gguf}"
NAME="${NAME:-$(basename "$MODEL" .gguf)}"
FOX_BIN="${FOX_BIN:?set FOX_BIN}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:?set LLAMA_SERVER_BIN}"
PORT="${PORT:-8360}"
URL="http://127.0.0.1:$PORT"
ROUNDS="${ROUNDS:-3}"
CONC="${CONC:-8}"
REPEATS="${REPEATS:-30}"   # ~1.8k-token shared system prompt (measured, see below)
MAXTOK="${MAXTOK:-64}"
# Per-sequence context, matched across both servers. llama-server splits n_ctx across
# --parallel slots, so its -c must be CTX_PER_SEQ * CONC to give each slot what fox
# gives each sequence. Sizing this wrong does not fail loudly in the same way on both:
# llama-server returns 400, while fox rolls the context window, which sets
# rolled_tokens and disables prompt reuse — so an undersized context would look like
# "fox cannot reuse prompts" when it actually means "the benchmark overflowed".
CTX_PER_SEQ="${CTX_PER_SEQ:-4096}"
CTX=$((CTX_PER_SEQ * CONC))

wait_up() { for _ in $(seq 1 90); do curl -sf -m 2 "$URL/health" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }
# Match by listening socket, never `pkill -f` — a pattern broad enough to catch the
# server also matches this script's own command line.
stop() { local p; p=$(ss -lptn "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$p" ] && kill "$p"; sleep 3; }

arm() {
  local label="$1"; shift
  stop
  "$@" > "$OUT/burst_$label.log" 2>&1 &
  disown
  wait_up || { echo "  $label: no arrancó (ver $OUT/burst_$label.log)"; stop; return 1; }
  local out
  out=$(python3 "$S/bench_burst.py" "$URL" "$NAME" "$CONC" "$REPEATS" "$MAXTOK" 2>&1) || {
    echo "  $label: el cliente falló"; echo "$out" | tail -3; stop; return 1; }
  # bench_burst.py añadió ITL p50/p99 al final de cada línea; hay que absorberlos o
  # `read` los mete en $ptok y la comprobación de tamaño de prompt compara basura.
  while read -r phase p50 p90 wall cached ptok itl50 itl99; do
    echo "$p50 $wall $cached" >> "$OUT/${phase}_$label.dat"
    printf "  %-13s %-5s TTFT p50 %6s ms  p90 %6s ms  wall %5ss  cached %6s  prompt %s tok\n" \
           "$label" "$phase" "$p50" "$p90" "$wall" "$cached" "$ptok"
    if (( ptok > CTX_PER_SEQ )); then
      echo "  AVISO: el prompt ($ptok) no cabe en el contexto por secuencia ($CTX_PER_SEQ) — sube CTX_PER_SEQ o baja REPEATS"
    fi
  done <<< "$out"
  stop
}

fox_arm() { arm fox env LD_LIBRARY_PATH="$(dirname "$FOX_BIN")" "$FOX_BIN" serve \
            --model-path "$MODEL" --host 127.0.0.1 --port "$PORT" \
            --max-context-len "$CTX_PER_SEQ" --max-batch-size "$CONC"; }
ls_arm()  { arm llama-server env LD_LIBRARY_PATH="$(dirname "$LLAMA_SERVER_BIN")" "$LLAMA_SERVER_BIN" \
            -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c "$CTX" -ngl 99 --parallel "$CONC"; }

rm -f "$OUT"/cold_*.dat "$OUT"/warm_*.dat
echo "=== ráfaga concurrente, prompt de sistema compartido ==="
echo "    $CONC clientes · $MAXTOK tokens de salida · ${CTX_PER_SEQ} ctx/secuencia · $ROUNDS rondas"
for r in $(seq 1 "$ROUNDS"); do
  echo "ronda $r/$ROUNDS:"
  if (( r % 2 == 1 )); then fox_arm; ls_arm; else ls_arm; fox_arm; fi
done

echo
python3 - "$OUT" <<'PY'
import statistics, sys
d = sys.argv[1]


def col(phase, label, idx):
    try:
        return [float(l.split()[idx]) for l in open(f"{d}/{phase}_{label}.dat") if l.strip()]
    except FileNotFoundError:
        return []


for phase in ("cold", "warm"):
    f, l = col(phase, "fox", 0), col(phase, "llama-server", 0)
    if not (f and l):
        continue
    mf, ml = statistics.median(f), statistics.median(l)
    print(f"  {phase.upper():<5} TTFT p50   fox {mf:8.0f} ms   llama-server {ml:8.0f} ms   "
          f"{'fox '+format(ml/mf,'.2f')+'x más rápido' if mf < ml else 'llama-server '+format(mf/ml,'.2f')+'x más rápido'}")
    print(f"        rangos     fox [{min(f):.0f}, {max(f):.0f}]   ls [{min(l):.0f}, {max(l):.0f}]"
          + ("   (disjuntos)" if min(f) > max(l) or min(l) > max(f) else "   SOLAPAN — no concluyente"))
    cf, cl = col(phase, "fox", 2), col(phase, "llama-server", 2)
    if cf and cl:
        print(f"        cached_tokens sumados   fox {statistics.median(cf):.0f}   ls {statistics.median(cl):.0f}")
PY
echo
echo "datos en $OUT"
