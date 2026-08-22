#!/usr/bin/env bash
# Is this machine in a state where a benchmark result means anything?
#
# WHY THIS EXISTS. On 2026-08-15 two separate optimisations were measured, written up as
# wins, and then withdrawn — a 2x from `FOX_KV_UNIFIED=0` and a 2x gap against
# llama-server under concurrency. Neither was real. This laptop's CPU had drifted from
# 2963 MHz to 1235 MHz over a long session, and the "improvement" was whichever arm
# happened to run while the clock was high. The reproducibility check that caught it
# showed fox's cold TTFT stable to 0.4% (1.01x over 5 identical runs) while wall time
# swung 1.55x — same binary, same workload, same everything.
#
# So: a number from this machine is only worth publishing alongside the state it was
# taken in. Call this before a benchmark, and record `sample_line` with every sample.
#
#   scripts/check_bench_env.sh            # report + exit 1 if unfit
#   scripts/check_bench_env.sh --sample   # one compact line, for per-sample logging
#   scripts/check_bench_env.sh --quiet    # exit code only
#
# Exit codes: 0 fit to measure · 1 something would distort the result.
set -uo pipefail

# Same reason ab_bench.sh does it: under a comma-decimal locale printf emits "107,8" and
# every awk comparison downstream silently reads it as 107 — or as 0. The first run of
# this script warned about "only 107,8G available" on a machine with 107.8 GB free.
export LC_ALL=C

MODE="${1:-}"

cpu_governor() { cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "?"; }
cpu_mhz()      { awk '/cpu MHz/{s+=$4; n++} END{if(n) printf "%.0f", s/n; else print "0"}' /proc/cpuinfo; }

# The frequency this machine can actually reach *right now*, not the one it idles at.
#
# Reading /proc/cpuinfo on an idle laptop under `powersave` reports ~1400 MHz on a part
# that runs at 5158, which says nothing about whether a benchmark will be throttled — it
# only says nobody is asking for cycles. So ask for some: a brief all-core spin, sample
# during it, then stop. Costs ~0.4s and is the difference between a useful warning and a
# false alarm on every idle machine.
cpu_mhz_under_load() {
  local pids=() i
  for ((i = 0; i < $(nproc); i++)); do
    ( while :; do :; done ) & pids+=($!)
  done
  sleep 0.4
  local m; m=$(cpu_mhz)
  kill "${pids[@]}" 2>/dev/null
  wait "${pids[@]}" 2>/dev/null
  echo "$m"
}
cpu_max_mhz()  { awk '{printf "%.0f", $1/1000}' /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0; }
gpu_temp()     { for f in /sys/class/drm/card*/device/hwmon/hwmon*/temp1_input; do
                   [ -r "$f" ] && { awk '{printf "%.0f", $1/1000}' "$f"; return; }; done; echo "?"; }
mem_avail_g()  { awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo; }

# Anything else on the GPU or eating cores makes the measurement about that instead.
competitors() {
  local list=""
  # -x (match the executable name) not -f (match the whole command line): `pgrep -f`
  # also matches the caller, because the caller's own command line contains the pattern.
  # Observed repeatedly on 2026-08-16 — this function reported "fox-serve" and "ollama"
  # as competitors when nothing at all was running, purely because the invoking shell
  # mentioned them. A gate that cries wolf is a gate that gets ignored.
  pgrep -x "fox"          >/dev/null 2>&1 && list="$list fox-serve"
  pgrep -x "llama-server" >/dev/null 2>&1 && list="$list llama-server"
  pgrep -x "ollama"       >/dev/null 2>&1 && list="$list ollama"
  pgrep -x "cargo|rustc|cc1plus|ninja|make" >/dev/null 2>&1 && list="$list build"
  docker ps -q 2>/dev/null | grep -q . && list="$list docker"
  echo "${list# }"
}

GOV=$(cpu_governor); MAXMHZ=$(cpu_max_mhz)
# --sample runs per measurement, so it must stay cheap: idle reading, no spin-up.
if [ "${1:-}" = "--sample" ]; then MHZ=$(cpu_mhz); else MHZ=$(cpu_mhz_under_load); fi
GPUT=$(gpu_temp); MEM=$(mem_avail_g); COMP=$(competitors)
RATIO=0
[ "$MAXMHZ" -gt 0 ] 2>/dev/null && RATIO=$(awk -v a="$MHZ" -v b="$MAXMHZ" 'BEGIN{printf "%.2f", a/b}')

if [ "$MODE" = "--sample" ]; then
  echo "gov=$GOV mhz=$MHZ ratio=$RATIO gpu=${GPUT}C mem=${MEM}G${COMP:+ busy=$COMP}"
  exit 0
fi

FIT=0
problems=()
# `powersave` on amd-pstate-epp is exactly what produced the drift described above.
[ "$GOV" != "performance" ] && { problems+=("gobernador '$GOV' (usa: sudo cpupower frequency-set -g performance)"); FIT=1; }
# Measured with all cores spinning, so this is the ceiling a benchmark would actually
# get — not an idle reading. Below ~45% the part is being held back and the run will be
# slower than the same run an hour earlier, for reasons that have nothing to do with fox.
awk -v r="$RATIO" 'BEGIN{exit !(r < 0.45)}' && { problems+=("bajo carga sólo alcanza $MHZ MHz de $MAXMHZ ($RATIO del techo) — limitada, deja enfriar"); FIT=1; }
[ -n "$COMP" ] && { problems+=("compitiendo por la máquina: $COMP"); FIT=1; }
awk -v m="$MEM" 'BEGIN{exit !(m < 20)}' && { problems+=("sólo ${MEM}G de RAM disponible"); FIT=1; }

if [ "$MODE" != "--quiet" ]; then
  echo "estado de la máquina: gobernador=$GOV · CPU bajo carga ${MHZ}/${MAXMHZ} MHz (${RATIO}) · GPU ${GPUT}C · RAM ${MEM}G"
  if [ "$FIT" -eq 0 ]; then
    echo "  apta para medir"
  else
    for p in "${problems[@]}"; do echo "  AVISO: $p"; done
    echo "  un resultado tomado así no es comparable con otro tomado en otro momento"
  fi
fi
exit "$FIT"
