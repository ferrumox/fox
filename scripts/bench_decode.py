#!/usr/bin/env python3
"""Decode-bound throughput — the neutral control for scripts/bench_engines.sh.

The burst benchmark sits deliberately on fox's strength: a long shared prefix is
exactly what its prefix cache exists for. A table containing only that workload is
marketing. This one is built to have nothing to reuse — N clients, N *different* short
prompts, no shared preamble — so almost all the time is spent decoding and the number
measures the sampling and batching path rather than prefill avoidance.

Two metrics, because they answer different questions and can disagree:

  per-request decode rate — tokens after the first, divided by the time between first
      token and last. Excludes prefill entirely, so a server that prefills slowly but
      decodes fast is not punished twice for the same thing.

  aggregate throughput — all completion tokens over the wall clock of the burst. This
      is what a deployment feels, and it does include prefill and scheduling overhead.

Measured completion tokens are reported, never assumed: with temperature > 0 a model
may stop early, and averaging over "max_tokens" when it actually produced half that
silently inflates the rate. If the token counts differ much between engines, the
comparison is of different amounts of work and the run should be repeated with prompts
that keep everyone generating.

Usage: bench_decode.py URL MODEL [CONCURRENCY] [MAX_TOKENS]
Prints: "decode <per_req_tps_p50> <aggregate_tps> <completion_tokens_median> <itl_p99_ms>"
"""
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MAXTOK = int(sys.argv[4]) if len(sys.argv) > 4 else 128

# Unrelated on purpose, and each asks for a long enumerated answer so the model runs to
# the token limit instead of stopping early and turning the measurement into a race to
# emit an EOS.
PROMPTS = [
    "List twelve distinct uses for a paperclip, one per line, with a short reason each.",
    "Describe the water cycle in twelve numbered steps, one sentence per step.",
    "Name twelve common cooking mistakes and how to avoid each, one per line.",
    "Explain twelve differences between a bicycle and a motorcycle, one per line.",
    "List twelve stages of building a wooden chair, one sentence each.",
    "Give twelve tips for keeping houseplants alive, one line each with a reason.",
    "Describe twelve steps to change a car tyre safely, one per line.",
    "List twelve landmarks of medieval architecture and what makes each distinctive.",
    "Name twelve ways to reduce household energy use, one per line with the saving.",
    "Explain twelve rules of basic chess strategy, one sentence each.",
    "List twelve steps to prepare a garden bed for planting, one per line.",
    "Give twelve techniques for remembering names, one line each.",
    "Describe twelve phases of a thunderstorm, one sentence per phase.",
    "List twelve tools every home should own and what each is for.",
    "Name twelve ways to make a small room feel larger, one per line.",
    "Explain twelve steps of making bread by hand, one sentence each.",
]


def one(i):
    # The index goes FIRST so that two clients never share a prefix. With 16 base
    # prompts and `i % 16`, concurrency 32 handed two clients byte-identical prompts —
    # and an identical prompt is precisely what fox's prefix cache is built to reuse,
    # so the neutral control quietly turned into the favourable workload at exactly the
    # concurrencies where the sweep was making its strongest claims. Leading with the
    # index costs one token of shared prefix instead of the whole prompt.
    prompt = f"[{i}] {PROMPTS[i % len(PROMPTS)]}"
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAXTOK,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    ctok = 0
    chunks = 0
    stamps = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = d.get("choices") or [{}]
            if ch and ch[0].get("delta", {}).get("content"):
                chunks += 1
                stamps.append(time.perf_counter())
                if ttft is None:
                    ttft = time.perf_counter() - t0
            usage = d.get("usage")
            if usage:
                ctok = usage.get("completion_tokens", 0) or 0
    end = time.perf_counter()
    # Not every engine returns usage on a streamed request; fall back to counting
    # content chunks, which is one token per chunk for all three servers here.
    if not ctok:
        ctok = chunks
    decode_s = end - t0 - (ttft or 0)
    # Guard the degenerate cases rather than dividing by ~0 and reporting a fantasy.
    tps = (ctok - 1) / decode_s if ctok > 1 and decode_s > 1e-6 else 0.0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return tps, ctok, gaps


t_start = time.perf_counter()
with ThreadPoolExecutor(max_workers=CONC) as ex:
    out = list(ex.map(one, range(CONC)))
wall = time.perf_counter() - t_start

rates = sorted(t for t, _, _ in out)
toks = [c for _, c, _ in out]
# The p99 gap is the stall a user sees, and it is invisible in a tokens/s average.
gaps = sorted(g for _, _, gg in out for g in gg)
itl99 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] * 1000 if gaps else 0
print(f"decode {statistics.median(rates):.1f} {sum(toks)/wall:.1f} "
      f"{statistics.median(toks):.0f} {itl99:.1f}")
