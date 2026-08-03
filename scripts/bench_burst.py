#!/usr/bin/env python3
"""Concurrent burst behind a shared system prompt — the agent/RAG-shaped workload.

The throughput bench fox already had sends short, unrelated prompts, which measures
decode speed and almost nothing else. It cannot see prompt reuse, because there is no
prompt worth reusing. This one is built to expose exactly that: N clients arrive at
once, all carrying the same long system prompt, each with a different short question.

Two bursts are measured per run and the difference between them is the point:

  cold — the server has just started, nothing is cached, and all N requests arrive
         while no sequence holds the shared prefix yet. A server that can only inherit
         an *idle* sequence has nothing to inherit from and prefills the prompt N
         times. A server that can copy from a *live* one pays for it once.

  warm — the same burst again. Now the previous requests have finished, their
         sequences are idle and hold the prefix, so slot-affinity reuse applies to
         both designs. This is where the two should converge, and reporting it is what
         keeps the cold number honest rather than cherry-picked.

TTFT is the headline metric, not tokens/s: prefill is what is being avoided, and it
lands entirely on time-to-first-token.

Usage: bench_burst.py URL MODEL [CONCURRENCY] [SYS_REPEATS] [MAX_TOKENS]
Prints two lines: "cold <ttft_p50_ms> <ttft_p90_ms> <wall_s> <cached_total> <prompt_tokens>
<itl_p50_ms> <itl_p99_ms>" and the same for "warm".
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL = sys.argv[1], sys.argv[2]
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
MAXTOK = int(sys.argv[5]) if len(sys.argv) > 5 else 64

# A plausible agent preamble rather than filler: the shared prefix in real deployments
# is a system prompt plus tool definitions, which is what makes it both long and
# byte-identical across concurrent users.
SYSTEM = (
    "You are a meticulous senior software engineer performing code review. "
    "Always cite the file and line number for any claim you make about the code. "
    "If you are uncertain, say so explicitly rather than guessing. "
    "Prefer concrete failure scenarios over abstract concerns. "
    "Never suggest a change without stating what breaks if it is not made. "
) * REPEATS


def delta_text(chunk):
    """Text produced by this chunk, whichever field the server put it in.

    `llama-server` routes a reasoning model's output through its reasoning parser and
    streams it as `reasoning_content`, not `content`. A driver that only reads
    `content` sees an empty stream and reports the *total* request time as TTFT, with
    zero inter-token gaps — which is exactly what happened on Qwen3.5-9B: 13214 ms
    "TTFT" and `ITL p50 0.0`, published as a fox loss before the zeros were noticed.
    Counting both fields is what makes engines with different reasoning handling
    comparable at all.
    """
    ch = chunk.get("choices") or [{}]
    if not ch:
        return None
    d = ch[0].get("delta") or {}
    return d.get("content") or d.get("reasoning_content")


def one(i):
    """One streaming request. Returns (ttft_seconds, cached_tokens)."""
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Review item number {i}. Answer in one sentence."},
        ],
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
    cached = 0
    ptok = 0
    # Inter-token gaps, not just their average: a stream that averages 20 ms but stalls
    # for 400 ms once is what a user actually notices, and a tok/s figure cannot show it.
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
            # First chunk carrying actual content is the first decoded token; role-only
            # and empty deltas are protocol noise and would understate TTFT.
            if delta_text(d):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                stamps.append(time.perf_counter())
            usage = d.get("usage")
            if usage:
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
                ptok = usage.get("prompt_tokens", 0) or 0
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    return (ttft if ttft is not None else time.perf_counter() - t0), cached, ptok, gaps


def burst(label):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        out = list(ex.map(one, range(CONC)))
    wall = time.perf_counter() - t0
    ttfts = sorted(t for t, _, _, _ in out)
    p50 = ttfts[len(ttfts) // 2]
    p90 = ttfts[min(len(ttfts) - 1, int(len(ttfts) * 0.9))]
    cached = sum(c for _, c, _, _ in out)
    gaps = sorted(g for _, _, _, gg in out for g in gg)
    # The prompt length is reported as measured, never estimated: an oversized prompt
    # does not fail loudly on both servers — llama-server returns 400, fox silently
    # rolls the context window, which sets rolled_tokens and disables reuse. Either way
    # the benchmark would be measuring truncation rather than prompt sharing.
    ptok = max(p for _, _, p, _ in out)
    itl50 = gaps[len(gaps) // 2] * 1000 if gaps else 0
    itl99 = gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))] * 1000 if gaps else 0
    print(f"{label} {p50*1000:.0f} {p90*1000:.0f} {wall:.2f} {cached} {ptok} "
          f"{itl50:.1f} {itl99:.1f}")


burst("cold")
burst("warm")
