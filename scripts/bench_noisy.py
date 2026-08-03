#!/usr/bin/env python3
"""Noisy neighbour — what one long prefill does to streams that are already running.

The workload nobody publishes and everybody hits. A handful of interactive chats are
streaming along nicely; then one request arrives carrying a huge prompt — a pasted file,
a RAG context, an agent's accumulated history. On a server that prefills a prompt in one
indivisible chunk, every interactive stream stops dead until that prefill finishes. On
one that splits prefill into pieces and interleaves them with decode, the streams keep
moving and merely slow down.

Both behaviours produce *identical* average tokens/s over the whole run. The difference
shows up only as a gap in the middle of somebody else's stream, which is why this is
measured as inter-token latency inside a time window rather than as throughput.

Method:

  1. N interactive clients stream short requests back to back, continuously.
  2. After BASELINE_S seconds of that, one long-prompt request is injected.
  3. Every inter-token gap is timestamped and sorted into two buckets: before the
     injection, and between the injection and the long request finishing.

The headline is the ratio of p99 gaps between the two buckets. 1.0 means the interactive
users never noticed; 20 means their streams froze for the duration.

The long prompt shares no prefix with the interactive ones on purpose. If it did, an
engine with prefix caching could skip the prefill entirely and the benchmark would
measure cache hits rather than prefill interference.

Usage: bench_noisy.py URL MODEL [INTERACTIVE] [LONG_REPEATS] [BASELINE_S] [TAIL_S]
Prints: "noisy <base_p50> <base_p99> <inj_p50> <inj_p99> <ratio_p99> <long_ttft_ms>
         <window_s> <n_base> <n_inj> <long_prompt_tokens>"   (latencies in ms)

The long prompt's measured token count is printed, never assumed: if it does not fit the
per-sequence context the engines diverge silently — llama-server returns 400 while fox
rolls the window — and the test would be measuring truncation instead of interference.
"""
import json
import sys
import threading
import time
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
INTERACTIVE = int(sys.argv[3]) if len(sys.argv) > 3 else 4
LONG_REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 110
BASELINE_S = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
TAIL_S = float(sys.argv[6]) if len(sys.argv) > 6 else 3.0

# ~3.5k tokens, and deliberately unlike the interactive prompts in both content and
# opening words so no prefix can be shared with them.
LONG_PROMPT = (
    "Section: inventory reconciliation notes. Warehouse 7 recorded a discrepancy "
    "between the counted stock and the ledger for the third consecutive quarter, "
    "with the largest variance in fasteners and sealing components. "
) * LONG_REPEATS

SHORT_PROMPTS = [
    "Name three uses for baking soda.",
    "What is the capital of Portugal, and why does it sit where it does?",
    "Explain gravity to a curious ten-year-old.",
    "Suggest two ways to keep coffee hot for longer.",
]

stop = threading.Event()
lock = threading.Lock()
gaps = []          # (timestamp, gap_seconds) for every interactive token gap
long_result = {}


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


def stream(prompt, maxtok, collect):
    """Stream one request. When collect is true, timestamp every inter-token gap."""
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": maxtok,
        "temperature": 0.8,
        "top_p": 0.9,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    prev = None
    ttft = None
    ptok = 0
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
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens", 0) or 0
            if not delta_text(d):
                continue
            now = time.perf_counter()
            if ttft is None:
                ttft = now - t0
            elif collect:
                with lock:
                    gaps.append((now, now - prev))
            prev = now
    return (ttft if ttft is not None else time.perf_counter() - t0), ptok


def interactive_worker(i):
    n = 0
    while not stop.is_set():
        try:
            stream(SHORT_PROMPTS[(i + n) % len(SHORT_PROMPTS)], 96, True)  # noqa
        except Exception:
            # One failed request must not silence a client for the rest of the run —
            # a quiet worker would shrink the sample exactly when the server is
            # struggling, which is when the sample matters most.
            time.sleep(0.2)
        n += 1


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))] * 1000


workers = [threading.Thread(target=interactive_worker, args=(i,), daemon=True)
           for i in range(INTERACTIVE)]
for w in workers:
    w.start()

time.sleep(BASELINE_S)

t_inject = time.perf_counter()
def run_long():
    ttft, ptok = stream(LONG_PROMPT, 16, False)
    long_result.update(ttft=ttft, ptok=ptok)


long_thread = threading.Thread(target=run_long)
long_thread.start()
long_thread.join()
t_done = time.perf_counter()

time.sleep(TAIL_S)
stop.set()
for w in workers:
    w.join(timeout=30)

with lock:
    base = [g for ts, g in gaps if ts < t_inject]
    inj = [g for ts, g in gaps if t_inject <= ts <= t_done]

bp50, bp99 = pct(base, 0.50), pct(base, 0.99)
ip50, ip99 = pct(inj, 0.50), pct(inj, 0.99)
ratio = (ip99 / bp99) if bp99 > 0 else 0.0
print(f"noisy {bp50:.1f} {bp99:.1f} {ip50:.1f} {ip99:.1f} {ratio:.2f} "
      f"{long_result.get('ttft', 0)*1000:.0f} {t_done - t_inject:.2f} {len(base)} {len(inj)} "
      f"{long_result.get('ptok', 0)}")
