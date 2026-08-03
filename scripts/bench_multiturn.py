#!/usr/bin/env python3
"""Multi-turn chat — the workload behind "conversations get faster over time".

The most-quoted product claim in fox's docs, and until now the only one with no number
under it. Everything else in this bank used a synthetic burst that re-sends identical
prompts, which is a *proxy* for a conversation, not a conversation.

What a real turn looks like: turn N's prompt is turn N-1's prompt, plus the assistant's
actual reply, plus a new user message. The history is therefore an exact prefix of the
next request, and it grows monotonically. An engine that keeps the sequence resident
prefills only the new message; one that does not re-reads the whole conversation.

BUILT TO BE ABLE TO DISAPPOINT. Between turns a conversation's sequence is *idle*, and
inheriting an idle slot is precisely what `llama-server` already does well — its
limitation is inheriting from a sequence that is still decoding. So the honest
expectation here is parity, not a fox win, and a result showing parity is a result. The
place fox should pull ahead is when there are more live conversations than there are
slots to hold them, so the reply is measured against concurrency too.

Replies are fed back verbatim rather than simulated: a synthetic "assistant" turn would
not tokenise like a real one, and the whole measurement is about prefix boundaries.

Each conversation opens with a unique marker so two conversations never share a prefix.
That isolates *within-conversation* reuse from the shared-system-prompt case the burst
benchmark already covers; conflating them would let one mechanism take credit for the
other's work.

Usage: bench_multiturn.py URL MODEL [CONVERSATIONS] [TURNS] [SYS_REPEATS] [MAX_TOKENS]
Prints one line per turn index:
  "turn <i> <ttft_p50_ms> <cached_p50> <prompt_tokens_p50> <n>"
"""
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, MODEL = sys.argv[1], sys.argv[2]
CONVERSATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
TURNS = int(sys.argv[4]) if len(sys.argv) > 4 else 6
SYS_REPEATS = int(sys.argv[5]) if len(sys.argv) > 5 else 8
MAXTOK = int(sys.argv[6]) if len(sys.argv) > 6 else 64

# A short per-conversation preamble. Kept modest on purpose: the point here is the
# history accumulating, not a long shared prompt — that is the burst benchmark's job.
PREAMBLE = (
    "You are helping plan a small project. Keep answers to one or two sentences. "
)

FOLLOWUPS = [
    "What should the first step be?",
    "What could go wrong with that?",
    "How would I know it worked?",
    "What would you do differently with half the time?",
    "Who else needs to be involved?",
    "What is the smallest version worth doing?",
    "What would make you abandon this approach?",
    "Summarise the plan so far in one sentence.",
]

results = []  # (turn_index, ttft_ms, cached, prompt_tokens)


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


def turn(messages):
    """One streaming turn. Returns (ttft_s, reply_text, cached, prompt_tokens)."""
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
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
    reply = []
    cached = 0
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
            piece = delta_text(d)
            if piece:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                reply.append(piece)
            usage = d.get("usage")
            if usage:
                cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0) or 0
                ptok = usage.get("prompt_tokens", 0) or 0
    return (ttft if ttft is not None else time.perf_counter() - t0), "".join(reply), cached, ptok


def conversation(c):
    # The index leads so no two conversations share even their first token.
    messages = [{"role": "system", "content": f"[conversation {c}] " + PREAMBLE * SYS_REPEATS},
                {"role": "user", "content": "I want to organise a small community library."}]
    out = []
    for t in range(TURNS):
        ttft, reply, cached, ptok = turn(messages)
        out.append((t, ttft * 1000, cached, ptok))
        # Feed the real reply back. An empty reply would silently shorten the history
        # and make later turns look cheaper than they are.
        messages.append({"role": "assistant", "content": reply or "(no reply)"})
        messages.append({"role": "user", "content": FOLLOWUPS[t % len(FOLLOWUPS)]})
    return out


with ThreadPoolExecutor(max_workers=CONVERSATIONS) as ex:
    for conv in ex.map(conversation, range(CONVERSATIONS)):
        results.extend(conv)

for t in range(TURNS):
    rows = [r for r in results if r[0] == t]
    if not rows:
        continue
    print(f"turn {t} "
          f"{statistics.median([r[1] for r in rows]):.0f} "
          f"{statistics.median([r[2] for r in rows]):.0f} "
          f"{statistics.median([r[3] for r in rows]):.0f} "
          f"{len(rows)}")
