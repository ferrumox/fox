#!/usr/bin/env python3
"""Does this server report prompt-cache hits, and does it report them where the
benchmark reads them?

Exists because `cached_tokens: 0` has two completely different meanings and the
benchmark table cannot tell them apart:

    the server reused nothing            — a real finding about the engine
    the server does not report the field — a finding about its API

Publishing the second as the first would be a fabricated result, and both Ollama and
vLLM produce a 0 in the burst benchmark while their warm TTFT drops sharply, which is
exactly the pattern that says the reuse happened and the field is simply absent.

It also checks streaming and non-streaming separately: several servers populate
`prompt_tokens_details` on a normal response and omit it from the final usage chunk of
a streamed one, and bench_burst.py necessarily reads the streamed path.

Usage: probe_cached_tokens.py URL MODEL [REPEATS]
"""
import json
import sys
import urllib.request

URL, MODEL = sys.argv[1], sys.argv[2]
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 30

# Long enough that a cache hit is unmistakable, and identical between the two calls.
SYSTEM = (
    "You are a meticulous senior software engineer performing code review. "
    "Always cite the file and line number for any claim you make about the code. "
    "If you are uncertain, say so explicitly rather than guessing. "
) * REPEATS


def body(stream):
    d = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Answer with the single word: ok."},
        ],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": stream,
    }
    if stream:
        d["stream_options"] = {"include_usage": True}
    return json.dumps(d).encode()


def usage_of(stream):
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=body(stream),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            return json.load(r).get("usage") or {}
        last = {}
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
                last = d["usage"]
        return last


def report(mode, u):
    ptok = u.get("prompt_tokens")
    det = u.get("prompt_tokens_details")
    if det is None:
        print(f"  {mode:<14} prompt_tokens={ptok}  prompt_tokens_details AUSENTE")
    else:
        print(f"  {mode:<14} prompt_tokens={ptok}  cached_tokens={det.get('cached_tokens')}")


for mode in ("no-stream", "stream"):
    stream = mode == "stream"
    # First call populates whatever cache exists; the second is the one that should hit.
    usage_of(stream)
    report(f"{mode} (2ª)", usage_of(stream))

print()
print("  'AUSENTE' significa que el 0 de la tabla es de la API, no del motor. Si el TTFT")
print("  en caliente baja pero esto sale AUSENTE, hubo reutilización y no se reporta.")
