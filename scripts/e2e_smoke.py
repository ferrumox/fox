#!/usr/bin/env python3
"""e2e_smoke.py — end-to-end smoke checks against a RUNNING fox server.

Exercises the cross-request lifecycle and every user-facing 0.13–0.15 feature over
real HTTP with a real model — the layer no unit/golden/stub test covers. This suite
exists because that exact blind spot hid two prefix-cache lifecycle bugs (see
CHANGELOG [0.15.0] Fixed).

Usage:  e2e_smoke.py [BASE_URL]          (default http://127.0.0.1:8199)

Requirements on the server side (the runner script handles this):
  - started with a real GGUF model (any small instruct model works)
  - --speculative true  (check 7 asserts drafts are proposed on repetitive output)

Exit code 0 = all checks passed; 1 = at least one failed. stdlib only.
"""

import json
import math
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8199"
TIMEOUT = 300  # generous: CI runners decode a 0.5B on CPU

ok_count = 0
fail_count = 0


def post(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as r:
        return r.read().decode()


def check(name, cond, detail=""):
    global ok_count, fail_count
    mark = "✅" if cond else "❌"
    if cond:
        ok_count += 1
    else:
        fail_count += 1
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


# Model name comes from the server itself (basename the server loaded).
MODEL = json.loads(get("/health"))["model_name"]
print(f"target: {BASE}  model: {MODEL}\n")

# ── 1) repeat requests: prefix-cache donate→hit lifecycle ────────────────────
# The exact scenario that exposed the poisoned-sequence bugs: a finished request
# donates its prefix, the next identical request hits the cache and reuses the seq.
# STRICT: the request must reach max_tokens (finish "length"). A request that dies
# after its prefill token still returns 200 with 1 token — that leniency previously
# masked a decode-after-hit failure, so 1 token is a FAIL here.
print("1) repeat chat ×3 (prefix-cache donate→hit)")
for i in range(3):
    st, r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": "Count from one to twenty in words."}
            ],
            "max_tokens": 12,
        },
    )
    n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
    finish = r["choices"][0]["finish_reason"] if st == 200 else "?"
    check(
        f"request {i + 1} decodes past prefill",
        st == 200 and n >= 12 and finish == "length",
        f"tokens={n} finish={finish}",
    )

# ── 2) guided decoding: JSON schema (enum + integer → short, deterministic) ──
print("2) guided decoding (json_schema)")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Is the sky blue? How many suns?"}],
        "max_tokens": 60,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "a",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"enum": ["yes", "no"]},
                        "count": {"type": "integer"},
                    },
                    "required": ["answer", "count"],
                },
            },
        },
    },
)
try:
    p = json.loads(r["choices"][0]["message"]["content"])
    check(
        "output parses and conforms",
        p["answer"] in ("yes", "no") and isinstance(p["count"], int),
        str(p),
    )
except Exception as e:  # noqa: BLE001 — any failure is a check failure
    check("output parses and conforms", False, f"{e} resp={str(r)[:200]}")

# ── 3) unconvertible schema must be a 400, not a silent fallback ─────────────
print("3) invalid schema rejected")
st, _ = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "x", "schema": {"type": "widget"}},
        },
    },
)
check("HTTP 400", st == 400, f"got {st}")

# ── 4) logprobs ──────────────────────────────────────────────────────────────
print("4) logprobs")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "The sky is"}],
        "max_tokens": 3,
        "logprobs": True,
        "top_logprobs": 2,
    },
)
try:
    entries = r["choices"][0]["logprobs"]["content"]
    first = entries[0]
    structural = (
        all(x["logprob"] <= 1e-4 for x in entries)
        and len(first["top_logprobs"]) == 2
    )
    check(
        "per-token logprobs + alternatives",
        structural,
        f"token={first['token']!r} p={math.exp(first['logprob']) * 100:.0f}%",
    )
except Exception as e:  # noqa: BLE001
    check("per-token logprobs + alternatives", False, f"{e} resp={str(r)[:200]}")

# ── 5) min_p / logit_bias / min_tokens ───────────────────────────────────────
print("5) sampling controls")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Say ok"}],
        "max_tokens": 8,
        "min_p": 0.05,
        "min_tokens": 3,
        "logit_bias": {"1000": -50},
    },
)
n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
check("accepted and min_tokens honoured (≥3)", st == 200 and n >= 3, f"tokens={n}")

# ── 6) Ollama surface: format "json" ─────────────────────────────────────────
print('6) Ollama format: "json"')
st, r = post(
    "/api/chat",
    {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "user", "content": "Give me a JSON object with a color key."}
        ],
    },
)
try:
    p = json.loads(r["message"]["content"])
    check("output parses as JSON", isinstance(p, (dict, list)), str(p)[:80])
except Exception as e:  # noqa: BLE001
    check("output parses as JSON", False, f"{e} resp={str(r)[:200]}")

# ── 7) speculative decoding proposes drafts on repetitive output ─────────────
# The prompt embeds a repeating n-gram, so prompt-lookup must find matches during
# generation regardless of what the (small) model produces.
print("7) speculative decoding (repetitive output)")
post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Write the phrase: ping pong ping pong ping pong, "
                "and continue that pattern for many lines.",
            }
        ],
        "max_tokens": 100,
        "temperature": 0,
    },
)
proposed = accepted = 0.0
for line in get("/metrics").splitlines():
    if line.startswith("ferrumox_spec_tokens_proposed_total"):
        proposed = float(line.split()[-1])
    if line.startswith("ferrumox_spec_tokens_accepted_total"):
        accepted = float(line.split()[-1])
check(
    "drafts proposed > 0",
    proposed > 0,
    f"proposed={proposed:.0f} accepted={accepted:.0f}",
)

print(f"\n{'=' * 50}\nRESULT: {ok_count} passed, {fail_count} failed")
sys.exit(1 if fail_count else 0)
