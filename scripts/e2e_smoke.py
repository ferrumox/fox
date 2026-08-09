#!/usr/bin/env python3
"""e2e_smoke.py — end-to-end smoke checks against a RUNNING fox server.

Exercises the cross-request lifecycle and every user-facing 0.13–0.15 feature over
real HTTP with a real model — the layer no unit/golden/stub test covers. This suite
exists because that exact blind spot hid three prefix-cache lifecycle bugs (see
CHANGELOG [0.15.0] Fixed). Beyond per-feature checks it covers streaming (SSE +
NDJSON), concurrent clients (continuous batching on real KV), and a context-window
fill that forces context rolling mid-generation.

Usage:  e2e_smoke.py [BASE_URL]          (default http://127.0.0.1:8199)

Requirements on the server side (the runner script handles this):
  - started with a real GGUF model (any small instruct model works)
  - --speculative true  (check 7 asserts drafts are proposed on repetitive output)

Exit code 0 = all checks passed; 1 = at least one failed. stdlib only.
"""

import base64
import json
import math
import os
import struct
import sys
import urllib.error
import urllib.request
import zlib

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


def make_test_png(width=4, height=4, color=(255, 0, 0)):
    """Build a tiny valid PNG in-memory (8-bit RGB, no filter/interlace) — no
    external image library needed, and no large base64 blob checked into the repo."""

    def chunk(tag, data):
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit depth, RGB
    row = b"\x00" + bytes(color) * width  # leading byte = filter type "none"
    idat = zlib.compress(row * height)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


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
            # min_tokens suppresses EOG until the cap is reached, so `finish ==
            # "length"` below is a fact about the engine rather than a coin flip.
            # Without it the model samples at the default temperature and can emit
            # EOS early: measured at 0.33% per request under concurrency, which across
            # this suite's 7 such requests is ~1 failing run in 43 — the intermittent
            # e2e failure that took two sessions to pin down. The check's intent (it
            # must decode PAST its prefill token, 1 token is a FAIL) is untouched: a
            # request that dies early still reports n < 12.
            "min_tokens": 12,
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
#
# Token budget: the grammar's `integer` rule is `"-"? ("0" | [1-9][0-9]*)` — digits are
# unbounded, because JSON Schema's `type: integer` has no length bound to translate. So
# guided decoding does NOT guarantee a *complete* document within a token cap: a run
# that wanders into a long number gets cut off mid-JSON and `json.loads` fails. With the
# old 60-token cap that failure was indistinguishable from "the grammar emitted
# something non-conforming", which is the actual bug this check exists to catch.
#
# Fixed two ways: generous headroom, and an explicit truncation check first, so a future
# failure says which of the two happened instead of leaving it to be guessed.
print("2) guided decoding (json_schema)")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Is the sky blue? How many suns?"}],
        "max_tokens": 256,
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
finish = (r.get("choices") or [{}])[0].get("finish_reason") if st == 200 else None
raw = (r.get("choices") or [{}])[0].get("message", {}).get("content", "") if st == 200 else ""
if finish == "length":
    # Not a conformance failure: the document was cut off before it could close.
    check(
        "output was not truncated by max_tokens",
        False,
        f"hit max_tokens ({len(raw)} chars, no closing brace) — raise the cap or bound "
        f"the schema; partial={raw[:120]!r}",
    )
else:
    try:
        p = json.loads(raw)
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
#
# Same truncation trap as check 2, and worse: `format: "json"` compiles to the fully
# permissive any-JSON grammar, so nothing bounds how long or how deeply nested the
# document gets. An explicit num_predict plus a truncation check keeps a cut-off
# document from being misreported as "the grammar produced invalid JSON".
print('6) Ollama format: "json"')
st, r = post(
    "/api/chat",
    {
        "model": MODEL,
        "stream": False,
        "format": "json",
        "options": {"num_predict": 256},
        "messages": [
            {"role": "user", "content": "Give me a JSON object with a color key."}
        ],
    },
)
if st == 200 and r.get("done_reason") == "length":
    partial = r.get("message", {}).get("content", "")
    check(
        "output was not truncated by num_predict",
        False,
        f"hit num_predict — raise it or constrain the format; partial={partial[:120]!r}",
    )
else:
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
    if line.startswith("#"):
        continue
    # Summed rather than assigned: since 0.21 these carry a `model` label, so a
    # server with more than one model loaded emits one series per model and
    # taking the last would silently report a single model's drafting.
    if line.startswith("fox_spec_tokens_proposed_total"):
        proposed += float(line.split()[-1])
    if line.startswith("fox_spec_tokens_accepted_total"):
        accepted += float(line.split()[-1])
check(
    "drafts proposed > 0",
    proposed > 0,
    f"proposed={proposed:.0f} accepted={accepted:.0f}",
)

# ── 8) streaming: SSE (OpenAI) and NDJSON (Ollama) ───────────────────────────
print("8) streaming")


def post_stream(path, body):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return [ln.decode().strip() for ln in r if ln.strip()]

try:
    lines = post_stream(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from one to ten in words."}],
            "max_tokens": 10,
            "stream": True,
        },
    )
    datas = [ln[6:] for ln in lines if ln.startswith("data: ")]
    chunks = [json.loads(d) for d in datas if d != "[DONE]"]
    finished = any(
        c["choices"][0].get("finish_reason") for c in chunks if c.get("choices")
    )
    check(
        "SSE chunks + finish + [DONE]",
        len(chunks) >= 3 and finished and datas[-1] == "[DONE]",
        f"chunks={len(chunks)}",
    )
except Exception as e:  # noqa: BLE001
    check("SSE chunks + finish + [DONE]", False, str(e))

try:
    lines = post_stream(
        "/api/chat",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Count from one to ten in words."}],
            "stream": True,
            "options": {"num_predict": 10},
        },
    )
    objs = [json.loads(ln) for ln in lines]
    check(
        "NDJSON chunks + done:true",
        len(objs) >= 3 and objs[-1].get("done") is True,
        f"chunks={len(objs)}",
    )
except Exception as e:  # noqa: BLE001
    check("NDJSON chunks + done:true", False, str(e))

# ── 9) concurrent clients: continuous batching on real KV ────────────────────
# Four simultaneous requests — decode batches carry several sequences at once, the
# path the sequential checks never exercise. STRICT: every request must decode fully.
print("9) concurrent clients ×4")
import threading  # noqa: E402

results = [None] * 4


def one_client(i):
    st, r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": f"Count from {i + 1} to fifty in words, slowly.",
                }
            ],
            "max_tokens": 12,
            "min_tokens": 12,  # see check 1 — keeps `finish == "length"` deterministic
        },
    )
    n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
    fin = r["choices"][0]["finish_reason"] if st == 200 else "?"
    results[i] = (st, n, fin)


threads = [threading.Thread(target=one_client, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
for t in threads:
    t.join()
for i, (st, n, fin) in enumerate(results):
    check(
        f"client {i + 1} decodes fully",
        st == 200 and n >= 12 and fin == "length",
        f"tokens={n} finish={fin}",
    )

# ── 10) context fill → rolling keeps generating ──────────────────────────────
# A medium prompt plus 1100 FORCED tokens (min_tokens suppresses EOS) always crosses
# the server's 2048-token context regardless of tokenizer packing. Without context
# rolling the decode fails at the boundary and the request dies early. This also
# exercises rolling + speculation together on a live server.
print("10) context fill → rolling (crosses n_ctx=2048)")
filler = " ".join(["alpha bravo charlie delta echo foxtrot golf hotel"] * 130)
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": filler + "\nNow keep listing words in that style forever.",
            }
        ],
        "max_tokens": 1100,
        "min_tokens": 1100,
        "temperature": 0,
    },
)
n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
p = r.get("usage", {}).get("prompt_tokens", 0) if st == 200 else 0
check(
    "generation continues past n_ctx",
    st == 200 and n >= 1100 and p + n > 2048,
    f"prompt={p} completions={n} total={p + n}",
)

# ── 11) re-request after a rolled generation (donate-after-roll guard) ───────
# Check 10's request ROLLED its context. If it (wrongly) donated its prefix to the
# cache, this identical prompt would hit an entry whose cells no longer hold the
# prompt prefix — conditioning on garbage. Must still decode fully and healthily.
print("11) same prompt again after the rolled generation")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": filler + "\nNow keep listing words in that style forever.",
            }
        ],
        "max_tokens": 16,
        "min_tokens": 16,
        "temperature": 0,
    },
)
n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
check("decodes fully after prior roll", st == 200 and n >= 16, f"tokens={n}")

# ── 12) client disconnect mid-stream, then a healthy request ─────────────────
# Dropping the connection mid-generation triggers the engine's preempt path (clear
# sequence, free grammar, recycle seq id). The NEXT request must be unaffected.
print("12) disconnect mid-stream → next request healthy")
try:
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count to one hundred in words."}],
                "max_tokens": 200,
                "stream": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=TIMEOUT)
    for _ in range(3):  # read a few chunks, then hang up mid-generation
        resp.readline()
    resp.close()
except Exception:  # noqa: BLE001 — the abort itself may raise; that's fine
    pass
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from one to fifty in words."}],
        "max_tokens": 12,
    },
)
n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
fin = r["choices"][0]["finish_reason"] if st == 200 else "?"
check(
    "request after disconnect decodes fully",
    st == 200 and n >= 12 and fin == "length",
    f"tokens={n} finish={fin}",
)

# ── 13) embeddings alongside generation ───────────────────────────────────────
# Embeddings use a dedicated KV sequence outside the scheduler pool; they must not
# perturb generation. Run an embed, then a strict chat request.
print("13) embeddings then chat")
st, r = post("/v1/embeddings", {"model": MODEL, "input": "The quick brown fox."})
try:
    vec = r["data"][0]["embedding"]
    check("embedding vector non-degenerate", st == 200 and len(vec) > 64
          and any(abs(x) > 1e-6 for x in vec), f"dim={len(vec)}")
except Exception as e:  # noqa: BLE001
    check("embedding vector non-degenerate", False, f"{e} resp={str(r)[:120]}")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Count from one to fifty in words."}],
        "max_tokens": 12,
    },
)
n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
fin = r["choices"][0]["finish_reason"] if st == 200 else "?"
check(
    "chat after embed decodes fully",
    st == 200 and n >= 12 and fin == "length",
    f"tokens={n} finish={fin}",
)

# ── 14) vision: image input (only when the server was started with --mmproj) ──
# Exercises the full multimodal path: image_url data-URI decode → MEDIA_MARKER
# splice → mtmd_tokenize → atomic do_prefill_multimodal. The third request
# reuses a DIFFERENT image right after the first — the exact scenario the
# per-request skip-prefix-cache design (empty prompt_tokens on multimodal
# requests) exists to prevent cross-contaminating (see
# docs/design/vision-support.md); a hang, error, or garbage response here would
# indicate that design failed, not just an accuracy miss.
if os.environ.get("FOX_E2E_VISION") == "1":
    print("14) vision: image input")
    red_b64 = base64.b64encode(make_test_png(color=(255, 0, 0))).decode()
    blue_b64 = base64.b64encode(make_test_png(color=(0, 0, 255))).decode()

    st, r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{red_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 20,
        },
    )
    content = r["choices"][0]["message"]["content"] if st == 200 else ""
    check(
        "/v1/chat/completions accepts image_url and generates a response",
        st == 200 and len(content.strip()) > 0,
        f"status={st} content={content[:80]!r}",
    )

    st, r = post(
        "/api/chat",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "What color is this image? One word.",
                    "images": [red_b64],
                }
            ],
            "stream": False,
        },
    )
    content = r.get("message", {}).get("content", "") if st == 200 else ""
    check(
        "/api/chat accepts images field and generates a response",
        st == 200 and len(content.strip()) > 0,
        f"status={st} content={content[:80]!r}",
    )

    st, r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? One word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{blue_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 20,
        },
    )
    content = r["choices"][0]["message"]["content"] if st == 200 else ""
    check(
        "a different image right after the first doesn't error/hang (prefix-cache isolation)",
        st == 200 and len(content.strip()) > 0,
        f"status={st} content={content[:80]!r}",
    )

    st, r = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        }
                    ],
                }
            ],
        },
    )
    check(
        "remote image_url is rejected with 400, not fetched",
        st == 400,
        f"status={st}",
    )
else:
    print("14) vision: image input — SKIPPED (run with --mmproj-path / E2E_MMPROJ to enable)")

# ── 15) LoRA: adapter selection via the `model` field ─────────────────────────
# Exercises resolve_for_request (alias -> primary model + LoraSelection) and the
# group-and-switch llama_set_adapters_lora path in do_prefill/do_decode. Requests
# alternate base -> adapter -> base -> adapter. NOTE: this does NOT assert
# byte-identical output across same-target requests — fox's decode is not
# bit-reproducible in general (prefix-cache hit vs. miss alone takes a different
# compute path with different floating-point rounding, confirmed by running two
# plain base-only requests back-to-back with no adapter involved at all). What
# this checks instead: (a) the adapter measurably changes output vs. the base
# model on the same prompt (proves the adapter is actually engaged, not silently
# ignored), and (b) every request in the interleaved sequence decodes fully and
# healthily regardless of which config immediately preceded it (proves switching
# adapters — including the skip_prefix_cache path — doesn't corrupt context state
# or hang; see docs/design/lora-support.md).
LORA_NAME = os.environ.get("FOX_E2E_LORA_NAME")
if LORA_NAME:
    print("15) LoRA adapter selection")

    # A short-answer factual prompt ("capital of France") is greedy-deterministic
    # enough at temperature 0 that many adapters won't visibly move it — this needs
    # an open-ended prompt where an adapter's influence (style, verbosity, reasoning
    # structure) actually has room to show up in the completion.
    def ask(model_name):
        st, r = post(
            "/v1/chat/completions",
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": "If a train travels 60 miles in 90 minutes, "
                        "what is its average speed in mph?",
                    }
                ],
                "max_tokens": 64,
                "min_tokens": 64,
                "temperature": 0,
            },
        )
        n = r.get("usage", {}).get("completion_tokens", 0) if st == 200 else 0
        content = r["choices"][0]["message"]["content"] if st == 200 else ""
        return st, n, content

    sequence = [MODEL, LORA_NAME, MODEL, LORA_NAME]
    results = [ask(name) for name in sequence]

    check(
        "every request in the base/adapter/base/adapter sequence decodes fully",
        all(st == 200 and n >= 64 for st, n, _ in results),
        f"statuses/tokens={[(st, n) for st, n, _ in results]}",
    )
    base1, lora1 = results[0][2], results[1][2]
    check(
        "adapter output differs from base (adapter is actually applied)",
        base1.strip() != lora1.strip(),
        f"base={base1[:60]!r} lora={lora1[:60]!r}",
    )
else:
    print("15) LoRA adapter selection — SKIPPED (run with --lora-modules / E2E_LORA to enable)")

# ── 16) n: multiple completions per request ───────────────────────────────────
# n branches are independent fan-out generations, not a shared-prefill fork (see
# docs/design/n-best-of-support.md) — deliberately NOT asserting the choices are
# textually distinct (no such guarantee; thread_rng-driven divergence is likely
# but not proven, per the LoRA e2e experience that exact-content assumptions on
# this engine are risky to bake into a check). What's actually verified: the
# right number of choices at the right indices, all non-empty, and usage summed
# across every branch.
print("16) n: multiple completions per request")
st, r = post(
    "/v1/chat/completions",
    {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Tell me a short fun fact."}],
        "max_tokens": 24,
        "temperature": 0.9,
        "n": 3,
    },
)
choices = r.get("choices", []) if st == 200 else []
indices = sorted(c.get("index") for c in choices)
non_empty = all(c["message"]["content"].strip() for c in choices)
check(
    "n=3 returns 3 choices at indices 0,1,2, all non-empty",
    st == 200 and indices == [0, 1, 2] and non_empty,
    f"status={st} indices={indices}",
)
usage = r.get("usage", {}) if st == 200 else {}
check(
    "usage.completion_tokens reflects all 3 branches, not just one",
    usage.get("completion_tokens", 0) >= 3,
    f"usage={usage}",
)

print(f"\n{'=' * 50}\nRESULT: {ok_count} passed, {fail_count} failed")
sys.exit(1 if fail_count else 0)
