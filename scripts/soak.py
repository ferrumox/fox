#!/usr/bin/env python3
"""Soak test — sustained mixed traffic, then check that the server came back clean.

The gap this fills: `make e2e` is 22 checks over a couple of minutes, and every test
below it runs against a fresh process. Nothing exercises fox for long enough to show a
leak, a pool that never returns, or latency that drifts. The three prefix-cache
lifecycle bugs recorded in CHANGELOG 0.15.1 were all found by real use rather than by
tests, which is the same thing said less deliberately.

The traffic is mixed on purpose, because each shape has broken something before:

  conversations   multi-turn, growing history — exercises parking, slot reuse and
                  (on hybrid models) the prefill checkpoint
  one-offs        unrelated short prompts that never repeat — pure churn through the
                  slot table
  cancellations   clients that hang up mid-stream — fox is supposed to preempt and
                  free the KV, and a leak here is invisible to any request-shaped test

What makes it a test rather than a demo is the verdict at the end. After the load
stops, the server is given time to drain and then must satisfy:

  - `kv_cache_usage` back to the parked floor. NOT zero: fox deliberately keeps a
    finished sequence's KV resident so the next turn can reuse it, so a drained server
    legitimately holds up to one slot's worth per slot. What a leak looks like is usage
    that keeps climbing across cycles, which is why this runs the drain check twice with
    load in between and compares the two.
  - `active_requests` 0 and `queue_depth` 0.
  - RSS in the last quarter no more than 10% above the first quarter *after warmup*.
    Growth during warmup is normal (weights, caches, allocator arenas); growth after it
    is what a leak looks like.
  - No request failures.
  - TTFT p50 in the last quarter no more than 50% above the first.

Usage: soak.py URL MODEL [MINUTES] [CONCURRENCY]
"""
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

URL = sys.argv[1]
MODEL = sys.argv[2]
MINUTES = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
CONC = int(sys.argv[4]) if len(sys.argv) > 4 else 4

DEADLINE = time.time() + MINUTES * 60
stop = threading.Event()
lock = threading.Lock()
samples = []      # (t, rss_mb, kv_usage, active, queued)
requests_done = 0
failures = []
ttfts = []        # (t, ttft_ms)

PROMPTS = [
    "Name three uses for baking soda.",
    "Explain the water cycle briefly.",
    "What makes bread rise?",
    "How does a bicycle stay upright?",
]
FOLLOWUPS = [
    "Why does that happen?",
    "What would change that?",
    "Give one concrete example.",
    "Summarise it in one line.",
]


def post(messages, maxtok, cancel_after=None):
    """One streaming request. Returns ttft_ms, or raises. Hangs up early if asked."""
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": maxtok,
        "temperature": 0.8, "stream": True,
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    reply = []
    n = 0
    with urllib.request.urlopen(req, timeout=300) as r:
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
            piece = (ch[0].get("delta") or {}).get("content") if ch else None
            if piece:
                n += 1
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                reply.append(piece)
                # Hanging up mid-stream is the point of this branch: closing the
                # response is how fox learns the client is gone.
                if cancel_after and n >= cancel_after:
                    return ttft, "".join(reply), True
    return ttft, "".join(reply), False


def record(ttft, cancelled):
    global requests_done
    with lock:
        requests_done += 1
        if ttft is not None:
            ttfts.append((time.time(), ttft))


def worker(i):
    turn = 0
    messages = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
    while not stop.is_set() and time.time() < DEADLINE:
        try:
            kind = turn % 5
            if kind == 4:
                # cancellation: a fresh short request the client abandons
                ttft, _, _ = post([{"role": "user", "content": PROMPTS[(i + turn) % len(PROMPTS)]}],
                                  64, cancel_after=3)
                record(ttft, True)
            elif kind == 3:
                # one-off, never repeated: churn through the slot table
                ttft, _, _ = post([{"role": "user", "content":
                                    f"[{i}-{turn}] {PROMPTS[turn % len(PROMPTS)]}"}], 24)
                record(ttft, False)
                messages = [{"role": "user", "content": PROMPTS[(i + turn) % len(PROMPTS)]}]
            else:
                # conversation: history grows, then resets so it cannot grow forever
                ttft, reply, _ = post(messages, 24)
                record(ttft, False)
                messages.append({"role": "assistant", "content": reply or "(none)"})
                messages.append({"role": "user", "content": FOLLOWUPS[turn % len(FOLLOWUPS)]})
                if len(messages) > 12:
                    messages = [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]
        except Exception as e:  # noqa: BLE001 — any failure is a finding, not a crash
            with lock:
                failures.append(f"{type(e).__name__}: {e}")
        turn += 1


def server_rss_mb():
    """RSS of whatever is listening on our port, via /proc — no psutil dependency."""
    import glob
    import os
    port_hex = format(int(URL.rsplit(":", 1)[1].split("/")[0]), "04X")
    inode = None
    for line in open("/proc/net/tcp"):
        f = line.split()
        if len(f) > 9 and f[1].endswith(":" + port_hex) and f[3] == "0A":
            inode = f[9]
            break
    if inode is None:
        return None
    for fd in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd) == f"socket:[{inode}]":
                pid = fd.split("/")[2]
                for line in open(f"/proc/{pid}/status"):
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except (OSError, PermissionError):
            continue
    return None


def sampler():
    while not stop.is_set() and time.time() < DEADLINE:
        try:
            with urllib.request.urlopen(URL + "/health", timeout=5) as r:
                h = json.load(r)
            with lock:
                samples.append((time.time(), server_rss_mb(), h.get("kv_cache_usage", 0),
                                h.get("active_requests", 0), h.get("queue_depth", 0)))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


print(f"soak: {MINUTES:g} min, {CONC} clientes, mezcla conversación/one-off/cancelación")
threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONC)]
threads.append(threading.Thread(target=sampler, daemon=True))
for t in threads:
    t.start()

last = 0
while time.time() < DEADLINE:
    time.sleep(15)
    with lock:
        done, fails = requests_done, len(failures)
        rss = samples[-1][1] if samples else None
    print(f"  {int(DEADLINE - time.time()):5d}s restantes · {done} peticiones "
          f"(+{done - last}) · {fails} fallos · RSS {rss:.0f} MB" if rss else
          f"  {int(DEADLINE - time.time()):5d}s restantes · {done} peticiones · {fails} fallos")
    last = done

stop.set()
for t in threads:
    t.join(timeout=60)

# Drain: the pool is only expected to settle once in-flight work finishes.
print("\ndrenando 20 s…")
time.sleep(20)


def health():
    with urllib.request.urlopen(URL + "/health", timeout=10) as r:
        return json.load(r)


try:
    final = health()
except Exception as e:  # noqa: BLE001
    print(f"VEREDICTO: FALLO — el servidor no responde al final ({e})")
    sys.exit(1)

# The drained pool is not empty and should not be: parked sequences hold their KV on
# purpose. So the question is not "is it zero" but "does it grow". Run a short second
# burst, drain again, and compare — a floor that climbs is the leak signature.
early_floor = final.get("kv_cache_usage", 0)
print(f"suelo tras drenar: {early_floor:.4f} — repitiendo carga para ver si sube…")
stop.clear()
DEADLINE = time.time() + 45
second = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(CONC)]
for t in second:
    t.start()
for t in second:
    t.join(timeout=120)
stop.set()
time.sleep(20)
try:
    final = health()
except Exception as e:  # noqa: BLE001
    print(f"VEREDICTO: FALLO — el servidor no responde tras el segundo ciclo ({e})")
    sys.exit(1)
late_floor = final.get("kv_cache_usage", 0)

with lock:
    warm = [s for s in samples if s[0] > samples[0][0] + 60] if samples else []
    rss_vals = [s[1] for s in warm if s[1] is not None]
    tt = sorted(ttfts, key=lambda x: x[0])

print("\n" + "=" * 60)
verdict_ok = True


def check(name, ok, detail):
    global verdict_ok
    print(f"  {'✅' if ok else '❌'} {name}: {detail}")
    if not ok:
        verdict_ok = False


check("peticiones", len(failures) == 0,
      f"{requests_done} completadas, {len(failures)} fallos"
      + (f" — p.ej. {failures[0][:80]}" if failures else ""))
# Parked KV is expected; a floor that rises between two drains is not.
check("KV no crece entre ciclos", late_floor <= max(early_floor * 1.5, early_floor + 0.02),
      f"suelo {early_floor:.4f} → {late_floor:.4f} (lo aparcado es legítimo; lo que sube, no)")
check("sin trabajo colgado", final.get("active_requests", 1) == 0 and final.get("queue_depth", 1) == 0,
      f"active={final.get('active_requests')} queue={final.get('queue_depth')}")

if len(rss_vals) >= 8:
    q = len(rss_vals) // 4
    first, lastq = statistics.median(rss_vals[:q]), statistics.median(rss_vals[-q:])
    growth = (lastq - first) / first * 100 if first else 0
    check("memoria estable", growth <= 10,
          f"{first:.0f} → {lastq:.0f} MB ({growth:+.1f}% tras el calentamiento)")
else:
    print(f"  ⚠️  memoria: sólo {len(rss_vals)} muestras tras el calentamiento — sube la duración")

if len(tt) >= 8:
    q = len(tt) // 4
    f_ttft = statistics.median([x[1] for x in tt[:q]])
    l_ttft = statistics.median([x[1] for x in tt[-q:]])
    drift = (l_ttft - f_ttft) / f_ttft * 100 if f_ttft else 0
    check("latencia estable", drift <= 50,
          f"TTFT p50 {f_ttft:.0f} → {l_ttft:.0f} ms ({drift:+.1f}%)")

print("=" * 60)
print("VEREDICTO:", "OK" if verdict_ok else "FALLO")
sys.exit(0 if verdict_ok else 1)
