# fox bench-prefill

Validate **chunked prefill**: prove that prefilling a long prompt no longer stalls a
concurrent short request. Weights are loaded once; a fresh llama.cpp context is created
for each `--max-prefill-chunk` value under test.

```
fox bench-prefill <MODEL> [OPTIONS]
```

---

## What it measures

The benchmark submits two requests at the same time:

- a **long prompt** whose prefill, unchunked, would monopolise the engine loop, and
- a **short request** that tries to generate tokens while the long prompt prefills.

It then reports the short request's **worst stall** — the largest gap between two of its
consecutive tokens, including the time-to-first-token (its first gap, measured from
submission). This is exactly the latency chunked prefill is designed to bound:

- With chunking **on**, the long prompt is prefilled a chunk at a time, so the short
  request slips a token in after every chunk and its worst stall is ≈ one chunk's
  prefill.
- With chunking **off** (`--chunks 0`, single-shot), the whole long prompt is one
  `llama_decode`, so the short request cannot produce its first token until that entire
  prefill finishes — its worst stall balloons to the full long-prompt prefill.

A single run prints both and their ratio.

---

## Basic usage

```bash
# Compare chunked (512) against single-shot on a 2048-token prompt (defaults)
fox bench-prefill llama3.2

# Heavier long prompt, more chunk sizes
fox bench-prefill llama3.2 --long-prompt-tokens 4096 --chunks 256,512,1024,0

# Longer short request, explicit context
fox bench-prefill llama3.2 --short-new-tokens 48 --max-context-len 8192
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | (required) | Model name, alias, or path to a GGUF file. |
| `--long-prompt-tokens <N>` | `2048` | Length of the competing long prompt, in tokens. |
| `--short-new-tokens <N>` | `24` | Tokens the short request generates during the long prefill. |
| `--chunks <LIST>` | `512,0` | Comma-separated `--max-prefill-chunk` values to compare. `0` = single-shot (chunking off). |
| `--max-context-len <N>` | auto | Context length used for each scenario. Defaults to the model's trained context. |
| `--main-gpu <N>` | `0` | Primary GPU index. |
| `--split-mode <MODE>` | `layer` | Multi-GPU split mode: `none`, `layer`, `row`. |
| `--tensor-split <RATIOS>` | auto | Comma-separated VRAM proportions (e.g. `"3,1"`). |
| `--moe-cpu` | `false` | Offload MoE expert layers to CPU RAM. |

---

## Sample output

A 4096-token prompt on CPU (llama-3.2-1B, single-threaded Docker) makes the effect
stark — single-shot prefill blocks the short request for ~22 seconds:

```
  🦊  llama-3.2-1b-instruct-q8_0  ·  chunked-prefill validation
  ──────────────────────────────────────────────────────────────

  Long prompt     4096 tokens
  Short request   24 generated tokens

  chunk      short-TTFT    worst-stall  short-tok/s  long-prefill
  ──────────────────────────────────────────────────────────────
    256          1.399s         1.399s          0.8       22.098s
    512          2.496s         3.415s          0.3       22.469s
    off         21.852s        21.852s          8.2       21.852s

  Chunked prefill (chunk=256) cut the short request's worst stall
  from 21.852s (single-shot) to 1.399s — 15.6× more responsive.

  worst-stall = largest gap the short request waited (incl. time-to-first-token)
```

Reading the table:

- **`worst-stall`** is the headline. Single-shot forces the short request to wait for
  the *entire* 4096-token prefill before its first token (21.9 s). Chunking bounds the
  wait to one chunk — 1.4 s at chunk=256, 3.4 s at chunk=512. Smaller chunks interleave
  more finely, so the stall shrinks with the chunk size.
- **`long-prefill`** stays ≈22 s regardless of chunk: chunking does not add prefill
  work, it only spreads it across steps so other requests slip in between.
- **`short-tok/s`** is *lower* with chunking, and that is expected, not a regression.
  Chunking front-loads the short request's first token by ~15× (TTFT 1.4 s vs 21.9 s),
  but each of its early tokens now shares a scheduler step with a prefill chunk, so
  those tokens arrive slower *while the long prefill is still draining*. The trade is
  dramatically better responsiveness (TTFT / tail latency) for the same total system
  throughput — exactly what a server under concurrent load wants.

The gap widens with `--long-prompt-tokens`: the bigger the long prompt, the longer a
single-shot prefill stalls everyone else, while chunked stays flat at ≈ one chunk.

---

## See also

- [fox bench](./bench.md) — single-request throughput and load time
- [fox bench-kv](./bench-kv.md) — compare KV cache quantization types
- [Serving robustness design](../design/serving-robustness.md) — why chunked prefill exists
