# fox bench-spec

Quantify **speculative decoding**: run the same generation with speculation off and on
and report tokens/s, the draft acceptance ratio, and the speedup. Weights are loaded
once; each scenario gets a fresh llama.cpp context.

```
fox bench-spec <MODEL> [OPTIONS]
```

---

## What it measures

Two workloads, each generated twice (speculation off, then on) with **greedy** sampling:

- **repetitive** — a prompt that makes the model echo a fixed cycle. This is n-gram
  lookup's best case: the next tokens keep re-appearing in the context, so drafts are
  frequently right.
- **prose** — an open-ended question. Free-form text repeats itself much less, so this
  is closer to speculation's worst case.

Because sampling is greedy, speculation **must not change the output** — the bench
verifies the off/on texts are byte-identical per workload and flags any difference as a
bug. Speculation only changes *speed*:

- **acceptance** = drafts the target model agreed with / drafts proposed. High acceptance
  → several tokens per forward pass → speedup. Low acceptance → wasted draft compute.
- **speedup** = tok/s (spec on) / tok/s (spec off), per workload.

---

## Basic usage

```bash
# Default: 192 tokens per run, ngram=2, draft_len=4
fox bench-spec llama3.2

# Longer runs, more aggressive drafting
fox bench-spec llama3.2 --max-new-tokens 256 --spec-draft-len 8
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | (required) | Model name, alias, or path to a GGUF file. |
| `--max-new-tokens <N>` | `192` | Tokens generated per run. |
| `--spec-ngram <N>` | `2` | Suffix length matched against the history when proposing. |
| `--spec-draft-len <N>` | `4` | Maximum draft tokens proposed per step. |
| `--max-context-len <N>` | auto | Context length per scenario. Defaults to the model's trained context. |
| `--main-gpu <N>` | `0` | Primary GPU index. |
| `--split-mode <MODE>` | `layer` | Multi-GPU split mode: `none`, `layer`, `row`. |
| `--tensor-split <RATIOS>` | auto | Comma-separated VRAM proportions (e.g. `"3,1"`). |
| `--moe-cpu` | `false` | Offload MoE expert layers to CPU RAM. |

---

## Sample output

llama-3.2-1B on CPU (single-threaded Docker):

```
  🦊  llama-3.2-1b-instruct-q8_0  ·  speculative-decoding validation
  ──────────────────────────────────────────────────────────────

  Config          ngram=2 draft_len=4 · 160 tokens/run · greedy

  workload       spec      tok/s    tokens   acceptance    speedup
  ──────────────────────────────────────────────────────────────
  repetitive      off       39.6       159            —          —
  repetitive       on       70.6       159          98%      1.78×
  prose           off       39.8       159            —          —
  prose            on       36.5       159           9%      0.92×

  acceptance = drafts the target model agreed with / drafts proposed
  greedy sampling: spec on/off produce identical text (verified per workload)
```

---

## Reading the results

- Expect a **clear speedup with high acceptance on the repetitive workload**, and little
  to no speedup (acceptance near zero) on prose — that asymmetry is inherent to
  prompt-lookup speculation and is exactly why `--speculative` is off by default: turn it
  on for workloads that echo their context (code edits, JSON transforms, RAG extraction,
  structured rewriting).
- The same numbers are visible on a running server via Prometheus:
  `ferrumox_spec_tokens_proposed_total`, `ferrumox_spec_tokens_accepted_total`, and the
  `ferrumox_spec_acceptance_ratio` gauge.

---

## See also

- [fox serve](./serve.md) — `--speculative`, `--spec-ngram`, `--spec-draft-len`
- [fox bench](./bench.md) — single-request throughput and load time
- [Speculative decoding design](https://github.com/ferrumox/fox/blob/main/docs/design/speculative-decoding.md) — how and why it's exact
