# fox bench

Measure model load time and inference throughput directly from the command line. No server is started — the model loads, runs the benchmark, prints results, and exits.

```
fox bench <MODEL> [OPTIONS]
```

---

## Overview

`fox bench` is a quick single-model profiler built into the `fox` binary. It measures:

- **Load time** — wall-clock time from `fox bench` start to first token ready
- **TTFT** — time to first token (prefill latency)
- **Generation speed** — tokens per second during the decode phase

For sustained load testing against a running server (multiple concurrent users, ramp-up, statistical percentiles), see [fox-bench](#fox-bench-standalone).

---

## Usage

```bash
# Benchmark with default prompt
fox bench llama3.2

# Benchmark with a custom prompt
fox bench llama3.2 --prompt "Write a Rust function that parses JSON"

# Average results over 3 runs
fox bench llama3.2 --runs 3

# Limit output length
fox bench llama3.2 --max-new-tokens 100

# Use a specific GGUF file
fox bench ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model name, alias, or path to a GGUF file. Resolved using the same alias → stem → prefix → contains order as `fox serve`. |

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--prompt <TEXT>` | `"Explain what a large language model is in two sentences."` | The prompt to run. Shorter prompts emphasise decode speed; longer prompts emphasise prefill speed. |
| `--runs <N>` | `1` | Number of complete inference runs to execute. Results are averaged. Use `--runs 3` or more for stable numbers. |
| `--max-new-tokens <N>` | `200` | Maximum tokens to generate per run. |
| `--max-context-len <N>` | `4096` | Context window size. Should match the value you use with `fox serve`. |
| `--alias-file <PATH>` | `~/.config/ferrumox/aliases.toml` | Aliases TOML file, if you use short model names. |

---

## Output

```
  🦊  Llama-3.2-3B-Instruct-Q4_K_M
  ────────────────────────────────────────────

  Prompt       14 tokens
  Load         0.84 s
  TTFT         0.031 s
  Generation   74.2 tok/s  (200 tokens · 2.70s)
```

When `--runs` is greater than 1, a `Runs` line is added and all values are averaged:

```
  🦊  Llama-3.2-3B-Instruct-Q4_K_M
  ────────────────────────────────────────────

  Prompt       14 tokens
  Load         0.82 s
  TTFT         0.029 s
  Generation   75.1 tok/s  (600 tokens · 2.66s)
  Runs         3
```

---

## Interpreting results

| Metric | What it measures | Tuning levers |
|--------|-----------------|---------------|
| Load | Time to map the GGUF file into memory and initialise GPU layers | Faster storage, more GPU VRAM for full offload |
| TTFT | Prefill latency — time to process the input prompt and produce the first output token | Shorter prompt, larger batch size (server mode) |
| Generation | Decode throughput — tokens per second for the autoregressive generation phase | Quantization, context length, `--max-new-tokens` |

A low TTFT matters most for interactive chat. High generation speed matters for long outputs.

---

## Comparing models

```bash
# Compare two quantizations
fox bench llama3.2 --runs 3
fox bench ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q8_0.gguf --runs 3
```

---

## fox-bench (standalone)

`fox-bench` is a separate load-testing tool for benchmarking a **running** fox server under concurrent load. Unlike `fox bench`, it supports:

- Multiple concurrent users (workers)
- Ramp-up periods
- Statistical output: P50, P90, P99 latencies
- CSV export for charting

See [Benchmarks](../benchmarks.md) for full documentation on `fox-bench`, how to run it with Docker Compose, and comparison tables.

---

## See also

- [fox serve](./serve.md) — start an HTTP server for production use
- [fox run](./run.md) — interactive REPL and one-shot inference
- [Benchmarks](../benchmarks.md) — load testing with fox-bench, performance results
