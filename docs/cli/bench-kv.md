# fox bench-kv

Compare KV cache quantization types side by side on a single model. Weights are loaded once; a fresh llama.cpp context is created for each KV type — no weight reload between runs.

```
fox bench-kv <MODEL> [OPTIONS]
```

---

## Basic usage

```bash
# Compare all supported types with default settings
fox bench-kv llama3.2

# Compare a specific subset
fox bench-kv llama3.2 --types f16,turbo3,turbo4

# Average over 3 runs with a larger context
fox bench-kv llama3.2 --types f16,q8_0,turbo3 --runs 3 --max-context-len 8192

# Use a custom prompt
fox bench-kv llama3.2 --prompt "Explain how transformers work."
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | (required) | Model name, alias, or path to a GGUF file. |
| `--types <LIST>` | `f16,q8_0,turbo3,turbo4,turbo2` | Comma-separated KV cache types to benchmark. |
| `--runs <N>` | `2` | Number of inference passes per type. Results are averaged. |
| `--max-context-len <N>` | auto | Context length used for each run. Defaults to the model's trained context. |
| `--max-new-tokens <N>` | `300` | Maximum tokens to generate per pass. |
| `--prompt <TEXT>` | `"Explain what a large language model is in two sentences."` | Prompt used for each inference run. |
| `--main-gpu <N>` | `0` | Primary GPU index. |
| `--split-mode <MODE>` | `layer` | Multi-GPU split mode: `none`, `layer`, `row`. |
| `--tensor-split <RATIOS>` | auto | Comma-separated VRAM proportions (e.g. `"3,1"`). |
| `--moe-cpu` | `false` | Offload MoE expert layers to CPU RAM. |

---

## Supported KV types

| Type | Bits/token | Compression vs f16 | Notes |
|------|-----------|-------------------|-------|
| `f16` | 16 | 1× (baseline) | Default, full precision |
| `q8_0` | 8 | ~2× | Good quality, half the VRAM |
| `q4_0` | 4 | ~4× | Larger quality trade-off |
| `turbo3` | ~3.1 | **~4.9×** | TurboQuant — recommended sweet spot |
| `turbo4` | ~4.25 | ~3.8× | TurboQuant — higher quality |
| `turbo2` | ~2.1 | ~6.4× | TurboQuant — maximum compression |

TurboQuant types (`turbo2`, `turbo3`, `turbo4`) require flash attention and `head_dim % 128 == 0`. Most modern models qualify.

---

## Sample output

```
  🦊  Llama-3.2-3B-Instruct-Q4_K_M  ·  KV cache benchmark
  ──────────────────────────────────────────────────────────────

  type        blocks    ctx-tokens     tok/s     TTFT   vs f16
  ──────────────────────────────────────────────────────────────
    f16          512         8 192     142.3    0.089s  baseline
    q8_0        1024        16 384     144.1    0.091s     2.0×
    q4_0        2048        32 768     141.8    0.090s     4.0×
  ✦ turbo3      2510        40 160     143.7    0.090s     4.9×
  ✦ turbo4      1966        31 456     143.2    0.089s     3.8×
  ✦ turbo2      3277        52 432     140.9    0.092s     6.4×

  blocks = available KV cache slots  ·  ctx-tokens = blocks × 16
  ✦ = TurboQuant (requires flash attention + head_dim % 128 == 0)
```

The `vs f16` column shows how many more KV blocks are available compared to the f16 baseline. More blocks means a longer effective context window.

---

## See also

- [fox serve](./serve.md) — `--type-kv`, `--type-k`, `--type-v` flags
- [Features — TurboQuant KV cache](../features.md#turboquant-kv-cache)
- [Benchmarks](../benchmarks.md)
