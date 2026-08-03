# Benchmarks

This page covers what has actually been measured, on what hardware, and how to
reproduce it. It also says where fox loses, because a benchmark page that only reports
wins is not useful for deciding whether to adopt something.

---

## What was measured

**Hardware:** AMD Radeon 890M (integrated), Vulkan backend
**Model:** Llama-3.2-1B-Instruct Q8_0
**Reference:** `llama-server` built from the same vendored llama.cpp fox links against

Both servers come from the same llama.cpp checkout and the same toolchain, so the
comparison is between the two serving layers rather than between two versions of an
engine. Exactly one server runs at a time (ggml's thread pool spin-waits, so an idle
second server still burns cores and skews the arm under test), arms alternate each round
so thermal drift cannot systematically favour one, and every figure below comes from 3
rounds with disjoint ranges.

### Concurrent requests behind a shared prompt

Eight and sixteen clients arriving together, each carrying the same 1856-token system
prompt and a different short question. This is the shape of agent and RAG traffic.

| | fox | llama-server | |
|---|---|---|---|
| 8 clients, cold — TTFT p50 | **1129 ms** | 4550 ms | fox 4.0× |
| 16 clients, cold — TTFT p50 | **1402 ms** | 8064 ms | fox 5.75× |
| 16 clients — whole burst wall clock | **3.8 s** | 16.2 s | |
| 8 clients, warm — TTFT p50 | **52 ms** | 193 ms | fox 3.7× |

Doubling the clients costs fox 24% more cold TTFT and `llama-server` 79%. Fox adds one
short suffix of prefill per extra client; `llama-server` adds a whole prompt.

The reason is structural rather than tuning. Slot affinity reuses an *idle* sequence, so
when requests sharing a prompt arrive together there is nothing idle to inherit and each
prefills the same tokens. `llama-server` skips busy slots in both its similarity pass and
its LRU fallback, and reports `cached_tokens` 0 on this workload. Fox copies the shared
prefix out of a sibling that is already decoding.

The warm row is the fair floor: once the earlier sequences go idle, both servers reuse
the prefix, and both report the same `cached_tokens`.

### Single requests with short prompts

| | fox | llama-server |
|---|---|---|
| 4 clients, unrelated short prompts — throughput | 96% | baseline |

Fox is about 4% behind here. Fox wraps llama.cpp, so a request decoding on its own runs
the same kernels `llama-server` runs; the gap is fox's serving layer, not the model. This
workload cannot see prompt reuse because there is no prompt worth reusing. If your traffic
looks like this, fox will not make it faster.

### Reproduce

```bash
scripts/ab_shared_prefix.sh    # concurrent burst behind a shared prompt
scripts/ab_bench.sh            # decode-bound throughput
```

Comparisons against Ollama are pending re-measurement. Figures previously published here
carried no round count or methodology and are not repeated.

## How these benchmarks were wrong before they were right

Kept here because both failures produced confident, plausible, wrong answers rather than
obvious breakage.

**A binary that predated the feature.** The first run of the shared-prompt benchmark
reported `llama-server` ahead and fox reusing nothing. The fox arm was a prebuilt bundle
from 31 minutes before the feature landed. What made it convincing was that the warm row
still looked right, because the slot table it depends on predated that bundle. If an arm
shows no effect at all, check the binary's timestamp against the commit.

**A metric that could not move.** Pool usage was read as the sum of per-slot block counts.
A shared block is counted once by every slot referencing it, so that sum cannot fall when
sharing works. It read as "sharing changed nothing" across two measurements. `/slots` now
reports `kv_blocks_used` and `kv_blocks_total`, which is the pool's own occupancy: 282 →
72 blocks on 6 clients behind a 673-token prompt.

**A prompt that did not fit.** An oversized prompt does not fail the same way on both
servers. `llama-server` returns 400; fox rolls the context window, which disables prompt
reuse. That would have read as "fox cannot reuse prompts". The driver now reports measured
prompt tokens and warns when they exceed the per-sequence context.

Full detail is in `docs/design/rocm-benchmarking-2026-08.md`.

---

## fox-bench

fox ships with a standalone benchmark tool: `fox-bench`. It sends concurrent requests to an inference server and reports latency percentiles, throughput, and token rates.

### Basic usage

```bash
fox-bench \
  --url http://localhost:8080 \
  --model llama3.2 \
  --concurrency 8 \
  --requests 100
```

### Options

| Flag | Default | Description |
|------|---------|---|
| `--url <URL>` | `http://localhost:8080` | Server URL to benchmark. |
| `--model <NAME>` | (required) | Model name to use in requests. |
| `--concurrency <N>` | `4` | Number of parallel workers. |
| `--requests <N>` | `50` | Total number of requests to send. |
| `--max-tokens <N>` | `128` | Max tokens to generate per request. |
| `--prompt <TEXT>` | (built-in) | Custom prompt to send in each request. |
| `--compare-url <URL>` | — | Second server to benchmark side-by-side. |
| `--label <TEXT>` | `"server"` | Label for the primary server in output. |
| `--compare-label <TEXT>` | `"compare"` | Label for the comparison server. |
| `--output <FORMAT>` | `table` | Output format: `table` or `json`. |
| `--warmup <N>` | `5` | Number of warmup requests to send before measuring. |

### Side-by-side comparison

Compare fox against another server (or another fox instance with different settings):

```bash
fox-bench \
  --url http://localhost:8080 \
  --label "fox" \
  --compare-url http://localhost:11434 \
  --compare-label "ollama" \
  --model llama3.2 \
  --concurrency 4 \
  --requests 50
```

Output:

```
fox vs ollama  •  llama3.2  •  concurrency=4  •  50 requests

                fox         ollama      improvement
TTFT P50        ...         ...         ...
TTFT P95        ...         ...         ...
Latency P50     ...         ...         ...
Latency P95     ...         ...         ...
Throughput      ...         ...         ...

(shape of the output; run it to get your own numbers)
```

### JSON output (for CI)

```bash
fox-bench \
  --url http://localhost:8080 \
  --model llama3.2 \
  --concurrency 8 \
  --requests 100 \
  --output json
```

```json
{
  "_comment": "illustrative values, not measurements — run fox-bench for real ones",
  "url": "http://localhost:8080",
  "model": "llama3.2",
  "concurrency": 8,
  "total_requests": 100,
  "ttft_p50_ms": 87,
  "ttft_p95_ms": 134,
  "latency_p50_ms": 412,
  "latency_p95_ms": 823,
  "throughput_tokens_per_sec": 312.4,
  "total_tokens_generated": 12800,
  "errors": 0
}
```

Use this output to track performance regressions in CI:

```bash
result=$(fox-bench --output json ...)
ttft=$(echo $result | jq '.ttft_p50_ms')
if [ "$ttft" -gt 200 ]; then
  echo "TTFT regression: ${ttft}ms > 200ms threshold"
  exit 1
fi
```

---

## Reproducible benchmark script

The repository includes `scripts/benchmark.sh`, a script that runs a controlled benchmark with fixed parameters for reproducible comparison across hardware or versions.

```bash
./scripts/benchmark.sh
```

The script:
1. Checks that `fox` and `fox-bench` are on your PATH
2. Pulls the reference model if not already downloaded
3. Starts a fox server with fixed settings
4. Runs `fox-bench` with standard parameters
5. Prints the results table
6. Shuts down the server

---

## Performance tuning guide

### GPU memory

More KV cache blocks = more concurrent sequences = higher throughput. If your model weights leave significant free VRAM, increase `--gpu-memory-fraction`:

```bash
fox serve --gpu-memory-fraction 0.92
```

Check `fox_kv_cache_usage` in `/metrics` to see if you are memory-constrained. If it frequently reaches 0.9+, you are likely queuing requests due to memory pressure.

### Context length

Each token of context occupies KV cache space for the duration of the request. Shorter contexts allow more concurrent sequences. If your workload uses short conversations, reduce `--max-context-len` to free up blocks for more parallel requests:

```bash
# Chat workload with short exchanges: 2048 tokens is often enough
fox serve --max-context-len 2048 --gpu-memory-fraction 0.85
```

### Batch size

`--max-batch-size` limits how many sequences are processed in a single forward pass. The default (32) is appropriate for most VRAM capacities. On cards with less than 8 GB, reduce it to 16 to avoid OOM:

```bash
fox serve --max-batch-size 16
```

On high-VRAM cards (40+ GB) serving many concurrent users, you can increase it:

```bash
fox serve --max-batch-size 64
```

### Prefix cache hit rate

Monitor `fox_prefix_cache_hit_ratio` in Prometheus. If it is low, consider:

- Using a consistent system prompt across requests (shared prefixes are cached)
- Keeping the same conversation structure across users
- Reducing `--block-size` to 8 for finer-grained caching (at some overhead cost)

High prefix cache hit rates directly reduce TTFT. A 70% hit rate means 70% of prompt tokens skip the forward pass entirely.

### Multi-model workloads

For workloads that mix several models, tune `--max-models` and `--keep-alive-secs` together:

```bash
# Serve 4 models, evict after 5 minutes idle
fox serve --max-models 4 --keep-alive-secs 300
```

Each loaded model occupies VRAM for its weights. With 4 models loaded simultaneously on a 24 GB card, you have less KV cache budget per model. Watch `fox_kv_cache_usage` per model and adjust the balance between `--max-models` and `--gpu-memory-fraction`.

---

## Expected performance by hardware

**These are estimates, not measurements.** Nobody has run these configurations for this
project; they are order-of-magnitude figures to help size hardware, and they should not be
quoted as fox benchmark results. The only measured numbers on this page are in
[What was measured](#what-was-measured).

### Llama-3.2-3B Q4_K_M (2 GB model)

| Hardware | Single request tok/s | Concurrent (4) tok/s |
|----------|---------------------|----------------------|
| RTX 3090 (24 GB) | ~120 | ~310 |
| RTX 4080 (16 GB) | ~110 | ~280 |
| RTX 3060 (12 GB) | ~85 | ~190 |
| M2 Pro (20 GB unified) | ~55 | ~120 |
| CPU (modern, 32 GB RAM) | ~12 | ~25 |

### Llama-3.1-8B Q4_K_M (5 GB model)

| Hardware | Single request tok/s | Concurrent (4) tok/s |
|----------|---------------------|----------------------|
| RTX 3090 (24 GB) | ~65 | ~140 |
| RTX 4080 (16 GB) | ~55 | ~115 |
| RTX 3060 (12 GB) | ~30 | ~65 |
| M2 Pro (20 GB unified) | ~28 | ~58 |

Numbers vary by prompt length, context length setting, and system load. Use `fox-bench` on your hardware for accurate measurements.
