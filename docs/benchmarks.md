# Benchmarks

fox is designed for high throughput and low latency. This page covers measured performance on reference hardware and explains how to run your own benchmarks.

---

## Reference results

**Hardware:** NVIDIA RTX 3090 (24 GB VRAM)
**Model:** Llama-3.2-3B-Instruct Q4_K_M
**Concurrency:** 4 workers
**Requests:** 50
**Max tokens per request:** 128

| Metric | fox | Ollama | Improvement |
|--------|-----|--------|-------------|
| TTFT P50 | 87 ms | 310 ms | −72% |
| TTFT P95 | 134 ms | 480 ms | −72% |
| Latency P50 | 412 ms | 890 ms | −54% |
| Latency P95 | 823 ms | 1740 ms | −53% |
| Throughput | 312 tok/s | 148 tok/s | +111% |

**TTFT** = Time to first token (from when the server receives the request to when it sends the first token).
**Latency** = End-to-end time for the complete response.
**Throughput** = Total tokens generated per second across all concurrent workers.

The throughput improvement (2.1×) comes primarily from continuous batching — all 4 workers' requests are batched into the same forward pass rather than processed sequentially. The TTFT improvement comes from prefix caching: after the first few requests, the system prompt and initial context are already in the KV cache.

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
TTFT P50        87ms        310ms       +72%
TTFT P95        134ms       480ms       +72%
Latency P50     412ms       890ms       +54%
Latency P95     823ms       1740ms      +53%
Throughput      312 tok/s   148 tok/s   +111%
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

These are rough guidelines based on typical GGUF model sizes and hardware capabilities.

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
