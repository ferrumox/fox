# How fox works

This page explains the core techniques fox uses to achieve high throughput and low latency. Understanding these concepts is not required to use fox, but it helps you tune the server for your hardware and workload.

---

## Continuous batching

Traditional inference servers process one request at a time: they wait for request A to finish before starting request B. This means the GPU sits idle for most of a request's lifetime — specifically, during the prompt processing phase of new requests and between generations.

fox uses **continuous batching**: multiple sequences are processed in a single forward pass, and new requests are inserted into the batch mid-generation without waiting for current requests to complete.

The mechanics work like this:

1. Each request occupies one or more **slots** in the batch
2. After every forward pass, the scheduler checks for new requests waiting in the queue
3. If there are free slots, the new requests are added to the batch for the next pass
4. If slots are full (all at `--max-batch-size`), new requests wait in a queue

This approach keeps the GPU saturated regardless of individual request completion times. Under load (multiple concurrent requests), continuous batching typically delivers 2–4× higher throughput than sequential processing.

When KV cache memory runs out and a new request cannot be admitted, fox uses **LIFO preemption**: the most recently added sequence is suspended and its KV cache is freed, allowing the system to continue making progress on older sequences. Preempted sequences are re-queued and reprocessed once memory becomes available.

---

## Prompt reuse

Every sequence remembers the tokens resident in its KV cache, including the ones it
generated. An arriving request is matched to the sequence sharing the longest prefix with
it and skips the prefill for that overlap. Matching is token-exact rather than aligned to
block boundaries, so a partial block at the end of the shared prefix still counts.

In a chat, this means the second turn does not re-read the first, and neither does the
tenth. The previous answer is part of the next prompt, and it is already resident.

**Copying from a live sequence.** Reuse of this kind normally only works against an
*idle* sequence, which is exactly when it is least useful: when eight requests carrying
the same system prompt arrive at once, none of them is idle and all eight prefill the same
tokens. Fox copies the shared prefix out of a sibling that is already decoding. Under a
unified KV cache the copy shares cells rather than duplicating the buffer, so it is
metadata work rather than a memory copy. Measured against `llama-server` on the same
llama.cpp: 4.0× faster time to first token at 8 concurrent clients behind a 1856-token
system prompt, 5.75× at 16. See [Benchmarks](benchmarks.md).

**Paid for once.** Sequences sharing a prefix share the block budget for it instead of
each reserving a copy, so the server admits as much concurrency as the hardware actually
holds rather than as much as a duplicated count suggests.

**Host-RAM cache.** `--cache-ram <MiB>` keeps reclaimed sequences in system memory instead
of discarding them, so a conversation can stay warm without holding GPU blocks. Off by
default.

`--kv-reuse false` disables all of it, which exists so the behaviour can be A/B'd.

`GET /slots` shows what each sequence currently holds and the KV pool's real occupancy.

---

## PagedAttention

Traditional attention implementations allocate contiguous memory for each sequence's KV cache, sized for the maximum possible context length. This wastes memory on sequences that don't use their full context, and it also creates fragmentation that limits how many sequences can be processed simultaneously.

fox implements **PagedAttention**: logical KV cache sequences are mapped to non-contiguous physical blocks, using a page table (similar to virtual memory in operating systems).

Key properties:

- **No pre-allocation waste** — a sequence only occupies the blocks it has actually filled
- **No fragmentation** — physical blocks can be allocated from a pool without requiring contiguous regions
- **Reference counting** — blocks shared between sequences (via prefix caching) are reference-counted; a block is only freed when all sequences that reference it have finished
- **Copy-on-write** — when a sequence that shares a prefix block needs to write to that block (at the boundary of the shared prefix), a new block is allocated and the data is copied before writing; the original shared block is preserved

The block pool is allocated at startup based on `--gpu-memory-fraction`. The total number of available blocks determines how many concurrent sequences can be in flight and how long each context can be.

---

## Multi-GPU support

fox automatically distributes model layers across all available GPUs when multiple GPUs are present — no configuration required for the default behaviour.

### Split modes

| Mode | Flag value | Description |
|------|-----------|-------------|
| Layer split | `layer` (default) | Consecutive transformer layers are assigned to GPUs proportionally to their available VRAM. Best for most workloads. |
| Row split | `row` | Each layer is split across GPUs using tensor parallelism. Lower latency for very large models but higher interconnect overhead. |
| Single GPU | `none` | Disables distribution; the model runs entirely on `--main-gpu`. |

```bash
# Layer split, auto-balance VRAM
fox serve --split-mode layer

# Two GPUs, manual 3:1 ratio
fox serve --split-mode layer --tensor-split "3,1"

# Use GPU 1 as primary for single-GPU mode
fox serve --split-mode none --main-gpu 1
```

### MoE CPU offload

For Mixture-of-Experts models (DeepSeek, Mixtral, and others), fox can offload the expert weight tensors to CPU RAM while keeping the attention layers on GPU:

```bash
fox serve --moe-cpu
```

This lets you run large MoE models on hardware where the full model wouldn't fit in VRAM. Throughput is lower than full-GPU operation because expert weights must be transferred from RAM on each forward pass, but quality is identical.

---

## KV cache quantization

Quantizing the KV cache reduces its memory footprint, letting you fit a longer context (or more concurrent requests) into the same VRAM, at a small quality cost.

| Type | Bits/token | Compression vs f16 | Recommended use |
|------|-----------|-------------------|-----------------|
| `f16` | 16 | 1× | Default — full precision, no quality loss |
| `q8_0` | ~8.5 | ~1.9× | Near-lossless; safe everywhere |
| `q4_0` | ~4.5 | ~3.5× | Aggressive compression; some quality degradation |

These are the standard llama.cpp KV types, available on every backend.

```bash
# 8-bit KV cache (near-lossless, ~2× longer context)
fox serve --type-kv q8_0

# Asymmetric: full precision for K, 4-bit for V
fox serve --type-k f16 --type-v q4_0
```

Use `fox bench-kv <model>` to measure the throughput and context gain of each KV type on your specific hardware.

---

## Multi-model serving

fox can hold multiple models in memory simultaneously, routing each request to the appropriate model by the `model` field in the request.

### Loading and eviction

Models are loaded lazily on the first request that names them. The `--max-models` flag controls capacity. When capacity is reached and a new model is needed, the **least-recently-used** model is evicted:

1. The LRU model's background engine task is aborted
2. The `EngineEntry` is dropped, which frees KV cache blocks and model weights
3. The new model is loaded and added to the registry

The LRU tracking is maintained in a concurrent `DashMap` (last-used timestamps) and an `LruCache`. The `get_or_load` function updates the LRU position on every access.

### Time-based eviction

In addition to capacity-based eviction, fox runs a background task every 60 seconds that evicts any model idle for longer than `--keep-alive-secs`. This prevents models from occupying memory indefinitely when traffic is sporadic.

Setting `--keep-alive-secs 0` disables time-based eviction — models are only evicted when the capacity limit is exceeded.

### Model name resolution

When a request arrives with a model name, fox resolves it in four steps:

1. **Alias match** — checks the `aliases.toml` file for an exact key
2. **Exact stem match** — checks for a file whose name (without `.gguf`) exactly matches
3. **Starts-with match** — checks for a file whose name starts with the given string
4. **Contains match** — checks for a file whose name contains the given string

This allows short, convenient names in API requests while still supporting full filenames.

---

## Request cancellation

When a client disconnects mid-stream, fox detects the broken connection immediately (`send()` returns an error on the SSE channel). At that point:

1. `clear_sequence()` is called to remove the sequence from the scheduler
2. The sequence is marked as `Finished(Preempt)`
3. All KV cache blocks occupied by that sequence are freed immediately

This means a disconnected client does not hold KV cache memory hostage. Under high load with many short-lived connections, this can significantly reduce latency for other requests.

---

## Sampling

At each generation step, fox samples the next token from the model's output logits. Several sampling strategies are supported and can be combined:

| Strategy | Parameter | Description |
|----------|-----------|-------------|
| Greedy | `temperature: 0.0` | Always select the highest-probability token. Fully deterministic. |
| Temperature | `temperature` | Divide logits by temperature before softmax. Lower = more focused, higher = more creative. |
| Top-p (nucleus) | `top_p` | Keep only the smallest set of tokens whose cumulative probability exceeds `top_p`. Removes the long tail of unlikely tokens. |
| Top-k | `top_k` | Keep only the `k` most probable tokens. |
| Repetition penalty | `repetition_penalty` | Reduce the probability of tokens that have already appeared in the output. |
| Fixed seed | `seed` | Fix the RNG seed for reproducible output given the same prompt and parameters. |

Sampling is applied in this order: temperature scaling → top-k filter → top-p filter → repetition penalty → sample.

---

## Function calling

fox supports the OpenAI tool use specification. When a request includes a `tools` array, fox:

1. Serialises the tool schemas into a JSON description and injects it into the system prompt
2. Generates a response
3. After generation, scans the output for a JSON tool call pattern
4. If a tool call is found, parses it and returns it in the `tool_calls` field with `finish_reason: "tool_calls"`
5. If no tool call is found, returns the text response normally

This approach works with any GGUF model that has been instruction-tuned to produce tool call syntax, without requiring special model variants.

---

## API key authentication

By default, fox accepts requests from any client without authentication. To restrict access, set `FOX_API_KEY` (or `--api-key` on `fox serve`). When set, every incoming request must include:

```
Authorization: Bearer <your-key>
```

Requests with a missing or incorrect key receive `HTTP 401 Unauthorized`. The health endpoint (`GET /health`) is exempt and always reachable without authentication.

```bash
# Start with authentication enabled
FOX_API_KEY=my-secret fox serve

# Query with the key
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer my-secret" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hi"}]}'
```

OpenAI SDK clients pass the key via `api_key`:

```python
client = OpenAI(base_url="http://localhost:8080/v1", api_key="my-secret")
```

Authentication is implemented as an Axum middleware layer — it runs before any route handler, so all endpoints are protected uniformly.

---

## Structured output (JSON mode)

When `response_format: {"type": "json_object"}` is set, fox injects a system instruction telling the model to respond exclusively with a valid JSON object. This increases the probability that the output is parseable JSON but does not guarantee it for all models — instruction-tuned models produce better results than base models.

---

## Observability

fox exposes metrics via the `GET /metrics` endpoint in Prometheus format:

| Metric | Type | Description |
|--------|------|-------------|
| `fox_requests_total` | Counter | Total requests, labelled by model and status. |
| `fox_request_duration_seconds` | Histogram | End-to-end request latency. |
| `fox_time_to_first_token_seconds` | Histogram | Time from request receipt to first generated token. |
| `fox_tokens_generated_total` | Counter | Total tokens generated across all requests. |
| `fox_kv_cache_usage` | Gauge | Fraction of KV cache blocks currently occupied (0–1). |
| `fox_prefix_cache_hit_ratio` | Gauge | Ratio of prompt blocks served from the prefix cache. |
| `fox_queue_depth` | Gauge | Number of requests currently waiting for a scheduler slot. |
| `fox_active_requests` | Gauge | Number of sequences currently being decoded. |

These metrics give you visibility into GPU utilisation (`kv_cache_usage`), the effectiveness of prefix caching (`prefix_cache_hit_ratio`), and throughput characteristics (tokens/s, TTFT distribution).

---

## Speculative decoding

Fox proposes several tokens per step and verifies them in one pass, so accepted proposals
cost a fraction of a normal decode. Two proposers:

- **N-gram** (`--spec-ngram`) needs no second model. It looks for repeats of the current
  suffix in the prompt and generated text, which fits code, structured output, and any
  workload that quotes its input back.
- **Draft model** (`--draft-model`) runs a small model to propose for a larger one.

`fox bench-spec` reports the acceptance rate, which is what decides whether either is
worth enabling for your workload.

---

## Vision

Multimodal models run through llama.cpp's mtmd. Pass the projector with `--mmproj` and
send images the OpenAI way, as `image_url` content parts. `fox pull gemma4-e2b` fetches a
model and projector that work together.

---

## LoRA adapters

Load adapters at startup with `--lora-modules name=path[:scale]`. A client selects one by
naming it in the `model` field, the same way it selects a model.

`GET /lora-adapters` lists what is loaded; `POST /lora-adapters` changes an adapter's
scale at runtime, without a restart. In-flight generations keep the scale they were
admitted with.

---

## Editor and retrieval endpoints

- **`POST /infill`** — fill-in-the-middle completion, taking `input_prefix` and
  `input_suffix`. This is what code-completion plugins call.
- **`POST /rerank`** and **`/v1/rerank`** — score documents against a query. Requires
  `--reranking`, because reranking needs RANK pooling and most reranker GGUFs do not
  declare a pooling type in their metadata.
- **`POST /tokenize`**, **`/detokenize`**, **`/apply-template`** — count tokens, inspect
  what the tokenizer did, and see exactly what the chat template renders.

---

## Introspection

`GET /props` reports the loaded model, its context size, chat template, special tokens,
and the sampling defaults in effect for each API surface. `GET /slots` reports per-sequence
state, resident token counts, and KV pool occupancy.

Both exist so that when throughput looks wrong you can find out what the server actually
believes, rather than inferring it from latency.
