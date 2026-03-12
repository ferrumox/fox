# fox serve

Start the HTTP inference server. This is the main command you will use to run fox as a persistent service.

```
fox serve [OPTIONS]
```

---

## Basic usage

```bash
# Start with defaults — lazy loading, port 8080
fox serve

# Pre-load a specific model at startup
fox serve --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf

# Serve on a different port
fox serve --port 11434

# Load up to 3 models simultaneously
fox serve --max-models 3

# Production setup
fox serve --json-logs --port 8080 --max-models 2 --keep-alive-secs 600
```

---

## Options

### Network

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--host <HOST>` | `FOX_HOST` | `0.0.0.0` | Address to bind to. Use `127.0.0.1` to accept local connections only. |
| `--port <PORT>` | `FOX_PORT` | `8080` | TCP port to listen on. |

### Model loading

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--model-path <PATH>` | `FOX_MODEL_PATH` | — | Path to a GGUF model file. If omitted, no model is loaded at startup (lazy loading). |
| `--max-models <N>` | `FOX_MAX_MODELS` | `1` | Maximum number of models held in memory at the same time. When this limit is reached, the least-recently-used model is evicted to make room. |
| `--keep-alive-secs <N>` | `FOX_KEEP_ALIVE_SECS` | `300` | How long (in seconds) a model stays loaded after its last request. Set to `0` to never evict models based on time. |
| `--alias-file <PATH>` | `FOX_ALIAS_FILE` | `~/.config/ferrumox/aliases.toml` | Path to the model aliases TOML file. See [Configuration](../configuration.md). |

### Inference engine

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--max-context-len <N>` | `FOX_MAX_CONTEXT_LEN` | `4096` | Maximum context length in tokens. Larger values require more KV cache memory. |
| `--max-batch-size <N>` | `FOX_MAX_BATCH_SIZE` | `32` | Maximum number of sequences processed in a single forward pass. |
| `--gpu-memory-fraction <F>` | `FOX_GPU_MEMORY_FRACTION` | `0.85` | Fraction of GPU VRAM reserved for the KV cache. Must be between 0.0 and 1.0. The remaining memory is left for model weights and other allocations. |
| `--block-size <N>` | `FOX_BLOCK_SIZE` | `16` | Number of tokens per KV cache block. Smaller blocks improve prefix cache granularity but add overhead. |
| `--swap-fraction <F>` | `FOX_SWAP_FRACTION` | `0.0` | Fraction of GPU memory reserved for CPU↔GPU KV swap. Currently a placeholder for an upcoming feature. |

### System prompt

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--system-prompt <TEXT>` | `FOX_SYSTEM_PROMPT` | `"You are a helpful assistant."` | Default system prompt injected at the start of every conversation that does not already include one. Pass an empty string (`""`) to disable injection. |

### Authentication

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--hf-token <TOKEN>` | `HF_TOKEN` | — | HuggingFace API token. Required for downloading private or gated models via `POST /api/pull`. |

### Observability

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--json-logs` | `FOX_JSON_LOGS` | `false` | Output logs as structured JSON (one object per line) instead of human-readable text. Recommended for production deployments where logs are ingested by a log aggregator. |

---

## Lazy loading

When `--model-path` is omitted, fox starts without loading any model. The first HTTP request that specifies a model by name triggers loading. This means:

- **The server starts instantly** regardless of model size
- **You can switch models at runtime** without restarting the server
- **Multiple models can be served** from the same process using `--max-models`

When a request arrives for a model that is not loaded, fox resolves the name against the files in your models directory, loads the matching file, and processes the request. Subsequent requests to the same model reuse the already-loaded engine.

```bash
fox serve --max-models 3

# First request to llama3.2 — loads the model, then answers
# First request to gemma3 — loads a second model, then answers
# First request to qwen2.5 — loads a third model, then answers
# Next request to llama3.2 — uses the already-loaded engine
# Next request to mistral — evicts the LRU model, loads mistral
```

---

## Multi-model serving

`--max-models` controls how many models can be resident in memory at once. When the limit is reached, the least-recently-used model is evicted synchronously before the new model is loaded. Eviction calls `abort()` on the background engine task and drops the `EngineEntry`, which frees KV cache blocks and model weights.

Combine `--max-models` with `--keep-alive-secs` to also evict idle models on a schedule:

```bash
# Keep up to 5 models loaded; evict any that haven't been used in 10 minutes
fox serve --max-models 5 --keep-alive-secs 600
```

The background eviction task runs every 60 seconds and evicts all models whose idle time exceeds `--keep-alive-secs`. Setting `--keep-alive-secs 0` disables time-based eviction entirely (models are only evicted when capacity is exceeded).

---

## GPU memory

The `--gpu-memory-fraction` flag controls how much of your GPU VRAM is allocated for the KV cache. Model weights are loaded first and take a fixed amount of VRAM; the KV cache pool is allocated from the remaining available memory, up to the specified fraction.

**Recommended values:**

| Scenario | Value |
|----------|-------|
| Single model, plenty of VRAM | `0.85` (default) |
| Multiple models, limited VRAM | `0.6` – `0.7` |
| Large context window | `0.90` – `0.95` |
| CPU inference | `0.0` (all KV cache goes to RAM) |

If you increase `--max-context-len` significantly, you may need to also increase `--gpu-memory-fraction` to ensure there are enough KV cache blocks to fill the context.

---

## System prompt

The default system prompt (`"You are a helpful assistant."`) is injected only when a request does not include a `system` message. If a request already contains a message with `role: "system"`, fox uses that message as-is and does not inject the default.

To disable the default system prompt entirely:

```bash
fox serve --system-prompt ""
```

To set a custom default:

```bash
fox serve --system-prompt "You are a concise technical assistant. Answer in plain text only."
```

The system prompt can also be set per-request by including a `system` message in the conversation.

---

## Graceful shutdown

fox handles `SIGTERM` and `SIGINT` (Ctrl+C) gracefully. On receiving either signal:

1. The HTTP server stops accepting new connections
2. In-flight requests are allowed to complete
3. The engine is shut down cleanly
4. The process exits with code 0

This means you can send `SIGTERM` to fox and it will drain active requests before exiting, which is important in containerised deployments where the orchestrator sends SIGTERM before SIGKILL.

---

## Examples

### Local development

```bash
fox serve
```

Starts the server on port 8080. Load a model by sending any request that names it.

### Serve a specific model, expanded context

```bash
fox serve \
  --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --max-context-len 8192 \
  --gpu-memory-fraction 0.9
```

### Multi-model server for a team

```bash
fox serve \
  --max-models 4 \
  --keep-alive-secs 900 \
  --alias-file ~/.config/ferrumox/aliases.toml \
  --port 8080
```

### Production deployment

```bash
fox serve \
  --json-logs \
  --host 0.0.0.0 \
  --port 8080 \
  --max-models 2 \
  --keep-alive-secs 600 \
  --gpu-memory-fraction 0.85 \
  --max-context-len 4096 \
  --hf-token "$HF_TOKEN"
```

---

## See also

- [Configuration](../configuration.md) — config file and aliases
- [fox run](./run.md) — run inference directly without a server
- [Deployment](../deployment.md) — Docker, systemd, reverse proxy
