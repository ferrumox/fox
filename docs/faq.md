# FAQ

Frequently asked questions, organized by topic.

---

## General

### What is fox?

fox is a local LLM inference server written in Rust. It loads GGUF models from disk, runs them on your CPU or GPU using llama.cpp, and exposes two HTTP APIs: one compatible with the OpenAI SDK and one compatible with Ollama. Any tool that works with either of those works with fox.

### How is fox different from Ollama?

| | fox | Ollama |
|--|-----|--------|
| Language | Rust | Go + Python |
| Continuous batching | Yes | No |
| Prefix caching | Block-level (vLLM-style) | No |
| Multi-model LRU | Yes | Yes |
| OpenAI API | Yes | Yes |
| Ollama API | Yes | Yes |
| Function calling | Yes | Yes |
| Prometheus metrics | Yes | No |

The key functional difference is that fox uses continuous batching and block-level prefix caching, which means throughput under concurrent load is significantly higher. See [Benchmarks](./benchmarks.md) for numbers.

### Can I use my existing Ollama models with fox?

No — Ollama stores models in its own format (`.bin` manifests in `~/.ollama/models/`). fox uses GGUF files stored in `~/.cache/ferrumox/models/`. You will need to re-download models using `fox pull`.

### Does fox work without a GPU?

Yes. fox detects and uses CUDA (NVIDIA), Metal (Apple Silicon), or Vulkan at runtime and falls back to CPU if none are available. CPU inference is much slower — a 7B model may generate 2–5 tok/s on CPU vs. 60–100+ tok/s on a mid-range GPU.

### What model formats does fox support?

GGUF only. EXL2, AWQ, GPTQ, and safetensors formats are not supported.

### Is fox production-ready?

fox v1.0.0 is stable. It is used in production deployments with Docker and systemd. The API is stable — changes between versions follow semantic versioning.

---

## Models

### What quantization should I choose?

A good starting point:

| Quantization | Quality | Size (7B) | Recommendation |
|---|---|---|---|
| Q2_K | Low | ~2.7 GB | Only if VRAM is very limited |
| Q4_K_M | Good | ~4.1 GB | **Best balance — start here** |
| Q5_K_M | Better | ~5.0 GB | If you have headroom |
| Q8_0 | Near-lossless | ~7.7 GB | For quality-critical use |
| F16 | Lossless | ~14 GB | Rarely needed |

`Q4_K_M` is the default when fox auto-selects a quantization during `fox pull`.

### What model size fits in my GPU?

Rule of thumb: quantized model size + ~20–30% for KV cache.

| VRAM | Fits comfortably |
|------|-----------------|
| 4 GB | 3B Q4_K_M |
| 8 GB | 7B Q4_K_M, 3B Q8_0 |
| 16 GB | 13B Q4_K_M, 7B Q8_0 |
| 24 GB | 13B Q8_0, 34B Q4_K_M |
| 48 GB | 70B Q4_K_M |

These are estimates. Exact fit depends on context length — a 32K context uses much more KV cache than a 4K context.

### Can I serve multiple models at the same time?

Yes. fox uses an LRU cache to keep multiple models loaded simultaneously:

```bash
fox serve --max-models 3
```

Models are loaded on first request and evicted from memory (after a keep-alive timeout) when the slot is needed for a new model. See [fox serve](./cli/serve.md) for details.

### Where are models stored?

By default: `~/.cache/ferrumox/models/`

You can change this with `--models-dir`:

```bash
fox serve --models-dir /data/models
```

Or set `FOX_MODELS_DIR` in your environment or config file.

### How do I give a model a short name?

Use aliases:

```bash
fox alias set llama3 Llama-3.2-3B-Instruct-Q4_K_M
```

The alias is then usable in all API calls and CLI commands. See [fox alias](./cli/alias.md).

---

## API

### Can I use fox without an API key?

Yes — by default, fox does not require authentication. To add authentication:

```bash
fox serve --api-key my-secret-key
```

Clients then need to send `Authorization: Bearer my-secret-key`.

### Is there rate limiting?

fox does not implement rate limiting itself. For production deployments, add rate limiting at the reverse proxy level (nginx, Caddy, Traefik). See [Deployment](./deployment.md).

### Does fox support function calling / tool use?

Yes. Pass a `tools` array in your `/v1/chat/completions` request. fox injects the tool definitions into the system message and parses tool calls from the model's output. See [OpenAI API Reference](./api/openai.md#function-calling) for examples.

### Does fox support structured output (JSON mode)?

Yes. Set `"response_format": {"type": "json_object"}` in your request. fox injects a system message instructing the model to respond in JSON.

### Can I use the OpenAI Python SDK with fox?

Yes. Point the base URL at your fox server:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="unused",  # fox doesn't require a key by default
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

### What Ollama API endpoints does fox support?

fox supports all Ollama endpoints that are relevant for inference: `POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `GET /api/ps`, `POST /api/show`, `DELETE /api/delete`, `POST /api/pull`, `POST /api/embed`, `GET /api/version`. See [Ollama API Reference](./api/ollama.md).

---

## Performance

### Why is the first token slow?

If the model is not pre-loaded, fox loads it on the first request (lazy loading). Subsequent requests to the same model are fast. To always pre-load a model:

```bash
fox serve --model-path ~/.cache/ferrumox/models/my-model.gguf
```

### What is prefix caching and does it help me?

When multiple requests share the same prefix (e.g., a shared system prompt), fox reuses the KV cache blocks for that prefix instead of recomputing them. This reduces TTFT for the second and subsequent requests with the same prefix. It is enabled automatically — no configuration needed.

### Why is single-request throughput lower than the benchmarks?

The benchmark numbers in [Benchmarks](./benchmarks.md) are measured with multiple concurrent workers, which allows continuous batching to combine requests. A single sequential request does not benefit from batching. Under real concurrent load, throughput increases significantly.

### Does fox support multi-GPU?

Not in the current release. GPU distribution across multiple devices is on the roadmap.

---

## Deployment

### Docker or systemd?

Both are supported and production-tested.

- **Docker**: easier to isolate dependencies, good for multi-service setups with Docker Compose
- **systemd**: lower overhead, simpler on single-server deployments, integrates with existing Linux service management

See [Deployment](./deployment.md) for both setups.

### Can I run fox behind a reverse proxy?

Yes — and you should for production (TLS termination, rate limiting, logging). See [Deployment — Reverse Proxy](./deployment.md#reverse-proxy). The key requirement is disabling response buffering so streaming works correctly.

### How do I expose fox to the internet securely?

1. Run fox on `--host 127.0.0.1` (bind only to localhost)
2. Use nginx or Caddy as a reverse proxy with TLS
3. Enable API key authentication (`--api-key`)
4. Add rate limiting at the proxy level

See [Deployment](./deployment.md) for a complete nginx example with TLS.

### Does fox support Kubernetes?

Yes. See [Deployment — Kubernetes](./deployment.md#kubernetes) for a minimal Deployment + Service manifest.

---

## Troubleshooting

### Something isn't working — where do I start?

See [Troubleshooting](./troubleshooting.md) for a full list of common problems and solutions.

For issues not covered there, open a [GitHub issue](https://github.com/ferrumox/fox/issues) with:
- `fox --version` output
- OS and GPU
- The exact error message
- Steps to reproduce
