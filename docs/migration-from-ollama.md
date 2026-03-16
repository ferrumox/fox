# Migrating from Ollama

fox is API-compatible with Ollama, so most tools that work with Ollama work with fox without code changes. This guide covers the practical differences you will encounter when switching.

---

## Quick checklist

- [ ] Change the base URL from `http://localhost:11434` to `http://localhost:8080`
- [ ] Download models with `fox pull` (fox cannot read Ollama's model format)
- [ ] (Optional) Set up aliases to match your existing Ollama model names
- [ ] (Optional) Configure authentication if fox will be network-accessible

---

## Port and URL

Ollama listens on port **11434** by default. fox listens on port **8080**.

```bash
# Ollama
curl http://localhost:11434/api/tags

# fox (same endpoint, different port)
curl http://localhost:8080/api/tags
```

If you cannot change the port in your tools, start fox on port 11434:

```bash
fox serve --port 11434
```

---

## Downloading models

Ollama downloads and manages models in its own internal format stored in `~/.ollama/`. fox uses standard GGUF files stored in `~/.cache/ferrumox/models/`.

**You cannot reuse Ollama's downloaded models.** Re-download them with fox:

```bash
# Ollama equivalent: ollama pull llama3.2
fox pull llama3.2

# Equivalent with explicit HuggingFace repo
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF
```

fox auto-selects the best quantization (Q4_K_M by default). To choose a specific one:

```bash
fox pull llama3.2:8b-q8
```

---

## Command equivalents

| Ollama | fox | Notes |
|--------|-----|-------|
| `ollama pull llama3.2` | `fox pull llama3.2` | Same name format |
| `ollama list` | `fox list` | Lists GGUF files, not Ollama manifests |
| `ollama show llama3.2` | `fox show llama3.2` | Shows architecture, quantization, size |
| `ollama rm llama3.2` | `fox rm llama3.2` | Removes from models dir |
| `ollama run llama3.2` | `fox run llama3.2` | Interactive REPL |
| `ollama serve` | `fox serve` | Starts HTTP server |
| `ollama ps` | `fox ps` | Lists loaded models |

---

## Model names

Ollama uses names like `llama3.2`, `mistral`, `codellama:13b`. fox uses the filename stem of the GGUF file, e.g. `Llama-3.2-3B-Instruct-Q4_K_M`.

To use the same short names as Ollama, define aliases:

```bash
fox alias set llama3.2 Llama-3.2-3B-Instruct-Q4_K_M
fox alias set mistral Mistral-7B-Instruct-v0.3-Q4_K_M
fox alias set codellama Codellama-13B-Instruct-Q4_K_M
```

After that, your existing API calls using those names work without changes.

---

## API differences

### Endpoints

All Ollama endpoints fox supports are at the same paths:

- `POST /api/chat`
- `POST /api/generate`
- `GET /api/tags`
- `GET /api/ps`
- `POST /api/show`
- `DELETE /api/delete`
- `POST /api/pull`
- `POST /api/embed`
- `GET /api/version`

See [Ollama API Reference](./api/ollama.md) for full documentation.

### `POST /api/pull` behaviour

Ollama's `pull` is synchronous (streams NDJSON progress events). fox's `/api/pull` behaves the same way — progress events are streamed as the download proceeds.

One difference: fox downloads from HuggingFace Hub, not the Ollama registry. Pass the model ID in the same format as `fox pull`:

```bash
curl -X POST http://localhost:8080/api/pull \
  -d '{"name": "bartowski/Llama-3.2-3B-Instruct-GGUF"}'
```

### `POST /api/show`

The response format matches Ollama's. The `details` field includes `architecture` and `quantization_level` parsed from the GGUF filename.

### Response format

fox's NDJSON responses match Ollama's schema. If you parse response fields, check the [Ollama API Reference](./api/ollama.md) for the exact field names fox uses.

---

## OpenAI API

If you currently use Ollama's OpenAI-compatible endpoint (`/v1/...`), fox supports the same endpoints. Change the base URL:

```python
# Before (Ollama)
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# After (fox)
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
```

---

## Open WebUI

Open WebUI uses the Ollama backend protocol. Change the Ollama URL in Open WebUI settings:

- **Old**: `http://localhost:11434`
- **New**: `http://localhost:8080`

Or, if fox is in a Docker container visible to Open WebUI:

```
http://fox:8080
```

See [Connecting Open WebUI to fox](../examples/openwebui.md) for a complete setup guide.

---

## LangChain / LlamaIndex

These frameworks have Ollama integration classes that accept a `base_url` parameter. Change it to point at fox:

```python
# LangChain
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:8080",
)

# LlamaIndex
from llama_index.llms.ollama import Ollama

llm = Ollama(model="llama3.2", base_url="http://localhost:8080")
```

---

## Features not in Ollama

After migrating, you gain access to features fox has that Ollama does not:

- **Continuous batching** — concurrent requests share a batch; throughput scales with concurrency
- **Block-level prefix caching** — shared system prompts are cached and reused
- **Prometheus metrics** — `GET /metrics` exposes request counts, TTFT histograms, KV cache utilization
- **`fox bench`** — built-in single-model profiler
- **`fox ps`** — live view of loaded models with KV cache usage

---

## See also

- [Quickstart](./quickstart.md) — get fox running in 5 minutes
- [Ollama API Reference](./api/ollama.md) — full endpoint documentation
- [FAQ](./faq.md) — common questions including Ollama comparisons
- [Troubleshooting](./troubleshooting.md) — if something doesn't work
