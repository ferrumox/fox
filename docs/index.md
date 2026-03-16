# fox documentation

fox is a local LLM inference server written in Rust. It runs GGUF models on your hardware and exposes two complete HTTP APIs: one compatible with the OpenAI SDK and one compatible with Ollama. Any tool that works with OpenAI or Ollama works with fox.

---

## Get started

```bash
# Install
curl -L https://github.com/ferrumox/fox/releases/latest/download/fox-linux-x86_64.tar.gz | tar xz
sudo mv fox /usr/local/bin/

# Download a model
fox pull llama3.2

# Start the server
fox serve

# Send a request
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hello!"}]}'
```

→ [Full Quick Start guide](./quickstart.md)

---

## Documentation

### Getting started
- [Introduction](./introduction.md) — what fox is and how it works
- [Installation](./installation.md) — binaries, Docker, build from source
- [Quick Start](./quickstart.md) — running fox in 5 minutes

### CLI reference
- [fox serve](./cli/serve.md) — start the inference server
- [fox run](./cli/run.md) — one-shot inference and interactive REPL
- [fox pull / list / show / rm / search / models](./cli/pull.md) — model management
- [fox alias](./cli/alias.md) — manage model name aliases
- [fox bench](./cli/bench.md) — measure load time and throughput

### Configuration
- [Configuration](./configuration.md) — config file, environment variables, model aliases

### API reference
- [OpenAI-Compatible API](./api/openai.md) — `/v1/chat/completions`, `/v1/embeddings`, function calling, streaming
- [Ollama-Compatible API](./api/ollama.md) — `/api/chat`, `/api/generate`, `/api/tags`, `/api/pull`

### Guides
- [How fox works](./features.md) — continuous batching, prefix caching, PagedAttention, multi-model serving
- [Integrations](./integrations.md) — OpenAI SDK, LangChain, Open WebUI, Continue.dev, curl
- [Benchmarks](./benchmarks.md) — performance results, fox-bench, tuning guide
- [Deployment](./deployment.md) — Docker, Docker Compose, systemd, nginx, Kubernetes
- [Migrate from Ollama](./migration-from-ollama.md) — switching from Ollama to fox
- [Troubleshooting](./troubleshooting.md) — common problems and solutions
- [FAQ](./faq.md) — frequently asked questions

---

## Why fox?

| | fox | Ollama |
|--|-----|--------|
| Language | Rust | Go + Python |
| Continuous batching | ✓ | — |
| Prefix caching | Block-level (vLLM-style) | — |
| Multi-model LRU | ✓ | ✓ |
| OpenAI API | ✓ | ✓ |
| Ollama API | ✓ | ✓ |
| Function calling | ✓ | ✓ |
| Prometheus metrics | ✓ | — |
| TTFT P50 (3B, 4 workers) | **87ms** | 310ms |
| Throughput (3B, 4 workers) | **312 tok/s** | 148 tok/s |

---

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for the full release history.

Current version: **v1.0.0**
