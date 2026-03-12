# Introduction

**fox** is a local LLM inference server built entirely in Rust. It runs GGUF-format models directly on your hardware — CPU or GPU — and exposes two complete HTTP APIs: one compatible with the OpenAI SDK, and one compatible with Ollama.

The goal is straightforward: run any model, serve any client, with no Python runtime, no complex setup, and the highest throughput your hardware allows.

---

## Why fox?

Most local inference tools are wrappers around Python runtimes. fox is different — it is a single self-contained binary compiled from Rust, with llama.cpp linked directly via FFI. There is no interpreter, no virtual environment, no dynamic dependency resolution at startup.

This design choice has concrete effects:

- **Cold start in under a second** — the binary loads, initialises the engine, and binds to a port with no interpreter warm-up
- **Predictable memory usage** — no garbage collector, no hidden heap growth
- **True parallelism** — Rust's async runtime (Tokio) handles concurrent requests without a GIL

---

## What fox does

fox is an inference server. You point it at a GGUF model file, it starts an HTTP server, and any client that speaks the OpenAI or Ollama protocol can use it immediately.

Beyond basic inference, fox implements several techniques from production inference systems:

**Continuous batching** — incoming requests are inserted into the current batch mid-generation, so the GPU is never idle waiting for a request to finish before it starts the next one. This is the same approach used by vLLM and TGI and is the primary reason fox achieves significantly higher throughput than single-request-at-a-time servers.

**Block-level prefix caching** — the KV cache is divided into fixed-size blocks (16 tokens by default). Blocks for repeated prompt prefixes are identified by a chain hash and reused across requests without recomputation. In practice, multi-turn conversations and RAG pipelines with a shared context see cache hit rates of 60–75%, which directly reduces time-to-first-token.

**PagedAttention** — logical KV cache sequences map to non-contiguous physical blocks, eliminating memory fragmentation. Reference counting and copy-on-write ensure correctness when multiple sequences share a prefix.

**Multi-model serving** — fox can hold multiple models in memory simultaneously, routing each request to the correct model by name. When capacity is reached, the least-recently-used model is evicted. New models are loaded lazily on the first request that names them.

---

## API compatibility

fox implements both the OpenAI REST API and the Ollama REST API. This means:

- Any application built with the OpenAI Python or JavaScript SDK works with fox by changing one line (the `base_url`)
- Any application built for Ollama works with fox by changing the Ollama host URL
- Tools like Open WebUI, Continue.dev, LangChain, LlamaIndex, and Cursor work out of the box

You do not need to choose between the two APIs — both run on the same server at the same time.

---

## Supported model formats

fox supports GGUF models as produced by [llama.cpp](https://github.com/ggerganov/llama.cpp). Any model in GGUF format — regardless of architecture — can be loaded and served. fox recognises architectures including Llama, Mistral, Gemma, Qwen, Phi, Falcon, DeepSeek, and others.

Quantization levels from Q2_K through Q8_0, F16, and F32 are all supported. When pulling models from HuggingFace, fox automatically selects a balanced quantization (preferring Q4_K_M) unless you specify otherwise.

---

## Versioning

The current stable release is **v1.0.0**.

| Version | Highlight |
|---------|-----------|
| v0.1.0 | Initial MVP: OpenAI API, continuous batching, KV cache |
| v0.2.0 | Stochastic sampling, Docker, integrated benchmark |
| v0.3.0 | PagedAttention, prefix caching, Prometheus metrics |
| v0.4.0 | Unified CLI (`serve`, `run`, `pull`), system prompt, JSON logs |
| v0.5.0 | Block-level prefix caching, CoW, swap scaffold |
| v0.6.0 | CLI visual overhaul, project renamed ferrumox/fox |
| v0.7.0 | Ollama-compatible API (`/api/tags`, `/api/show`, etc.) |
| v0.8.0 | Embeddings API, `POST /api/pull`, release binaries |
| v0.9.0 | Multi-model registry, LRU eviction, `GET /api/ps` |
| v0.10.0 | Tool use, structured output, config file, request cancellation |
| **v1.0.0** | HuggingFace search, benchmark comparison mode, full docs |

---

## License

fox is dual-licensed under MIT and Apache 2.0. You may choose either license.
