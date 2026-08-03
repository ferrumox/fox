# Quick Start

This guide gets you from zero to a running inference server in under five minutes.

---

## 1. Install fox

```bash
curl -L https://github.com/ferrumox/fox/releases/latest/download/fox-linux-x86_64.tar.gz \
  | tar xz
sudo mv fox /usr/local/bin/
```

See [Installation](./installation.md) for other platforms and Docker.

---

## 2. Download a model

fox can search HuggingFace Hub and download models with a single command. We will use
Qwen3.6 35B-A3B, the current flagship in the built-in catalogue. It is a mixture of
experts, so despite the 35B it activates only about 3B parameters per token and runs on
far less hardware than the size suggests — pair it with `--moe-cpu` to keep the expert
layers in RAM.

It is a 22 GB download. If you want something quicker to try, `fox pull qwen3.5` is
Qwen3.5 4B at 2.7 GB and every command below works the same way with it.

```bash
fox pull qwen3.6
```

You will see a progress bar as the model downloads to `~/.cache/ferrumox/models/`.

```
Searching HuggingFace for "qwen3.6"...
  → unsloth/Qwen3.6-35B-A3B-GGUF (UD-Q4_K_M)
Downloading Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  [████████████████████] 22.1 GB / 22.1 GB  •  12.4 MB/s  •  done
```

You can also pull larger models or specify a quantization:

```bash
fox pull gemma3:12b          # 12B Gemma 3
fox pull qwen2.5:7b          # 7B Qwen 2.5
fox pull qwen3.5:9b-q8       # Qwen3.5 9B, Q8 quantization
```

---

## 3. Start the server

```bash
fox serve
```

By default, the server binds to `0.0.0.0:8080`. Models load lazily on the first request — you will see a log line when the model is loaded.

```
INFO fox::api: listening on 0.0.0.0:8080
INFO fox::engine: loading Qwen3.6-35B-A3B-UD-Q4_K_M  [on first request]
```

If you want to pre-load a model at startup so the first request has no delay:

```bash
fox serve --model-path ~/.cache/ferrumox/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

---

## 4. Send a request

In another terminal, send a chat request using curl:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role": "user", "content": "Explain what a KV cache is in one paragraph."}
    ]
  }'
```

You will get a response in OpenAI format:

```json
{
  "id": "chatcmpl-a1b2c3",
  "object": "chat.completion",
  "created": 1741824000,
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "A KV cache (key-value cache) stores the intermediate attention keys and values computed during a transformer's forward pass..."
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 22,
    "completion_tokens": 87,
    "total_tokens": 109
  }
}
```

---

## 5. Use your existing tools

Because fox speaks the OpenAI and Ollama protocols, your existing tools work immediately.

**OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="qwen3.6",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

**Ollama CLI:**

```bash
OLLAMA_HOST=http://localhost:8080 ollama run qwen3.6
```

**Open WebUI:**

```bash
docker run -e OLLAMA_BASE_URL=http://host.docker.internal:8080 \
  -p 3000:8080 ghcr.io/open-webui/open-webui:main
```

Then open `http://localhost:3000` in your browser.

---

## Next steps

| Topic | Link |
|-------|------|
| All CLI commands and flags | [CLI Reference](./cli/serve.md) |
| Full API documentation | [API Reference](./api/openai.md) |
| Running inference without a server | [fox run](./cli/run.md) |
| Managing models | [fox pull / list / show](./cli/pull.md) |
| Connecting to LangChain, Open WebUI, etc. | [Integrations](./integrations.md) |
| Deploying with Docker or systemd | [Deployment](./deployment.md) |
| Performance tuning | [Benchmarks](./benchmarks.md) |

---

## Troubleshooting

**The server starts but returns 404 on the first request**

Make sure the model name in your request matches a file in `~/.cache/ferrumox/models/`. Run `fox list` to see what is available.

**Out of memory on model load**

Try a smaller quantization. `fox pull qwen3.6:4b-q4` uses roughly half the memory of `q8`. You can also reduce the context length: `fox serve --max-context-len 2048`.

**Port 8080 is already in use**

Change the port: `fox serve --port 9090`

**CUDA not detected**

Make sure `nvidia-smi` works in your terminal — no special build flags are required since fox detects the GPU at runtime. If your VRAM is tight, try `--gpu-memory-fraction 0.7` or reduce the KV cache size with `--type-kv q8_0`.
