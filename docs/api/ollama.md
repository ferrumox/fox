# Ollama-Compatible API

fox implements the Ollama REST API, allowing any tool built for Ollama to work with fox without code changes. To switch, point the Ollama host at your fox instance.

**Base URL:** `http://localhost:8080`

**Authentication:** If `FOX_API_KEY` is configured, include `Authorization: Bearer <key>` in every request. Ollama-based tools that support custom headers (e.g., Open WebUI) can be configured to send this header. Tools that do not support auth headers should connect to a fox instance without `FOX_API_KEY`, or use a reverse proxy to inject the header.

---

## Switching from Ollama to fox

Most Ollama-based tools accept a host or base URL configuration. Set it to your fox server address:

```bash
# Ollama CLI
OLLAMA_HOST=http://localhost:8080 ollama run llama3.2

# Open WebUI
OLLAMA_BASE_URL=http://localhost:8080 open-webui serve

# Continue.dev (config.json)
"apiBase": "http://localhost:8080"
```

Note: fox uses the same model names as the files on disk (e.g., `Llama-3.2-3B-Instruct-Q4_K_M`). If you use the [aliases file](../configuration.md#aliases), short names like `llama3.2` work too. The `GET /api/tags` endpoint lists all available model names.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Chat completion (NDJSON streaming) |
| `POST` | `/api/generate` | Text generation (NDJSON streaming) |
| `POST` | `/api/embed` | Embedding generation |
| `GET` | `/api/tags` | List models on disk |
| `GET` | `/api/ps` | List loaded models |
| `POST` | `/api/show` | Show model details |
| `DELETE` | `/api/delete` | Delete a model from disk |
| `POST` | `/api/pull` | Pull a model from HuggingFace (SSE) |
| `GET` | `/api/version` | Server version |

---

## POST /api/chat

Multi-turn conversation using the Ollama chat format. Supports streaming (NDJSON) and non-streaming modes.

### Request

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "stream": true,
  "options": {
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "repeat_penalty": 1.0,
    "seed": null,
    "num_predict": 256,
    "stop": ["</s>"]
  }
}
```

### Request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | (required) | Model name. |
| `messages` | `array` | (required) | Conversation history. Each item has `role` and `content`. |
| `stream` | `boolean` | `true` | Stream responses as NDJSON. Set to `false` for a single JSON response. |
| `options` | `object` | `{}` | Sampling and generation options (see table below). |

**`options` fields:**

| Field | Default | Description |
|-------|---------|-------------|
| `temperature` | `1.0` | Sampling temperature. |
| `top_p` | `1.0` | Nucleus sampling probability. |
| `top_k` | `0` | Top-K filter. `0` = disabled. |
| `repeat_penalty` | `1.0` | Repetition penalty. |
| `seed` | `null` | RNG seed. |
| `num_predict` | `256` | Maximum tokens to generate (equivalent to `max_tokens`). |
| `stop` | `[]` | Stop sequences. |

### Response (streaming)

One JSON object per line. The final object has `"done": true` and includes timing and token count statistics.

```
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","message":{"role":"assistant","content":"The"},"done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","message":{"role":"assistant","content":" sky"},"done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","message":{"role":"assistant","content":" appears"},"done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":412000000,"load_duration":0,"prompt_eval_count":12,"eval_count":48}
```

### Response (non-streaming)

When `"stream": false`, a single JSON object is returned:

```json
{
  "model": "llama3.2",
  "created_at": "2026-03-12T10:00:00Z",
  "message": {
    "role": "assistant",
    "content": "The sky appears blue because of a phenomenon called Rayleigh scattering..."
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 412000000,
  "load_duration": 0,
  "prompt_eval_count": 12,
  "eval_count": 48
}
```

**Response fields:**

| Field | Description |
|-------|-------------|
| `model` | Model that generated the response. |
| `created_at` | ISO 8601 timestamp. |
| `message.content` | Generated text for this chunk. Empty string on the final chunk. |
| `done` | `true` on the final message. |
| `done_reason` | `"stop"` or `"length"`. Present on the final message only. |
| `total_duration` | Total request duration in nanoseconds. |
| `load_duration` | Time spent loading the model (nanoseconds). `0` if already loaded. |
| `prompt_eval_count` | Number of prompt tokens processed. |
| `eval_count` | Number of tokens generated. |

---

## POST /api/generate

Single-turn text generation. Accepts a raw prompt string rather than a messages array.

### Request

```json
{
  "model": "llama3.2",
  "prompt": "The three laws of robotics are:",
  "system": "You are a science fiction encyclopedia.",
  "stream": true,
  "options": {
    "temperature": 0.5,
    "num_predict": 200
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | (required) | Model name. |
| `prompt` | `string` | (required) | Prompt text. |
| `system` | `string` | — | System prompt. Overrides the server's default. |
| `stream` | `boolean` | `true` | NDJSON streaming. |
| `options` | `object` | `{}` | Same options object as `/api/chat`. |

### Response (streaming)

```
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","response":"First","done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","response":",","done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","response":" a","done":false}
{"model":"llama3.2","created_at":"2026-03-12T10:00:00Z","response":"","done":true,"done_reason":"stop","total_duration":312000000,"prompt_eval_count":15,"eval_count":42}
```

Note: the token text is in the `response` field (not `message.content` as in `/api/chat`).

---

## POST /api/embed

Generate embeddings for one or more inputs.

### Request

```json
{
  "model": "llama3.2",
  "input": "The quick brown fox jumps over the lazy dog"
}
```

Batch input:

```json
{
  "model": "llama3.2",
  "input": ["first sentence", "second sentence", "third sentence"]
}
```

### Response

```json
{
  "model": "llama3.2",
  "embeddings": [
    [0.0123, -0.0456, 0.0789, ...]
  ]
}
```

For batch input, `embeddings` contains one array per input string, in the same order.

---

## GET /api/tags

List all GGUF model files in the models directory with metadata.

### Response

```json
{
  "models": [
    {
      "name": "Llama-3.2-3B-Instruct-Q4_K_M",
      "size": 2148532224,
      "digest": "sha256:a1b2c3d4e5f6...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "unknown",
        "quantization_level": "Q4_K_M"
      },
      "modified_at": "2026-03-12T10:00:00Z"
    },
    {
      "name": "gemma-3-12b-it-Q4_K_M",
      "size": 7835181056,
      "digest": "sha256:b2c3d4e5f6a1...",
      "details": {
        "format": "gguf",
        "family": "gemma",
        "parameter_size": "unknown",
        "quantization_level": "Q4_K_M"
      },
      "modified_at": "2026-03-10T15:30:00Z"
    }
  ]
}
```

| Field | Description |
|-------|-------------|
| `name` | Model identifier (filename stem without `.gguf`). Use this as the `model` field in requests. |
| `size` | File size in bytes. |
| `digest` | SHA-256 digest of the file. Cached after first computation. |
| `details.family` | Model architecture (inferred from filename). |
| `details.quantization_level` | Quantization type (inferred from filename). |
| `modified_at` | File modification timestamp in ISO 8601 format. |

---

## GET /api/ps

List all models currently loaded in memory.

### Response

```json
{
  "models": [
    {
      "name": "Llama-3.2-3B-Instruct-Q4_K_M",
      "size": 2148532224,
      "digest": "sha256:a1b2c3d4e5f6...",
      "details": {
        "format": "gguf",
        "family": "llama",
        "parameter_size": "unknown",
        "quantization_level": "Q4_K_M"
      },
      "expires_at": "0001-01-01T00:00:00Z",
      "size_vram": 0
    }
  ]
}
```

`expires_at` reflects the keep-alive expiry time. `size_vram` is reserved for future use.

---

## POST /api/show

Show detailed metadata about a specific model.

### Request

```json
{
  "name": "llama3.2"
}
```

### Response

```json
{
  "modelfile": "# GGUF model: Llama-3.2-3B-Instruct-Q4_K_M",
  "parameters": "",
  "template": "",
  "details": {
    "format": "gguf",
    "family": "llama",
    "parameter_size": "unknown",
    "quantization_level": "Q4_K_M"
  },
  "model_info": {
    "general.architecture": "llama",
    "general.quantization": "Q4_K_M",
    "general.size": "2.0 GB",
    "general.digest": "sha256:a1b2c3d4e5f6...",
    "general.modified_at": "2026-03-12T10:00:00Z"
  }
}
```

---

## DELETE /api/delete

Delete a model file from disk. The model is also evicted from memory if it is currently loaded.

### Request

```json
{
  "name": "llama3.2"
}
```

Returns `200 OK` with no body on success. Returns `404` if the model is not found.

---

## POST /api/pull

Pull a model from HuggingFace Hub. Progress events are streamed as Server-Sent Events (SSE).

### Request

```json
{
  "name": "llama3.2"
}
```

The `name` field accepts the same formats as `fox pull`: friendly names, HuggingFace repo IDs, or search queries.

### Response (SSE stream)

```
event: message
data: {"status":"pulling manifest","digest":null,"total":null,"completed":null}

event: message
data: {"status":"downloading","digest":"sha256:a1b2c3d4...","total":2148532224,"completed":0}

event: message
data: {"status":"downloading","digest":"sha256:a1b2c3d4...","total":2148532224,"completed":536870912}

event: message
data: {"status":"downloading","digest":"sha256:a1b2c3d4...","total":2148532224,"completed":1073741824}

event: message
data: {"status":"downloading","digest":"sha256:a1b2c3d4...","total":2148532224,"completed":2148532224}

event: message
data: {"status":"verifying sha256 digest","digest":"sha256:a1b2c3d4...","total":null,"completed":null}

event: message
data: {"status":"success","digest":null,"total":null,"completed":null}
```

| Field | Description |
|-------|-------------|
| `status` | Current phase: `"pulling manifest"`, `"downloading"`, `"verifying sha256 digest"`, `"success"`, or `"error"`. |
| `digest` | SHA-256 digest of the file being downloaded (present during download phase). |
| `total` | Total file size in bytes (present during download). |
| `completed` | Bytes downloaded so far (present during download). |

Use `completed / total * 100` to compute percentage progress.

To use a HuggingFace token with `POST /api/pull`, start the server with `--hf-token` or `$HF_TOKEN`.

---

## GET /api/version

Returns the server version. Used by Ollama clients for compatibility detection.

### Response

```json
{
  "version": "1.0.0"
}
```

---

## Compatibility notes

fox aims for full API-level compatibility with Ollama. The following features are supported:

| Feature | Supported |
|---------|-----------|
| `/api/chat` streaming | ✓ |
| `/api/generate` streaming | ✓ |
| `/api/embed` (single and batch) | ✓ |
| `/api/tags` | ✓ |
| `/api/ps` | ✓ |
| `/api/show` | ✓ |
| `/api/delete` | ✓ |
| `/api/pull` with progress | ✓ |
| `/api/version` | ✓ |
| Function calling (via `/v1/chat/completions`) | ✓ |
| Image/multimodal input | — (text models only) |
| Model creation (`/api/create`) | — |
| Blob endpoints (`/api/blobs`) | — |
