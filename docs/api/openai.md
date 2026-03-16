# OpenAI-Compatible API

fox implements the OpenAI REST API. Any application built with the OpenAI Python SDK, JavaScript SDK, or any HTTP client targeting OpenAI's API can point at fox by changing one setting — the base URL.

**Base URL:** `http://localhost:8080/v1`

**Authentication:** By default, fox requires no API key. If `FOX_API_KEY` is set when the server starts, every request must include `Authorization: Bearer <key>`. When using an OpenAI SDK, pass the key as `api_key`; when authentication is disabled, any non-empty string (e.g., `"none"`) satisfies SDK validation.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completion (streaming and non-streaming) |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/embeddings` | Text embedding generation |
| `GET` | `/v1/models` | List available models |
| `GET` | `/health` | Server health and status |
| `GET` | `/metrics` | Prometheus metrics |

---

## POST /v1/chat/completions

The primary inference endpoint. Accepts a conversation history and returns a model-generated reply.

### Request

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the Rust borrow checker?"},
    {"role": "assistant", "content": "The borrow checker is..."},
    {"role": "user",   "content": "Can you give an example?"}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 1.0,
  "top_k": 0,
  "repetition_penalty": 1.0,
  "seed": null,
  "stop": null,
  "stream": false,
  "tools": null,
  "tool_choice": null,
  "response_format": null
}
```

### Request fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | (required) | Model name. Resolved using aliases and filename matching. See [Configuration](../configuration.md). |
| `messages` | `array` | (required) | Conversation history. Each item has `role` (`system`, `user`, `assistant`, or `tool`) and `content`. |
| `max_tokens` | `integer` | `256` | Maximum number of tokens to generate. |
| `temperature` | `float` | `1.0` | Sampling temperature. `0.0` = greedy, higher = more random. |
| `top_p` | `float` | `1.0` | Nucleus sampling probability. `1.0` = disabled. |
| `top_k` | `integer` | `0` | Top-K filter. `0` = disabled. |
| `repetition_penalty` | `float` | `1.0` | Repetition penalty. `1.0` = disabled. Values above `1.0` reduce repetition. |
| `seed` | `integer` | `null` | RNG seed for reproducible output. |
| `stop` | `string \| array` | `null` | Stop sequences. Generation halts when any of these strings is produced. |
| `stream` | `boolean` | `false` | Whether to stream tokens as they are generated (SSE). |
| `tools` | `array` | `null` | List of tools (functions) the model can call. See [Function calling](#function-calling). |
| `tool_choice` | `string \| object` | `null` | Controls tool selection. `"auto"`, `"none"`, or `{"type":"function","function":{"name":"..."}}`. |
| `response_format` | `object` | `null` | Output format constraint. `{"type":"json_object"}` forces valid JSON output. |

### Response (non-streaming)

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1741824000,
  "model": "Llama-3.2-3B-Instruct-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The borrow checker is a compile-time analysis pass in the Rust compiler..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 28,
    "completion_tokens": 94,
    "total_tokens": 122
  }
}
```

| Field | Description |
|-------|-------------|
| `id` | Unique completion ID. |
| `object` | Always `"chat.completion"`. |
| `created` | Unix timestamp of the request. |
| `model` | Resolved model name (may differ from request if an alias was used). |
| `choices[].message.content` | Generated text. `null` when a tool call was made instead. |
| `choices[].finish_reason` | `"stop"` (natural end), `"length"` (hit `max_tokens`), or `"tool_calls"` (tool was invoked). |
| `usage` | Token counts for the entire request. |

### Response (streaming)

Set `"stream": true`. The server responds with `Content-Type: text/event-stream` and sends `data:` lines as tokens are generated, ending with `data: [DONE]`.

```
data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1741824000,"model":"Llama-3.2-3B-Instruct-Q4_K_M","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1741824000,"model":"Llama-3.2-3B-Instruct-Q4_K_M","choices":[{"index":0,"delta":{"content":" borrow"},"finish_reason":null}]}

data: {"id":"chatcmpl-a1b2c3d4","object":"chat.completion.chunk","created":1741824000,"model":"Llama-3.2-3B-Instruct-Q4_K_M","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}],"usage":{"prompt_tokens":28,"completion_tokens":94,"total_tokens":122}}

data: [DONE]
```

The final chunk (with `"finish_reason"`) includes a `usage` field. All other chunks have `usage: null`.

---

## POST /v1/completions

Text completion. Accepts a raw text prompt instead of a message array. Internally, fox wraps the prompt as a user message and routes it through the same engine.

### Request

```json
{
  "model": "llama3.2",
  "prompt": "The capital of France is",
  "max_tokens": 32,
  "temperature": 0.0,
  "stream": false
}
```

### Response

```json
{
  "id": "cmpl-a1b2c3d4",
  "object": "text_completion",
  "created": 1741824000,
  "model": "Llama-3.2-3B-Instruct-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "text": " Paris, the city of light.",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 6,
    "total_tokens": 14
  }
}
```

---

## POST /v1/embeddings

Generate vector embeddings for one or more texts. Useful for semantic search, clustering, and RAG pipelines.

### Request

Single input:

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
  "input": [
    "First document to embed",
    "Second document to embed",
    "Third document to embed"
  ]
}
```

### Response

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.0123, -0.0456, 0.0789, ...],
      "index": 0
    }
  ],
  "model": "Llama-3.2-3B-Instruct-Q4_K_M",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

The embedding vector dimension depends on the model architecture.

---

## GET /v1/models

List all GGUF model files in the models directory.

### Response

```json
{
  "object": "list",
  "data": [
    {"id": "Llama-3.2-3B-Instruct-Q4_K_M", "object": "model"},
    {"id": "gemma-3-12b-it-Q4_K_M", "object": "model"},
    {"id": "Mistral-7B-Instruct-v0.3-Q4_K_M", "object": "model"}
  ]
}
```

The `id` field is the filename stem (without `.gguf`). Use this value as the `model` field in inference requests.

---

## GET /health

Returns the current server status.

### Response

```json
{
  "status": "ok",
  "kv_cache_usage": 0.12,
  "queue_depth": 2,
  "active_requests": 1,
  "model_name": "Llama-3.2-3B-Instruct-Q4_K_M",
  "started_at": 1741824000
}
```

| Field | Description |
|-------|-------------|
| `status` | Always `"ok"` if the server is running. |
| `kv_cache_usage` | Fraction of KV cache blocks currently occupied (0.0–1.0). High values indicate memory pressure. |
| `queue_depth` | Number of requests waiting to be scheduled. |
| `active_requests` | Number of sequences currently being decoded. |
| `model_name` | Name of the currently active model (in single-model mode). |
| `started_at` | Unix timestamp of when the server started. |

---

## GET /metrics

Prometheus metrics scrape endpoint. Returns all metrics in Prometheus text exposition format.

```
# HELP fox_requests_total Total number of inference requests
# TYPE fox_requests_total counter
fox_requests_total{model="llama3.2",status="ok"} 142
fox_requests_total{model="llama3.2",status="error"} 2

# HELP fox_request_duration_seconds Request latency histogram
# TYPE fox_request_duration_seconds histogram
fox_request_duration_seconds_bucket{le="0.1"} 34
fox_request_duration_seconds_bucket{le="0.5"} 98
fox_request_duration_seconds_bucket{le="1.0"} 130
fox_request_duration_seconds_bucket{le="+Inf"} 144
fox_request_duration_seconds_sum 78.2
fox_request_duration_seconds_count 144

# HELP fox_kv_cache_usage KV cache utilization ratio
# TYPE fox_kv_cache_usage gauge
fox_kv_cache_usage 0.12

# HELP fox_prefix_cache_hit_ratio Prefix cache hit ratio
# TYPE fox_prefix_cache_hit_ratio gauge
fox_prefix_cache_hit_ratio 0.68

# HELP fox_queue_depth Current number of queued requests
# TYPE fox_queue_depth gauge
fox_queue_depth 0

# HELP fox_tokens_generated_total Total tokens generated
# TYPE fox_tokens_generated_total counter
fox_tokens_generated_total 48291

# HELP fox_time_to_first_token_seconds Time to first token histogram
# TYPE fox_time_to_first_token_seconds histogram
fox_time_to_first_token_seconds_bucket{le="0.05"} 12
fox_time_to_first_token_seconds_bucket{le="0.1"} 89
fox_time_to_first_token_seconds_bucket{le="0.25"} 138
fox_time_to_first_token_seconds_bucket{le="+Inf"} 144
```

Add this endpoint to your Prometheus scrape config:

```yaml
scrape_configs:
  - job_name: fox
    static_configs:
      - targets: ["localhost:8080"]
    metrics_path: /metrics
```

---

## Function calling

fox supports the OpenAI tool use spec. Include a `tools` array in your request with one or more function definitions, and set `tool_choice` to control when tools are used.

### Request with tools

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is the weather like in Madrid right now?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather for a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and country, e.g. Madrid, Spain"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### Response when the model calls a tool

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1741824000,
  "model": "Llama-3.2-3B-Instruct-Q4_K_M",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Madrid, Spain\", \"unit\": \"celsius\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 78,
    "completion_tokens": 24,
    "total_tokens": 102
  }
}
```

### Sending the tool result back

Add the tool call and its result to the conversation, then make another request:

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "What is the weather like in Madrid right now?"},
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "get_current_weather",
            "arguments": "{\"location\": \"Madrid, Spain\", \"unit\": \"celsius\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_abc123",
      "content": "{\"temperature\": 18, \"condition\": \"Partly cloudy\", \"humidity\": 55}"
    }
  ],
  "tools": [...]
}
```

### `tool_choice` values

| Value | Description |
|-------|-------------|
| `"auto"` | The model decides whether to call a tool or respond directly. |
| `"none"` | The model must respond with text and cannot call any tool. |
| `{"type": "function", "function": {"name": "..."}}` | Force the model to call a specific function. |

---

## Structured output (JSON mode)

Set `response_format` to force the model to produce valid JSON output. fox injects a system instruction that constrains the model's output format.

```json
{
  "model": "llama3.2",
  "messages": [
    {"role": "user", "content": "Give me a JSON object describing a fictional person, with keys: name, age, occupation, city."}
  ],
  "response_format": {"type": "json_object"},
  "temperature": 0.3
}
```

Response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "{\"name\": \"Elena Vasquez\", \"age\": 34, \"occupation\": \"Marine biologist\", \"city\": \"Lisbon\"}"
    },
    "finish_reason": "stop"
  }]
}
```

The `content` field is always a valid JSON string when `json_object` mode is active. Parse it with `JSON.parse()` / `json.loads()` after receiving it.

---

## Stop sequences

Use the `stop` field to end generation when specific strings are produced.

```json
{
  "model": "llama3.2",
  "messages": [{"role": "user", "content": "Count from 1 to 10"}],
  "stop": ["5", "five"]
}
```

The model stops as soon as it generates any of the listed strings. The stop string itself is not included in the output. You can pass a single string or an array of up to several strings.

---

## Using with OpenAI SDKs

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none"
)

# Non-streaming
response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256
)
print(response.choices[0].message.content)

# Streaming
with client.chat.completions.stream(
    model="llama3.2",
    messages=[{"role": "user", "content": "Tell me a joke"}]
) as stream:
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

### JavaScript / TypeScript

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "none",
});

// Non-streaming
const response = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Hello!" }],
});
console.log(response.choices[0].message.content);

// Streaming
const stream = await client.chat.completions.create({
  model: "llama3.2",
  messages: [{ role: "user", content: "Tell me a joke" }],
  stream: true,
});
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

---

## Error responses

| HTTP status | Meaning |
|-------------|---------|
| `200 OK` | Success. |
| `400 Bad Request` | Malformed JSON or invalid field values. |
| `401 Unauthorized` | Missing or invalid `Authorization: Bearer` header (only when `FOX_API_KEY` is set). |
| `404 Not Found` | Model not found. Check `fox list` and your model name. |
| `500 Internal Server Error` | Engine error. Check server logs for details. |
| `503 Service Unavailable` | KV cache exhausted (all blocks occupied). Retry after in-flight requests complete. |

Error body:

```json
{
  "error": {
    "message": "Model 'gpt-4' not found. Available models: llama3.2, gemma3",
    "type": "invalid_request_error"
  }
}
```
