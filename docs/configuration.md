# Configuration

fox can be configured through three mechanisms, applied in this order of precedence:

1. **CLI flags** — highest priority, override everything
2. **Environment variables** — override config file values
3. **Config file** — base configuration, loaded at startup

---

## Config file

The config file is a TOML file read at startup before command-line arguments are parsed. Environment variables from the config are injected into the process environment, so they are then picked up as if you had set them in your shell.

**Default location:** `~/.config/ferrumox/config.toml`

Override the location with the `FOX_CONFIG` environment variable:

```bash
FOX_CONFIG=/etc/ferrumox/config.toml fox serve
```

### Full config file reference

```toml
# Network
host = "0.0.0.0"
port = 8080

# Authentication (optional — omit to leave the server open)
# api_key = "your-secret-key"

# Model loading
max_models = 1
keep_alive_secs = 300

# Engine
max_batch_size = 32
gpu_memory_fraction = 0.85
block_size = 16
swap_fraction = 0.0
type_kv = "f16"   # f16 | q8_0 | q4_0 | turbo3 | turbo4 | turbo2
# type_k = "f16"          # override K cache type independently
# type_v = "turbo3"       # override V cache type independently
# max_context_len = 8192  # auto-detected from model if omitted

# Multi-GPU
# split_mode = "layer"    # none | layer | row  (default: layer)
# main_gpu = 0
# tensor_split = "3,1"    # manual VRAM weights (omit = auto-balance)
# moe_cpu = false         # offload MoE expert layers to CPU RAM

# System prompt
system_prompt = "You are a helpful assistant."

# Logging
json_logs = false
```

All fields are optional. Missing fields fall back to their default values (shown above).

### Example configurations

**Development:**

```toml
port = 8080
max_models = 3
keep_alive_secs = 0
max_context_len = 8192
json_logs = false
```

**Production:**

```toml
host = "0.0.0.0"
port = 8080
api_key = "your-secret-key"
max_models = 2
keep_alive_secs = 600
gpu_memory_fraction = 0.85
type_kv = "q8_0"
max_context_len = 4096
json_logs = true
system_prompt = ""
```

**Low-memory machine:**

```toml
max_models = 1
keep_alive_secs = 120
max_context_len = 2048
gpu_memory_fraction = 0.70
max_batch_size = 16
```

---

## Environment variables

Every `fox serve` option can be set via an environment variable. Environment variables are read after the config file and before CLI flags.

| Variable | Corresponding flag | Default |
|---|---|---|
| `FOX_CONFIG` | — | `~/.config/ferrumox/config.toml` |
| `FOX_HOST` | `--host` | `0.0.0.0` |
| `FOX_PORT` | `--port` | `8080` |
| `FOX_API_KEY` | `--api-key` | — |
| `FOX_MODEL_PATH` | `--model-path` | — |
| `FOX_MAX_MODELS` | `--max-models` | `1` |
| `FOX_KEEP_ALIVE_SECS` | `--keep-alive-secs` | `300` |
| `FOX_MAX_CONTEXT_LEN` | `--max-context-len` | auto |
| `FOX_MAX_BATCH_SIZE` | `--max-batch-size` | `32` |
| `FOX_GPU_MEMORY_FRACTION` | `--gpu-memory-fraction` | `0.85` |
| `FOX_TYPE_KV` | `--type-kv` | `f16` |
| `FOX_TYPE_K` | `--type-k` | — |
| `FOX_TYPE_V` | `--type-v` | — |
| `FOX_MAIN_GPU` | `--main-gpu` | `0` |
| `FOX_SPLIT_MODE` | `--split-mode` | `layer` |
| `FOX_TENSOR_SPLIT` | `--tensor-split` | — |
| `FOX_MOE_CPU` | `--moe-cpu` | `false` |
| `FOX_BLOCK_SIZE` | `--block-size` | `16` |
| `FOX_SWAP_FRACTION` | `--swap-fraction` | `0.0` |
| `FOX_SYSTEM_PROMPT` | `--system-prompt` | `"You are a helpful assistant."` |
| `FOX_ALIAS_FILE` | `--alias-file` | `~/.config/ferrumox/aliases.toml` |
| `FOX_JSON_LOGS` | `--json-logs` | `false` |
| `HF_TOKEN` | `--hf-token` | — |

Example:

```bash
export FOX_PORT=11434
export FOX_MAX_MODELS=3
export FOX_JSON_LOGS=true
fox serve
```

---

## Aliases

The aliases file maps short names to model filename stems. This lets you refer to models by a memorable name in API requests and CLI commands instead of typing the full filename.

**Default location:** `~/.config/ferrumox/aliases.toml`

Override with `--alias-file` or `FOX_ALIAS_FILE`.

### Format

```toml
[aliases]
"llama3"   = "Llama-3.2-3B-Instruct-Q4_K_M"
"llama3:8b" = "Llama-3.1-8B-Instruct-Q4_K_M"
"gemma3"   = "gemma-3-12b-it-Q4_K_M"
"mistral"  = "Mistral-7B-Instruct-v0.3-Q4_K_M"
"qwen"     = "Qwen2.5-7B-Instruct-Q4_K_M"
"code"     = "Qwen2.5-Coder-7B-Instruct-Q4_K_M"
```

The key is the short name you use in API requests. The value is the filename stem (filename without `.gguf` extension).

### How name resolution works

When a request arrives with `"model": "llama3"`, fox resolves the name in four steps:

1. **Alias match** — checks the aliases file for an exact key match (`"llama3"` → `"Llama-3.2-3B-Instruct-Q4_K_M"`)
2. **Exact stem match** — checks if any file in the models directory has this exact stem
3. **Starts-with match** — checks if any filename starts with the given string
4. **Contains match** — checks if any filename contains the given string

The first match wins. If no match is found, the server returns a 404 with an informative message listing available models.

### Examples

With this aliases file:

```toml
[aliases]
"llama3" = "Llama-3.2-3B-Instruct-Q4_K_M"
"fast"   = "Llama-3.2-1B-Instruct-Q4_K_M"
```

All of these request `"model"` values resolve correctly:

```
"llama3"                             → alias match → Llama-3.2-3B-Instruct-Q4_K_M
"Llama-3.2-3B-Instruct-Q4_K_M"      → exact stem match
"Llama-3.2"                         → starts-with match
"3B-Instruct"                       → contains match
"fast"                              → alias match → Llama-3.2-1B-Instruct-Q4_K_M
```

---

## Precedence example

Suppose you have:

**`~/.config/ferrumox/config.toml`:**
```toml
port = 9090
max_models = 2
```

**Shell environment:**
```bash
export FOX_PORT=11434
```

**CLI:**
```bash
fox serve --port 8080
```

The effective port will be **8080** — CLI flag wins over environment variable wins over config file.

---

## Checking effective configuration

Run `fox serve --help` to see all available flags and their current default values. To see what the server actually started with, check the startup log lines:

```
INFO fox::cli::serve: config file loaded path="~/.config/ferrumox/config.toml"
INFO fox::cli::serve: starting server host="0.0.0.0" port=8080 max_models=2 keep_alive_secs=300
```

With `--json-logs`, these appear as structured JSON objects.
