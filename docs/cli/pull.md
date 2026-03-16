# Model Management

fox includes a complete set of CLI commands for managing GGUF models: downloading from HuggingFace, listing what you have, inspecting individual models, and removing them when you no longer need them.

---

## fox pull

Download a GGUF model from HuggingFace Hub.

```
fox pull <MODEL_ID> [OPTIONS]
```

### Model ID formats

fox accepts several formats for the model identifier:

| Format | Example | Behaviour |
|--------|---------|-----------|
| Friendly name | `llama3.2` | Searches HuggingFace for a matching GGUF repo, picks the one with the most downloads |
| Name with size hint | `llama3.2:8b` | Searches for "llama3.2 8b" |
| Name with size and quant | `llama3.2:8b-q4` | Searches for "llama3.2 8b", then filters for a Q4 variant |
| Exact HF repo | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Uses the exact repo without searching |
| Exact HF repo with quant | `bartowski/Llama-3.2-3B-Instruct-GGUF:q4` | Exact repo, filtered to Q4 variant |

### Quantization auto-selection

When you do not specify a filename with `-f`, fox automatically selects a quantization variant from the files in the repo. The priority order is:

1. `Q4_K_M` — best balance of quality and size for most use cases
2. `Q4_K_S` — slightly smaller, marginally lower quality
3. `Q5_K_M` — higher quality, larger file
4. `Q4_0` — older format, widely compatible
5. `Q8_0` — near-lossless, large file
6. First available file — fallback if none of the above match

To override this selection, use `-f` to specify the exact filename you want.

### Options

| Flag | Short | Default | Description |
|------|-------|---------|---|
| `-f, --filename <NAME>` | — | — | Exact GGUF filename to download. Skips auto-selection entirely. |
| `--output-dir <PATH>` | — | `~/.cache/ferrumox/models` | Directory where the model file is saved. Created if it does not exist. |
| `--hf-token <TOKEN>` | — | `$HF_TOKEN` | HuggingFace API token. Required for private or gated models (e.g., Llama 3.1 from Meta). |

### Examples

```bash
# Search and download — fox picks the best match
fox pull llama3.2
fox pull gemma3
fox pull qwen2.5

# Include a size hint to narrow the search
fox pull gemma3:12b
fox pull qwen2.5:72b

# Include size and quantization hints
fox pull llama3.1:8b-q8
fox pull gemma3:27b-q6

# Exact HuggingFace repository
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF
fox pull bartowski/gemma-3-12b-it-GGUF

# Exact repo with quantization filter
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF:q4_k_m

# Specific filename (no auto-selection)
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF \
  -f Llama-3.2-3B-Instruct-Q8_0.gguf

# Save to a custom directory
fox pull llama3.2 --output-dir /data/models

# Gated or private model (requires HuggingFace token)
fox pull meta-llama/Llama-3.1-8B-Instruct-GGUF \
  --hf-token hf_xxxxxxxxxxxx
```

You can also set your token as an environment variable to avoid passing it on every command:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
fox pull meta-llama/Llama-3.1-8B-Instruct-GGUF
```

### Download behaviour

fox downloads files to a `.part` suffix first (`model.gguf.part`), then atomically renames to the final name on completion. This means a partially downloaded file never silently replaces a valid existing file.

If the target file already exists with the same name, fox skips the download and prints a message.

---

## fox list

List all GGUF models in your models directory.

```
fox list [--path <DIR>]
```

| Flag | Default | Description |
|------|---------|---|
| `--path <DIR>` | `~/.cache/ferrumox/models` | Directory to list models from. |

Output:

```
NAME                                         SIZE      MODIFIED
Llama-3.2-3B-Instruct-Q4_K_M.gguf          2.0 GB    2 days ago
gemma-3-12b-it-Q4_K_M.gguf                 7.3 GB    5 hours ago
Mistral-7B-Instruct-v0.3-Q4_K_M.gguf       4.1 GB    1 week ago
```

---

## fox show

Show detailed information about a specific model.

```
fox show <MODEL_NAME> [--path <DIR>]
```

| Flag | Default | Description |
|------|---------|---|
| `--path <DIR>` | `~/.cache/ferrumox/models` | Directory to look up the model in. |

The `MODEL_NAME` argument is matched against filenames using the same resolution order as the server (alias → exact stem → starts-with → contains), so you do not need to type the full filename.

```bash
fox show llama3.2
fox show Llama-3.2-3B-Instruct-Q4_K_M
```

Output:

```
Name          Llama-3.2-3B-Instruct-Q4_K_M
Architecture  llama
Quantization  Q4_K_M
Size          2.0 GB
Modified      2 days ago
Path          /home/user/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Recognised architectures:** codellama, deepseek, wizardlm, internlm, stablelm, baichuan, mistral, vicuna, alpaca, gemma, llama, qwen, falcon, bloom, orca, phi, mpt, gpt, yi

**Recognised quantizations:** Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_0, Q4_1, Q4_K_S, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K, Q8_0, F16, F32, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS

---

## fox rm

Remove a downloaded model from disk.

```
fox rm <MODEL_NAME> [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|---|
| `-y, --yes` | — | `false` | Skip the confirmation prompt. |
| `--path <DIR>` | — | `~/.cache/ferrumox/models` | Directory to look up the model in. |

By default, fox shows the model name and file size and asks you to confirm before deleting.

```bash
fox rm llama3.2
# Remove Llama-3.2-3B-Instruct-Q4_K_M.gguf (2.0 GB)? [y/N]: y
# Removed.

# Skip confirmation (useful in scripts)
fox rm llama3.2 -y
```

You can also delete models through the API (Ollama-compatible):

```bash
curl -X DELETE http://localhost:8080/api/delete \
  -H "Content-Type: application/json" \
  -d '{"name": "llama3.2"}'
```

---

## fox search

Search HuggingFace Hub for GGUF models without downloading them. Useful for discovering what is available before committing to a download.

```
fox search <QUERY> [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|---|
| `--limit <N>` | `-l` | `15` | Maximum number of results to display. |
| `--sort <FIELD>` | — | `downloads` | Sort results by `downloads` or `likes`. |
| `--hf-token <TOKEN>` | — | `$HF_TOKEN` | HuggingFace token (for searching private models). |

The query can be multiple words:

```bash
fox search gemma
fox search "small reasoning model" --sort likes --limit 5
fox search llama 3b q4 --limit 10
```

Output:

```
REPO                                              DOWNLOADS    LIKES
bartowski/Llama-3.2-3B-Instruct-GGUF             1.2M         4.8k   ✓
bartowski/Llama-3.2-1B-Instruct-GGUF             890k         2.1k
unsloth/Llama-3.2-3B-Instruct-GGUF               650k         1.3k
```

The `✓` mark indicates models that are already downloaded in your models directory.

After searching, pull any result with its exact repo ID:

```bash
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF
```

---

## fox models

List the built-in curated model registry — a hand-picked selection of recommended models with descriptions.

```
fox models
```

Output:

```
NAME              SIZE     TAGS                      DESCRIPTION
llama3.2:3b       2.0 GB   chat, fast                Meta Llama 3.2 3B Instruct
llama3.2:1b       1.2 GB   chat, fast, tiny          Meta Llama 3.2 1B Instruct
gemma3:4b         3.3 GB   chat, multilingual        Google Gemma 3 4B Instruct
gemma3:12b        7.3 GB   chat, multilingual        Google Gemma 3 12B Instruct
qwen2.5:7b        4.7 GB   chat, code, multilingual  Alibaba Qwen 2.5 7B Instruct
...
```

Use the NAME column as the argument to `fox pull`:

```bash
fox pull llama3.2:3b
```

---

## fox ps

Show models currently loaded in a running server instance.

```
fox ps [--port <PORT>]
```

| Flag | Default | Description |
|------|---------|---|
| `--port <N>` | `8080` | Port the server is listening on. |

`fox ps` queries `GET /api/ps` on the local server and formats the response as a table.

```
NAME                              STATUS    PORT   KV CACHE   QUEUE   UPTIME
Llama-3.2-3B-Instruct-Q4_K_M     loaded    8080   12%        0       2h 14m
gemma-3-12b-it-Q4_K_M             loaded    8080   3%         0       45m
```

---

## See also

- [fox serve](./serve.md) — start the inference server
- [Configuration](../configuration.md) — setting up aliases
- [API Reference: Ollama](../api/ollama.md) — model management via API
