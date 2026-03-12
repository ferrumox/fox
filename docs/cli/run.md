# fox run

Run inference directly from the command line. No HTTP server is started — the model loads, generates output, and the process exits (or you get an interactive session).

This command is useful for scripting, one-off queries, and exploring a model before serving it.

```
fox run --model-path <PATH> [OPTIONS] [PROMPT]
```

---

## Modes

### One-shot

Provide a prompt as a positional argument. fox loads the model, generates a response, prints it to stdout, and exits.

```bash
fox run --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  "What is the capital of France?"
```

Output is streamed token by token to stdout.

### Interactive REPL

Omit the prompt to start a multi-turn conversation in your terminal. Type a message, press Enter, and read the response. Your conversation history is maintained across turns.

```bash
fox run --model-path ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

```
Fox 1.0.0  •  Llama-3.2-3B-Instruct-Q4_K_M
Type /bye or Ctrl+D to exit.

▸ Tell me about prefix caching
Fox: Prefix caching is a technique used in transformer inference to avoid
recomputing attention keys and values for repeated prompt prefixes...

▸ How does it work with multiple users?
Fox: When multiple users share the same system prompt or context, prefix
caching allows the KV cache blocks for that shared prefix to be reused...
```

To exit the REPL, type `/bye`, `/exit`, `exit`, `quit`, or press `Ctrl+D`.

---

## Options

### Required

| Flag | Env variable | Description |
|------|---|---|
| `--model-path <PATH>` | `FOX_MODEL_PATH` | Path to the GGUF model file to load. This is the only required argument. |

### Prompt and output

| Flag | Default | Description |
|------|---------|---|
| `[PROMPT]` | — | The prompt text. If omitted, interactive REPL starts. |
| `--max-new-tokens <N>` | `512` | Maximum number of tokens to generate per turn. Generation stops when this limit is reached, even if the model hasn't produced a stop token. |
| `--show-thinking` | `false` | When using a reasoning model (e.g., DeepSeek-R1), stream the `<think>…</think>` block to stdout. By default, thinking tokens are hidden and only the final answer is shown. |

### Sampling

These parameters control how tokens are selected from the model's probability distribution. They correspond directly to the same parameters in the API.

| Flag | Default | Description |
|------|---------|---|
| `--temperature <F>` | `1.0` | Sampling temperature. Lower values (e.g., 0.2) make output more focused and deterministic. Higher values (e.g., 1.5) make it more creative and random. `0.0` selects the single most-probable token at each step (greedy). |
| `--top-p <F>` | `1.0` | Nucleus sampling. At each step, consider only the smallest set of tokens whose cumulative probability exceeds this value. `1.0` disables the filter. Typical values: `0.9`–`0.95`. |
| `--top-k <N>` | `0` | At each step, consider only the top K most probable tokens. `0` disables the filter. Typical values: `40`–`100`. |
| `--repetition-penalty <F>` | `1.0` | Penalise tokens that have already appeared in the output. `1.0` means no penalty. Values above `1.0` reduce repetition (try `1.1`–`1.3`). |
| `--seed <N>` | — | Fix the random number generator seed. With a fixed seed, identical prompts and sampling parameters will produce identical output. Useful for reproducible testing. |

### System prompt

| Flag | Env variable | Default | Description |
|------|---|---|---|
| `--system-prompt <TEXT>` | `FOX_SYSTEM_PROMPT` | `"You are a helpful assistant."` | Prepended to the conversation as a system message. |
| `--no-system-prompt` | — | `false` | Disable system prompt injection entirely. The model receives only your prompt. |

### Engine

| Flag | Default | Description |
|------|---------|---|
| `--max-context-len <N>` | `4096` | Maximum context length in tokens. For long conversations or documents, increase this value. Requires more memory. |
| `--gpu-memory-fraction <F>` | `0.85` | Fraction of GPU VRAM to reserve for the KV cache. |
| `--block-size <N>` | `16` | Tokens per KV cache block. |

### Debugging

| Flag | Default | Description |
|------|---------|---|
| `--verbose` | `false` | Show engine log output (model loading progress, token throughput, etc.). Hidden by default to keep output clean. |

---

## Examples

### One-shot queries

```bash
# Ask a factual question
fox run --model-path model.gguf "What year did the Berlin Wall fall?"

# Deterministic output (temperature 0, fixed seed)
fox run --model-path model.gguf \
  --temperature 0 \
  --seed 42 \
  "Write a haiku about compilers"

# Creative writing with higher temperature
fox run --model-path model.gguf \
  --temperature 1.2 \
  --top-p 0.92 \
  --max-new-tokens 1024 \
  "Write a short story about a robot who discovers poetry"

# Reduce repetition in long outputs
fox run --model-path model.gguf \
  --repetition-penalty 1.15 \
  --max-new-tokens 2048 \
  "Explain the history of type systems in programming languages"
```

### Working with reasoning models

```bash
# Show chain-of-thought reasoning
fox run --model-path deepseek-r1.gguf \
  --show-thinking \
  "Solve: if 3x + 7 = 22, what is x?"

# Hide reasoning, show only final answer (default behaviour)
fox run --model-path deepseek-r1.gguf \
  "Solve: if 3x + 7 = 22, what is x?"
```

### Custom system prompts

```bash
# Code assistant
fox run --model-path model.gguf \
  --system-prompt "You are an expert Rust programmer. Provide concise, idiomatic code." \
  "Write a function that reads a CSV file and returns a Vec<HashMap<String, String>>"

# Disable system prompt for raw completion
fox run --model-path model.gguf \
  --no-system-prompt \
  "The quick brown fox"
```

### Scripting

Because output is streamed to stdout, you can pipe it to other tools:

```bash
# Save output to a file
fox run --model-path model.gguf "Summarise the Rust ownership model" > summary.txt

# Use in a pipeline
echo "What is 42?" | xargs -I {} fox run --model-path model.gguf {}

# Combine with jq-style tools
fox run --model-path model.gguf \
  --no-system-prompt \
  'Return a JSON object with keys "name" and "age" for a fictional person' \
  | python3 -m json.tool
```

### Interactive REPL

```bash
# Start REPL with a custom role
fox run --model-path model.gguf \
  --system-prompt "You are a Socratic tutor. Never give direct answers — guide the student with questions."

# REPL with extended context for long conversations
fox run --model-path model.gguf \
  --max-context-len 8192 \
  --max-new-tokens 1024
```

---

## Context management

In the interactive REPL, all turns accumulate in the context window. When the total token count approaches `--max-context-len`, fox automatically clears the conversation history (keeping the system prompt) and starts a fresh context. You will see a notice in the terminal when this happens.

If you need a longer conversation without resets, increase `--max-context-len`. This requires more GPU memory.

---

## See also

- [fox serve](./serve.md) — start an HTTP server instead
- [fox pull](./pull.md) — download models from HuggingFace
- [API Reference](../api/openai.md) — the same sampling parameters are available via the API
