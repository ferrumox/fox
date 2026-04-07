# fox models

Browse the curated model catalogue — a built-in registry of popular GGUF models with size, tags, and a short description.

```
fox models
```

---

## Usage

```bash
fox models
```

No flags. Prints a table to stdout and exits.

---

## Sample output

```
NAME            SIZE    TAGS                        DESCRIPTION
--------------------------------------------------------------------------------------------
llama3.2         2.0 GB  chat, instruct, fast        Meta Llama 3.2 3B — fast and capable small model
llama3.1         4.7 GB  chat, instruct              Meta Llama 3.1 8B instruction-tuned
gemma3          7.3 GB  chat, instruct, google       Google Gemma 3 12B instruction-tuned
qwen2.5         4.7 GB  chat, instruct, code         Alibaba Qwen 2.5 7B — strong reasoning and code
deepseek-r1     4.7 GB  chat, reasoning, moe         DeepSeek R1 Distill 8B — open reasoning model
mistral         4.1 GB  chat, instruct               Mistral 7B v0.3 instruction-tuned
phi4            8.5 GB  chat, instruct, microsoft    Microsoft Phi-4 14B
```

---

## Pulling a model from the catalogue

Use `fox pull` to download any model listed here:

```bash
fox pull llama3.2       # top HuggingFace result, balanced quant auto-selected
fox pull gemma3:12b     # specific size
fox pull qwen2.5:7b-q4  # specific size and quantization
```

---

## See also

- [fox pull](./pull.md) — download models from HuggingFace
- [fox search](./pull.md#fox-search) — search HuggingFace for GGUF models
- [fox list](./pull.md#fox-list) — list already-downloaded models
