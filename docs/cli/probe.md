# fox probe

Load a model and print its resolved **`ModelInfo`** — the facts fox reads from the
GGUF metadata and the llama.cpp API — then flag any **contradictions** between
those facts and the formulas fox uses internally.

Unlike `fox show`, which infers architecture and quantization from the *filename*,
`fox probe` actually loads the model and reports the truth.

```bash
# By name/stem (looked up in the models directory)
fox probe llama-3.2-1b-instruct-q8_0

# By path to a .gguf file
fox probe ./models/gemma-4-E2B-it-Q4_K_M.gguf

# Search a custom directory
fox probe my-model --path /data/models
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | (required) | Model name, stem, filename, or path to a GGUF file. |
| `--path <DIR>` | `~/.cache/ferrumox/models` | Directory to search when a name (not a path) is given. |

---

## Output

`fox probe` prints, among others: the GGUF architecture, `n_embd`, head counts
(`n_head` / `n_head_kv`), `head_dim`, layer count, vocab size, trained context
length, EOS token, whether an embedded chat template is present, native-thinking
detection, KV sequence-copy support, and the sampling parameters recommended by
the model's metadata (if any).

It then reports **contradictions** — cases where a metadata-derived fact disagrees
with a formula used elsewhere in fox. An empty list means the model is coherent
with fox's assumptions.

### Example

```
  Model           gemma-4-E2B-it-Q4_K_M
  Architecture    gemma4
  n_embd          1536
  Heads           8 (kv: 1)
  Head dim        512
  Layers          35
  Trained ctx     131072
  Chat template   embedded

⚠ 2 contradiction(s) detected:
  - head_dim = 512 (metadata); n_embd/n_head = 1536/8 = 192 — the fallback formula would mis-size the KV cache
  - embedding_dim: n_head*head_dim = 8*512 = 4096 ≠ n_embd = 1536 — the num_heads*head_dim reconstruction breaks embeddings for this model
```

A coherent model (e.g. Llama 3.2, where `head_dim == n_embd/n_head`) reports
`✓ No contradictions`.

---

## See also

- [fox serve](./serve.md) — run the inference server
- [fox bench-kv](./bench-kv.md) — compare KV cache quantization types
