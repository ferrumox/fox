# fox rm

Remove a downloaded model file from disk.

```
fox rm <MODEL> [OPTIONS]
```

---

## Usage

```bash
# Interactive confirmation (default)
fox rm llama3.2

# Skip confirmation prompt
fox rm llama3.2 -y

# Search in a custom directory
fox rm llama3.2 --path /data/models
```

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `<MODEL>` | (required) | Model name, alias, or filename stem to remove. Resolved using the same matching rules as `fox serve` (alias → exact stem → starts-with → contains). |
| `-y`, `--yes` | `false` | Skip the confirmation prompt and remove immediately. |
| `--path <DIR>` | `~/.cache/ferrumox/models` | Directory to search for the model file. |

---

## Confirmation prompt

By default, fox shows the resolved file path and asks for confirmation before deleting:

```
  Found: ~/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf (2.0 GB)
  Remove? [y/N]:
```

Type `y` (or `yes`) to confirm. Any other input cancels the operation.

---

## Notes

- `fox rm` only removes the `.gguf` file. It does not affect loaded models — if the model is currently running in `fox serve`, the server continues to use it until the process restarts.
- If multiple files match the name, fox picks the first match using the same resolution order as inference requests.
- Use `fox list` to see all downloaded models before removing.

---

## See also

- [fox list](./pull.md#fox-list) — list downloaded models
- [fox pull](./pull.md) — download models
