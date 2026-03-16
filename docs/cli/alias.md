# fox alias

Manage short alias names for models. Aliases let you refer to a model by a short, memorable name instead of its full filename stem.

Aliases are stored in `~/.config/ferrumox/aliases.toml` (or the path set via `--alias-file` / `FOX_ALIAS_FILE`). The `fox serve` command reads the same file at startup, so any alias you define is immediately available to the server.

```
fox alias <SUBCOMMAND> [OPTIONS]
```

---

## Subcommands

### fox alias set

Add a new alias or update an existing one.

```
fox alias set <NAME> <MODEL>
```

| Argument | Description |
|----------|-------------|
| `<NAME>` | Short alias name. Can be any string. Use quotes for names containing spaces or dots. |
| `<MODEL>` | The model stem or full filename to map to. Matched using the same resolution order as the server (exact stem → starts-with → contains). |

```bash
# Simple alias
fox alias set llama3 Llama-3.2-3B-Instruct-Q4_K_M

# Alias with dot notation
fox alias set "llama3.2" Llama-3.2-3B-Instruct-Q4_K_M

# Point to a quantization variant
fox alias set phi phi-4-mini-instruct-Q4_K_M

# Update an existing alias (same command — prints "Updated" instead of "Added")
fox alias set phi phi-4-mini-instruct-Q8_0
```

---

### fox alias list

Print all defined aliases as a table.

```
fox alias list
```

Output:

```
ALIAS        MODEL
──────────────────────────────────────────────
llama3       Llama-3.2-3B-Instruct-Q4_K_M
mistral      Mistral-7B-Instruct-v0.3-Q4_K_M
phi          phi-4-mini-instruct-Q4_K_M
```

If no aliases are defined, a message is printed with a hint on how to add one.

---

### fox alias rm

Remove an alias by name.

```
fox alias rm <NAME>
```

```bash
fox alias rm phi
# Removed alias phi
```

Exits with an error if the alias does not exist.

---

## Options

| Flag | Env variable | Description |
|------|---|---|
| `--alias-file <PATH>` | `FOX_ALIAS_FILE` | Path to the aliases TOML file. Defaults to `~/.config/ferrumox/aliases.toml`. |

The `--alias-file` flag is global and applies to all subcommands:

```bash
fox alias --alias-file /etc/fox/aliases.toml list
fox alias --alias-file /etc/fox/aliases.toml set prod-llama Llama-3.2-3B-Instruct-Q4_K_M
```

---

## Aliases TOML format

fox writes aliases in a simple TOML format:

```toml
[aliases]
llama3 = "Llama-3.2-3B-Instruct-Q4_K_M"
mistral = "Mistral-7B-Instruct-v0.3-Q4_K_M"
phi = "phi-4-mini-instruct-Q4_K_M"
"llama3.2" = "Llama-3.2-3B-Instruct-Q4_K_M"
```

You can edit this file by hand — fox reads it on every command. Keys containing characters other than letters, digits, `_`, and `-` are automatically quoted.

---

## How aliases are resolved

When you pass a model name to any fox command or API endpoint, fox resolves it in this order:

1. Exact alias match (this file)
2. Exact filename stem match
3. Filename starts-with match
4. Filename contains match

This means an alias always takes priority over a partial filename match.

---

## Use cases

### Shorter names in API calls

```bash
# Without alias — verbose
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"Llama-3.2-3B-Instruct-Q4_K_M","messages":[...]}'

# With alias "llama3" defined
curl http://localhost:8080/v1/chat/completions \
  -d '{"model":"llama3","messages":[...]}'
```

### Pin a model version for production

```bash
# Production always uses this specific quantization
fox alias set prod-model Llama-3.1-8B-Instruct-Q4_K_M

# Test with Q8 without changing production config
fox alias set staging-model Llama-3.1-8B-Instruct-Q8_0
```

### Open WebUI model selector names

Aliases appear in Open WebUI's model list alongside the full filenames. Define aliases to give models readable display names.

---

## See also

- [fox serve](./serve.md) — `--alias-file` flag loads aliases at server startup
- [Configuration](../configuration.md) — full configuration reference including alias file location
