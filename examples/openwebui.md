# Connecting Open WebUI to ferrumox

ferrumox is fully compatible with Open WebUI's Ollama backend. No plugins or
extensions needed — just point Open WebUI at your ferrumox server.

## Quick start

### 1. Start ferrumox

```bash
# Pull a model (if you don't have one)
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF

# Start the server
fox serve
# Server listens on http://localhost:8080
```

### 2. Start Open WebUI

```bash
docker run -d \
  -p 3000:8080 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:8080 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

Open `http://localhost:3000` in your browser. Open WebUI will automatically
discover models via `GET /api/tags`.

### 3. (Optional) systemd service

To keep ferrumox running in the background:

```bash
sudo cp fox.service /etc/systemd/system/
sudo systemctl enable --now fox
```

## Multiple models

ferrumox supports serving multiple models simultaneously:

```bash
fox serve --max-models 3
```

Open WebUI will list all `.gguf` files in `~/.cache/ferrumox/models/` and let
you switch between them without restarting the server.

## Aliases

Create `~/.config/ferrumox/aliases.toml` to give short names to models:

```toml
[aliases]
"llama3" = "Llama-3.2-3B-Instruct-Q4_K_M"
"mistral" = "Mistral-7B-Instruct-v0.3-Q4_K_M"
```

These aliases appear in the Open WebUI model selector.

## Verified compatibility

| Feature | Status |
|---------|--------|
| Model list | ✓ |
| Chat | ✓ |
| Streaming | ✓ |
| RAG / Embeddings | ✓ |
| Image upload | — (text models only) |
| Function calling | ✓ (via OpenAI endpoint) |
