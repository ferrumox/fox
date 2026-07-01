# Troubleshooting

Common problems and how to resolve them.

---

## Server won't start

### Port already in use

```
Error: Address already in use (os error 98)
```

Another process is already listening on port 8080 (or whichever port you configured).

```bash
# Find what is using the port
lsof -i :8080
# or
ss -tlnp | grep 8080

# Kill it, or use a different port
fox serve --port 9090
```

### Permission denied binding to port

Ports below 1024 require root on Linux. Use a port ≥ 1024, or grant the binary the capability:

```bash
sudo setcap 'cap_net_bind_service=+ep' $(which fox)
```

### Config file parse error

```
Error: failed to parse config: expected string, found integer at line 4
```

fox reads `~/.config/ferrumox/config.toml` before CLI flags. Open the file and check for type mismatches (all string values must be quoted).

---

## Model won't load

### File not found

```
Error: model file not found: /home/user/.cache/ferrumox/models/my-model.gguf
```

Check that the path exists and the filename matches exactly (case-sensitive on Linux):

```bash
fox list
ls ~/.cache/ferrumox/models/
```

If you're using an alias, verify it points to an existing model:

```bash
fox alias list
fox show <alias-name>
```

### Out of memory / allocation failed

```
Error: failed to allocate KV cache: not enough memory
```

The model does not fit in available GPU VRAM or RAM.

Options:
- Use a smaller quantization (Q4_K_M instead of Q8_0, or Q3_K_M)
- Reduce `--gpu-memory-fraction` (default 0.85) to leave more headroom
- Reduce `--max-context-len` (e.g., 2048 instead of 4096)
- Use a smaller model variant (3B instead of 7B)

```bash
fox serve --max-context-len 2048 --gpu-memory-fraction 0.75
```

### GGUF format error / unsupported architecture

```
Error: unsupported model architecture: exl2
```

fox only supports GGUF format files. EXL2, AWQ, GPTQ, and safetensors formats are not supported. Download a GGUF variant from HuggingFace:

```bash
fox pull bartowski/Llama-3.2-3B-Instruct-GGUF
```

---

## GPU not detected

### No CUDA device found

```
Warning: no CUDA device found, falling back to CPU
```

Possible causes:

1. **No NVIDIA GPU** — fox runs on CPU. Performance will be significantly lower.
2. **CUDA drivers not installed** — install the NVIDIA driver and CUDA toolkit for your OS.
3. **Wrong CUDA version** — fox is built against a specific CUDA version. Check that your driver supports it:
   ```bash
   nvidia-smi
   # Check "CUDA Version" in the top-right corner
   ```
4. **`libcuda.so` not in library path** — add the CUDA library directory:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```

### GPU detected but not used

If `fox serve --verbose` shows `GPU 0 layers` despite a capable GPU:

```bash
# Check VRAM available
nvidia-smi --query-gpu=memory.free,memory.total --format=csv

# Increase the GPU memory fraction (default: 0.85)
fox serve --gpu-memory-fraction 0.9
```

### Metal (macOS) not working

On Apple Silicon, fox uses Metal automatically. If inference is unexpectedly slow, check that the binary was built for `aarch64-apple-darwin` (not Rosetta):

```bash
file $(which fox)
# should say: Mach-O 64-bit executable arm64
```

---

## API authentication errors

### 401 Unauthorized

```json
{"error": {"message": "invalid api key", "type": "invalid_request_error"}}
```

You started fox with `--api-key` but the client is not sending the key, or is sending the wrong one.

Client must send: `Authorization: Bearer <your-key>`

```bash
# Correct
curl http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer my-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2","messages":[{"role":"user","content":"Hi"}]}'
```

If you do not want authentication, start fox without `--api-key`.

---

## Model not found errors

### 404 from the API

```json
{"error": {"message": "model 'gpt-4' not found", "type": "not_found"}}
```

The model name in the API request does not resolve to any loaded or on-disk model.

Resolution order: alias → exact stem → starts-with → contains

```bash
# Check what model names are available
fox list
fox alias list

# Check what is currently loaded (server must be running)
fox ps
```

Use a name that appears in `fox list` or define an alias:

```bash
fox alias set gpt4-local Llama-3.2-3B-Instruct-Q4_K_M
```

---

## Streaming issues

### Response arrives all at once instead of streaming

Some reverse proxies (nginx, Caddy, HAProxy) buffer responses by default. For SSE/streaming to work, disable buffering:

**nginx:**
```nginx
location / {
    proxy_pass http://localhost:8080;
    proxy_buffering off;
    proxy_cache off;
    proxy_set_header X-Accel-Buffering no;
}
```

**Caddy:**
```caddyfile
reverse_proxy localhost:8080 {
    flush_interval -1
}
```

### `[DONE]` never arrives / stream hangs

This usually means the client closed the connection before generation finished, or the model hit the max token limit. Check `--max-new-tokens` in `fox run` or `max_tokens` in your API request. The default is 512.

---

## Docker networking

### Cannot reach fox from inside a container

If fox is running on the host and a container needs to reach it:

- **Linux**: use `http://172.17.0.1:8080` (host IP on the default bridge network) or `--network host`
- **Docker Desktop (macOS/Windows)**: use `http://host.docker.internal:8080`

```bash
docker run -e OLLAMA_BASE_URL=http://host.docker.internal:8080 open-webui
```

### Model files not available inside container

Mount the models directory as a volume:

```bash
docker run \
  -v ~/.cache/ferrumox/models:/root/.cache/ferrumox/models \
  ferrumox/fox:latest fox serve
```

Or set a custom path:

```bash
docker run \
  -v /data/models:/models \
  ferrumox/fox:latest fox serve --models-dir /models
```

---

## Performance issues

### First request is slow, subsequent requests are fast

This is expected. The first request triggers model loading (lazy load). Subsequent requests reuse the loaded model.

To pre-load a model at startup:

```bash
fox serve --model-path ~/.cache/ferrumox/models/my-model.gguf
```

### KV cache full — requests queuing

```
Warning: KV cache utilization at 100%, request queued
```

The context windows of all active requests have filled the KV cache. Options:

- Reduce `--max-context-len` (smaller context per request, more concurrent requests fit)
- Reduce `--gpu-memory-fraction` so fox claims less VRAM (counterintuitively — check total available first)
- Upgrade to a GPU with more VRAM
- Use a smaller quantization to fit more KV cache blocks

### Low throughput / slow generation

1. **Check that the GPU is being used**: `fox serve --verbose` should show GPU layers > 0
2. **Check quantization**: Q8_0 is much slower than Q4_K_M on memory-bandwidth-limited hardware
3. **Reduce context length**: a 32K context takes significantly more memory than 4K
4. **Batch size**: under a running server, concurrent requests are batched together automatically — single-request throughput will always be lower than the numbers in `fox-bench` results

---

## HuggingFace download problems

### 401 / 403 on `fox pull`

The model is gated (requires accepting a license) or private. You need a HuggingFace token:

```bash
# Get a token at https://huggingface.co/settings/tokens
export HF_TOKEN=hf_xxxxxxxxxxxx
fox pull meta-llama/Llama-3.1-8B-Instruct-GGUF
```

### Download interrupted / partial file

fox writes to a `.part` file and renames only on success. Re-run `fox pull` — it will skip completed files and resume from scratch for incomplete ones.

### SSL certificate errors

```
Error: SSL handshake failed
```

Your system's CA bundle may be outdated. On Debian/Ubuntu:

```bash
sudo apt install --reinstall ca-certificates
```

---

## Checking logs

fox logs to stderr. Capture them for debugging:

```bash
fox serve 2>fox.log &
tail -f fox.log
```

Use `--verbose` to enable detailed engine output (model loading, token counts, GPU layer allocation):

```bash
fox serve --verbose
fox run --model-path model.gguf --verbose "test prompt"
```

---

## See also

- [FAQ](./faq.md) — general questions and quick answers
- [Configuration](./configuration.md) — all configuration options
- [Deployment](./deployment.md) — Docker, systemd, reverse proxy setup
- [GitHub Issues](https://github.com/ferrumox/fox/issues) — report a bug
