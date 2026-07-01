# Deployment

This page covers running fox in production: Docker, Docker Compose, systemd, and reverse proxy configuration.

---

## Docker

The official Docker image contains the fox binary, CUDA libraries, and a minimal Debian base.

### Quick start

```bash
docker run -d \
  --name fox \
  --gpus all \
  -p 8080:8080 \
  -v ~/.cache/ferrumox:/root/.cache/ferrumox \
  ferrumox/fox:latest
```

The `-v` mount preserves downloaded models between container restarts. Without it, models are lost when the container is removed.

### Available tags

| Tag | Description |
|-----|-------------|
| `latest` | Latest stable release |
| `1.0.0` | Pinned release version |

> The image includes CUDA support and falls back to CPU automatically when no GPU is available. No separate CPU image is needed.

### Environment variables in Docker

Pass configuration via environment variables:

```bash
docker run -d \
  --name fox \
  --gpus all \
  -p 8080:8080 \
  -v ~/.cache/ferrumox:/root/.cache/ferrumox \
  -e FOX_API_KEY=your-secret-key \
  -e FOX_MAX_MODELS=2 \
  -e FOX_KEEP_ALIVE_SECS=600 \
  -e FOX_MAX_CONTEXT_LEN=8192 \
  -e FOX_JSON_LOGS=true \
  -e HF_TOKEN=hf_xxxxxxxxxxxx \
  ferrumox/fox:latest
```

### Pre-loading a model at container startup

Mount the model and pass the path:

```bash
docker run -d \
  --name fox \
  --gpus all \
  -p 8080:8080 \
  -v ~/.cache/ferrumox:/root/.cache/ferrumox \
  -e FOX_MODEL_PATH=/root/.cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  ferrumox/fox:latest
```

---

## Docker Compose

Docker Compose is the recommended way to run fox alongside other services (e.g., Open WebUI, Prometheus, Grafana).

### fox only

```yaml
# compose.yml
services:
  fox:
    image: ferrumox/fox:latest
    container_name: fox
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - fox-models:/root/.cache/ferrumox
    environment:
      FOX_MAX_MODELS: "2"
      FOX_KEEP_ALIVE_SECS: "600"
      FOX_JSON_LOGS: "true"
      FOX_MAX_CONTEXT_LEN: "4096"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  fox-models:
```

### fox + Open WebUI

```yaml
services:
  fox:
    image: ferrumox/fox:latest
    container_name: fox
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - fox-models:/root/.cache/ferrumox
    environment:
      FOX_MAX_MODELS: "3"
      FOX_JSON_LOGS: "true"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    restart: unless-stopped
    ports:
      - "3000:8080"
    volumes:
      - open-webui-data:/app/backend/data
    environment:
      OLLAMA_BASE_URL: "http://fox:8080"
    depends_on:
      - fox

volumes:
  fox-models:
  open-webui-data:
```

### fox + Prometheus + Grafana

```yaml
services:
  fox:
    image: ferrumox/fox:latest
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - fox-models:/root/.cache/ferrumox
    environment:
      FOX_JSON_LOGS: "true"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus

  grafana:
    image: grafana/grafana:latest
    restart: unless-stopped
    ports:
      - "3001:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      GF_SECURITY_ADMIN_PASSWORD: "admin"

volumes:
  fox-models:
  prometheus-data:
  grafana-data:
```

`prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: fox
    static_configs:
      - targets: ["fox:8080"]
    metrics_path: /metrics
```

---

## systemd service

For Linux servers without Docker, the repository includes a `fox.service` systemd unit file.

### Installation

```bash
# Copy binary
sudo cp target/release/fox /usr/local/bin/

# Create service user
sudo useradd -r -s /bin/false fox

# Create models directory
sudo mkdir -p /var/lib/ferrumox/models
sudo chown fox:fox /var/lib/ferrumox/models

# Copy service file
sudo cp fox.service /etc/systemd/system/

# Reload and enable
sudo systemctl daemon-reload
sudo systemctl enable --now fox
```

### Service file

```ini
[Unit]
Description=fox LLM inference server
After=network.target
Documentation=https://ferrumox.dev/docs

[Service]
Type=simple
User=fox
Group=fox
ExecStart=/usr/local/bin/fox serve \
  --host 0.0.0.0 \
  --port 8080 \
  --max-models 2 \
  --keep-alive-secs 600 \
  --json-logs
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

# Environment
Environment=FOX_MAX_CONTEXT_LEN=4096
Environment=FOX_GPU_MEMORY_FRACTION=0.85
EnvironmentFile=-/etc/ferrumox/env

# Security
NoNewPrivileges=true
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

Put sensitive values in `/etc/ferrumox/env`:

```bash
# /etc/ferrumox/env
HF_TOKEN=hf_xxxxxxxxxxxx
FOX_API_KEY=your-secret-key
```

### Managing the service

```bash
# Start
sudo systemctl start fox

# Stop
sudo systemctl stop fox

# Restart
sudo systemctl restart fox

# Status
sudo systemctl status fox

# Logs (live)
sudo journalctl -u fox -f

# Logs (last 100 lines)
sudo journalctl -u fox -n 100
```

---

## Reverse proxy

Running fox behind a reverse proxy lets you add TLS, authentication, rate limiting, and load balancing.

### nginx

```nginx
upstream fox {
    server 127.0.0.1:8080;
    keepalive 32;
}

server {
    listen 80;
    server_name api.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    # Required for SSE and NDJSON streaming
    proxy_buffering off;
    proxy_cache off;

    location / {
        proxy_pass http://fox;
        proxy_http_version 1.1;

        # Keep-alive for streaming connections
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Increase timeouts for long-running inference
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        proxy_connect_timeout 10s;
    }

    # Rate limiting (optional)
    limit_req_zone $binary_remote_addr zone=fox_api:10m rate=10r/s;
    location /v1/chat/completions {
        limit_req zone=fox_api burst=20 nodelay;
        proxy_pass http://fox;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_read_timeout 300s;
    }
}
```

### Caddy

```caddy
api.example.com {
    reverse_proxy localhost:8080 {
        flush_interval -1   # required for streaming
        transport http {
            read_timeout 5m
        }
    }
}
```

### Traefik

```yaml
# docker-compose with Traefik labels
services:
  fox:
    image: ferrumox/fox:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.fox.rule=Host(`api.example.com`)"
      - "traefik.http.routers.fox.entrypoints=websecure"
      - "traefik.http.routers.fox.tls.certresolver=letsencrypt"
      - "traefik.http.services.fox.loadbalancer.server.port=8080"
      # Disable buffering for streaming
      - "traefik.http.middlewares.fox-headers.headers.customresponseheaders.X-Accel-Buffering=no"
```

---

## Health checks

Use the `/health` endpoint for container health checks and load balancer probes.

### Docker health check

```yaml
services:
  fox:
    image: ferrumox/fox:latest
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

### Kubernetes liveness/readiness probe

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 30
  periodSeconds: 15

readinessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## Logging

### JSON logs (production)

Enable structured JSON logs with `--json-logs` or `FOX_JSON_LOGS=true`. Each log line is a JSON object:

```json
{"timestamp":"2026-03-12T10:00:00Z","level":"INFO","target":"fox::api","message":"request completed","model":"llama3.2","tokens":94,"duration_ms":412}
```

These can be ingested by any log aggregator that handles NDJSON (Loki, Elasticsearch, Splunk, CloudWatch).

### Shipping logs to Loki

```yaml
# Promtail config
scrape_configs:
  - job_name: fox
    static_configs:
      - targets: ["localhost"]
        labels:
          job: fox
          __path__: /var/log/fox/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            message: message
            model: model
      - labels:
          level:
          model:
```

Or if using Docker, use the Loki logging driver:

```yaml
services:
  fox:
    image: ferrumox/fox:latest
    logging:
      driver: loki
      options:
        loki-url: "http://loki:3100/loki/api/v1/push"
        loki-labels: "job=fox"
```
