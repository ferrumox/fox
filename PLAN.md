# Plan — Ferrum Engine

## Visión
Motor de inferencia de LLMs en Rust, open source, con mayor throughput que vLLM y distribución más simple que Ollama. Diferencial clave: **un solo binario, sin Python, sin dependencias de runtime**.

---

## Versiones lanzadas

### v0.1.0 (2026-03-08) — MVP funcional
- API OpenAI-compatible: `POST /v1/chat/completions` (SSE streaming + non-streaming), `POST /v1/completions`, `GET /v1/models`, `GET /health`
- Inference engine: llama.cpp FFI, GGUF, continuous batching con LIFO preemption
- KV cache manager: block pool con free_list, cálculo automático desde GPU memory
- Muestreo: temperature, top-p
- Config completa: CLI flags + env vars
- Shutdown graceful, logging estructurado (tracing, JSON mode)

### v0.2.0 (2026-03-09) — Performance y observabilidad
- Muestreo estocástico completo: top-K, repetition penalty, seed para reproducibilidad
- Fix crítico: `kv_seq_id` estable por request (evita crashes de llama_decode con >1 request)
- Docker support: Dockerfile multi-stage + docker-compose
- `ferrum-bench`: benchmark integrado con TTFT P50/P95, throughput, latencia P50/P95/P99
- `SamplingParams` struct unificado para agrupar hyper-params de sampling

### v0.3.0 (en progreso) — Optimización de memoria y API completa
Ver Fase 2 (ítems marcados con `[x]`).

---

## Fases

### Fase 1 — MVP funcional `(Mes 1-3)` ✅ COMPLETADA
**Meta: algo que funcione y se pueda mostrar**

```
Semana 1-2: Setup y FFI
  [x] Proyecto Rust base con Cargo workspace
  [x] build.rs con bindgen → llama.cpp
  [x] Cargar modelo GGUF y generar tokens
  [x] llama.cpp backend funcional

Semana 3-4: Servidor básico
  [x] axum con POST /v1/chat/completions y /v1/completions
  [x] API OpenAI-compatible con streaming SSE
  [x] Config por CLI con clap + env vars
  [x] curl funcionando contra el servidor

Semana 5-6: KV-Cache Manager
  [x] PhysicalBlock pool con free_list
  [x] Allocate / free / can_allocate
  [x] Cálculo automático de bloques desde memoria GPU disponible
  [x] Tests unitarios del manager

Semana 7-8: Scheduler + Continuous Batching
  [x] InferenceRequest con estados (Waiting→Prefilling→Decoding→Finished)
  [x] schedule_step() con admit / evict / preempt LIFO
  [x] Engine loop principal con tokio + spawn_blocking para FFI
  [x] Streaming SSE en la API
  [x] kv_seq_id estable (fix crash multi-request)

Semana 9-10: Pulido MVP
  [x] /health endpoint con métricas básicas
  [x] GET /v1/models
  [x] Logging estructurado con tracing (pretty + JSON)
  [x] README con instalación y ejemplos
  [x] Docker image oficial + docker-compose
  [x] ferrum-bench: benchmark integrado (TTFT, throughput, latencia percentiles)
  [ ] Benchmark comparativo publicado vs Ollama (pendiente ejecución)
```

**Entregable:** servidor funcional, OpenAI-compatible, con continuous batching básico

---

### Fase 2 — Performance `(Mes 3-6)`
**Meta: superar a vLLM en throughput en hardware equivalente**

```
Mes 3-4: Optimizaciones de memoria (v0.3.0)
  [x] PageTable: mapping explícito lógico→físico por request
  [x] ref_count por bloque (infraestructura para CoW real)
  [x] Prefix caching: reutilizar KV cache entre requests con prompt idéntico
  [x] copy_on_write: infraestructura para bloques compartidos
  [ ] Swap CPU↔GPU para requests preemptados

Mes 3-4: Extensiones de API (v0.3.0)
  [x] Stop sequences (parámetro stop: string[])
  [x] Endpoint GET /metrics (Prometheus scrape format)
  [ ] Streaming: incluir usage en el último chunk

Mes 4-5: Optimizaciones de cómputo
  [ ] Flash Attention 2 via FFI (kernels C++ de Tri Dao)
  [ ] Quantización AWQ y GPTQ (además de GGUF)
  [ ] CUDA Graphs para decode steps repetitivos
  [ ] Overlap prefill/decode con CUDA streams

Mes 5-6: Escalado
  [ ] Tensor parallelism (multi-GPU, un modelo en varias GPUs)
  [ ] Speculative decoding (modelo draft pequeño + modelo grande)
  [ ] Prefill/decode disaggregation (máquinas separadas)
  [ ] Benchmark completo: latencia P50/P95/P99, throughput, memory usage
```

**Entregable:** benchmarks publicados, blog post técnico, primeros stars en GitHub

---

### Fase 3 — Producto completo `(Mes 6-12)`
**Meta: alternativa real con comunidad**

```
Mes 6-7: Experiencia de usuario
  [ ] CLI tipo Ollama (ferrum pull llama3, ferrum run llama3)
  [ ] Descarga automática de modelos desde HuggingFace Hub
  [ ] Web UI básica para probar modelos (opcional)

Mes 7-8: Ecosistema de modelos
  [ ] Soporte safetensors nativo (sin llama.cpp para modelos HF)
  [ ] candle como segundo backend (puro Rust)
  [ ] Soporte Vision models (LLaVA, Qwen-VL)
  [ ] Soporte Embedding models

Mes 8-9: Observabilidad y producción
  [ ] Métricas Prometheus completas (ya iniciado en v0.3.0)
  [ ] Distributed tracing con OpenTelemetry
  [ ] Rate limiting por API key
  [ ] Autenticación básica

Mes 9-12: Comunidad
  [ ] Documentación completa (mdBook)
  [ ] Plugin system para custom samplers
  [ ] Soporte Apple Silicon (Metal backend)
  [ ] CI/CD con benchmarks automáticos en cada PR
```

---

## Arquitectura actual (v0.3.0)

```
┌─────────────────────────────────────────────────────┐
│                   Cliente                           │
│         (curl, OpenAI SDK, LangChain...)            │
└──────────────────────┬──────────────────────────────┘
                       │ HTTP / SSE
┌──────────────────────▼──────────────────────────────┐
│              API Layer (axum)                       │
│  /v1/chat/completions  /v1/completions              │
│  /v1/models  /health  /metrics                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│            Scheduler (tokio async)                  │
│  Continuous batching · LIFO preemption              │
│  PrefixCache (hash-based, exact prompt match)       │
└───────────┬──────────────────────┬──────────────────┘
            │                      │
┌───────────▼──────────┐ ┌────────▼─────────────────┐
│   KV Cache Manager   │ │     Inference Engine      │
│   PageTable per req  │ │   prefill() + decode()    │
│   ref_count / block  │ │   stop sequences          │
│   copy_on_write infra│ │   prefix KV copy          │
└──────────────────────┘ └────────┬─────────────────┘
                                  │
                       ┌──────────▼──────────────────┐
                       │       Model Backend          │
                       │  llama.cpp FFI (GGUF)        │
                       └──────────┬──────────────────┘
                                  │
                       ┌──────────▼──────────────────┐
                       │      GPU / CPU              │
                       │  CUDA · CPU-only            │
                       └─────────────────────────────┘
```

---

## Métricas de éxito por fase

| Métrica | Fase 1 | Fase 2 | Fase 3 |
|---|---|---|---|
| Tokens/segundo (7B, A100) | >500 | >3000 | >5000 |
| Latencia TTFT | <500ms | <100ms | <50ms |
| Requests concurrentes | 8 | 64 | 256 |
| Modelos soportados | GGUF | GGUF + AWQ | GGUF + safetensors |
| GitHub stars | — | 500+ | 2000+ |

---

## Comandos CLI

```
ferrum serve       # arrancar el servidor
ferrum-bench       # benchmark integrado (disponible desde v0.2.0)
ferrum pull        # descargar modelo (pendiente Fase 3)
ferrum run         # inferencia rápida desde CLI (pendiente Fase 3)
```