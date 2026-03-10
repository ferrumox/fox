// Axum routes for OpenAI-compatible API.

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{delete, get, post},
    Json, Router,
};
use bytes::Bytes;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::convert::Infallible;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::cli::show::{parse_architecture, parse_quantization};
use crate::cli::{format_size, list_models};
use crate::model_registry::ModelRegistry;
use crate::scheduler::{InferenceRequest, SamplingParams, StopReason, Token};

use super::types::*;

/// Shared state for all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// Stem of the model supplied via `--model-path` (pre-loaded at startup).
    pub primary_model: String,
    /// Injected as the first message when no system message is present.
    pub system_prompt: Option<String>,
    /// Unix timestamp (seconds) when the server started.
    pub started_at: u64,
    /// Directory where `.gguf` model files are stored.
    pub models_dir: PathBuf,
    /// Cache of SHA256 digests keyed by file path. Computed once per file.
    pub digest_cache: Arc<Mutex<HashMap<PathBuf, String>>>,
    /// HuggingFace API token for authenticated model pulls.
    pub hf_token: Option<String>,
}

pub fn router(
    registry: Arc<ModelRegistry>,
    primary_model: String,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
    hf_token: Option<String>,
) -> Router {
    let state = AppState {
        registry,
        primary_model,
        system_prompt,
        started_at,
        models_dir,
        digest_cache: Arc::new(Mutex::new(HashMap::new())),
        hf_token,
    };
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/v1/embeddings", post(v1_embeddings))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        // Ollama-compatible endpoints
        .route("/api/version", get(ollama_version))
        .route("/api/tags", get(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/delete", delete(ollama_delete))
        .route("/api/embed", post(ollama_embed))
        .route("/api/generate", post(ollama_generate))
        .route("/api/chat", post(ollama_chat))
        .route("/api/pull", post(super::pull_handler::ollama_pull))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Returns the current time as a minimal RFC 3339 UTC string.
fn now_rfc3339() -> String {
    let s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    let year = 1970u64 + days / 365;
    let doy = days % 365;
    let month = doy / 30 + 1;
    let day = doy % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert a `StopReason` to the Ollama `done_reason` string.
fn ollama_done_reason(reason: &Option<StopReason>) -> String {
    match reason {
        Some(StopReason::Length) => "length".to_string(),
        _ => "stop".to_string(),
    }
}

/// Build `SamplingParams` from Ollama options, applying Ollama-style defaults.
fn sampling_from_ollama(opts: Option<&OllamaOptions>) -> (SamplingParams, usize) {
    let (temp, top_p, top_k, rep, seed, max_tokens, stop) = match opts {
        Some(o) => (
            o.temperature.unwrap_or(0.8),
            o.top_p.unwrap_or(0.9),
            o.top_k.unwrap_or(40),
            o.repeat_penalty.unwrap_or(1.1),
            o.seed,
            o.num_predict.unwrap_or(128) as usize,
            o.stop.clone(),
        ),
        None => (0.8, 0.9, 40, 1.1, None, 128, None),
    };
    (
        SamplingParams {
            temperature: temp,
            top_p,
            top_k,
            repetition_penalty: rep,
            seed,
            stop,
            show_thinking: false,
        },
        max_tokens,
    )
}

/// Build a response body with `application/x-ndjson` content-type for Ollama streaming.
fn ndjson_response(
    stream: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> axum::response::Response {
    axum::response::Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

/// Try to parse `response` text as a JSON tool call.
/// Returns `(content, tool_calls)` — when a tool call is detected, `content` is empty.
fn try_parse_tool_call(response: &str) -> (String, Option<Vec<ToolCall>>) {
    let trimmed = response.trim();
    let value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => return (response.to_string(), None),
    };

    // Pattern: {"name": "...", "arguments": {...}}
    if let (Some(name), Some(args)) = (
        value.get("name").and_then(|n| n.as_str()),
        value.get("arguments"),
    ) {
        let call = ToolCall {
            id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        };
        return (String::new(), Some(vec![call]));
    }

    // Pattern: {"tool_calls": [...]}
    if let Some(calls) = value.get("tool_calls").and_then(|tc| tc.as_array()) {
        let tool_calls: Vec<ToolCall> = calls
            .iter()
            .filter_map(|c| {
                let name = c.get("name")?.as_str()?.to_string();
                let args = c.get("arguments")?.to_string();
                Some(ToolCall {
                    id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name,
                        arguments: args,
                    },
                })
            })
            .collect();
        if !tool_calls.is_empty() {
            return (String::new(), Some(tool_calls));
        }
    }

    (response.to_string(), None)
}

/// Build a system message that describes available tools.
fn tools_system_message(tools: &[Tool]) -> String {
    let json = serde_json::to_string_pretty(tools).unwrap_or_default();
    format!(
        "You have access to the following tools:\n{json}\n\n\
         When you want to call a tool, respond ONLY with a JSON object:\n\
         {{\"name\": \"<tool_name>\", \"arguments\": {{<key>: <value>}}}}\n\n\
         If you don't need a tool, respond normally."
    )
}

// ---------------------------------------------------------------------------
// Prometheus
// ---------------------------------------------------------------------------

async fn metrics_handler() -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut body = String::new();
    if let Err(e) = encoder.encode_utf8(&metric_families, &mut body) {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("metrics encoding error: {e}"),
        );
    }
    (
        axum::http::StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

// ---------------------------------------------------------------------------
// Core handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let entry = state.registry.get_or_load(&state.primary_model).await.ok();
    let (kv_cache_usage, queue_depth, active_requests, model_name) = match entry {
        Some(e) => (
            e.engine.kv_cache_usage(),
            e.engine.queue_depth(),
            e.engine.active_requests(),
            e.engine.model_name().to_string(),
        ),
        None => (0.0, 0, 0, state.primary_model.clone()),
    };
    Json(HealthResponse {
        status: "ok".to_string(),
        kv_cache_usage,
        queue_depth,
        active_requests,
        model_name,
        started_at: state.started_at,
    })
}

/// Lists all `.gguf` models available on disk (OpenAI format).
async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let entries = list_models(&state.models_dir).unwrap_or_default();
    let data = entries
        .iter()
        .filter_map(|(path, _)| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|stem| ModelInfo {
                    id: stem.to_string(),
                    object: "model".to_string(),
                })
        })
        .collect();
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

// ---------------------------------------------------------------------------
// Ollama compatibility handlers
// ---------------------------------------------------------------------------

async fn ollama_version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Compute SHA256 of a file, returning `"sha256:<hex>"`.
async fn file_digest(path: PathBuf, cache: Arc<Mutex<HashMap<PathBuf, String>>>) -> String {
    if let Some(cached) = cache.lock().unwrap().get(&path).cloned() {
        return cached;
    }
    let digest = tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
        Ok::<String, std::io::Error>(format!("sha256:{}", hex::encode(hasher.finalize())))
    })
    .await
    .ok()
    .and_then(|r| r.ok())
    .unwrap_or_else(|| "sha256:unknown".to_string());
    digest
}

async fn get_digest(path: &PathBuf, cache: &Arc<Mutex<HashMap<PathBuf, String>>>) -> String {
    if let Some(cached) = cache.lock().unwrap().get(path).cloned() {
        return cached;
    }
    let path_clone = path.clone();
    let cache_clone = cache.clone();
    let digest = file_digest(path_clone.clone(), cache_clone.clone()).await;
    cache_clone.lock().unwrap().insert(path_clone, digest.clone());
    digest
}

fn modified_at_rfc3339(meta: &std::fs::Metadata) -> String {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| {
            let s = d.as_secs();
            let sec = s % 60;
            let min = (s / 60) % 60;
            let hour = (s / 3600) % 24;
            let days_since_epoch = s / 86400;
            let year = 1970u64 + days_since_epoch / 365;
            let day_of_year = days_since_epoch % 365;
            let month = day_of_year / 30 + 1;
            let day = day_of_year % 30 + 1;
            format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
        })
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".to_string())
}

async fn ollama_tags(State(state): State<AppState>) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let mut models = Vec::with_capacity(entries.len());
    for (path, meta) in &entries {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        let digest = get_digest(path, &state.digest_cache).await;
        models.push(OllamaModel {
            name: stem.to_string(),
            size: meta.len(),
            digest,
            details: OllamaDetails {
                format: "gguf".to_string(),
                family: parse_architecture(stem).unwrap_or("unknown").to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: parse_quantization(stem).unwrap_or("unknown").to_string(),
            },
            modified_at: modified_at_rfc3339(meta),
        });
    }

    Json(TagsResponse { models }).into_response()
}

async fn ollama_ps(State(state): State<AppState>) -> Json<PsResponse> {
    let loaded = state.registry.loaded();
    let mut ps_entries = Vec::with_capacity(loaded.len());

    for (name, entry) in &loaded {
        let file_info = list_models(&state.models_dir)
            .ok()
            .and_then(|entries| {
                entries.into_iter().find(|(path, _)| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|stem| stem == name.as_str())
                        .unwrap_or(false)
                })
            });

        let (size, digest) = if let Some((path, meta)) = file_info {
            let d = get_digest(&path, &state.digest_cache).await;
            (meta.len(), d)
        } else {
            (0u64, "sha256:unknown".to_string())
        };

        let model_name = entry.engine.model_name().to_string();
        ps_entries.push(PsEntry {
            name: model_name.clone(),
            size,
            digest,
            details: OllamaDetails {
                format: "gguf".to_string(),
                family: parse_architecture(&model_name).unwrap_or("unknown").to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: parse_quantization(&model_name)
                    .unwrap_or("unknown")
                    .to_string(),
            },
            expires_at: "0001-01-01T00:00:00Z".to_string(),
            size_vram: 0,
        });
    }

    Json(PsResponse { models: ps_entries })
}

async fn ollama_show(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let found = entries.into_iter().find(|(path, _)| {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        stem == req.name || name == req.name
    });

    match found {
        None => (
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.name),
        )
            .into_response(),
        Some((path, meta)) => {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
            let arch = parse_architecture(stem).unwrap_or("unknown");
            let quant = parse_quantization(stem).unwrap_or("unknown");
            let size_str = format_size(meta.len());
            let digest = get_digest(&path, &state.digest_cache).await;
            let resp = ShowResponse {
                modelfile: format!("# GGUF model: {}", stem),
                parameters: String::new(),
                template: String::new(),
                details: OllamaDetails {
                    format: "gguf".to_string(),
                    family: arch.to_string(),
                    parameter_size: "unknown".to_string(),
                    quantization_level: quant.to_string(),
                },
                model_info: serde_json::json!({
                    "general.architecture": arch,
                    "general.quantization": quant,
                    "general.size": size_str,
                    "general.digest": digest,
                    "general.modified_at": modified_at_rfc3339(&meta),
                    "general.path": path.display().to_string(),
                }),
            };
            Json(resp).into_response()
        }
    }
}

async fn ollama_delete(
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let found = entries.into_iter().find(|(path, _)| {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        stem == req.name || name == req.name
    });

    match found {
        None => (
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.name),
        )
            .into_response(),
        Some((path, _)) => {
            state.registry.unload(&req.name);
            match std::fs::remove_file(&path) {
                Ok(_) => StatusCode::OK.into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to delete model: {e}"),
                )
                    .into_response(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ollama /api/generate
// ---------------------------------------------------------------------------

async fn ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> axum::response::Response {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let engine = &entry.engine;

    // Build message list (system + user prompt).
    let mut messages: Vec<(String, String)> = Vec::new();
    if let Some(ref sys) = req.system {
        messages.push(("system".to_string(), sys.clone()));
    }
    messages.push(("user".to_string(), req.prompt.clone()));

    let prompt = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens: Vec<i32> = engine.tokenize(&prompt).unwrap_or_else(|_| {
        prompt.bytes().map(|b| b as i32).take(4096).collect()
    });
    let prompt_tokens_len = prompt_tokens.len();

    let (sampling, max_tokens) = sampling_from_ollama(req.options.as_ref());
    let req_id = engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    engine.submit_request(InferenceRequest::new(
        req_id,
        prompt_tokens,
        max_tokens,
        sampling,
        tx,
    ));

    let model_name = req.model.clone();
    let stream = req.stream.unwrap_or(true);

    if stream {
        let start = Instant::now();
        let stream = async_stream::stream! {
            let mut eval_count: u32 = 0;
            while let Some(token) = rx.recv().await {
                let is_done = token.stop_reason.is_some();
                let chunk = OllamaGenerateChunk {
                    model: model_name.clone(),
                    created_at: now_rfc3339(),
                    response: token.text.clone(),
                    done: is_done,
                    done_reason: if is_done { Some(ollama_done_reason(&token.stop_reason)) } else { None },
                    total_duration: if is_done { Some(start.elapsed().as_nanos() as u64) } else { None },
                    load_duration: if is_done { Some(0) } else { None },
                    prompt_eval_count: if is_done { Some(prompt_tokens_len as u32) } else { None },
                    eval_count: if is_done { Some(eval_count) } else { None },
                };
                eval_count += 1;
                let mut line = serde_json::to_string(&chunk).unwrap_or_default();
                line.push('\n');
                yield Ok::<_, Infallible>(Bytes::from(line.into_bytes()));
                if is_done { break; }
            }
        };
        ndjson_response(stream)
    } else {
        let start = Instant::now();
        let mut full_response = String::new();
        let mut eval_count: u32 = 0;
        let mut stop_reason = None;
        while let Some(token) = rx.recv().await {
            full_response.push_str(&token.text);
            eval_count += 1;
            if token.stop_reason.is_some() {
                stop_reason = token.stop_reason;
                break;
            }
        }
        let chunk = OllamaGenerateChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: full_response,
            done: true,
            done_reason: Some(ollama_done_reason(&stop_reason)),
            total_duration: Some(start.elapsed().as_nanos() as u64),
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            eval_count: Some(eval_count),
        };
        let mut line = serde_json::to_string(&chunk).unwrap_or_default();
        line.push('\n');
        axum::response::Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(line))
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// Ollama /api/chat
// ---------------------------------------------------------------------------

async fn ollama_chat(
    State(state): State<AppState>,
    Json(req): Json<OllamaChatRequest>,
) -> axum::response::Response {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let engine = &entry.engine;

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let prompt = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens: Vec<i32> = engine.tokenize(&prompt).unwrap_or_else(|_| {
        prompt.bytes().map(|b| b as i32).take(4096).collect()
    });
    let prompt_tokens_len = prompt_tokens.len();

    let (sampling, max_tokens) = sampling_from_ollama(req.options.as_ref());
    let req_id = engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    engine.submit_request(InferenceRequest::new(
        req_id,
        prompt_tokens,
        max_tokens,
        sampling,
        tx,
    ));

    let model_name = req.model.clone();
    let stream = req.stream.unwrap_or(true);

    if stream {
        let start = Instant::now();
        let stream = async_stream::stream! {
            let mut eval_count: u32 = 0;
            while let Some(token) = rx.recv().await {
                let is_done = token.stop_reason.is_some();
                let chunk = OllamaChatChunk {
                    model: model_name.clone(),
                    created_at: now_rfc3339(),
                    message: OllamaChatMessage {
                        role: "assistant".to_string(),
                        content: token.text.clone(),
                    },
                    done: is_done,
                    done_reason: if is_done { Some(ollama_done_reason(&token.stop_reason)) } else { None },
                    total_duration: if is_done { Some(start.elapsed().as_nanos() as u64) } else { None },
                    load_duration: if is_done { Some(0) } else { None },
                    prompt_eval_count: if is_done { Some(prompt_tokens_len as u32) } else { None },
                    eval_count: if is_done { Some(eval_count) } else { None },
                };
                eval_count += 1;
                let mut line = serde_json::to_string(&chunk).unwrap_or_default();
                line.push('\n');
                yield Ok::<_, Infallible>(Bytes::from(line.into_bytes()));
                if is_done { break; }
            }
        };
        ndjson_response(stream)
    } else {
        let start = Instant::now();
        let mut full_content = String::new();
        let mut eval_count: u32 = 0;
        let mut stop_reason = None;
        while let Some(token) = rx.recv().await {
            full_content.push_str(&token.text);
            eval_count += 1;
            if token.stop_reason.is_some() {
                stop_reason = token.stop_reason;
                break;
            }
        }
        let chunk = OllamaChatChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content: full_content,
            },
            done: true,
            done_reason: Some(ollama_done_reason(&stop_reason)),
            total_duration: Some(start.elapsed().as_nanos() as u64),
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            eval_count: Some(eval_count),
        };
        let mut line = serde_json::to_string(&chunk).unwrap_or_default();
        line.push('\n');
        axum::response::Response::builder()
            .status(200)
            .header(header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(line))
            .unwrap()
    }
}

// ---------------------------------------------------------------------------
// OpenAI embeddings
// ---------------------------------------------------------------------------

async fn v1_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let engine = &entry.engine;
    let inputs = req.input.into_vec();
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0u32;

    for (i, text) in inputs.iter().enumerate() {
        match engine.embed(text).await {
            Ok(embedding) => {
                let tokens = engine.tokenize(text).map(|t| t.len()).unwrap_or(0) as u32;
                total_tokens += tokens;
                data.push(EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding,
                    index: i,
                });
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    })
    .into_response()
}

async fn ollama_embed(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbedRequest>,
) -> impl IntoResponse {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let engine = &entry.engine;
    let inputs = req.input.into_vec();
    let mut embeddings = Vec::with_capacity(inputs.len());

    for text in &inputs {
        match engine.embed(text).await {
            Ok(embedding) => embeddings.push(embedding),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    Json(OllamaEmbedResponse {
        model: req.model,
        embeddings,
    })
    .into_response()
}

// ---------------------------------------------------------------------------
// OpenAI chat/completions (with tool calling and structured output)
// ---------------------------------------------------------------------------

fn finish_reason_str(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Eos => "stop",
        StopReason::Length => "length",
        StopReason::Preempt => "stop",
        StopReason::StopSequence => "stop",
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
    };
    let engine = &entry.engine;
    let id = Uuid::new_v4().to_string();
    let req_id = engine.next_request_id();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    // Inject system prompt when configured and none is present in the request.
    if let Some(ref sp) = state.system_prompt {
        if messages.first().map(|(r, _)| r.as_str()) != Some("system") {
            messages.insert(0, ("system".to_string(), sp.clone()));
        }
    }

    // Inject tool descriptions as a system message.
    if let Some(ref tools) = req.tools {
        let tool_msg = tools_system_message(tools);
        if messages.first().map(|(r, _)| r.as_str()) == Some("system") {
            messages[0].1.push_str(&format!("\n\n{tool_msg}"));
        } else {
            messages.insert(0, ("system".to_string(), tool_msg));
        }
    }

    // Inject JSON-mode instruction.
    if let Some(ref rf) = req.response_format {
        if rf.format_type == "json_object" {
            let json_instr =
                "Respond ONLY with valid JSON. Do not include any explanation or markdown.";
            if messages.first().map(|(r, _)| r.as_str()) == Some("system") {
                messages[0].1.push_str(&format!("\n\n{json_instr}"));
            } else {
                messages.insert(0, ("system".to_string(), json_instr.to_string()));
            }
        }
    }

    let prompt = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens: Vec<i32> = engine.tokenize(&prompt).unwrap_or_else(|_| {
        if prompt.is_empty() {
            vec![0]
        } else {
            prompt.bytes().map(|b| b as i32).take(4096).collect()
        }
    });

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0).max(0.0),
        top_p: req.top_p.unwrap_or(1.0).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0).max(1.0),
        seed: req.seed,
        stop: req.stop.clone(),
        show_thinking: false,
    };
    let prompt_tokens_len = prompt_tokens.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();

    let inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, sampling, tx);
    engine.submit_request(inference_req);

    // When tools are present, force non-streaming so we can parse the full response.
    let effective_stream = req.stream && req.tools.is_none();

    if effective_stream {
        let stream = async_stream::stream! {
            let mut completion_tokens: u32 = 0;
            while let Some(token) = rx.recv().await {
                let content = token.text.clone();
                let is_done = token.stop_reason.is_some();
                let finish_reason = token.stop_reason.as_ref().map(finish_reason_str).map(str::to_string);
                completion_tokens += 1;
                let usage = if is_done {
                    Some(Usage {
                        prompt_tokens: prompt_tokens_len as u32,
                        completion_tokens,
                        total_tokens: prompt_tokens_len as u32 + completion_tokens,
                    })
                } else {
                    None
                };
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: req.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatMessageDelta {
                            role: None,
                            content: Some(content),
                        },
                        finish_reason,
                    }],
                    usage,
                };
                let event = Event::default()
                    .json_data(chunk)
                    .unwrap_or_else(|_| Event::default().data(""));
                tokio::task::yield_now().await;
                yield Ok::<_, std::convert::Infallible>(event);
                if is_done {
                    break;
                }
            }
        };

        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut full_content = String::new();
        let mut completion_tokens = 0u32;
        let mut final_finish_reason = "stop".to_string();
        while let Some(token) = rx.recv().await {
            full_content.push_str(&token.text);
            completion_tokens += 1;
            if let Some(ref reason) = token.stop_reason {
                final_finish_reason = finish_reason_str(reason).to_string();
                break;
            }
        }

        // Attempt to parse tool calls if tools were provided.
        let (content, tool_calls) = if req.tools.is_some() {
            try_parse_tool_call(&full_content)
        } else {
            (full_content, None)
        };

        let finish_reason = if tool_calls.is_some() {
            "tool_calls".to_string()
        } else {
            final_finish_reason
        };

        let response = ChatCompletionResponse {
            id: id.clone(),
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content,
                    tool_calls,
                },
                finish_reason: Some(finish_reason),
            }],
            usage: Some(Usage {
                prompt_tokens: prompt_tokens_len as u32,
                completion_tokens,
                total_tokens: prompt_tokens_len as u32 + completion_tokens,
            }),
        };

        Json(response).into_response()
    }
}

async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let chat_req = ChatCompletionRequest {
        model: req.model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: req.prompt,
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        seed: None,
        stop: None,
        stream: req.stream,
        tools: None,
        tool_choice: None,
        response_format: None,
    };

    chat_completions(State(state), Json(chat_req)).await
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::StopReason;

    // --- now_rfc3339 ---

    #[test]
    fn test_now_rfc3339_format() {
        let s = now_rfc3339();
        // Basic shape: YYYY-MM-DDTHH:MM:SSZ
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z'));
        assert!(s.contains('T'));
        let parts: Vec<&str> = s.splitn(2, 'T').collect();
        assert_eq!(parts[0].split('-').count(), 3); // year-month-day
    }

    // --- ollama_done_reason ---

    #[test]
    fn test_done_reason_eos() {
        assert_eq!(ollama_done_reason(&Some(StopReason::Eos)), "stop");
    }

    #[test]
    fn test_done_reason_length() {
        assert_eq!(ollama_done_reason(&Some(StopReason::Length)), "length");
    }

    #[test]
    fn test_done_reason_stop_sequence() {
        assert_eq!(ollama_done_reason(&Some(StopReason::StopSequence)), "stop");
    }

    #[test]
    fn test_done_reason_none() {
        assert_eq!(ollama_done_reason(&None), "stop");
    }

    // --- sampling_from_ollama ---

    #[test]
    fn test_sampling_from_ollama_defaults() {
        let (params, max_tokens) = sampling_from_ollama(None);
        assert_eq!(max_tokens, 128);
        assert!((params.temperature - 0.8).abs() < f32::EPSILON);
        assert!((params.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
    }

    #[test]
    fn test_sampling_from_ollama_custom() {
        use crate::api::types::OllamaOptions;
        let opts = OllamaOptions {
            temperature: Some(0.3),
            top_p: Some(0.5),
            top_k: Some(10),
            repeat_penalty: Some(1.2),
            seed: Some(42),
            num_predict: Some(64),
            stop: None,
        };
        let (params, max_tokens) = sampling_from_ollama(Some(&opts));
        assert_eq!(max_tokens, 64);
        assert!((params.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(params.seed, Some(42));
        assert_eq!(params.top_k, 10);
    }

    // --- try_parse_tool_call ---

    #[test]
    fn test_try_parse_tool_call_valid_single() {
        let response = r#"{"name":"get_weather","arguments":{"city":"Madrid"}}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_try_parse_tool_call_plain_text() {
        let response = "The sky is blue because of Rayleigh scattering.";
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_invalid_json() {
        let response = "not { json at all }";
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_json_no_name() {
        // Valid JSON but not a tool call pattern
        let response = r#"{"answer": "42"}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_tool_calls_array() {
        let response = r#"{"tool_calls":[{"name":"search","arguments":{"query":"rust"}}]}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert!(content.is_empty());
        let calls = calls.expect("should detect tool_calls array");
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_try_parse_tool_call_whitespace_trimmed() {
        let response = "  \n{\"name\":\"foo\",\"arguments\":{}}\n  ";
        let (_content, calls) = try_parse_tool_call(response);
        assert!(calls.is_some());
    }

    // --- tools_system_message ---

    #[test]
    fn test_tools_system_message_contains_tool_name() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: None,
            },
        }];
        let msg = tools_system_message(&tools);
        assert!(msg.contains("get_weather"));
        assert!(msg.contains("tool"));
        assert!(msg.contains("JSON"));
    }

    #[test]
    fn test_tools_system_message_empty_tools() {
        let msg = tools_system_message(&[]);
        // Should still produce a valid (if empty) instruction block
        assert!(msg.contains("JSON"));
    }

    // -----------------------------------------------------------------------
    // Integration tests — HTTP handlers via tower::ServiceExt::oneshot
    // -----------------------------------------------------------------------

    use axum::body::Body;
    use axum::http::{Method, Request};
    use std::collections::HashMap;
    use tower::ServiceExt; // for `.oneshot()`

    use crate::model_registry::{EngineEntry, ModelRegistry, RegistryConfig};

    /// Build a minimal test registry backed by `StubModel`.
    /// Creates a dummy `.gguf` file at `<dir>/<name>.gguf` so that
    /// `resolve_model_name` can locate the model on disk.
    fn make_test_registry(
        name: &str,
        dir: &std::path::Path,
    ) -> (Arc<ModelRegistry>, Arc<EngineEntry>) {
        // Create the dummy file so list_models() finds it.
        std::fs::write(dir.join(format!("{name}.gguf")), b"").unwrap();

        let cfg = RegistryConfig {
            models_dir: dir.to_path_buf(),
            max_models: 4,
            max_batch_size: 4,
            max_context_len: 512,
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        let entry = EngineEntry::for_test(name);
        registry.preload_for_test(name, entry.clone());
        (registry, entry)
    }

    /// Build a test `AppState` with one preloaded stub model.
    fn make_test_state(
        name: &str,
        dir: &std::path::Path,
    ) -> (AppState, Arc<EngineEntry>) {
        let (registry, entry) = make_test_registry(name, dir);
        let state = AppState {
            registry,
            primary_model: name.to_string(),
            system_prompt: None,
            started_at: 0,
            models_dir: dir.to_path_buf(),
            digest_cache: Arc::new(Mutex::new(HashMap::new())),
            hf_token: None,
        };
        (state, entry)
    }

    /// POST a JSON body and return the full response bytes.
    async fn post_json(app: Router, path: &str, body: serde_json::Value) -> axum::response::Response {
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        app.oneshot(req).await.unwrap()
    }

    /// GET and return the full response.
    async fn get(app: Router, path: &str) -> axum::response::Response {
        let req = Request::builder()
            .method(Method::GET)
            .uri(path)
            .body(Body::empty())
            .unwrap();
        app.oneshot(req).await.unwrap()
    }

    /// Collect response body bytes.
    async fn body_bytes(resp: axum::response::Response) -> bytes::Bytes {
        use http_body_util::BodyExt;
        resp.into_body().collect().await.unwrap().to_bytes()
    }

    // -- /api/version ---------------------------------------------------------

    #[tokio::test]
    async fn test_api_version_returns_version() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );

        let resp = get(app, "/api/version").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let version = v["version"].as_str().unwrap();
        // Version should be non-empty and look like semver
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }

    // -- /api/ps --------------------------------------------------------------

    #[tokio::test]
    async fn test_api_ps_empty_registry() {
        let dir = tempfile::tempdir().unwrap();
        // Empty registry — no models loaded
        let cfg = RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 4,
            max_batch_size: 4,
            max_context_len: 512,
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        let app = router(
            registry,
            "none".to_string(),
            None,
            0,
            dir.path().to_path_buf(),
            None,
        );
        let resp = get(app, "/api/ps").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_api_ps_with_loaded_model() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("my-model", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let resp = get(app, "/api/ps").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"].as_str().unwrap(), "my-model");
    }

    // -- /v1/models -----------------------------------------------------------

    #[tokio::test]
    async fn test_v1_models_lists_disk_files() {
        let dir = tempfile::tempdir().unwrap();
        // Create two dummy .gguf files
        std::fs::write(dir.path().join("alpha.gguf"), b"").unwrap();
        std::fs::write(dir.path().join("beta.gguf"), b"").unwrap();

        let cfg = RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 4,
            max_batch_size: 4,
            max_context_len: 512,
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        let app = router(
            registry,
            "alpha".to_string(),
            None,
            0,
            dir.path().to_path_buf(),
            None,
        );
        let resp = get(app, "/v1/models").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"alpha"));
        assert!(ids.contains(&"beta"));
    }

    // -- /health --------------------------------------------------------------

    #[tokio::test]
    async fn test_health_with_primary_model() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("primary", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let resp = get(app, "/health").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"].as_str().unwrap(), "ok");
        assert_eq!(v["model_name"].as_str().unwrap(), "primary");
    }

    // -- /v1/chat/completions (non-streaming) ---------------------------------

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );

        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 4
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
        let content = v["choices"][0]["message"]["content"].as_str().unwrap();
        // StubModel emits "hi " for token 65
        assert!(!content.is_empty());
        assert_eq!(v["choices"][0]["finish_reason"].as_str().unwrap(), "stop");
    }

    // -- unknown model → 404 --------------------------------------------------

    #[tokio::test]
    async fn test_unknown_model_returns_404() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let body = serde_json::json!({
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 404);
    }

    // -- /api/generate (non-streaming) ----------------------------------------

    #[tokio::test]
    async fn test_ollama_generate_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let body = serde_json::json!({
            "model": "stub",
            "prompt": "Hello",
            "stream": false
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["done"].as_bool().unwrap());
        assert_eq!(v["model"].as_str().unwrap(), "stub");
        // StubModel generates "hi "
        let response = v["response"].as_str().unwrap();
        assert!(!response.is_empty());
    }

    // -- /api/chat (non-streaming) --------------------------------------------

    #[tokio::test]
    async fn test_ollama_chat_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        });
        let resp = post_json(app, "/api/chat", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["done"].as_bool().unwrap());
        let content = v["message"]["content"].as_str().unwrap();
        assert!(!content.is_empty());
        assert_eq!(v["message"]["role"].as_str().unwrap(), "assistant");
    }

    // -- /v1/chat/completions with response_format json_object ----------------

    #[tokio::test]
    async fn test_chat_completions_json_mode() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = router(
            state.registry.clone(),
            state.primary_model.clone(),
            None,
            state.started_at,
            state.models_dir.clone(),
            None,
        );
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "response_format": {"type": "json_object"}
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        // Should succeed — JSON mode just injects a system message
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }

    // -- /v1/chat/completions with system_prompt injection --------------------

    #[tokio::test]
    async fn test_chat_completions_system_prompt_injected() {
        let dir = tempfile::tempdir().unwrap();
        let (registry, _entry) = make_test_registry("stub", dir.path());
        let app = router(
            registry,
            "stub".to_string(),
            Some("You are a helpful assistant.".to_string()),
            0,
            dir.path().to_path_buf(),
            None,
        );
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // Response should have a valid completion
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }

}
