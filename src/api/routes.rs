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
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Read as _;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::cli::show::{parse_architecture, parse_quantization};
use crate::cli::{format_size, list_models};
use crate::engine::InferenceEngine;
use crate::scheduler::{InferenceRequest, SamplingParams, StopReason, Token};

use super::types::*;

/// Shared state for all route handlers.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<InferenceEngine>,
    /// Injected as the first message when no system message is present.
    /// `None` disables injection entirely.
    pub system_prompt: Option<String>,
    /// Unix timestamp (seconds) when the server started.
    pub started_at: u64,
    /// Directory where `.gguf` model files are stored.
    pub models_dir: PathBuf,
    /// Cache of SHA256 digests keyed by file path. Computed once per file.
    pub digest_cache: Arc<Mutex<HashMap<PathBuf, String>>>,
}

pub fn router(
    engine: Arc<InferenceEngine>,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
) -> Router {
    let state = AppState {
        engine,
        system_prompt,
        started_at,
        models_dir,
        digest_cache: Arc::new(Mutex::new(HashMap::new())),
    };
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        // Ollama-compatible endpoints
        .route("/api/tags", get(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/delete", delete(ollama_delete))
        .with_state(state)
}

/// Prometheus text-format scrape endpoint.
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

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let engine = &state.engine;
    Json(HealthResponse {
        status: "ok".to_string(),
        kv_cache_usage: engine.kv_cache_usage(),
        queue_depth: engine.queue_depth(),
        active_requests: engine.active_requests(),
        model_name: engine.model_name().to_string(),
        started_at: state.started_at,
    })
}

async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.engine.model_name().to_string(),
            object: "model".to_string(),
        }],
    })
}

// --- Ollama-compatible handlers ---

/// Compute SHA256 of a file, returning `"sha256:<hex>"`.
/// Uses the cache to avoid re-hashing large files on every request.
async fn file_digest(path: PathBuf, cache: Arc<Mutex<HashMap<PathBuf, String>>>) -> String {
    if let Some(cached) = cache.lock().unwrap().get(&path).cloned() {
        return cached;
    }
    let digest = tokio::task::spawn_blocking(move || {
        let mut file = std::fs::File::open(&path)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; 1024 * 1024]; // 1 MiB chunks
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

    // Store in cache (re-acquire lock after spawn_blocking)
    // Note: path was moved, so we rebuild from the digest string which is fine — we only
    // need the value. The caller holds the original path, so we skip re-caching here.
    // Instead we return the digest; caching is handled by the callers that have `path`.
    digest
}

/// Compute and cache digest for a given path.
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
            // Minimal RFC 3339 UTC from Unix timestamp
            let s = d.as_secs();
            let sec = s % 60;
            let min = (s / 60) % 60;
            let hour = (s / 3600) % 24;
            // Rough date (good enough for compatibility; not fully accurate for leap years)
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
    let engine = &state.engine;
    let name = engine.model_name().to_string();

    // Look up the file in models_dir to get real size and digest.
    let file_info = list_models(&state.models_dir)
        .ok()
        .and_then(|entries| {
            entries.into_iter().find(|(path, _)| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|stem| stem == name)
                    .unwrap_or(false)
            })
        });

    let (size, digest) = if let Some((path, meta)) = file_info {
        let d = get_digest(&path, &state.digest_cache).await;
        (meta.len(), d)
    } else {
        (0u64, "sha256:unknown".to_string())
    };

    let entry = PsEntry {
        name: name.clone(),
        size,
        digest,
        details: OllamaDetails {
            format: "gguf".to_string(),
            family: parse_architecture(&name).unwrap_or("unknown").to_string(),
            parameter_size: "unknown".to_string(),
            quantization_level: parse_quantization(&name).unwrap_or("unknown").to_string(),
        },
        expires_at: "0001-01-01T00:00:00Z".to_string(),
        size_vram: 0,
    };

    Json(PsResponse {
        models: vec![entry],
    })
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
        Some((path, _)) => match std::fs::remove_file(&path) {
            Ok(_) => StatusCode::OK.into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to delete model: {e}"),
            )
                .into_response(),
        },
    }
}

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
    let engine = &state.engine;
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
        show_thinking: false, // API responses never include raw thinking tokens
    };
    let prompt_tokens_len = prompt_tokens.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();

    let inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, sampling, tx);
    engine.submit_request(inference_req);

    if req.stream {
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
        let response = ChatCompletionResponse {
            id: id.clone(),
            object: "chat.completion".to_string(),
            created,
            model: req.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: full_content,
                },
                finish_reason: Some(final_finish_reason),
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
    };

    chat_completions(State(state), Json(chat_req)).await
}
