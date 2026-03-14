// Axum routes for OpenAI-compatible API.

use axum::{
    extract::State,
    http::header,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::engine::InferenceEngine;
use crate::scheduler::{InferenceRequest, SamplingParams, StopReason, Token};

use super::types::*;

/// Shared state for all route handlers: the engine plus the configured system prompt.
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<InferenceEngine>,
    /// Injected as the first message when no system message is present.
    /// `None` disables injection entirely.
    pub system_prompt: Option<String>,
    /// Unix timestamp (seconds) when the server started.
    pub started_at: u64,
}

pub fn router(
    engine: Arc<InferenceEngine>,
    system_prompt: Option<String>,
    started_at: u64,
) -> Router {
    let state = AppState {
        engine,
        system_prompt,
        started_at,
    };
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
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
