// Axum routes for OpenAI-compatible API.

use axum::{
    extract::State,
    response::{IntoResponse, sse::{Event, KeepAlive, Sse}},
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::engine::InferenceEngine;
use crate::scheduler::{InferenceRequest, StopReason, Token};

use super::types::*;

pub fn router(engine: Arc<InferenceEngine>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/models", get(models))
        .route("/health", get(health))
        .with_state(engine)
}

async fn health(State(engine): State<Arc<InferenceEngine>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        kv_cache_usage: engine.kv_cache_usage(),
        queue_depth: engine.queue_depth(),
        active_requests: engine.active_requests(),
    })
}

async fn models(State(engine): State<Arc<InferenceEngine>>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: engine.model_name().to_string(),
            object: "model".to_string(),
        }],
    })
}

fn finish_reason_str(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Eos => "stop",
        StopReason::Length => "length",
        StopReason::Preempt => "stop",
    }
}

async fn chat_completions(
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let id = Uuid::new_v4().to_string();
    let req_id = engine.next_request_id();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Build prompt: apply chat template if available, else simple "role: content" concatenation
    let mut messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    // Prepend system message if missing (helps Qwen, Llama, etc. generate properly)
    if messages.first().map(|(r, _)| r.as_str()) != Some("system") {
        messages.insert(0, ("system".to_string(), "You are a helpful assistant.".to_string()));
    }
    let prompt = engine
        .apply_chat_template(&messages)
        .unwrap_or_else(|_| {
            messages
                .iter()
                .map(|(r, c)| format!("{}: {}", r, c))
                .collect::<Vec<_>>()
                .join("\n")
        });

    // Tokenize using model's vocabulary
    let prompt_tokens: Vec<i32> = engine.tokenize(&prompt).unwrap_or_else(|_| {
        if prompt.is_empty() {
            vec![0]
        } else {
            prompt.bytes().map(|b| b as i32).take(4096).collect()
        }
    });

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let temperature = req.temperature.unwrap_or(1.0).max(0.0);
    let top_p = req.top_p.unwrap_or(1.0).clamp(0.0, 1.0);
    let prompt_tokens_len = prompt_tokens.len();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();

    let inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, temperature, top_p, tx);

    engine.submit_request(inference_req);

    if req.stream {
        let stream = async_stream::stream! {
            while let Some(token) = rx.recv().await {
                let content = token.text.clone();
                let is_done = token.stop_reason.is_some();
                let finish_reason = token.stop_reason.as_ref().map(finish_reason_str).map(str::to_string);
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
        // Collect full response
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
    State(engine): State<Arc<InferenceEngine>>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    // Convert to chat format and delegate
    let chat_req = ChatCompletionRequest {
        model: req.model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: req.prompt,
        }],
        max_tokens: req.max_tokens,
        temperature: req.temperature,
        top_p: None,
        stream: req.stream,
    };

    chat_completions(State(engine), Json(chat_req)).await
}
