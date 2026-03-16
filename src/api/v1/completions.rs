// POST /v1/completions handler — delegates to chat_completions.

use axum::{extract::State, Json};

use crate::api::router::AppState;
use crate::api::types::{ChatCompletionRequest, ChatMessage, CompletionRequest};

use super::chat::chat_completions;

pub async fn completions(
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
