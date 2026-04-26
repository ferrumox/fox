// POST /v1/completions handler — delegates to chat_completions.

use axum::{extract::State, Json};

use crate::api::router::AppState;
use crate::api::types::{ChatCompletionRequest, ChatMessage, CompletionRequest, MessageContent};

use super::chat::chat_completions;

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> axum::response::Response {
    let chat_req = ChatCompletionRequest {
        model: req.model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: Some(MessageContent::Text(req.prompt)),
            tool_call_id: None,
            tool_calls: None,
            name: None,
        }],
        max_tokens: req.max_tokens,
        max_completion_tokens: None,
        temperature: req.temperature,
        top_p: None,
        top_k: None,
        repetition_penalty: None,
        seed: None,
        stop: None,
        stream: req.stream,
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        response_format: None,
        stream_options: None,
        min_p: None,
        frequency_penalty: None,
        presence_penalty: None,
        user: None,
    };
    chat_completions(State(state), Json(chat_req)).await
}
