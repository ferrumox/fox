// POST /v1/chat/completions handler.

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::shared::inference::{prepare_prompt, try_parse_tool_call};
use crate::api::shared::streaming::finish_reason_str;
use crate::api::types::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessageDelta, ChatMessageResponse, Usage,
};
use crate::scheduler::{InferenceRequest, SamplingParams, Token};

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    let start = Instant::now();
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    let id = Uuid::new_v4().to_string();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let json_mode = req
        .response_format
        .as_ref()
        .map(|rf| rf.format_type == "json_object")
        .unwrap_or(false);

    let (prompt_tokens, prompt_tokens_len) = prepare_prompt(
        &entry,
        messages,
        state.system_prompt.as_deref(),
        req.tools.as_deref(),
        json_mode,
        entry.engine.supports_thinking(),
    );

    let max_tokens = req.max_tokens.unwrap_or(256) as usize;
    let supports_thinking = entry.engine.supports_thinking();
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.8).max(0.0),
        top_p: req.top_p.unwrap_or(0.9).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0).max(1.0),
        seed: req.seed,
        stop: req.stop.clone(),
        // OpenAI has no `thinking` field: the model reasons (initial_in_thinking injects <think>)
        // but the output filter suppresses the thinking block, returning only visible content.
        show_thinking: false,
        initial_in_thinking: supports_thinking,
        max_thinking_chars: 8192,
    };

    let req_id = entry.engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    entry.engine.submit_request(InferenceRequest::new(
        req_id,
        prompt_tokens,
        max_tokens,
        sampling,
        tx,
    ));

    // Tools force a single non-streaming response so we can parse the full output.
    let effective_stream = req.stream && req.tools.is_none();

    tracing::info!(
        model = %req.model,
        stream = effective_stream,
        prompt_tokens = prompt_tokens_len,
        thinking = supports_thinking,
        "request"
    );

    if effective_stream {
        let log_model = req.model.clone();
        let log_prompt = prompt_tokens_len;
        let stream = async_stream::stream! {
            let mut completion_tokens: u32 = 0;
            while let Some(token) = rx.recv().await {
                let content = token.text.clone();
                let is_done = token.stop_reason.is_some();
                let finish_reason = token.stop_reason.as_ref().map(finish_reason_str).map(str::to_string);
                completion_tokens += 1;
                if is_done {
                    tracing::info!(
                        model = %log_model,
                        stream = true,
                        prompt_tokens = log_prompt,
                        completion_tokens,
                        duration_ms = start.elapsed().as_millis() as u64,
                        finish_reason = %finish_reason.as_deref().unwrap_or("stop"),
                        "done"
                    );
                }
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

        tracing::info!(
            model = %req.model,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens,
            duration_ms = start.elapsed().as_millis() as u64,
            finish_reason = %finish_reason,
            "done"
        );

        Json(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
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
        })
        .into_response()
    }
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
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
        assert!(!v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty());
        assert_eq!(v["choices"][0]["finish_reason"].as_str().unwrap(), "stop");
    }

    #[tokio::test]
    async fn test_unknown_model_returns_404() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 404);
    }

    #[tokio::test]
    async fn test_chat_completions_json_mode() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "response_format": {"type": "json_object"}
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }

    #[tokio::test]
    async fn test_chat_completions_system_prompt_injected() {
        use crate::api::router::router;

        let dir = tempfile::tempdir().unwrap();
        let (registry, _entry) = make_test_registry("stub", dir.path());
        let app = router(
            registry,
            "stub".to_string(),
            Some("You are a helpful assistant.".to_string()),
            0,
            dir.path().to_path_buf(),
            None,
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
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }
}
