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
use crate::api::shared::inference::{
    prepare_prompt, try_parse_tool_call, MessageForTemplate,
};
use crate::api::shared::streaming::finish_reason_str;
use crate::api::types::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessageDelta, ChatMessageResponse, ToolCallDelta,
    ToolCallFunctionDelta, Usage,
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

    // Build MessageForTemplate from request messages (supports tool/multi-turn history).
    let messages: Vec<MessageForTemplate> = req
        .messages
        .iter()
        .map(|m| MessageForTemplate {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect();

    let (prompt_tokens, prompt_tokens_len) = prepare_prompt(
        &entry,
        messages,
        state.system_prompt.as_deref(),
        req.tools.as_deref(),
        req.response_format.as_ref(),
        entry.engine.supports_thinking(),
    );

    // max_completion_tokens is an alias for max_tokens (newer OpenAI API name).
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(256) as usize;

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

    let has_tools = req.tools.is_some();

    tracing::info!(
        model = %req.model,
        stream = req.stream,
        prompt_tokens = prompt_tokens_len,
        thinking = supports_thinking,
        has_tools,
        "request"
    );

    if req.stream {
        if has_tools {
            // Buffer all tokens, parse tool call, then emit as SSE deltas.
            // This lets streaming clients (Google ADK, LangChain) get a proper
            // streaming response even when tool use is involved.
            let mut full_content = String::new();
            let mut completion_tokens = 0u32;
            let mut final_stop_reason = None;
            while let Some(token) = rx.recv().await {
                full_content.push_str(&token.text);
                completion_tokens += 1;
                if let Some(ref reason) = token.stop_reason {
                    final_stop_reason = Some(reason.clone());
                    break;
                }
            }

            let (content, tool_calls) = try_parse_tool_call(&full_content);
            let finish_reason = if tool_calls.is_some() {
                "tool_calls".to_string()
            } else {
                final_stop_reason
                    .as_ref()
                    .map(finish_reason_str)
                    .unwrap_or("stop")
                    .to_string()
            };

            tracing::info!(
                model = %req.model,
                stream = true,
                prompt_tokens = prompt_tokens_len as u32,
                completion_tokens,
                duration_ms = start.elapsed().as_millis() as u64,
                finish_reason = %finish_reason,
                "done"
            );

            let usage = Usage {
                prompt_tokens: prompt_tokens_len as u32,
                completion_tokens,
                total_tokens: prompt_tokens_len as u32 + completion_tokens,
            };

            let id_c = id.clone();
            let model_c = req.model.clone();
            let finish_c = finish_reason.clone();
            let stream = async_stream::stream! {
                // First chunk: role announcement.
                let first = ChatCompletionChunk {
                    id: id_c.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_c.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta: ChatMessageDelta {
                            role: Some("assistant".to_string()),
                            content: None,
                            tool_calls: None,
                        },
                        finish_reason: None,
                    }],
                    usage: None,
                };
                yield Ok::<_, std::convert::Infallible>(
                    Event::default().json_data(first).unwrap_or_else(|_| Event::default().data(""))
                );
                tokio::task::yield_now().await;

                // Final chunk: tool_calls delta or content, with usage.
                let delta = if let Some(ref tcs) = tool_calls {
                    ChatMessageDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(
                            tcs.iter()
                                .enumerate()
                                .map(|(i, tc)| ToolCallDelta {
                                    index: i as u32,
                                    id: Some(tc.id.clone()),
                                    call_type: Some("function".to_string()),
                                    function: ToolCallFunctionDelta {
                                        name: Some(tc.function.name.clone()),
                                        arguments: Some(tc.function.arguments.clone()),
                                    },
                                })
                                .collect(),
                        ),
                    }
                } else {
                    ChatMessageDelta {
                        role: None,
                        content: Some(content),
                        tool_calls: None,
                    }
                };

                let final_chunk = ChatCompletionChunk {
                    id: id_c,
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: model_c,
                    choices: vec![ChatCompletionChunkChoice {
                        index: 0,
                        delta,
                        finish_reason: Some(finish_c),
                    }],
                    usage: Some(usage),
                };
                yield Ok::<_, std::convert::Infallible>(
                    Event::default().json_data(final_chunk).unwrap_or_else(|_| Event::default().data(""))
                );
            };

            return Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response();
        }

        // Normal streaming path (no tools).
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
                        prompt_tokens: log_prompt as u32,
                        completion_tokens,
                        total_tokens: log_prompt as u32 + completion_tokens,
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
                            tool_calls: None,
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
                    content: if tool_calls.is_some() {
                        None
                    } else {
                        Some(content)
                    },
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
    async fn test_chat_completions_json_schema() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": true, "schema": {"type": "object"}}
            }
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

    #[tokio::test]
    async fn test_chat_with_tool_result_in_history() {
        // Multi-turn: history contains a tool result message (role: "tool").
        // Fox should accept this without errors and return a normal response.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 22°C"}
            ],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }

    #[tokio::test]
    async fn test_compat_fields_accepted() {
        // Verify that OpenAI-compat fields like stream_options, parallel_tool_calls,
        // frequency_penalty, presence_penalty, user don't cause 422.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "parallel_tool_calls": false,
            "stream_options": {"include_usage": true},
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "user": "test-user"
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_max_completion_tokens_alias() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "max_completion_tokens": 4
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }
}
