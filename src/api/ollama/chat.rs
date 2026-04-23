// POST /api/chat handler.

use axum::extract::State;
use uuid::Uuid;

use crate::api::shared::extractor::LenientJson;
use bytes::Bytes;
use std::time::Instant;

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::shared::image::resolve_image_bytes;
use crate::api::shared::inference::{
    extract_thinking, prepare_prompt, prepare_vision_prompt, resolve_tool_choice,
    sampling_from_ollama, try_parse_tool_call, MessageForTemplate,
};
use crate::api::shared::streaming::{
    collect_tokens, ndjson_response, ndjson_stream, now_rfc3339, ollama_done_reason,
};
use crate::api::types::{
    OllamaChatChunk, OllamaChatMessage, OllamaChatRequest, OllamaToolCall, OllamaToolCallFunction,
    ToolCall, ToolCallFunction,
};
use std::sync::Arc;

use crate::scheduler::{InferenceRequest, Token};

pub async fn ollama_chat(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<OllamaChatRequest>,
) -> axum::response::Response {
    let start = Instant::now();
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    // Detect first image across all messages (Ollama sends base64 in `images` array).
    let first_image_b64: Option<&str> = req
        .messages
        .iter()
        .filter_map(|m| m.images.as_ref()?.first().map(|s| s.as_str()))
        .next();

    let has_image = first_image_b64.is_some();
    let is_vision = has_image && entry.engine.supports_vision();

    if has_image && !entry.engine.supports_vision() {
        let err = serde_json::json!({"error": "this model does not support images (no multimodal projector loaded)"});
        return axum::response::Response::builder()
            .status(400)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(err.to_string()))
            .unwrap();
    }

    let image_bytes: Option<Arc<Vec<u8>>> = if let Some(b64) = first_image_b64 {
        match resolve_image_bytes(b64).await {
            Ok(bytes) => Some(Arc::new(bytes)),
            Err(e) => {
                let err = serde_json::json!({"error": format!("failed to decode image: {e}")});
                return axum::response::Response::builder()
                    .status(400)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(axum::body::Body::from(err.to_string()))
                    .unwrap();
            }
        }
    } else {
        None
    };

    // Convert Ollama messages → MessageForTemplate (handles tool history).
    let messages: Vec<MessageForTemplate> = req
        .messages
        .iter()
        .map(|m| {
            let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                tcs.iter()
                    .map(|tc| ToolCall {
                        id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
                        call_type: "function".to_string(),
                        function: ToolCallFunction {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.to_string(),
                        },
                    })
                    .collect::<Vec<_>>()
            });
            let has_this_image = m
                .images
                .as_ref()
                .map(|imgs| !imgs.is_empty())
                .unwrap_or(false);
            MessageForTemplate {
                role: m.role.clone(),
                content: if m.content.is_empty() {
                    None
                } else {
                    Some(m.content.clone())
                },
                tool_calls,
                tool_call_id: m.tool_call_id.clone(),
                image_data: if has_this_image {
                    image_bytes.clone()
                } else {
                    None
                },
            }
        })
        .collect();

    // JSON mode from the `format` field.
    let json_mode = req
        .format
        .as_ref()
        .map(|f| f.as_str() == Some("json") || f.is_object())
        .unwrap_or(false);

    let response_format = if json_mode {
        Some(crate::api::types::ResponseFormat {
            format_type: "json_object".to_string(),
            json_schema: None,
        })
    } else {
        None
    };

    // Resolve tools (Ollama chat supports tools with "auto" behaviour only).
    let tc = resolve_tool_choice(req.tools.as_deref(), None);
    let eff_tools = tc.tools.as_deref();
    let has_tools = eff_tools.is_some();

    // Determine thinking: explicit `think` field OR model capability.
    let supports_thinking = entry.engine.supports_thinking();
    let think_requested = req.think.as_ref().map(|v| {
        v.as_bool().unwrap_or(true)
            || v.as_str()
                .map(|s| !matches!(s, "false" | "none"))
                .unwrap_or(false)
    });
    let use_thinking = think_requested.unwrap_or(supports_thinking) && supports_thinking;

    let stream_mode = req.stream.unwrap_or(true);
    let show_thinking_in_output = use_thinking && !stream_mode;

    let (mut sampling, max_tokens) =
        sampling_from_ollama(req.options.as_ref(), show_thinking_in_output);
    sampling.initial_in_thinking = use_thinking;

    let (prompt_tokens, prompt_tokens_len, vision_prompt) = if is_vision {
        let vp = prepare_vision_prompt(
            &entry,
            messages.clone(),
            state.system_prompt.as_deref(),
            eff_tools,
            false,
            None,
            response_format.as_ref(),
            show_thinking_in_output,
        );
        let estimate = entry.engine.tokenize(&vp).map(|t| t.len()).unwrap_or(256);
        (vec![0i32; estimate], estimate, Some(vp))
    } else {
        let (tokens, len) = prepare_prompt(
            &entry,
            messages,
            state.system_prompt.as_deref(),
            eff_tools,
            false,
            None,
            response_format.as_ref(),
            show_thinking_in_output,
        );
        (tokens, len, None)
    };

    let req_id = entry.engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    let mut inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, sampling, tx);
    if is_vision {
        inference_req.vision_image = image_bytes;
        inference_req.vision_prompt = vision_prompt;
    }
    entry.engine.submit_request(inference_req);

    let model_name = req.model.clone();

    tracing::info!(
        model = %req.model,
        stream = stream_mode,
        prompt_tokens = prompt_tokens_len,
        thinking = use_thinking,
        has_tools,
        "request"
    );

    if stream_mode && !has_tools {
        // Normal streaming (no tools) — emit NDJSON token by token.
        let log_model = model_name.clone();
        let log_prompt = prompt_tokens_len;
        let stream = ndjson_stream(rx, move |token: Token, eval_count: u32, elapsed_ns: u64| {
            let is_done = token.stop_reason.is_some();
            if is_done {
                tracing::info!(
                    model = %log_model,
                    stream = true,
                    prompt_tokens = log_prompt as u32,
                    completion_tokens = eval_count,
                    duration_ms = elapsed_ns / 1_000_000,
                    finish_reason = %ollama_done_reason(&token.stop_reason),
                    "done"
                );
            }
            OllamaChatChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                message: OllamaChatMessage {
                    role: "assistant".to_string(),
                    content: token.text.clone(),
                    thinking: None,
                    tool_calls: None,
                    tool_call_id: None,
                    images: None,
                },
                done: is_done,
                done_reason: if is_done {
                    Some(ollama_done_reason(&token.stop_reason))
                } else {
                    None
                },
                total_duration: if is_done { Some(elapsed_ns) } else { None },
                load_duration: if is_done { Some(0) } else { None },
                prompt_eval_count: if is_done {
                    Some(log_prompt as u32)
                } else {
                    None
                },
                prompt_eval_duration: if is_done { Some(0) } else { None },
                eval_count: if is_done { Some(eval_count) } else { None },
                eval_duration: if is_done { Some(elapsed_ns) } else { None },
            }
        });
        ndjson_response(stream)
    } else {
        // Non-streaming OR streaming with tools (buffer everything, then respond).
        let (full_content, eval_count, stop_reason) = collect_tokens(&mut rx).await;
        let elapsed_ns = start.elapsed().as_nanos() as u64;

        let (thinking, visible) = extract_thinking(&full_content);

        let (content, ollama_tool_calls) = if has_tools {
            let (text, oa_calls) = try_parse_tool_call(&visible, eff_tools);
            let ollama_calls = oa_calls.map(|calls| {
                calls
                    .into_iter()
                    .map(|tc| OllamaToolCall {
                        function: OllamaToolCallFunction {
                            name: tc.function.name,
                            arguments: serde_json::from_str(&tc.function.arguments)
                                .unwrap_or(serde_json::Value::Object(Default::default())),
                        },
                    })
                    .collect::<Vec<_>>()
            });
            (text, ollama_calls)
        } else {
            (visible, None)
        };

        let done_reason = if ollama_tool_calls.is_some() {
            "tool_calls".to_string()
        } else {
            ollama_done_reason(&stop_reason)
        };

        tracing::info!(
            model = %model_name,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens = eval_count,
            duration_ms = elapsed_ns / 1_000_000,
            finish_reason = %done_reason,
            "done"
        );

        let chunk = OllamaChatChunk {
            model: model_name,
            created_at: now_rfc3339(),
            message: OllamaChatMessage {
                role: "assistant".to_string(),
                content,
                thinking,
                tool_calls: ollama_tool_calls,
                tool_call_id: None,
                images: None,
            },
            done: true,
            done_reason: Some(done_reason),
            total_duration: Some(elapsed_ns),
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            prompt_eval_duration: Some(0),
            eval_count: Some(eval_count),
            eval_duration: Some(elapsed_ns),
        };
        let mut line = serde_json::to_string(&chunk).unwrap_or_default();
        line.push('\n');
        // Use NDJSON content-type when the client requested stream=true (even though
        // we buffer for tool calls): the client expects NDJSON framing.
        let ct = if stream_mode {
            "application/x-ndjson"
        } else {
            "application/json"
        };
        axum::response::Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, ct)
            .body(axum::body::Body::from(Bytes::from(line.into_bytes())))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn test_ollama_chat_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
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
        assert!(!v["message"]["content"].as_str().unwrap().is_empty());
        assert_eq!(v["message"]["role"].as_str().unwrap(), "assistant");
        // Ollama duration fields present
        assert!(v["prompt_eval_duration"].is_number());
        assert!(v["eval_duration"].is_number());
    }

    #[tokio::test]
    async fn test_ollama_chat_with_tool_result_in_history() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": {}}}]
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 22°C"}
            ],
            "stream": false
        });
        let resp = post_json(app, "/api/chat", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["done"].as_bool().unwrap());
    }

    #[tokio::test]
    async fn test_ollama_chat_format_json() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "format": "json"
        });
        let resp = post_json(app, "/api/chat", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_ollama_chat_keep_alive_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "keep_alive": "5m"
        });
        let resp = post_json(app, "/api/chat", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_ollama_chat_think_field_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "think": false
        });
        let resp = post_json(app, "/api/chat", body).await;
        assert_eq!(resp.status(), 200);
    }
}
