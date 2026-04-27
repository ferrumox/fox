//! `POST /v1/chat/completions/auto` — chat completions with **server-side**
//! resolution of tool calls.
//!
//! When the model emits a `tool_call` whose `name` is registered on the
//! [`crate::tools::ToolBoard`], the handler executes the tool, appends the
//! result back into the conversation as a `tool` message and re-runs the
//! model. The loop terminates when the model produces a turn without tool
//! calls or when `max_tool_iters` is reached.
//!
//! Streaming is intentionally not supported here — the loop produces 1..N
//! intermediate completions and a streaming response would have to expose
//! that structure to the client. Stick to `POST /v1/chat/completions` with
//! `stream: true` when streaming is required.
//!
//! Tool calls whose `name` is **not** in the board pass through unchanged in
//! the final response, exactly like the regular endpoint — the client can
//! still execute them.

use axum::{extract::State, http::StatusCode, Json};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::api::error::{load_model_or_respond, AppError};
use crate::api::router::AppState;
use crate::api::shared::inference::{
    prepare_prompt, resolve_tool_choice, try_parse_tool_call, MessageForTemplate,
};
use crate::api::shared::streaming::finish_reason_str;
use crate::api::types::{
    ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessageResponse,
    ToolCall, Usage,
};
use crate::scheduler::{InferenceRequest, SamplingParams, Token};
use crate::tools::ToolCtx;

/// Hard cap on round-trips between the model and the tool board. Stops a
/// pathological prompt from looping forever; matches the convention used by
/// most agent frameworks.
const MAX_TOOL_ITERS_DEFAULT: u32 = 5;

pub async fn chat_auto_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, axum::response::Response> {
    use axum::response::IntoResponse;
    if let Err(msg) = req.validate() {
        return Err(AppError::BadRequest(msg).into_response());
    }
    if req.stream {
        return Err(AppError::BadRequest(
            "/v1/chat/completions/auto does not support stream=true (use /v1/chat/completions)".into(),
        )
        .into_response());
    }

    let start = Instant::now();
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return Err(r),
    };

    let id = Uuid::new_v4().to_string();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Working conversation that grows with assistant + tool messages between
    // iterations.
    let mut messages: Vec<MessageForTemplate> = req
        .messages
        .iter()
        .map(|m| MessageForTemplate {
            role: m.role.clone(),
            content: m
                .content
                .as_ref()
                .map(|c| c.as_text())
                .filter(|s| !s.is_empty()),
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect();

    let tc = resolve_tool_choice(req.tools.as_deref(), req.tool_choice.as_ref());
    let eff_tools = tc.tools.as_deref();
    let tool_required = tc.required;
    let specific_tool = tc.specific.as_deref();
    let supports_thinking = entry.engine.supports_thinking();
    let max_tokens = req.max_tokens.or(req.max_completion_tokens).unwrap_or(256) as usize;

    let max_iters = MAX_TOOL_ITERS_DEFAULT;
    let mut total_prompt_tokens: u32 = 0;
    let mut total_completion_tokens: u32 = 0;
    let mut last_finish_reason = "stop".to_string();
    let mut last_content: Option<String> = None;
    let mut last_tool_calls: Option<Vec<ToolCall>> = None;

    for iter in 0..max_iters {
        let (prompt_tokens, prompt_tokens_len) = prepare_prompt(
            &entry,
            messages.clone(),
            state.system_prompt.as_deref(),
            eff_tools,
            tool_required,
            specific_tool,
            req.response_format.as_ref(),
            supports_thinking,
        );

        let sampling = build_sampling(&req, supports_thinking, &entry);

        let req_id = entry.engine.next_request_id();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
        entry.engine.submit_request(InferenceRequest::new(
            req_id,
            prompt_tokens,
            max_tokens,
            sampling,
            tx,
        ));

        let (full_content, completion_tokens, stop_reason) = buffer_tokens(&mut rx).await;
        total_prompt_tokens += prompt_tokens_len as u32;
        total_completion_tokens += completion_tokens;

        let (content, tool_calls) = try_parse_tool_call(&full_content, eff_tools);
        let finish_reason = if tool_calls.is_some() {
            "tool_calls".to_string()
        } else {
            stop_reason
                .as_ref()
                .map(finish_reason_str)
                .unwrap_or("stop")
                .to_string()
        };

        let content_opt = if content.is_empty() { None } else { Some(content.clone()) };
        last_content = content_opt.clone();
        last_tool_calls = tool_calls.clone();
        last_finish_reason = finish_reason.clone();

        // Decide whether to loop or stop.
        let calls = match tool_calls {
            Some(c) if !c.is_empty() => c,
            _ => break,
        };

        // Partition into "executed by board" vs "passed through".
        let mut executed_any = false;
        let mut tool_messages: Vec<MessageForTemplate> = Vec::new();
        for call in &calls {
            if !state.tool_board.contains(&call.function.name) {
                // Hand back to the client unchanged, as the regular endpoint does.
                continue;
            }
            executed_any = true;
            // Parse the JSON arguments — they're stored as a string blob in the
            // OpenAI shape. Tools that fail to parse the args themselves will
            // still get a Value, which they can validate.
            let args = serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                .unwrap_or(serde_json::Value::Null);
            let result = match state
                .tool_board
                .invoke(&call.function.name, args, &ToolCtx::default())
                .await
            {
                Ok(v) => v,
                Err(e) => serde_json::json!({"error": e.to_string()}),
            };
            tool_messages.push(MessageForTemplate {
                role: "tool".to_string(),
                content: Some(result.to_string()),
                tool_calls: None,
                tool_call_id: Some(call.id.clone()),
            });
        }

        if !executed_any {
            // Every tool the model asked for is something the client must run.
            // Return the response with the tool_calls intact.
            break;
        }

        // Feed the assistant turn (with the tool_calls that triggered execution)
        // and the per-tool responses back into the conversation, then loop.
        messages.push(MessageForTemplate {
            role: "assistant".to_string(),
            content: content_opt,
            tool_calls: Some(calls),
            tool_call_id: None,
        });
        messages.extend(tool_messages);

        tracing::info!(
            iter = iter + 1,
            executed = executed_any,
            "auto chat round-trip"
        );
    }

    tracing::info!(
        model = %req.model,
        prompt_tokens = total_prompt_tokens,
        completion_tokens = total_completion_tokens,
        duration_ms = start.elapsed().as_millis() as u64,
        finish_reason = %last_finish_reason,
        "auto chat done"
    );

    let response = ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: req.model.clone(),
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessageResponse {
                role: "assistant".to_string(),
                content: if last_finish_reason == "tool_calls" {
                    None
                } else {
                    last_content
                },
                tool_calls: last_tool_calls,
            },
            finish_reason: Some(last_finish_reason),
        }],
        usage: Some(Usage {
            prompt_tokens: total_prompt_tokens,
            completion_tokens: total_completion_tokens,
            total_tokens: total_prompt_tokens + total_completion_tokens,
        }),
        system_fingerprint: None,
    };
    Ok(Json(response))
}

fn build_sampling(
    req: &ChatCompletionRequest,
    supports_thinking: bool,
    entry: &crate::model_registry::EngineEntry,
) -> SamplingParams {
    fn parse_logit_bias(
        raw: &Option<std::collections::HashMap<String, f32>>,
    ) -> std::collections::HashMap<i32, f32> {
        match raw {
            None => std::collections::HashMap::new(),
            Some(map) => map
                .iter()
                .filter_map(|(k, v)| k.parse::<i32>().ok().map(|id| (id, *v)))
                .collect(),
        }
    }
    SamplingParams {
        temperature: req.temperature.unwrap_or(0.8).max(0.0),
        top_p: req.top_p.unwrap_or(0.9).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(0),
        min_p: req.min_p.unwrap_or(0.0).clamp(0.0, 1.0),
        repetition_penalty: req.repetition_penalty.unwrap_or(1.0).max(1.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        seed: req.seed,
        stop: req.stop.clone(),
        show_thinking: false,
        initial_in_thinking: supports_thinking && !entry.engine.uses_channel_thinking(),
        max_thinking_chars: 8192,
        mirostat_tau: req.mirostat_tau.unwrap_or(0.0),
        mirostat_eta: req.mirostat_eta.unwrap_or(0.1),
        logit_bias: parse_logit_bias(&req.logit_bias),
        dynamic_temp_low: req.dynamic_temp_low.unwrap_or(0.0),
        dynamic_temp_high: req.dynamic_temp_high.unwrap_or(0.0),
    }
}

async fn buffer_tokens(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Token>,
) -> (String, u32, Option<crate::scheduler::StopReason>) {
    let mut text = String::new();
    let mut count = 0u32;
    let mut stop_reason = None;
    while let Some(token) = rx.recv().await {
        text.push_str(&token.text);
        count += 1;
        if token.stop_reason.is_some() {
            stop_reason = token.stop_reason;
            break;
        }
    }
    (text, count, stop_reason)
}

/// Suppress unused-import lints when the helper above isn't needed during
/// trimmed test builds. Module-private — never called.
#[allow(dead_code)]
fn _force_use_status_code() -> StatusCode {
    StatusCode::OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::test_helpers::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode as Code};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    #[tokio::test]
    async fn auto_endpoint_rejects_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/auto")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), Code::BAD_REQUEST);
    }

    #[tokio::test]
    async fn auto_endpoint_returns_completion_when_model_does_not_call_tools() {
        // The stub model always emits a plain assistant turn with no tool
        // call, so the auto endpoint must short-circuit on the first iter.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions/auto")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), Code::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    }
}
