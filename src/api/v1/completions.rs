// POST /v1/completions handler — OpenAI's legacy text-completion endpoint.
//
// Generation is delegated to `chat_completions` (one synthetic `user` message), but the
// *response* must be the legacy `text_completion` shape, not `chat.completion`: clients
// of this endpoint read `choices[].text`, not `choices[].message.content`. fox used to
// return the chat shape verbatim and drop nearly every sampling parameter on the way in,
// which made the endpoint unusable for its actual consumers (older LangChain/llama-index
// stacks, code-completion plugins).

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{extract::State, Json};
use futures::StreamExt as _;

use crate::api::router::AppState;
use crate::api::types::{ChatCompletionRequest, ChatMessage, CompletionRequest, MessageContent};

use super::chat::chat_completions;

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    // `echo` and `suffix` are accepted by OpenAI but have no fox implementation.
    // Reject rather than silently ignore — a client asking to echo the prompt and
    // getting a completion without it has no way to notice (llama-server rejects
    // both the same way, server-common.cpp:805-815).
    if req.echo.unwrap_or(false) {
        return bad_request("`echo` is not supported");
    }
    if req.suffix.is_some() {
        return bad_request("`suffix` is not supported (use /infill-style prompting)");
    }

    let streaming = req.stream;
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
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        repeat_last_n: req.repeat_last_n,
        top_n_sigma: req.top_n_sigma,
        min_keep: req.min_keep,
        seed: req.seed,
        stop: req.stop,
        stream: req.stream,
        // Tool calling has no meaning on a raw-text endpoint.
        tools: None,
        tool_choice: None,
        parallel_tool_calls: None,
        response_format: None,
        grammar: req.grammar,
        logprobs: req.logprobs.map(|_| true),
        top_logprobs: req.logprobs,
        logit_bias: req.logit_bias,
        min_p: req.min_p,
        min_tokens: None,
        think: None,
        stream_options: req.stream_options,
        frequency_penalty: req.frequency_penalty,
        presence_penalty: req.presence_penalty,
        user: req.user,
        n: req.n,
        best_of: req.best_of,
    };

    let resp = chat_completions(State(state), Json(chat_req)).await;
    if resp.status() != StatusCode::OK {
        return resp; // error bodies pass through untouched
    }
    if streaming {
        to_text_completion_stream(resp)
    } else {
        to_text_completion_json(resp).await
    }
}

fn bad_request(msg: &str) -> Response {
    crate::api::error::AppError::BadRequest(msg.to_string()).into_response()
}

/// Rewrite one `chat.completion`-shaped JSON object into `text_completion`:
/// `choices[].message.content` → `choices[].text`, and the `object` discriminator.
/// Everything else (`id`, `created`, `model`, `usage`, `finish_reason`, `logprobs`)
/// is already shared between the two shapes and passes through unchanged.
fn rewrite_object(mut v: serde_json::Value, object_name: &str) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.insert(
            "object".to_string(),
            serde_json::Value::String(object_name.to_string()),
        );
        if let Some(choices) = obj.get_mut("choices").and_then(|c| c.as_array_mut()) {
            for choice in choices {
                let Some(c) = choice.as_object_mut() else {
                    continue;
                };
                // Non-streaming carries `message`, streaming carries `delta`; both
                // hold the text under `content`.
                let text = c
                    .remove("message")
                    .or_else(|| c.remove("delta"))
                    .and_then(|m| m.get("content").cloned())
                    .unwrap_or(serde_json::Value::Null);
                c.insert(
                    "text".to_string(),
                    match text {
                        serde_json::Value::String(s) => serde_json::Value::String(s),
                        // A `null` content (e.g. a role-only opening delta) is an
                        // empty string here — `text` is not nullable in this shape.
                        _ => serde_json::Value::String(String::new()),
                    },
                );
            }
        }
    }
    v
}

async fn to_text_completion_json(resp: Response) -> Response {
    let (parts, body) = resp.into_parts();
    let bytes = match axum::body::to_bytes(body, usize::MAX).await {
        Ok(b) => b,
        Err(e) => return bad_request(&format!("failed to read upstream body: {e}")),
    };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        // Not JSON we understand — hand it back rather than mangling it.
        return Response::from_parts(parts, Body::from(bytes));
    };
    let out = rewrite_object(v, "text_completion");
    let body = serde_json::to_vec(&out).unwrap_or_else(|_| bytes.to_vec());
    Response::from_parts(parts, Body::from(body))
}

/// Rewrite an SSE stream chunk by chunk.
///
/// Chunk boundaries are not guaranteed to align with SSE events, so this buffers
/// until it has a complete `\n\n`-terminated event before rewriting. `data: [DONE]`
/// is passed through verbatim — it is a sentinel, not JSON.
fn to_text_completion_stream(resp: Response) -> Response {
    let (parts, body) = resp.into_parts();
    let mut buf = String::new();

    let stream = body.into_data_stream().map(move |chunk| {
        let chunk = chunk?;
        buf.push_str(&String::from_utf8_lossy(&chunk));

        let mut out = String::new();
        while let Some(idx) = buf.find("\n\n") {
            let event: String = buf.drain(..idx + 2).collect();
            out.push_str(&rewrite_event(&event));
        }
        Ok::<_, axum::Error>(bytes::Bytes::from(out))
    });

    Response::from_parts(parts, Body::from_stream(stream))
}

fn rewrite_event(event: &str) -> String {
    let mut out = String::with_capacity(event.len());
    for line in event.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        match trimmed.strip_prefix("data: ") {
            Some(payload) if payload != "[DONE]" => {
                match serde_json::from_str::<serde_json::Value>(payload) {
                    Ok(v) => {
                        let rewritten = rewrite_object(v, "text_completion");
                        out.push_str("data: ");
                        out.push_str(&serde_json::to_string(&rewritten).unwrap_or_default());
                        out.push('\n');
                    }
                    // Unparseable payload: pass through rather than drop it.
                    Err(_) => out.push_str(line),
                }
            }
            _ => out.push_str(line),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rewrite_maps_message_content_to_text() {
        let v = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        });
        let out = rewrite_object(v, "text_completion");
        assert_eq!(out["object"], "text_completion");
        assert_eq!(out["choices"][0]["text"], "hi");
        assert!(out["choices"][0].get("message").is_none());
        // Shared fields survive untouched.
        assert_eq!(out["choices"][0]["finish_reason"], "stop");
        assert_eq!(out["usage"]["total_tokens"], 2);
        assert_eq!(out["id"], "chatcmpl-1");
    }

    #[test]
    fn rewrite_maps_streaming_delta_to_text() {
        let v = json!({
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "tok"}, "finish_reason": null}]
        });
        let out = rewrite_object(v, "text_completion");
        assert_eq!(out["choices"][0]["text"], "tok");
        assert!(out["choices"][0].get("delta").is_none());
    }

    #[test]
    fn rewrite_turns_null_content_into_empty_string() {
        // The opening delta carries a role and a null content; `text` is not
        // nullable in the legacy shape.
        let v = json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": null}}]});
        let out = rewrite_object(v, "text_completion");
        assert_eq!(out["choices"][0]["text"], "");
    }

    #[test]
    fn rewrite_event_passes_done_sentinel_through() {
        assert_eq!(rewrite_event("data: [DONE]\n\n"), "data: [DONE]\n\n");
    }

    #[test]
    fn rewrite_event_rewrites_a_data_payload() {
        let ev = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n";
        let out = rewrite_event(ev);
        assert!(out.starts_with("data: "), "{out}");
        assert!(out.contains("\"text\":\"x\""), "{out}");
        assert!(out.ends_with("\n\n"), "{out:?}");
    }
}
