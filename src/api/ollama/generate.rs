// POST /api/generate handler.

use axum::extract::State;

use crate::api::shared::extractor::LenientJson;
use bytes::Bytes;
use std::time::Instant;

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::shared::inference::{prepare_prompt, sampling_from_ollama, MessageForTemplate};
use crate::api::shared::streaming::{
    collect_tokens, ndjson_response, ndjson_stream, now_rfc3339, ollama_done_reason,
};
use crate::api::types::{OllamaGenerateChunk, OllamaGenerateRequest};
use crate::scheduler::{InferenceRequest, Token};

pub async fn ollama_generate(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<OllamaGenerateRequest>,
) -> axum::response::Response {
    let start = Instant::now();
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    // Build MessageForTemplate list (system prompt + user prompt).
    let mut messages: Vec<MessageForTemplate> = Vec::new();
    if let Some(ref sys) = req.system {
        messages.push(MessageForTemplate {
            role: "system".to_string(),
            content: Some(sys.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    messages.push(MessageForTemplate {
        role: "user".to_string(),
        content: Some(req.prompt.clone()),
        tool_calls: None,
        tool_call_id: None,
    });

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

    let supports_thinking = entry.engine.supports_thinking();

    // /api/generate always suppresses thinking from output (no `thinking` field in response).
    let (mut sampling, max_tokens) = sampling_from_ollama(req.options.as_ref(), false);
    sampling.initial_in_thinking = supports_thinking;

    let (prompt_tokens, prompt_tokens_len) = prepare_prompt(
        &entry,
        messages,
        state.system_prompt.as_deref(),
        None,  // no tools on /api/generate
        false,
        None,
        response_format.as_ref(),
        false, // show_thinking always false for /api/generate
    );

    let stream_mode = req.stream.unwrap_or(true);

    let req_id = entry.engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    entry.engine.submit_request(InferenceRequest::new(
        req_id,
        prompt_tokens,
        max_tokens,
        sampling,
        tx,
    ));

    let model_name = req.model.clone();

    tracing::info!(
        model = %req.model,
        stream = stream_mode,
        prompt_tokens = prompt_tokens_len,
        thinking = supports_thinking,
        "request"
    );

    if stream_mode {
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
            OllamaGenerateChunk {
                model: model_name.clone(),
                created_at: now_rfc3339(),
                response: token.text.clone(),
                done: is_done,
                done_reason: if is_done {
                    Some(ollama_done_reason(&token.stop_reason))
                } else {
                    None
                },
                total_duration: if is_done { Some(elapsed_ns) } else { None },
                load_duration: if is_done { Some(0) } else { None },
                prompt_eval_count: if is_done { Some(log_prompt as u32) } else { None },
                prompt_eval_duration: if is_done { Some(0) } else { None },
                eval_count: if is_done { Some(eval_count) } else { None },
                eval_duration: if is_done { Some(elapsed_ns) } else { None },
            }
        });
        ndjson_response(stream)
    } else {
        let (full_response, eval_count, stop_reason) = collect_tokens(&mut rx).await;
        let elapsed_ns = start.elapsed().as_nanos() as u64;
        tracing::info!(
            model = %model_name,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens = eval_count,
            duration_ms = elapsed_ns / 1_000_000,
            finish_reason = %ollama_done_reason(&stop_reason),
            "done"
        );
        let chunk = OllamaGenerateChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: full_response,
            done: true,
            done_reason: Some(ollama_done_reason(&stop_reason)),
            total_duration: Some(elapsed_ns),
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            prompt_eval_duration: Some(0),
            eval_count: Some(eval_count),
            eval_duration: Some(elapsed_ns),
        };
        let mut line = serde_json::to_string(&chunk).unwrap_or_default();
        line.push('\n');
        axum::response::Response::builder()
            .status(200)
            .header(axum::http::header::CONTENT_TYPE, "application/json")
            .body(axum::body::Body::from(Bytes::from(line.into_bytes())))
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn test_ollama_generate_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "prompt": "Hello",
            "stream": false
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["done"].as_bool().unwrap());
        assert_eq!(v["model"].as_str().unwrap(), "stub");
        assert!(!v["response"].as_str().unwrap().is_empty());
        assert!(v["prompt_eval_duration"].is_number());
        assert!(v["eval_duration"].is_number());
    }

    #[tokio::test]
    async fn test_ollama_generate_format_json() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "prompt": "Return JSON",
            "stream": false,
            "format": "json"
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_ollama_generate_keep_alive_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "prompt": "Hi",
            "stream": false,
            "keep_alive": "5m"
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
    }
}
