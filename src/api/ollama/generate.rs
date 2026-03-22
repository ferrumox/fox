// POST /api/generate handler.

use axum::extract::State;

use crate::api::shared::extractor::LenientJson;
use bytes::Bytes;
use std::time::Instant;

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::shared::inference::sampling_from_ollama;
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

    let mut messages: Vec<(String, String)> = Vec::new();
    if let Some(ref sys) = req.system {
        messages.push(("system".to_string(), sys.clone()));
    }
    messages.push(("user".to_string(), req.prompt.clone()));

    let supports_thinking = entry.engine.supports_thinking();
    let mut prompt = entry
        .engine
        .apply_chat_template(&messages)
        .unwrap_or_else(|_| {
            messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        });
    if supports_thinking {
        prompt.push_str("<think>\n");
    }

    let prompt_tokens: Vec<i32> = entry
        .engine
        .tokenize(&prompt)
        .unwrap_or_else(|_| prompt.bytes().map(|b| b as i32).take(4096).collect());
    let prompt_tokens_len = prompt_tokens.len();

    let stream_mode = req.stream.unwrap_or(true);

    // /api/generate has no `thinking` field: always suppress thinking from output.
    // The model still reasons because <think>\n was injected into the prompt.
    let (mut sampling, max_tokens) = sampling_from_ollama(req.options.as_ref(), false);
    sampling.initial_in_thinking = supports_thinking;

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
                prompt_eval_count: if is_done {
                    Some(prompt_tokens_len as u32)
                } else {
                    None
                },
                eval_count: if is_done { Some(eval_count) } else { None },
            }
        });
        ndjson_response(stream)
    } else {
        let (full_response, eval_count, stop_reason) = collect_tokens(&mut rx).await;
        tracing::info!(
            model = %model_name,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens = eval_count,
            duration_ms = start.elapsed().as_millis() as u64,
            finish_reason = %ollama_done_reason(&stop_reason),
            "done"
        );
        let chunk = OllamaGenerateChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: full_response,
            done: true,
            done_reason: Some(ollama_done_reason(&stop_reason)),
            total_duration: Some(start.elapsed().as_nanos() as u64), // reuses handler-level start
            load_duration: Some(0),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            eval_count: Some(eval_count),
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
    }
}
