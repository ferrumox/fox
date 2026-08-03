// POST /api/generate handler.

use axum::extract::State;
use axum::response::IntoResponse;

use crate::api::shared::extractor::LenientJson;
use bytes::Bytes;
use std::time::Instant;

use base64::{engine::general_purpose::STANDARD, Engine as _};

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::shared::inference::{
    prepare_multimodal_prompt, prepare_prompt, sampling_from_ollama, MessageForTemplate,
};
use crate::api::shared::streaming::{
    collect_tokens_timed, ndjson_response, ndjson_stream, now_rfc3339, ollama_done_reason,
    GenTimings,
};
use crate::api::types::{OllamaGenerateChunk, OllamaGenerateRequest};
use crate::engine::model::MEDIA_MARKER;
use crate::scheduler::{InferenceRequest, Token};

pub async fn ollama_generate(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<OllamaGenerateRequest>,
) -> axum::response::Response {
    let start = Instant::now();
    let (entry, lora) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    // Honour the request's `keep_alive` (previously parsed and thrown away, so a
    // client asking to keep a model warm — or drop it promptly — could not tell that
    // nothing happened). `Immediate` is applied after the response instead, below.
    let keep_alive = crate::api::types::parse_keep_alive(req.keep_alive.as_ref());
    match keep_alive {
        Some(crate::api::types::KeepAlive::Forever) => {
            state.registry.set_keep_alive(&req.model, None)
        }
        Some(crate::api::types::KeepAlive::Secs(n)) => state
            .registry
            .set_keep_alive(&req.model, Some(std::time::Duration::from_secs(n))),
        // `0` = "unload once this request is done". Expressed as a zero TTL rather
        // than an explicit unload: the eviction pass already refuses to drop a model
        // with work in flight (`is_busy`), so this cannot kill the very request that
        // asked for it — including a stream the handler has already returned.
        // Divergence from Ollama, deliberate: the unload lands on the next eviction
        // tick (within 60s) rather than the instant the response ends.
        Some(crate::api::types::KeepAlive::Immediate) => state
            .registry
            .set_keep_alive(&req.model, Some(std::time::Duration::ZERO)),
        None => {}
    }

    // Real `load_duration`: ~0 when the model was already resident, which is the
    // honest answer, and the actual load cost when this request triggered it.
    let load_ns = start.elapsed().as_nanos() as u64;

    let has_images = req.images.as_ref().is_some_and(|v| !v.is_empty());
    let use_vision = has_images && entry.engine.supports_vision();
    if has_images && !use_vision {
        tracing::warn!(
            model = %req.model,
            "dropped image content — model has no vision support (no mmproj loaded)"
        );
    }

    // Build MessageForTemplate list (system prompt + user prompt). When
    // `use_vision`, each base64 image is decoded and appended to the user
    // prompt as a MEDIA_MARKER, mirroring the OpenAI/chat handlers.
    let mut images: Vec<Vec<u8>> = Vec::new();
    let mut messages: Vec<MessageForTemplate> = Vec::new();
    if let Some(ref sys) = req.system {
        messages.push(MessageForTemplate {
            role: "system".to_string(),
            content: Some(sys.clone()),
            tool_calls: None,
            tool_call_id: None,
        });
    }
    let mut prompt = req.prompt.clone();
    if use_vision {
        for b64 in req.images.iter().flatten() {
            match STANDARD.decode(b64) {
                Ok(bytes) => {
                    images.push(bytes);
                    if !prompt.is_empty() {
                        prompt.push(' ');
                    }
                    prompt.push_str(MEDIA_MARKER);
                }
                Err(e) => tracing::warn!("skipping malformed base64 image: {e}"),
            }
        }
    }
    messages.push(MessageForTemplate {
        role: "user".to_string(),
        content: Some(prompt),
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
    if let Some(unsupported) = req
        .options
        .as_ref()
        .map(|o| o.unsupported_options())
        .filter(|v| !v.is_empty())
    {
        tracing::warn!(
            model = %req.model,
            options = %unsupported.join(", "),
            "ignoring unsupported Ollama options — fox accepts them for compatibility \
             but does not act on them"
        );
    }

    let (mut sampling, max_tokens) =
        sampling_from_ollama(req.options.as_ref(), false, state.repeat_last_n);
    sampling.initial_in_thinking = supports_thinking;

    // Guided decoding from the `format` field (`"json"` or a JSON schema object).
    match crate::api::shared::json_schema::grammar_from_ollama_format(req.format.as_ref()) {
        Ok(g) => sampling.grammar = g,
        Err(e) => {
            return crate::api::error::AppError::BadRequest(format!("invalid format: {e}"))
                .into_response()
        }
    }

    let (prompt_tokens, prompt_tokens_len, multimodal) = if use_vision {
        match prepare_multimodal_prompt(
            &entry,
            messages,
            state.system_prompt.as_deref(),
            None, // no tools on /api/generate
            false,
            None,
            response_format.as_ref(),
            false, // show_thinking always false for /api/generate
            &images,
        ) {
            Ok(chunks) => {
                let n = chunks.n_positions();
                (Vec::new(), n, Some(chunks))
            }
            Err(e) => {
                return crate::api::error::AppError::BadRequest(format!(
                    "failed to encode image(s): {e}"
                ))
                .into_response()
            }
        }
    } else {
        let (tokens, len) = prepare_prompt(
            &entry,
            messages,
            state.system_prompt.as_deref(),
            None, // no tools on /api/generate
            false,
            None,
            response_format.as_ref(),
            false, // show_thinking always false for /api/generate
        );
        (tokens, len, None)
    };

    let stream_mode = req.stream.unwrap_or(true);

    let req_id = entry.engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    let mut inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, sampling, tx);
    if let Some(chunks) = multimodal {
        inference_req = inference_req.with_multimodal(chunks);
    }
    if let Some(selection) = lora {
        inference_req = inference_req.with_lora(selection);
    }
    if let Err(e) = entry.engine.submit_request(inference_req) {
        entry
            .engine
            .record_rejection(crate::api::error::rejection_reason_label(&e));
        return crate::api::error::AppError::from(e).into_response();
    }

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
        let stream = ndjson_stream(
            rx,
            load_ns,
            move |token: Token, eval_count: u32, t: GenTimings| {
                let is_done = token.stop_reason.is_some();
                if is_done {
                    tracing::info!(
                        model = %log_model,
                        stream = true,
                        prompt_tokens = log_prompt as u32,
                        completion_tokens = eval_count,
                        duration_ms = t.total_ns / 1_000_000,
                        prefill_ms = t.prompt_eval_ns / 1_000_000,
                        decode_ms = t.eval_ns / 1_000_000,
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
                    total_duration: if is_done { Some(t.total_ns) } else { None },
                    load_duration: if is_done { Some(t.load_ns) } else { None },
                    prompt_eval_count: if is_done {
                        Some(log_prompt as u32)
                    } else {
                        None
                    },
                    prompt_eval_duration: if is_done {
                        Some(t.prompt_eval_ns)
                    } else {
                        None
                    },
                    eval_count: if is_done { Some(eval_count) } else { None },
                    eval_duration: if is_done { Some(t.eval_ns) } else { None },
                }
            },
        );
        ndjson_response(stream)
    } else {
        let (full_response, eval_count, stop_reason, prompt_eval_ns) =
            collect_tokens_timed(&mut rx).await;
        let t = GenTimings::new(
            load_ns,
            prompt_eval_ns,
            start.elapsed().as_nanos() as u64 - load_ns,
        );
        tracing::info!(
            model = %model_name,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens = eval_count,
            duration_ms = t.total_ns / 1_000_000,
            prefill_ms = t.prompt_eval_ns / 1_000_000,
            decode_ms = t.eval_ns / 1_000_000,
            finish_reason = %ollama_done_reason(&stop_reason),
            "done"
        );
        let chunk = OllamaGenerateChunk {
            model: model_name,
            created_at: now_rfc3339(),
            response: full_response,
            done: true,
            done_reason: Some(ollama_done_reason(&stop_reason)),
            total_duration: Some(t.total_ns),
            load_duration: Some(t.load_ns),
            prompt_eval_count: Some(prompt_tokens_len as u32),
            prompt_eval_duration: Some(t.prompt_eval_ns),
            eval_count: Some(eval_count),
            eval_duration: Some(t.eval_ns),
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
    async fn generate_reports_a_real_prefill_decode_split() {
        // Regression: `load_duration` and `prompt_eval_duration` were hard-coded to 0,
        // and `total_duration`/`eval_duration` were the same wall clock, so a client
        // could not tell prefill cost from decode cost at all.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub", "prompt": "Hello", "stream": false,
            "options": {"num_predict": 4}
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

        let total = v["total_duration"].as_u64().expect("total_duration");
        let prompt_eval = v["prompt_eval_duration"]
            .as_u64()
            .expect("prompt_eval_duration");
        let eval = v["eval_duration"].as_u64().expect("eval_duration");
        assert!(
            v["load_duration"].is_number(),
            "load_duration must be present"
        );

        assert!(
            prompt_eval > 0,
            "prefill must be measured, not reported as 0"
        );
        assert!(
            total >= prompt_eval + eval,
            "total ({total}) must cover prefill ({prompt_eval}) + decode ({eval})"
        );
        assert_ne!(
            total, eval,
            "total and eval must no longer be the same clock"
        );
    }

    #[tokio::test]
    async fn generate_stream_reports_a_real_split_on_the_done_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub", "prompt": "Hello", "stream": true,
            "options": {"num_predict": 4}
        });
        let resp = post_json(app, "/api/generate", body).await;
        assert_eq!(resp.status(), 200);
        let text = String::from_utf8(body_bytes(resp).await.to_vec()).unwrap();
        let done: serde_json::Value = text
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .find(|c| c["done"] == true)
            .expect("a done chunk");
        assert!(
            done["prompt_eval_duration"].as_u64().unwrap() > 0,
            "streamed prefill must be measured: {done}"
        );
        assert_ne!(done["total_duration"], done["eval_duration"]);
    }

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
    async fn keep_alive_is_applied_not_just_accepted() {
        // Regression: `keep_alive` was parsed and thrown away, and the test only
        // asserted a 200 — which it returned either way, so the field being inert
        // was invisible.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let body = |ka: serde_json::Value| serde_json::json!({"model": "stub", "prompt": "Hi", "stream": false, "keep_alive": ka});

        let resp = post_json(
            make_router(&state),
            "/api/generate",
            body(serde_json::json!("5m")),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            state
                .registry
                .keep_alive_override
                .get("stub")
                .map(|e| *e.value()),
            Some(Some(std::time::Duration::from_secs(300))),
            "\"5m\" must reach the registry"
        );

        // A negative value pins the model against timed eviction.
        let resp = post_json(
            make_router(&state),
            "/api/generate",
            body(serde_json::json!(-1)),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            state
                .registry
                .keep_alive_override
                .get("stub")
                .map(|e| *e.value()),
            Some(None),
            "a negative keep_alive must mean never evict"
        );

        // Zero becomes a zero TTL, so the next eviction pass drops it.
        let resp = post_json(
            make_router(&state),
            "/api/generate",
            body(serde_json::json!(0)),
        )
        .await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            state
                .registry
                .keep_alive_override
                .get("stub")
                .map(|e| *e.value()),
            Some(Some(std::time::Duration::ZERO))
        );
    }
}
