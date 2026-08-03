// POST /infill — fill-in-the-middle completion (llama-server's `/infill`,
// server-context.cpp:4614-4690).
//
// This is what editor plugins call: given the code before and after the cursor, write
// what goes between. It is NOT chat with a clever prompt — FIM models are trained on
// an explicit token layout, and a model without those tokens in its vocabulary has no
// notion of "join up to this suffix". Such a request is rejected rather than answered
// with plausible-looking text that ignores the suffix entirely.

use axum::extract::State;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

use crate::api::error::{load_model_or_respond, AppError};
use crate::api::router::AppState;
use crate::api::shared::extractor::LenientJson;
use crate::api::shared::sampling_defaults as defaults;
use crate::api::shared::streaming::{collect_tokens, finish_reason_str};
use crate::scheduler::{InferenceRequest, SamplingParams, Token};

#[derive(Debug, Deserialize)]
pub struct InfillRequest {
    pub model: String,
    /// Text before the cursor.
    #[serde(default)]
    pub input_prefix: String,
    /// Text after the cursor.
    #[serde(default)]
    pub input_suffix: String,
    /// Additional context files, each `{filename?, text}`. Prepended ahead of the
    /// prefix, which is where llama.cpp's own repo-level FIM format puts them.
    #[serde(default)]
    pub input_extra: Option<Vec<InfillExtra>>,
    #[serde(default)]
    pub n_predict: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default, deserialize_with = "crate::api::types::deserialize_stop")]
    pub stop: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct InfillExtra {
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub text: String,
}

#[derive(Debug, Serialize)]
pub struct InfillResponse {
    /// The generated middle section — just the fill, not prefix+fill+suffix.
    pub content: String,
    pub tokens_predicted: u32,
    pub tokens_evaluated: u32,
    pub stop_type: String,
    pub model: String,
}

pub async fn infill(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<InfillRequest>,
) -> axum::response::Response {
    let (entry, lora) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    let Some(fim) = entry.engine.fim_tokens() else {
        return AppError::BadRequest(format!(
            "model '{}' has no fill-in-the-middle tokens in its vocabulary — /infill \
             needs a FIM-trained model (Qwen2.5-Coder, CodeLlama, DeepSeek-Coder, …). \
             Use /v1/completions for a plain continuation.",
            req.model
        ))
        .into_response();
    };

    if req.input_prefix.is_empty() && req.input_suffix.is_empty() {
        return AppError::BadRequest(
            "at least one of `input_prefix` or `input_suffix` must be non-empty".to_string(),
        )
        .into_response();
    }

    // Repo-level context first, as llama.cpp's FIM format places it: each extra file
    // ahead of the prefix, with its name as a comment so the model can tell them apart.
    let mut leading = String::new();
    for extra in req.input_extra.iter().flatten() {
        if let Some(name) = &extra.filename {
            leading.push_str(&format!("// {name}\n"));
        }
        leading.push_str(&extra.text);
        if !extra.text.ends_with('\n') {
            leading.push('\n');
        }
    }

    let tokenize = |s: &str| -> Result<Vec<i32>, String> {
        if s.is_empty() {
            return Ok(Vec::new());
        }
        entry
            .engine
            .tokenize(s)
            .map_err(|e| format!("tokenize failed: {e}"))
    };
    let (prefix_toks, suffix_toks) = match (
        tokenize(&format!("{leading}{}", req.input_prefix)),
        tokenize(&req.input_suffix),
    ) {
        (Ok(p), Ok(s)) => (p, s),
        (Err(e), _) | (_, Err(e)) => return AppError::BadRequest(e).into_response(),
    };

    // Suffix BEFORE prefix — the order FIM models are trained on, and the whole point
    // of the layout: the model reads what it has to join up to before it starts
    // writing. Prefix-first still generates fluent text, it just ignores the suffix.
    let mut prompt_tokens = Vec::with_capacity(prefix_toks.len() + suffix_toks.len() + 3);
    prompt_tokens.push(fim.suffix);
    prompt_tokens.extend_from_slice(&suffix_toks);
    prompt_tokens.push(fim.prefix);
    prompt_tokens.extend_from_slice(&prefix_toks);
    prompt_tokens.push(fim.middle);

    let prompt_len = prompt_tokens.len();
    let max_tokens = req.n_predict.unwrap_or(128) as usize;
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(defaults::TEMPERATURE).max(0.0),
        top_p: req.top_p.unwrap_or(defaults::TOP_P).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(defaults::openai::TOP_K),
        seed: req.seed,
        stop: req.stop.clone(),
        repeat_last_n: state.repeat_last_n,
        ..Default::default()
    };

    let req_id = entry.engine.next_request_id();
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
    let mut inference_req = InferenceRequest::new(req_id, prompt_tokens, max_tokens, sampling, tx);
    if let Some(selection) = lora {
        inference_req = inference_req.with_lora(selection);
    }
    if let Err(e) = entry.engine.submit_request(inference_req) {
        entry
            .engine
            .record_rejection(crate::api::error::rejection_reason_label(&e));
        return AppError::from(e).into_response();
    }

    let (content, tokens_predicted, stop_reason) = collect_tokens(&mut rx).await;
    tracing::info!(
        model = %req.model,
        prompt_tokens = prompt_len,
        completion_tokens = tokens_predicted,
        "infill done"
    );

    axum::Json(InfillResponse {
        content,
        tokens_predicted,
        tokens_evaluated: prompt_len as u32,
        stop_type: stop_reason
            .as_ref()
            .map(finish_reason_str)
            .unwrap_or("stop")
            .to_string(),
        model: req.model,
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn infill_rejects_a_model_without_fim_tokens() {
        // The stub reports no FIM tokens, like every chat model. Answering anyway
        // would return fluent text that ignores the suffix entirely — worse than an
        // error, because the caller cannot tell.
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/infill",
            serde_json::json!({
                "model": "stub",
                "input_prefix": "fn add(a: i32, b: i32) -> i32 {",
                "input_suffix": "}",
            }),
        )
        .await;
        assert_eq!(resp.status(), 400);
        let body = String::from_utf8(body_bytes(resp).await.to_vec()).unwrap();
        assert!(
            body.contains("fill-in-the-middle"),
            "the error must say why, not just 400: {body}"
        );
    }

    #[tokio::test]
    async fn infill_unknown_model_is_404() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/infill",
            serde_json::json!({"model": "nope", "input_prefix": "a", "input_suffix": "b"}),
        )
        .await;
        assert_eq!(resp.status(), 404);
    }
}
