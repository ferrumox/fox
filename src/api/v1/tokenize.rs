// POST /tokenize, /detokenize, /apply-template — llama-server's tokenizer utilities
// (server-context.cpp:4899-4956, 4846-4856).
//
// These need no inference, only the loaded model's vocabulary and chat template, and
// they are what clients use to count tokens before sending a request or to debug why
// a template renders the way it does. fox had the underlying pieces
// (`InferenceEngine::tokenize`, `build_prompt_tokens`) and simply never routed them.

use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};

use crate::api::error::{load_model_or_respond, AppError};
use crate::api::router::AppState;
use crate::api::shared::extractor::LenientJson;
use crate::api::types::ChatMessage;
use axum::response::IntoResponse;

#[derive(Debug, Deserialize)]
pub struct TokenizeRequest {
    pub model: String,
    pub content: String,
    /// Return `{id, piece}` objects instead of bare ids.
    #[serde(default)]
    pub with_pieces: bool,
}

/// One token when `with_pieces` is set. A token holding only part of a multi-byte
/// codepoint has no valid UTF-8 piece, so its raw bytes are reported instead —
/// dropping to a replacement character would misrepresent the vocabulary.
#[derive(Debug, Serialize)]
pub struct TokenPiece {
    pub id: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub piece: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum TokenizeResponse {
    Ids { tokens: Vec<i32> },
    Pieces { tokens: Vec<TokenPiece> },
}

#[derive(Debug, Deserialize)]
pub struct DetokenizeRequest {
    pub model: String,
    pub tokens: Vec<i32>,
}

#[derive(Debug, Serialize)]
pub struct DetokenizeResponse {
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ApplyTemplateRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
}

#[derive(Debug, Serialize)]
pub struct ApplyTemplateResponse {
    pub prompt: String,
}

pub async fn tokenize(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<TokenizeRequest>,
) -> axum::response::Response {
    let (entry, _) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    let tokens = match entry.engine.tokenize(&req.content) {
        Ok(t) => t,
        Err(e) => return AppError::BadRequest(format!("tokenize failed: {e}")).into_response(),
    };

    if !req.with_pieces {
        return Json(TokenizeResponse::Ids { tokens }).into_response();
    }
    let pieces = tokens
        .into_iter()
        .map(|id| {
            let raw = entry.engine.token_piece_bytes(id);
            match String::from_utf8(raw.clone()) {
                Ok(piece) => TokenPiece {
                    id,
                    piece: Some(piece.replace('\u{2581}', " ")),
                    bytes: None,
                },
                Err(_) => TokenPiece {
                    id,
                    piece: None,
                    bytes: Some(raw),
                },
            }
        })
        .collect();
    Json(TokenizeResponse::Pieces { tokens: pieces }).into_response()
}

pub async fn detokenize(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<DetokenizeRequest>,
) -> axum::response::Response {
    let (entry, _) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    Json(DetokenizeResponse {
        content: entry.engine.detokenize(&req.tokens),
    })
    .into_response()
}

pub async fn apply_template(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<ApplyTemplateRequest>,
) -> axum::response::Response {
    let (entry, _) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    let flat: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| {
            let text = m.content.as_ref().map(|c| c.as_text()).unwrap_or_default();
            (m.role.clone(), text)
        })
        .collect();

    // Render through the SAME path a real request takes (`build_prompt_tokens` →
    // the model's own Jinja template), then detokenize. Going via tokens rather
    // than returning the pre-tokenization string is deliberate: it shows exactly
    // what the model will receive, control tokens included, so this endpoint can
    // actually be used to debug a misbehaving template.
    let tokens = match entry.engine.build_prompt_tokens(&flat, false, None) {
        Ok(t) => t,
        Err(e) => {
            return AppError::BadRequest(format!("template render failed: {e}")).into_response()
        }
    };
    Json(ApplyTemplateResponse {
        prompt: entry.engine.detokenize(&tokens),
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn tokenize_returns_ids() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/tokenize",
            serde_json::json!({"model": "stub", "content": "hello world"}),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["tokens"].is_array(), "{v}");
    }

    #[tokio::test]
    async fn tokenize_with_pieces_returns_objects() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/tokenize",
            serde_json::json!({"model": "stub", "content": "hi", "with_pieces": true}),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        let first = &v["tokens"][0];
        assert!(
            first["id"].is_number(),
            "expected {{id, piece}} objects: {v}"
        );
    }

    #[tokio::test]
    async fn detokenize_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/detokenize",
            serde_json::json!({"model": "stub", "tokens": [1, 2, 3]}),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["content"].is_string(), "{v}");
    }

    #[tokio::test]
    async fn apply_template_returns_a_prompt() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/apply-template",
            serde_json::json!({
                "model": "stub",
                "messages": [{"role": "user", "content": "Hi"}]
            }),
        )
        .await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert!(v["prompt"].is_string(), "{v}");
    }

    #[tokio::test]
    async fn unknown_model_is_404() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/tokenize",
            serde_json::json!({"model": "nope", "content": "x"}),
        )
        .await;
        assert_eq!(resp.status(), 404);
    }
}
