// POST /api/embed handler.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::api::shared::extractor::LenientJson;

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::types::{OllamaEmbedRequest, OllamaEmbedResponse};

pub async fn ollama_embed(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<OllamaEmbedRequest>,
) -> impl IntoResponse {
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    let engine = &entry.engine;
    let inputs = req.input.into_vec();
    let mut embeddings = Vec::with_capacity(inputs.len());

    for text in &inputs {
        match engine.embed(text).await {
            Ok(embedding) => embeddings.push(embedding),
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    Json(OllamaEmbedResponse {
        model: req.model,
        embeddings,
    })
    .into_response()
}
