// POST /api/embed handler.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::api::router::AppState;
use crate::api::types::{OllamaEmbedRequest, OllamaEmbedResponse};

pub async fn ollama_embed(
    State(state): State<AppState>,
    Json(req): Json<OllamaEmbedRequest>,
) -> impl IntoResponse {
    let entry = match state.registry.get_or_load(&req.model).await {
        Ok(e) => e,
        Err(e) => return (StatusCode::NOT_FOUND, e.to_string()).into_response(),
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
