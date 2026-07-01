// POST /v1/embeddings handler.

use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};

use crate::api::error::load_model_or_respond;
use crate::api::router::AppState;
use crate::api::types::{EmbeddingObject, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage};

pub async fn v1_embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let entry = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };
    let engine = &entry.engine;
    let inputs = req.input.into_vec();
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_tokens = 0u32;

    for (i, text) in inputs.iter().enumerate() {
        match engine.embed(text).await {
            Ok(embedding) => {
                let tokens = engine.tokenize(text).map(|t| t.len()).unwrap_or(0) as u32;
                total_tokens += tokens;
                data.push(EmbeddingObject {
                    object: "embedding".to_string(),
                    embedding,
                    index: i,
                });
            }
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embedding failed: {e}"),
                )
                    .into_response();
            }
        }
    }

    Json(EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    })
    .into_response()
}
