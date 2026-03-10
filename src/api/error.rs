// Centralised error type for all API handlers.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

/// All failure modes that HTTP handlers can produce.
#[allow(dead_code)]
pub enum AppError {
    ModelNotFound(String),
    EmbeddingFailed(String),
    InternalError(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            AppError::ModelNotFound(m) => (StatusCode::NOT_FOUND, m),
            AppError::EmbeddingFailed(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
            AppError::InternalError(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (status, Json(json!({"error": msg}))).into_response()
    }
}
