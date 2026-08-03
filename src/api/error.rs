// Centralised error type for all API handlers.

use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::model_registry::{EngineEntry, ModelRegistry};

/// All failure modes that HTTP handlers can produce.
#[allow(dead_code)]
pub enum AppError {
    BadRequest(String),
    ModelNotFound(String),
    ModelLoadFailed(String),
    EmbeddingFailed(String),
    InternalError(String),
    /// The scheduler's waiting queue is full (`--max-queue-depth`).
    ServerBusy(String),
    /// The request needs more KV blocks than the pool will ever have, even empty.
    RequestTooLarge(String),
}

impl From<crate::scheduler::SubmitError> for AppError {
    fn from(e: crate::scheduler::SubmitError) -> Self {
        match e {
            crate::scheduler::SubmitError::QueueFull { .. } => AppError::ServerBusy(e.to_string()),
            crate::scheduler::SubmitError::TooLarge { .. } => {
                AppError::RequestTooLarge(e.to_string())
            }
        }
    }
}

/// The metric label to record for a rejected submission.
pub fn rejection_reason_label(e: &crate::scheduler::SubmitError) -> &'static str {
    match e {
        crate::scheduler::SubmitError::QueueFull { .. } => "queue_full",
        crate::scheduler::SubmitError::TooLarge { .. } => "too_large",
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, msg, err_type, code) = match self {
            AppError::BadRequest(m) => (
                StatusCode::BAD_REQUEST,
                m,
                "invalid_request_error",
                "invalid_request",
            ),
            AppError::ModelNotFound(m) => (
                StatusCode::NOT_FOUND,
                m,
                "invalid_request_error",
                "model_not_found",
            ),
            AppError::ModelLoadFailed(m) => (
                StatusCode::SERVICE_UNAVAILABLE,
                m,
                "server_error",
                "model_load_failed",
            ),
            AppError::EmbeddingFailed(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                "server_error",
                "embedding_failed",
            ),
            AppError::InternalError(m) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                m,
                "server_error",
                "internal_error",
            ),
            AppError::ServerBusy(m) => (
                StatusCode::TOO_MANY_REQUESTS,
                m,
                "server_error",
                "queue_full",
            ),
            AppError::RequestTooLarge(m) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                m,
                "invalid_request_error",
                "request_too_large",
            ),
        };
        (
            status,
            Json(json!({
                "error": {
                    "message": msg,
                    "type": err_type,
                    "param": null,
                    "code": code
                }
            })),
        )
            .into_response()
    }
}

/// Resolve and load a model, logging the right message depending on whether the
/// failure is "model not on disk" (404) or "model failed to load" (e.g. OOM → 503).
///
/// Returns `Ok(entry)` or an `Err(Response)` ready to return from a handler.
pub async fn load_model_or_respond(
    registry: &ModelRegistry,
    model: &str,
) -> Result<Arc<EngineEntry>, Response> {
    // First check if the model exists on disk — gives a clean 404 when it doesn't.
    if let Err(e) = registry.resolve_model_name(model) {
        tracing::warn!(model = %model, error = %e, "model not found");
        return Err(AppError::ModelNotFound(e.to_string()).into_response());
    }

    // Model exists — try to load it. Failures here are load errors (OOM, corrupt file…).
    match registry.get_or_load(model).await {
        Ok(entry) => Ok(entry),
        Err(e) => {
            tracing::error!(model = %model, error = %e, "failed to load model");
            Err(AppError::ModelLoadFailed(e.to_string()).into_response())
        }
    }
}
