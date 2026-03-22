// Lenient JSON extractor that accepts any Content-Type (or none).
//
// Axum's built-in `Json<T>` rejects requests that don't carry
// `Content-Type: application/json`, but the real Ollama server is permissive
// about this header. LiteLLM and other clients sometimes omit it or send
// `application/x-ndjson`, so Ollama-compatible endpoints use this extractor
// instead.

use axum::{
    async_trait,
    body::Body,
    extract::FromRequest,
    http::{Request, StatusCode},
    response::{IntoResponse, Response},
};
use serde::de::DeserializeOwned;

pub struct LenientJson<T>(pub T);

#[async_trait]
impl<T, S> FromRequest<S> for LenientJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request(req: Request<Body>, state: &S) -> Result<Self, Self::Rejection> {
        let bytes = axum::body::Bytes::from_request(req, state)
            .await
            .map_err(|e| {
                (StatusCode::BAD_REQUEST, e.to_string()).into_response()
            })?;

        let value: T = serde_json::from_slice(&bytes).map_err(|e| {
            (
                StatusCode::UNPROCESSABLE_ENTITY,
                format!("invalid JSON: {e}"),
            )
                .into_response()
        })?;

        Ok(LenientJson(value))
    }
}
