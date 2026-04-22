// Bearer token authentication middleware.
//
// When `--api-key` is set, every request must carry
// `Authorization: Bearer <key>`.  Requests without a valid token receive
// 401 Unauthorized.  When no API key is configured the middleware is a
// transparent no-op.

use axum::{
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::Response,
};

use super::router::AppState;

pub async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, StatusCode> {
    let Some(ref expected_key) = state.api_key else {
        // No key configured — pass through.
        return Ok(next.run(request).await);
    };

    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(value) if value.starts_with("Bearer ") => {
            let token = &value["Bearer ".len()..];
            if token != expected_key {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
        _ => return Err(StatusCode::UNAUTHORIZED),
    }

    Ok(next.run(request).await)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, middleware, routing::get, Router};
    use std::{
        collections::HashMap,
        path::PathBuf,
        sync::{Arc, Mutex},
    };
    use tower::ServiceExt;

    fn make_state(api_key: Option<&str>) -> AppState {
        AppState {
            registry: Arc::new(crate::model_registry::ModelRegistry::new(
                crate::model_registry::RegistryConfig {
                    models_dir: PathBuf::from("/tmp"),
                    max_models: 1,
                    max_batch_size: 1,
                    max_context_len: Some(512),
                    block_size: 16,
                    gpu_memory_bytes: 0,
                    gpu_memory_fraction: 0.85,
                    metrics: None,
                    keep_alive_secs: 0,
                    type_k: 1,
                    type_v: 1,
                    main_gpu: 0,
                    split_mode: 1,
                    tensor_split: vec![],
                    moe_offload_cpu: false,
                    mmproj_path: None,
                },
                Default::default(),
            )),
            primary_model: String::new(),
            system_prompt: None,
            started_at: 0,
            models_dir: PathBuf::from("/tmp"),
            digest_cache: Arc::new(Mutex::new(HashMap::new())),
            hf_token: None,
            api_key: api_key.map(str::to_string),
        }
    }

    async fn dummy_handler() -> &'static str {
        "ok"
    }

    fn app(state: AppState) -> Router {
        Router::new()
            .route("/test", get(dummy_handler))
            .layer(middleware::from_fn_with_state(
                state.clone(),
                auth_middleware,
            ))
            .with_state(state)
    }

    #[tokio::test]
    async fn no_key_configured_passes_through() {
        let response = app(make_state(None))
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn valid_bearer_token_passes() {
        let response = app(make_state(Some("secret")))
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn wrong_token_returns_401() {
        let response = app(make_state(Some("secret")))
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header("Authorization", "Bearer wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn missing_header_returns_401() {
        let response = app(make_state(Some("secret")))
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn non_bearer_scheme_returns_401() {
        let response = app(make_state(Some("secret")))
            .oneshot(
                axum::http::Request::builder()
                    .uri("/test")
                    .header("Authorization", "Basic dXNlcjpwYXNz")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }
}
