// Router assembly and AppState definition.

use axum::{
    http::{HeaderValue, Method},
    middleware,
    routing::{delete, get, post},
    Router,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use crate::model_registry::ModelRegistry;

fn build_cors_layer(origins: &str) -> CorsLayer {
    let methods = [Method::GET, Method::POST, Method::DELETE, Method::OPTIONS];
    if origins == "*" {
        return CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(methods)
            .allow_headers(Any);
    }
    let parsed: Vec<HeaderValue> = origins
        .split(',')
        .filter_map(|o| o.trim().parse().ok())
        .collect();
    CorsLayer::new()
        .allow_origin(parsed)
        .allow_methods(methods)
        .allow_headers(Any)
}

/// Shared state injected into every route handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// Stem of the model supplied via `--model-path` (pre-loaded at startup).
    pub primary_model: String,
    /// Injected as the first message when no system message is present.
    pub system_prompt: Option<String>,
    /// Unix timestamp (seconds) when the server started.
    pub started_at: u64,
    /// Directory where `.gguf` model files are stored.
    pub models_dir: PathBuf,
    /// Cache of SHA256 digests keyed by file path. Computed once per file.
    pub digest_cache: Arc<Mutex<HashMap<PathBuf, String>>>,
    /// HuggingFace API token for authenticated model pulls.
    pub hf_token: Option<String>,
    /// Optional Bearer token required on every request (`--api-key` / `FOX_API_KEY`).
    pub api_key: Option<String>,
}

pub fn router(
    registry: Arc<ModelRegistry>,
    primary_model: String,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
    hf_token: Option<String>,
    api_key: Option<String>,
    cors_origins: &str,
) -> Router {
    let state = AppState {
        registry,
        primary_model,
        system_prompt,
        started_at,
        models_dir,
        digest_cache: Arc::new(Mutex::new(HashMap::new())),
        hf_token,
        api_key,
    };

    Router::new()
        // OpenAI-compatible
        .route(
            "/v1/chat/completions",
            post(crate::api::v1::chat::chat_completions),
        )
        .route(
            "/v1/completions",
            post(crate::api::v1::completions::completions),
        )
        .route("/v1/models", get(crate::api::v1::models::models))
        .route(
            "/v1/models/:model_id",
            get(crate::api::v1::models::model_by_id),
        )
        .route(
            "/v1/embeddings",
            post(crate::api::v1::embeddings::v1_embeddings),
        )
        .route("/health", get(crate::api::v1::models::health))
        .route("/metrics", get(crate::api::v1::models::metrics_handler))
        // Ollama-compatible
        .route(
            "/api/version",
            get(crate::api::ollama::management::ollama_version),
        )
        .route(
            "/api/tags",
            get(crate::api::ollama::management::ollama_tags),
        )
        .route("/api/ps", get(crate::api::ollama::management::ollama_ps))
        .route(
            "/api/show",
            post(crate::api::ollama::management::ollama_show),
        )
        .route(
            "/api/delete",
            delete(crate::api::ollama::management::ollama_delete),
        )
        .route("/api/embed", post(crate::api::ollama::embed::ollama_embed))
        .route(
            "/api/generate",
            post(crate::api::ollama::generate::ollama_generate),
        )
        .route("/api/chat", post(crate::api::ollama::chat::ollama_chat))
        .route(
            "/api/copy",
            post(crate::api::ollama::management::ollama_copy),
        )
        .route(
            "/api/create",
            post(crate::api::ollama::management::ollama_create),
        )
        .route("/api/pull", post(crate::api::pull_handler::ollama_pull))
        .route(
            "/api/models/:name/load",
            post(crate::api::ollama::management::api_model_load),
        )
        .route(
            "/api/models/:name/unload",
            post(crate::api::ollama::management::api_model_unload),
        )
        .route("/", get(|| async { "Fox is running" }))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            crate::api::auth::auth_middleware,
        ))
        .layer(build_cors_layer(cors_origins))
        .with_state(state)
}
