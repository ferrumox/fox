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
use crate::tools::ToolBoard;

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
    /// Pluggable tool registry exposed via `/v1/tools` and consumed by the
    /// function-calling path during chat completion.
    pub tool_board: ToolBoard,
}

/// Configuration for the CORS layer applied to every route.
///
/// `Any` keeps the historical behaviour where any origin is allowed — fine
/// for desktop clients hitting `localhost`, dangerous when the server is
/// reachable from the public internet. Operators that expose fox to a
/// browser-shaped client should pass `Origins(vec![…])` with the exact
/// origins their UI loads from.
#[derive(Debug, Clone, Default)]
pub enum CorsConfig {
    #[default]
    Any,
    Origins(Vec<String>),
}

pub fn router(
    registry: Arc<ModelRegistry>,
    primary_model: String,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
    hf_token: Option<String>,
    api_key: Option<String>,
) -> Router {
    router_with_cors(
        registry,
        primary_model,
        system_prompt,
        started_at,
        models_dir,
        hf_token,
        api_key,
        CorsConfig::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn router_with_cors(
    registry: Arc<ModelRegistry>,
    primary_model: String,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
    hf_token: Option<String>,
    api_key: Option<String>,
    cors: CorsConfig,
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
        tool_board: crate::tools::default_board(),
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
        // Tool registry
        .route("/v1/tools", get(crate::api::v1::tools::list_tools))
        .route(
            "/v1/tools/execute",
            post(crate::api::v1::tools::execute_tool),
        )
        // Pipelines DAG
        .route(
            "/v1/pipelines/run",
            post(crate::api::v1::pipelines::run_pipeline),
        )
        // Chat completions with server-side tool execution loop
        .route(
            "/v1/chat/completions/auto",
            post(crate::api::v1::chat_auto::chat_auto_completions),
        )
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
        .layer(build_cors_layer(&cors))
        .with_state(state)
}

fn build_cors_layer(cfg: &CorsConfig) -> CorsLayer {
    let base = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
        .allow_headers(Any);
    match cfg {
        CorsConfig::Any => base.allow_origin(Any),
        CorsConfig::Origins(origins) => {
            let parsed: Vec<HeaderValue> = origins
                .iter()
                .filter_map(|o| HeaderValue::from_str(o).ok())
                .collect();
            if parsed.is_empty() {
                // Empty / all-invalid list collapses to `Any` so the server
                // doesn't end up unreachable due to a typo.
                base.allow_origin(Any)
            } else {
                base.allow_origin(parsed)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_cors_layer_compiles_for_any_and_origins() {
        let _ = build_cors_layer(&CorsConfig::Any);
        let _ = build_cors_layer(&CorsConfig::Origins(vec![
            "https://example.com".to_string(),
            "http://localhost:5173".to_string(),
        ]));
        // Empty list falls back to Any.
        let _ = build_cors_layer(&CorsConfig::Origins(Vec::new()));
        // Invalid origin is silently dropped.
        let _ = build_cors_layer(&CorsConfig::Origins(vec!["not a header value\n".into()]));
    }
}
