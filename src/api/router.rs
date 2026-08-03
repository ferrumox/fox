// Router assembly and AppState definition.

use axum::{
    http::Method,
    middleware,
    routing::{delete, get, post},
    Router,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tower_http::cors::{Any, CorsLayer};

use crate::model_registry::ModelRegistry;

/// Shared state injected into every route handler.
#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<ModelRegistry>,
    /// Stem of the model supplied via `--model-path` (pre-loaded at startup).
    pub primary_model: String,
    /// Injected as the first message when no system message is present.
    pub system_prompt: Option<String>,
    /// Which format to parse tool calls from: `auto` (default), `generic`, `hermes`.
    /// `auto` picks Hermes when the loaded model's own chat template natively
    /// formats tool calls, generic prompt-based parsing otherwise.
    pub tool_call_parser: String,
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
    /// Server-wide default for how far back the repetition/frequency/presence
    /// penalties look, in generated tokens (`--repeat-last-n`). `-1` = whole
    /// history (fox's historical behaviour), `0` = disabled, `n` = last `n`.
    /// A request that sets the field explicitly always wins.
    pub repeat_last_n: i32,
}

#[allow(clippy::too_many_arguments)]
pub fn router(
    registry: Arc<ModelRegistry>,
    primary_model: String,
    system_prompt: Option<String>,
    started_at: u64,
    models_dir: PathBuf,
    hf_token: Option<String>,
    api_key: Option<String>,
    tool_call_parser: String,
    repeat_last_n: i32,
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
        tool_call_parser,
        repeat_last_n,
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
        .route(
            "/lora-adapters",
            axum::routing::get(super::v1::lora::list_lora_adapters)
                .post(super::v1::lora::set_lora_adapters),
        )
        .route("/props", axum::routing::get(super::v1::props::props))
        .route("/slots", axum::routing::get(super::v1::props::slots))
        .route("/infill", axum::routing::post(super::v1::infill::infill))
        .route("/rerank", axum::routing::post(super::v1::rerank::rerank))
        .route("/v1/rerank", axum::routing::post(super::v1::rerank::rerank))
        // Tokenizer utilities (llama-server parity) — no inference, just the
        // loaded model's vocabulary and chat template.
        .route(
            "/tokenize",
            axum::routing::post(super::v1::tokenize::tokenize),
        )
        .route(
            "/detokenize",
            axum::routing::post(super::v1::tokenize::detokenize),
        )
        .route(
            "/apply-template",
            axum::routing::post(super::v1::tokenize::apply_template),
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
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
                .allow_headers(Any),
        )
        .with_state(state)
}
