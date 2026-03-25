// Shared test fixtures — only compiled in test mode.
//
// Handler-level integration tests import these via `use crate::api::test_helpers::*;`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::{
    body::Body,
    http::{Method, Request},
    Router,
};
use tower::ServiceExt;

use crate::api::router::{router, AppState};
use crate::model_registry::{EngineEntry, ModelRegistry, RegistryConfig};

/// Build a minimal test registry backed by `StubModel`.
/// Creates a dummy `.gguf` file at `<dir>/<name>.gguf` so that
/// `resolve_model_name` can locate the model on disk.
pub fn make_test_registry(
    name: &str,
    dir: &std::path::Path,
) -> (Arc<ModelRegistry>, Arc<EngineEntry>) {
    std::fs::write(dir.join(format!("{name}.gguf")), b"").unwrap();
    let cfg = RegistryConfig {
        models_dir: dir.to_path_buf(),
        max_models: 4,
        max_batch_size: 4,
        max_context_len: Some(512),
        block_size: 16,
        gpu_memory_bytes: 4 * 1024 * 1024,
        gpu_memory_fraction: 0.9,
        metrics: None,
        keep_alive_secs: 0,
        type_kv: 1,
        main_gpu: 0,
        split_mode: 1,
        tensor_split: vec![],
        moe_offload_cpu: false,
    };
    let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
    let entry = EngineEntry::for_test(name);
    registry.preload_for_test(name, entry.clone());
    (registry, entry)
}

/// Build a test `AppState` with one preloaded stub model.
pub fn make_test_state(name: &str, dir: &std::path::Path) -> (AppState, Arc<EngineEntry>) {
    let (registry, entry) = make_test_registry(name, dir);
    let state = AppState {
        registry,
        primary_model: name.to_string(),
        system_prompt: None,
        started_at: 0,
        models_dir: dir.to_path_buf(),
        digest_cache: Arc::new(Mutex::new(HashMap::new())),
        hf_token: None,
        api_key: None,
    };
    (state, entry)
}

/// Build a test `AppState` backed by `ThinkingStubModel`.
/// The model reports `supports_thinking() = true` and produces the token
/// sequence: "thought" → "</think>" → "answer" → EOS.
pub fn make_test_state_thinking(name: &str, dir: &std::path::Path) -> (AppState, Arc<EngineEntry>) {
    std::fs::write(dir.join(format!("{name}.gguf")), b"").unwrap();
    let cfg = crate::model_registry::RegistryConfig {
        models_dir: dir.to_path_buf(),
        max_models: 4,
        max_batch_size: 4,
        max_context_len: Some(512),
        block_size: 16,
        gpu_memory_bytes: 4 * 1024 * 1024,
        gpu_memory_fraction: 0.9,
        metrics: None,
        keep_alive_secs: 0,
        type_kv: 1,
        main_gpu: 0,
        split_mode: 1,
        tensor_split: vec![],
        moe_offload_cpu: false,
    };
    let registry = Arc::new(crate::model_registry::ModelRegistry::new(
        cfg,
        HashMap::new(),
    ));
    let entry = EngineEntry::for_test_thinking(name);
    registry.preload_for_test(name, entry.clone());
    let state = AppState {
        registry,
        primary_model: name.to_string(),
        system_prompt: None,
        started_at: 0,
        models_dir: dir.to_path_buf(),
        digest_cache: Arc::new(Mutex::new(HashMap::new())),
        hf_token: None,
        api_key: None,
    };
    (state, entry)
}

/// Build a router from an existing `AppState`.
pub fn make_router(state: &AppState) -> Router {
    router(
        state.registry.clone(),
        state.primary_model.clone(),
        state.system_prompt.clone(),
        state.started_at,
        state.models_dir.clone(),
        state.hf_token.clone(),
        state.api_key.clone(),
    )
}

/// POST a JSON body and return the full response.
pub async fn post_json(
    app: Router,
    path: &str,
    body: serde_json::Value,
) -> axum::response::Response {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(&body).unwrap()))
        .unwrap();
    app.oneshot(req).await.unwrap()
}

/// GET and return the full response.
pub async fn get_req(app: Router, path: &str) -> axum::response::Response {
    let req = Request::builder()
        .method(Method::GET)
        .uri(path)
        .body(Body::empty())
        .unwrap();
    app.oneshot(req).await.unwrap()
}

/// Collect response body bytes.
pub async fn body_bytes(resp: axum::response::Response) -> bytes::Bytes {
    use http_body_util::BodyExt;
    resp.into_body().collect().await.unwrap().to_bytes()
}
