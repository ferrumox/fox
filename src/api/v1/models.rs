// GET /v1/models, GET /health, GET /metrics handlers.

use axum::{extract::State, http::header, response::IntoResponse, Json};

use crate::api::router::AppState;
use crate::api::types::{HealthResponse, ModelInfo, ModelsResponse};
use crate::cli::list_models;

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let entry = state.registry.get_or_load(&state.primary_model).await.ok();
    let (kv_cache_usage, queue_depth, active_requests, model_name) = match entry {
        Some(e) => (
            e.engine.kv_cache_usage(),
            e.engine.queue_depth(),
            e.engine.active_requests(),
            e.engine.model_name().to_string(),
        ),
        None => (0.0, 0, 0, state.primary_model.clone()),
    };
    Json(HealthResponse {
        status: "ok".to_string(),
        kv_cache_usage,
        queue_depth,
        active_requests,
        model_name,
        started_at: state.started_at,
    })
}

/// Lists all `.gguf` models available on disk (OpenAI format).
pub async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    let entries = list_models(&state.models_dir).unwrap_or_default();
    let data = entries
        .iter()
        .filter_map(|(path, _)| {
            path.file_stem()
                .and_then(|s| s.to_str())
                .map(|stem| ModelInfo {
                    id: stem.to_string(),
                    object: "model".to_string(),
                })
        })
        .collect();
    Json(ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = prometheus::TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut body = String::new();
    if let Err(e) = encoder.encode_utf8(&metric_families, &mut body) {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(header::CONTENT_TYPE, "text/plain")],
            format!("metrics encoding error: {e}"),
        );
    }
    (
        axum::http::StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        body,
    )
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn test_health_with_primary_model() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("primary", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/health").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["status"].as_str().unwrap(), "ok");
        assert_eq!(v["model_name"].as_str().unwrap(), "primary");
    }

    #[tokio::test]
    async fn test_v1_models_lists_disk_files() {
        use std::collections::HashMap;
        use std::sync::Arc;
        use crate::api::router::router;
        use crate::model_registry::{ModelRegistry, RegistryConfig};

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("alpha.gguf"), b"").unwrap();
        std::fs::write(dir.path().join("beta.gguf"), b"").unwrap();
        let cfg = RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 4,
            max_batch_size: 4,
            max_context_len: 512,
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
        };
        let reg = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        let app = router(reg, "alpha".to_string(), None, 0, dir.path().to_path_buf(), None);
        let resp = get_req(app, "/v1/models").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let data = v["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"alpha"));
        assert!(ids.contains(&"beta"));
    }
}
