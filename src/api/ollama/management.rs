// GET /api/version, /api/tags, /api/ps, POST /api/show, DELETE /api/delete handlers.

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};

use crate::api::router::AppState;
use crate::api::shared::digest::{get_digest, modified_at_rfc3339};
use crate::api::types::{
    DeleteRequest, OllamaDetails, OllamaModel, PsEntry, PsResponse, ShowRequest, ShowResponse,
    TagsResponse, VersionResponse,
};
use crate::cli::{format_size, list_models};
use crate::cli::show::{parse_architecture, parse_quantization};

pub async fn ollama_version() -> Json<VersionResponse> {
    Json(VersionResponse {
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn ollama_tags(State(state): State<AppState>) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let mut models = Vec::with_capacity(entries.len());
    for (path, meta) in &entries {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
        let digest = get_digest(path, &state.digest_cache).await;
        models.push(OllamaModel {
            name: stem.to_string(),
            size: meta.len(),
            digest,
            details: OllamaDetails {
                format: "gguf".to_string(),
                family: parse_architecture(stem).unwrap_or("unknown").to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: parse_quantization(stem).unwrap_or("unknown").to_string(),
            },
            modified_at: modified_at_rfc3339(meta),
        });
    }

    Json(TagsResponse { models }).into_response()
}

pub async fn ollama_ps(State(state): State<AppState>) -> Json<PsResponse> {
    let loaded = state.registry.loaded();
    let mut ps_entries = Vec::with_capacity(loaded.len());

    for (name, entry) in &loaded {
        let file_info = list_models(&state.models_dir)
            .ok()
            .and_then(|entries| {
                entries.into_iter().find(|(path, _)| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|stem| stem == name.as_str())
                        .unwrap_or(false)
                })
            });

        let (size, digest) = if let Some((path, meta)) = file_info {
            let d = get_digest(&path, &state.digest_cache).await;
            (meta.len(), d)
        } else {
            (0u64, "sha256:unknown".to_string())
        };

        let model_name = entry.engine.model_name().to_string();
        ps_entries.push(PsEntry {
            name: model_name.clone(),
            size,
            digest,
            details: OllamaDetails {
                format: "gguf".to_string(),
                family: parse_architecture(&model_name).unwrap_or("unknown").to_string(),
                parameter_size: "unknown".to_string(),
                quantization_level: parse_quantization(&model_name)
                    .unwrap_or("unknown")
                    .to_string(),
            },
            expires_at: "0001-01-01T00:00:00Z".to_string(),
            size_vram: 0,
        });
    }

    Json(PsResponse { models: ps_entries })
}

pub async fn ollama_show(
    State(state): State<AppState>,
    Json(req): Json<ShowRequest>,
) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let found = entries.into_iter().find(|(path, _)| {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        stem == req.name || name == req.name
    });

    match found {
        None => (
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.name),
        )
            .into_response(),
        Some((path, meta)) => {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
            let arch = parse_architecture(stem).unwrap_or("unknown");
            let quant = parse_quantization(stem).unwrap_or("unknown");
            let size_str = format_size(meta.len());
            let digest = get_digest(&path, &state.digest_cache).await;
            let resp = ShowResponse {
                modelfile: format!("# GGUF model: {}", stem),
                parameters: String::new(),
                template: String::new(),
                details: OllamaDetails {
                    format: "gguf".to_string(),
                    family: arch.to_string(),
                    parameter_size: "unknown".to_string(),
                    quantization_level: quant.to_string(),
                },
                model_info: serde_json::json!({
                    "general.architecture": arch,
                    "general.quantization": quant,
                    "general.size": size_str,
                    "general.digest": digest,
                    "general.modified_at": modified_at_rfc3339(&meta),
                    "general.path": path.display().to_string(),
                }),
            };
            Json(resp).into_response()
        }
    }
}

pub async fn ollama_delete(
    State(state): State<AppState>,
    Json(req): Json<DeleteRequest>,
) -> impl IntoResponse {
    let entries = match list_models(&state.models_dir) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list models: {e}"),
            )
                .into_response()
        }
    };

    let found = entries.into_iter().find(|(path, _)| {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        stem == req.name || name == req.name
    });

    match found {
        None => (
            StatusCode::NOT_FOUND,
            format!("model '{}' not found", req.name),
        )
            .into_response(),
        Some((path, _)) => {
            state.registry.unload(&req.name);
            match std::fs::remove_file(&path) {
                Ok(_) => StatusCode::OK.into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to delete model: {e}"),
                )
                    .into_response(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;
    use crate::api::test_helpers::*;
    use crate::api::router::router;
    use crate::model_registry::{ModelRegistry, RegistryConfig};

    fn empty_registry(dir: &std::path::Path) -> Arc<ModelRegistry> {
        let cfg = RegistryConfig {
            models_dir: dir.to_path_buf(),
            max_models: 4,
            max_batch_size: 4,
            max_context_len: 512,
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
        };
        Arc::new(ModelRegistry::new(cfg, HashMap::new()))
    }

    #[tokio::test]
    async fn test_api_version_returns_version() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/api/version").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let version = v["version"].as_str().unwrap();
        assert!(!version.is_empty());
        assert!(version.contains('.'));
    }

    #[tokio::test]
    async fn test_api_ps_empty_registry() {
        let dir = tempfile::tempdir().unwrap();
        let reg = empty_registry(dir.path());
        let app = router(reg, "none".to_string(), None, 0, dir.path().to_path_buf(), None);
        let resp = get_req(app, "/api/ps").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["models"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn test_api_ps_with_loaded_model() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("my-model", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/api/ps").await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let models = v["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"].as_str().unwrap(), "my-model");
    }
}
