// GET/POST /lora-adapters — inspect and re-scale loaded LoRA adapters at runtime
// (llama-server's server-context.cpp:5042-5102).
//
// fox loads adapters at startup with `--lora-modules name=path[:scale]` and a client
// selects one by naming it in the `model` field. What was missing is the ability to see
// what is loaded, and to change an adapter's strength without restarting the server —
// the whole point of a scale being a number rather than a rebuild.
//
// Only the *default* scale is mutable. A request's `LoraSelection` already carries its
// own scale down to `llama_set_adapters_lora`, so overriding what gets copied into it
// is the entire mechanism; the model's adapter handles stay immutable after load.

use axum::extract::State;
use axum::response::IntoResponse;
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::api::error::AppError;
use crate::api::router::AppState;
use crate::api::shared::extractor::LenientJson;

#[derive(Debug, Serialize)]
pub struct LoraAdapter {
    /// Positional id, matching llama-server's wire format. Stable for the process
    /// lifetime, since the adapter list is fixed at startup.
    pub id: usize,
    /// The name a client passes in the `model` field to select this adapter.
    pub name: String,
    pub path: String,
    /// The scale in effect now — the runtime override if one was set, else the
    /// value from `--lora-modules`.
    pub scale: f32,
}

/// One entry of a `POST /lora-adapters` body. Either `id` or `name` identifies the
/// adapter: llama-server addresses them by id, but fox's own `--lora-modules` and the
/// `model` field are name-based, so accepting both avoids making callers translate.
#[derive(Debug, Deserialize)]
pub struct LoraScaleUpdate {
    #[serde(default)]
    pub id: Option<usize>,
    #[serde(default)]
    pub name: Option<String>,
    pub scale: f32,
}

pub async fn list_lora_adapters(State(state): State<AppState>) -> Json<Vec<LoraAdapter>> {
    Json(
        state
            .registry
            .lora_adapters()
            .into_iter()
            .enumerate()
            .map(|(id, (name, path, scale))| LoraAdapter {
                id,
                name,
                path: path.display().to_string(),
                scale,
            })
            .collect(),
    )
}

pub async fn set_lora_adapters(
    State(state): State<AppState>,
    LenientJson(updates): LenientJson<Vec<LoraScaleUpdate>>,
) -> axum::response::Response {
    let loaded = state.registry.lora_adapters();
    if loaded.is_empty() {
        return AppError::BadRequest(
            "no LoRA adapters are loaded — start the server with --lora-modules".to_string(),
        )
        .into_response();
    }

    // Resolve every update before applying any: a body naming one valid and one
    // unknown adapter should change nothing, rather than leave the server in a state
    // the caller did not ask for and cannot infer from the error.
    let mut resolved = Vec::with_capacity(updates.len());
    for u in &updates {
        let name = match (&u.name, u.id) {
            (Some(n), _) => n.clone(),
            (None, Some(id)) => match loaded.get(id) {
                Some((n, _, _)) => n.clone(),
                None => {
                    return AppError::BadRequest(format!(
                        "no LoRA adapter with id {id} (server has {})",
                        loaded.len()
                    ))
                    .into_response()
                }
            },
            (None, None) => {
                return AppError::BadRequest(
                    "each entry needs `id` or `name` to identify the adapter".to_string(),
                )
                .into_response()
            }
        };
        if !loaded.iter().any(|(n, _, _)| *n == name) {
            return AppError::BadRequest(format!("LoRA adapter '{name}' is not loaded"))
                .into_response();
        }
        if !u.scale.is_finite() {
            return AppError::BadRequest(format!("scale for '{name}' must be finite"))
                .into_response();
        }
        resolved.push((name, u.scale));
    }

    for (name, scale) in &resolved {
        if let Err(e) = state.registry.set_lora_scale(name, *scale) {
            return AppError::BadRequest(e.to_string()).into_response();
        }
        tracing::info!(adapter = %name, scale, "LoRA scale changed at runtime");
    }

    // Note for callers: this changes the default applied to *subsequent* requests.
    // Generations already in flight keep the scale they were admitted with, since the
    // adapter set is a property of the batch being decoded.
    Json(serde_json::json!({ "success": true, "updated": resolved.len() })).into_response()
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn listing_is_empty_without_lora_modules() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/lora-adapters").await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
        assert_eq!(v.as_array().unwrap().len(), 0, "{v}");
    }

    #[tokio::test]
    async fn setting_a_scale_without_adapters_is_a_clear_400() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/lora-adapters",
            serde_json::json!([{"name": "nope", "scale": 0.5}]),
        )
        .await;
        assert_eq!(resp.status(), 400);
        let body = String::from_utf8(body_bytes(resp).await.to_vec()).unwrap();
        assert!(
            body.contains("--lora-modules"),
            "must say how to fix it: {body}"
        );
    }
}
