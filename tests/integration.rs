// End-to-end HTTP integration tests using StubModel (no real GPU required).
//
// Run with: FOX_SKIP_LLAMA=1 cargo test --all --features test-helpers

use axum::{body::Body, http::Method};
use tower::ServiceExt;

use ferrumox::api::router as build_router;
use ferrumox::api::test_helpers::*;
use ferrumox::model_registry::EngineEntry;

// ─────────────────────────────────────────────────────────────────────────────
// POST /v1/chat/completions
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_completions_happy_path() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(!v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .is_empty());
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_completions_unknown_model_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "does-not-exist-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
        }),
    )
    .await;
    assert_eq!(resp.status(), 404);
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /v1/embeddings
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn embeddings_happy_path() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({
            "model": "stub",
            "input": "Hello world",
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["object"], "list");
    assert!(!v["data"].as_array().unwrap().is_empty());
    assert_eq!(v["data"][0]["object"], "embedding");
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /v1/models
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn list_models_returns_disk_files() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = get_req(app, "/v1/models").await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["object"], "list");
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"stub"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /health
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = get_req(app, "/health").await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["status"], "ok");
    assert_eq!(v["model_name"], "stub");
}

// ─────────────────────────────────────────────────────────────────────────────
// Auth middleware on real routes
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn auth_missing_token_returns_401_on_real_route() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, _) = make_test_state("stub", dir.path());
    state.api_key = Some("s3cr3t".to_string());
    let app = make_router(&state);
    let req = axum::http::Request::builder()
        .method(Method::GET)
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn auth_wrong_token_returns_401_on_real_route() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, _) = make_test_state("stub", dir.path());
    state.api_key = Some("s3cr3t".to_string());
    let app = make_router(&state);
    let req = axum::http::Request::builder()
        .method(Method::GET)
        .uri("/health")
        .header("Authorization", "Bearer wrong-token")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 401);
}

#[tokio::test]
async fn auth_correct_token_passes_on_real_route() {
    let dir = tempfile::tempdir().unwrap();
    let (mut state, _) = make_test_state("stub", dir.path());
    state.api_key = Some("s3cr3t".to_string());
    let app = make_router(&state);
    let req = axum::http::Request::builder()
        .method(Method::GET)
        .uri("/health")
        .header("Authorization", "Bearer s3cr3t")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 200);
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-model: two models loaded, each request routes to the correct engine
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn multi_model_each_request_routes_to_correct_engine() {
    let dir = tempfile::tempdir().unwrap();

    // Build a shared registry with two preloaded stub models.
    let (registry, _) = make_test_registry("model-a", dir.path());
    std::fs::write(dir.path().join("model-b.gguf"), b"").unwrap();
    let entry_b = EngineEntry::for_test("model-b");
    registry.preload_for_test("model-b", entry_b);

    let app = build_router(
        registry,
        "model-a".to_string(),
        None,
        0,
        dir.path().to_path_buf(),
        None,
        None,
    );

    let resp_a = post_json(
        app.clone(),
        "/v1/chat/completions",
        serde_json::json!({
            "model": "model-a",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": false,
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(resp_a.status(), 200);
    let v_a: serde_json::Value = serde_json::from_slice(&body_bytes(resp_a).await).unwrap();
    assert_eq!(v_a["model"], "model-a");

    let resp_b = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "model-b",
            "messages": [{"role": "user", "content": "ping"}],
            "stream": false,
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(resp_b.status(), 200);
    let v_b: serde_json::Value = serde_json::from_slice(&body_bytes(resp_b).await).unwrap();
    assert_eq!(v_b["model"], "model-b");
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/chat (Ollama)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn ollama_chat_non_streaming_happy_path() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let resp = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["done"], true);
    assert_eq!(v["message"]["role"], "assistant");
    assert!(!v["message"]["content"].as_str().unwrap().is_empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// DELETE /api/delete
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn delete_nonexistent_model_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);
    let req = axum::http::Request::builder()
        .method(Method::DELETE)
        .uri("/api/delete")
        .header("content-type", "application/json")
        .body(Body::from(
            serde_json::to_vec(&serde_json::json!({"name": "does-not-exist"})).unwrap(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), 404);
}
