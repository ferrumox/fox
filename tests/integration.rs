// End-to-end HTTP integration tests using StubModel (no real GPU required).
//
// Run with: FOX_SKIP_LLAMA=1 cargo test --all --features test-helpers

use axum::{body::Body, http::Method};
use tower::ServiceExt;

use ferrumox::api::router as build_router;
use ferrumox::api::test_helpers::*;
use ferrumox::model_registry::EngineEntry;

// ─────────────────────────────────────────────────────────────────────────────
// Thinking suppression helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Collect all content delta strings from an SSE stream.
fn sse_content(bytes: &[u8]) -> String {
    parse_sse_chunks(bytes)
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect()
}

/// Collect all content strings from an NDJSON Ollama chat stream.
fn ndjson_chat_content(bytes: &[u8]) -> String {
    parse_ndjson_lines(bytes)
        .iter()
        .filter_map(|c| c["message"]["content"].as_str())
        .collect()
}

/// Collect all response strings from an NDJSON Ollama generate stream.
fn ndjson_generate_response(bytes: &[u8]) -> String {
    parse_ndjson_lines(bytes)
        .iter()
        .filter_map(|c| c["response"].as_str())
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse SSE body bytes into a list of JSON values from "data: " lines.
/// Skips comment lines and the "[DONE]" sentinel.
fn parse_sse_chunks(bytes: &[u8]) -> Vec<serde_json::Value> {
    let body = std::str::from_utf8(bytes).expect("SSE body is not UTF-8");
    body.lines()
        .filter(|l| l.starts_with("data: "))
        .filter_map(|l| {
            let payload = &l["data: ".len()..];
            if payload == "[DONE]" {
                return None;
            }
            serde_json::from_str(payload).ok()
        })
        .collect()
}

/// Parse NDJSON body bytes into a list of JSON values (one per non-empty line).
fn parse_ndjson_lines(bytes: &[u8]) -> Vec<serde_json::Value> {
    let body = std::str::from_utf8(bytes).expect("NDJSON body is not UTF-8");
    body.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).expect("NDJSON line is not valid JSON"))
        .collect()
}

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

// ─────────────────────────────────────────────────────────────────────────────
// SSE streaming – POST /v1/chat/completions  (stream: true)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_v1_chat_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 4,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/event-stream"),
        "expected text/event-stream, got: {ct}"
    );

    let bytes = body_bytes(resp).await;
    let chunks = parse_sse_chunks(&bytes);
    assert!(
        !chunks.is_empty(),
        "expected at least one SSE chunk, body: {}",
        std::str::from_utf8(&bytes).unwrap_or("<binary>")
    );

    // Every chunk must have the right object type.
    for chunk in &chunks {
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert!(chunk["choices"].as_array().is_some());
    }

    // The last chunk must have a finish_reason.
    let last = &chunks[chunks.len() - 1];
    let finish_reason = last["choices"][0]["finish_reason"].as_str();
    assert!(
        finish_reason.is_some(),
        "last SSE chunk has no finish_reason: {last}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// NDJSON streaming – POST /api/chat  (stream: true)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ollama_chat_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("ndjson"),
        "expected application/x-ndjson, got: {ct}"
    );

    let bytes = body_bytes(resp).await;
    let lines = parse_ndjson_lines(&bytes);
    assert!(
        !lines.is_empty(),
        "expected at least one NDJSON line, body: {}",
        std::str::from_utf8(&bytes).unwrap_or("<binary>")
    );

    // Every line must have model and message fields.
    for line in &lines {
        assert_eq!(line["model"], "stub");
        assert!(line["message"]["role"].as_str().is_some());
    }

    // The last line must be the done sentinel.
    let last = &lines[lines.len() - 1];
    assert_eq!(last["done"], true);
    assert!(last["done_reason"].as_str().is_some());
}

// ─────────────────────────────────────────────────────────────────────────────
// NDJSON streaming – POST /api/generate  (stream: true)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ollama_generate_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "stub",
            "prompt": "Explain Rust ownership.",
            "stream": true,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("ndjson"), "expected ndjson, got: {ct}");

    let bytes = body_bytes(resp).await;
    let lines = parse_ndjson_lines(&bytes);
    assert!(!lines.is_empty());

    for line in &lines {
        assert_eq!(line["model"], "stub");
        assert!(line["response"].as_str().is_some());
    }

    let last = &lines[lines.len() - 1];
    assert_eq!(last["done"], true);
    assert!(last["done_reason"].as_str().is_some());
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/generate  (non-streaming)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ollama_generate_non_streaming() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "stub",
            "prompt": "Hello",
            "stream": false,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["done"], true);
    assert_eq!(v["model"], "stub");
    assert!(v["response"].as_str().is_some());
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool calling – POST /v1/chat/completions  (tools present)
// With StubModel the model output is plain text, so finish_reason stays "stop".
// The test verifies that:
//   • the endpoint accepts a request with tools without error (200)
//   • stream is forced to non-streaming (response is a single JSON object)
//   • response shape is valid
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_tool_calling_endpoint_accepts_tools() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "What is the weather in Madrid?"}],
            "stream": true,   // must be forced off by the handler when tools are present
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather for a city",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"}
                            }
                        }
                    }
                }
            ],
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);

    // With tools + stream: true, the handler buffers tokens, parses the tool
    // call, and emits the result as SSE chunks followed by data: [DONE].
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/event-stream"),
        "expected SSE stream when tools present with stream:true, got: {ct}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stop sequences – POST /v1/chat/completions  (stop field present)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_stop_sequences_accepted() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "count to 10"}],
            "stream": false,
            "stop": ["END", "STOP"],
            "max_tokens": 16,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["choices"][0]["finish_reason"].as_str().is_some());
}

// ─────────────────────────────────────────────────────────────────────────────
// Ollama embed – POST /api/embed
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ollama_embed() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/embed",
        serde_json::json!({
            "model": "stub",
            "input": "Hello world",
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert_eq!(v["model"], "stub");
    let embeddings = v["embeddings"]
        .as_array()
        .expect("embeddings must be array");
    assert!(!embeddings.is_empty());
    // Each embedding must be a non-empty float array.
    let first = embeddings[0]
        .as_array()
        .expect("embedding[0] must be array");
    assert!(!first.is_empty());
}

#[tokio::test]
async fn test_ollama_embed_multiple_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/embed",
        serde_json::json!({
            "model": "stub",
            "input": ["first sentence", "second sentence"],
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let embeddings = v["embeddings"]
        .as_array()
        .expect("embeddings must be array");
    assert_eq!(embeddings.len(), 2, "expected one embedding per input");
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/version
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_api_version() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = get_req(app, "/api/version").await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let version = v["version"].as_str().expect("version must be a string");
    assert!(!version.is_empty());
    assert!(
        version.contains('.'),
        "version should be semver-like, got: {version}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/tags
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_api_tags_lists_models_on_disk() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = get_req(app, "/api/tags").await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let models = v["models"].as_array().expect("models must be array");
    assert!(!models.is_empty(), "expected at least one model");

    // The stub model file was written to disk by make_test_state.
    let names: Vec<&str> = models.iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert!(
        names.contains(&"stub"),
        "expected 'stub' in /api/tags, got: {names:?}"
    );

    // Each model entry must have mandatory fields.
    for m in models {
        assert!(m["name"].as_str().is_some());
        assert!(m["digest"].as_str().is_some());
        assert!(m["details"]["format"].as_str().is_some());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /api/ps
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_api_ps_lists_loaded_models() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = get_req(app, "/api/ps").await;
    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let models = v["models"].as_array().expect("models must be array");
    assert_eq!(models.len(), 1, "expected exactly one loaded model");
    assert_eq!(models[0]["name"], "stub");
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /api/show
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_api_show_known_model() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(app, "/api/show", serde_json::json!({"name": "stub"})).await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    assert!(v["modelfile"].as_str().is_some());
    assert!(v["details"]["format"].as_str().is_some());
    assert_eq!(v["details"]["format"], "gguf");
}

#[tokio::test]
async fn test_api_show_unknown_model_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/show",
        serde_json::json!({"name": "nonexistent-xyz"}),
    )
    .await;

    assert_eq!(resp.status(), 404);
}

// ─────────────────────────────────────────────────────────────────────────────
// GET /metrics
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_metrics_endpoint() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = get_req(app, "/metrics").await;
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ct.contains("text/plain"),
        "expected text/plain content-type, got: {ct}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// POST /v1/completions
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_v1_completions() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "stub",
            "prompt": "Once upon a time",
            "max_tokens": 4,
            "stream": false,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    // /v1/completions delegates to chat_completions → same response shape
    assert_eq!(v["object"], "chat.completion");
    assert!(v["choices"][0]["finish_reason"].as_str().is_some());
}

#[tokio::test]
async fn test_v1_completions_unknown_model_returns_404() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state("stub", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "no-such-model",
            "prompt": "test",
        }),
    )
    .await;

    assert_eq!(resp.status(), 404);
}

// ─────────────────────────────────────────────────────────────────────────────
// Thinking suppression — ThinkingStubModel produces:
//   prefill → "thought"  (inside thinking block)
//   decode  → "</think>" → "answer" → EOS
//
// Expected behaviour per endpoint:
//   OpenAI /v1/chat/completions  (any stream mode) : content = "answer"   (no <think> tags)
//   Ollama /api/chat  stream=false                  : thinking = "thought", content = "answer"
//   Ollama /api/chat  stream=true                   : content = "answer"   (no <think> tags)
//   Ollama /api/generate (any stream mode)          : response = "answer"  (no <think> tags)
// ─────────────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn thinking_openai_non_streaming_no_tags_in_content() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "think",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "max_tokens": 16,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let content = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(
        content, "answer",
        "OpenAI non-stream: thinking must be suppressed"
    );
    assert!(
        !content.contains("<think>"),
        "OpenAI content must not contain <think>"
    );
    assert!(
        !content.contains("</think>"),
        "OpenAI content must not contain </think>"
    );
    assert!(
        !content.contains("thought"),
        "OpenAI content must not contain thinking text"
    );
}

#[tokio::test]
async fn thinking_openai_streaming_no_tags_in_content() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/v1/chat/completions",
        serde_json::json!({
            "model": "think",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
            "max_tokens": 16,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let bytes = body_bytes(resp).await;
    let content = sse_content(&bytes);
    assert_eq!(
        content, "answer",
        "OpenAI stream: thinking must be suppressed"
    );
    assert!(
        !content.contains("<think>"),
        "stream content must not contain <think>"
    );
    assert!(
        !content.contains("thought"),
        "stream content must not contain thinking text"
    );
}

#[tokio::test]
async fn thinking_ollama_chat_non_streaming_separates_thinking_field() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "think",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

    let content = v["message"]["content"].as_str().unwrap();
    assert_eq!(
        content, "answer",
        "Ollama non-stream: visible content must be 'answer'"
    );
    assert!(
        !content.contains("<think>"),
        "content must not contain <think>"
    );
    assert!(
        !content.contains("thought"),
        "content must not contain thinking text"
    );

    let thinking = v["message"]["thinking"].as_str().unwrap_or("");
    assert_eq!(
        thinking, "thought",
        "Ollama non-stream: thinking field must contain reasoning"
    );
}

#[tokio::test]
async fn thinking_ollama_chat_streaming_no_tags_in_content() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/chat",
        serde_json::json!({
            "model": "think",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let bytes = body_bytes(resp).await;
    let content = ndjson_chat_content(&bytes);
    assert_eq!(
        content, "answer",
        "Ollama chat stream: thinking must be suppressed"
    );
    assert!(
        !content.contains("<think>"),
        "stream content must not contain <think>"
    );
    assert!(
        !content.contains("thought"),
        "stream content must not contain thinking text"
    );
}

#[tokio::test]
async fn thinking_ollama_generate_non_streaming_no_tags_in_response() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "think",
            "prompt": "Hi",
            "stream": false,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();
    let response = v["response"].as_str().unwrap();
    assert_eq!(
        response, "answer",
        "Ollama generate non-stream: thinking must be suppressed"
    );
    assert!(
        !response.contains("<think>"),
        "response must not contain <think>"
    );
    assert!(
        !response.contains("thought"),
        "response must not contain thinking text"
    );
}

#[tokio::test]
async fn thinking_ollama_generate_streaming_no_tags_in_response() {
    let dir = tempfile::tempdir().unwrap();
    let (state, _) = make_test_state_thinking("think", dir.path());
    let app = make_router(&state);

    let resp = post_json(
        app,
        "/api/generate",
        serde_json::json!({
            "model": "think",
            "prompt": "Hi",
            "stream": true,
        }),
    )
    .await;

    assert_eq!(resp.status(), 200);
    let bytes = body_bytes(resp).await;
    let response = ndjson_generate_response(&bytes);
    assert_eq!(
        response, "answer",
        "Ollama generate stream: thinking must be suppressed"
    );
    assert!(
        !response.contains("<think>"),
        "stream response must not contain <think>"
    );
    assert!(
        !response.contains("thought"),
        "stream response must not contain thinking text"
    );
}
