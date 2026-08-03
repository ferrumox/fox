// POST /v1/rerank, /rerank — score documents against a query with a reranker model
// (llama-server's /rerank, server-context.cpp:4962-5040).
//
// Reranking is not embedding similarity: a reranker model reads the query and the
// document *together* through cross-attention and emits one relevance score from a
// classification head. That head is what makes it work, and it is also what most
// models do not have.
//
// Reading that head requires the context to be created with RANK pooling, which is what
// `--reranking` does. It cannot be auto-detected: a reranker GGUF does not reliably
// carry a `<arch>.pooling_type` key (jina-reranker-v1-tiny-en has none), so llama.cpp's
// UNSPECIFIED fallback resolves to NONE. llama-server takes a flag for the same reason
// (arg.cpp:3067-3070).
//
// Without RANK pooling `llama_get_embeddings_seq` returns NULL, and that NULL is the
// signal used to reject the request rather than answer it with a number derived from a
// mean-pooled vector — which would look like a ranking and rank nothing.

use axum::extract::State;
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};

use crate::api::error::{load_model_or_respond, AppError};
use crate::api::router::AppState;
use crate::api::shared::extractor::LenientJson;

#[derive(Debug, Deserialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    /// Jina/Cohere spelling.
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    /// TEI spelling — accepted as an alias so either client works unmodified.
    #[serde(default)]
    pub texts: Option<Vec<String>>,
    /// Return only the best `top_n` after scoring. Absent = all.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// Echo each document back in its result entry.
    #[serde(default)]
    pub return_text: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct RerankResult {
    /// Index into the request's `documents`, preserved across sorting so the caller
    /// can map a result back to its input.
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct RerankResponse {
    pub model: String,
    pub object: String,
    pub results: Vec<RerankResult>,
    pub usage: RerankUsage,
}

#[derive(Debug, Serialize)]
pub struct RerankUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

pub async fn rerank(
    State(state): State<AppState>,
    LenientJson(req): LenientJson<RerankRequest>,
) -> axum::response::Response {
    let documents = match (req.documents, req.texts) {
        (Some(d), _) | (None, Some(d)) => d,
        (None, None) => {
            return AppError::BadRequest("one of `documents` or `texts` is required".to_string())
                .into_response()
        }
    };
    if documents.is_empty() {
        return AppError::BadRequest("`documents` must not be empty".to_string()).into_response();
    }
    if req.query.trim().is_empty() {
        return AppError::BadRequest("`query` must not be empty".to_string()).into_response();
    }

    let (entry, _) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    // Prompt layout mirrors llama.cpp's `format_prompt_rerank`
    // (server-common.h:370): the pair is one sequence, joined by SEP, so
    // cross-attention can see both halves at once.
    let sep = entry.engine.sep_token_id();
    let query_tokens = match entry.engine.tokenize(&req.query) {
        Ok(t) => t,
        Err(e) => return AppError::BadRequest(format!("tokenize query: {e}")).into_response(),
    };

    let model_name = req.model.clone();
    let return_text = req.return_text.unwrap_or(false);
    let mut results = Vec::with_capacity(documents.len());
    let mut prompt_tokens_total = 0u32;

    for (index, doc) in documents.iter().enumerate() {
        let doc_tokens = match entry.engine.tokenize(doc) {
            Ok(t) => t,
            Err(e) => {
                return AppError::BadRequest(format!("tokenize document {index}: {e}"))
                    .into_response()
            }
        };
        let mut pair = Vec::with_capacity(query_tokens.len() + doc_tokens.len() + 1);
        pair.extend_from_slice(&query_tokens);
        if let Some(sep) = sep {
            pair.push(sep);
        }
        pair.extend_from_slice(&doc_tokens);
        prompt_tokens_total += pair.len() as u32;

        // Scoring is a blocking forward pass, like `embed`.
        let engine = entry.engine.clone();
        let score = match tokio::task::spawn_blocking(move || engine.rerank_score(&pair)).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                return AppError::BadRequest(format!(
                    "model '{model_name}' cannot rerank: {e}. Two things are needed: a \
                     reranker model (bge-reranker, jina-reranker, mxbai-rerank, …), and \
                     the server started with `--reranking true` so its context is \
                     created with RANK pooling. fox cannot infer the second — reranker \
                     GGUFs do not reliably declare their pooling type."
                ))
                .into_response()
            }
            Err(e) => {
                return AppError::InternalError(format!("rerank task failed: {e}")).into_response()
            }
        };

        results.push(RerankResult {
            index,
            relevance_score: score,
            document: return_text.then(|| doc.clone()),
        });
    }

    // Highest relevance first. `total_cmp` rather than `partial_cmp().unwrap()`: a
    // NaN score from a misbehaving head would panic the request otherwise.
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(n) = req.top_n {
        results.truncate(n);
    }

    axum::Json(RerankResponse {
        model: req.model,
        object: "list".to_string(),
        results,
        usage: RerankUsage {
            prompt_tokens: prompt_tokens_total,
            total_tokens: prompt_tokens_total,
        },
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn rerank_rejects_a_model_without_a_relevance_head() {
        // The stub is not a reranker, like every chat and embedding model. Scoring it
        // anyway would return numbers that look like a ranking and rank nothing.
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/v1/rerank",
            serde_json::json!({
                "model": "stub",
                "query": "what is rust",
                "documents": ["a systems language", "a fungus"],
            }),
        )
        .await;
        assert_eq!(resp.status(), 400);
        let body = String::from_utf8(body_bytes(resp).await.to_vec()).unwrap();
        assert!(
            body.contains("reranker"),
            "the error must name what is missing: {body}"
        );
    }

    #[tokio::test]
    async fn rerank_validates_its_input_before_loading_a_model() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        for body in [
            serde_json::json!({"model": "stub", "query": "q"}),
            serde_json::json!({"model": "stub", "query": "q", "documents": []}),
            serde_json::json!({"model": "stub", "query": "  ", "documents": ["d"]}),
        ] {
            let resp = post_json(make_router(&state), "/v1/rerank", body.clone()).await;
            assert_eq!(resp.status(), 400, "should reject {body}");
        }
    }

    #[tokio::test]
    async fn rerank_accepts_the_tei_texts_spelling() {
        // `texts` is TEI's name for the same field; a client using it must not get a
        // "documents is required" error.
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = post_json(
            app,
            "/rerank",
            serde_json::json!({"model": "stub", "query": "q", "texts": ["d"]}),
        )
        .await;
        // Reaches the model (and fails there, since the stub has no head) rather than
        // being rejected as a malformed request.
        let body = String::from_utf8(body_bytes(resp).await.to_vec()).unwrap();
        assert!(body.contains("reranker"), "{body}");
    }
}
