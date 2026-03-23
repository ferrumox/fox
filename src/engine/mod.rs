// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
mod logits;
pub mod model;
mod output_filter;
mod run;

use output_filter::PerRequestState;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::kv_cache::KVCacheManager;
use crate::metrics::Metrics;
use crate::scheduler::InferenceRequest;

use self::model::Model;

/// SentencePiece uses U+2581 (▁) for word boundaries. Replace with space so words don't concatenate.
const SPM_SPACE: char = '\u{2581}';

/// Main inference engine coordinating scheduler, model, and KV cache.
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    scheduler: Arc<crate::scheduler::Scheduler>,
    kv_cache: Arc<KVCacheManager>,
    next_request_id: AtomicU64,
    /// Per-request mutable state for output filtering and stop sequence detection.
    per_request_state: Arc<Mutex<HashMap<u64, PerRequestState>>>,
    /// Human-readable model identifier (basename of the loaded model file).
    model_name: String,
    /// Prometheus metrics (optional — None disables recording).
    metrics: Option<Arc<Metrics>>,
    /// Whether the loaded model supports KV-cache sequence copying (llama_memory_seq_cp).
    /// False for recurrent/hybrid models (Mamba, Qwen3.5, etc.); prefix caching is disabled.
    supports_prefix_cache: bool,
    /// Text forms of the model's EOS and EOT tokens, used as base stop sequences.
    model_stop_tokens: Vec<String>,
}

impl InferenceEngine {
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
        model_name: String,
        metrics: Option<Arc<Metrics>>,
    ) -> Self {
        let supports_prefix_cache = model.supports_seq_copy();
        if supports_prefix_cache {
            tracing::info!("prefix caching enabled (model supports seq_cp)");
        } else {
            tracing::info!(
                "prefix caching disabled (model uses recurrent/hybrid memory — seq_cp unsupported)"
            );
        }
        let model_stop_tokens = model.stop_tokens();
        if !model_stop_tokens.is_empty() {
            tracing::info!("model stop tokens: {:?}", model_stop_tokens);
        }
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            per_request_state: Arc::new(Mutex::new(HashMap::new())),
            model_name,
            metrics,
            supports_prefix_cache,
            model_stop_tokens,
        }
    }

    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        self.model.tokenize(text)
    }

    pub fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }

    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let tokens = self.model.tokenize(text)?;
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || model.get_embeddings(&tokens))
            .await
            .map_err(|e| anyhow::anyhow!("embed spawn_blocking: {}", e))?
    }

    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> anyhow::Result<String> {
        self.model.apply_chat_template(messages)
    }

    pub fn submit_request(&self, req: InferenceRequest) {
        self.scheduler.submit(req);
    }

    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Effective per-sequence context length (tokens) this engine was configured with.
    pub fn context_len(&self) -> u32 {
        self.model.context_len()
    }

    /// Whether the loaded model has native thinking support (`<think>` special token).
    pub fn supports_thinking(&self) -> bool {
        self.model.supports_thinking()
    }

    /// Sampling parameters recommended by the model's GGUF metadata, if any.
    pub fn recommended_sampling(&self) -> Option<model::RecommendedSampling> {
        self.model.recommended_sampling()
    }

    pub fn kv_cache_usage(&self) -> f32 {
        self.kv_cache.memory_usage()
    }

    pub fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }

    pub fn active_requests(&self) -> usize {
        self.scheduler.active_requests()
    }

    pub fn prefix_cache_hits(&self) -> u64 {
        self.scheduler.prefix_hits.load(Ordering::Relaxed)
    }

    pub fn prefix_cache_misses(&self) -> u64 {
        self.scheduler.prefix_misses.load(Ordering::Relaxed)
    }
}
