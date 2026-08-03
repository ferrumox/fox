// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
mod logits;
pub mod model;
mod output_filter;
mod run;
mod speculative;

use output_filter::PerRequestState;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;

use crate::kv_cache::KVCacheManager;
use crate::metrics::Metrics;
use crate::scheduler::InferenceRequest;

use self::model::Model;
use self::speculative::{DraftModelProposer, NgramProposer, Proposer};

/// SentencePiece uses U+2581 (▁) for word boundaries. Replace with space so words don't concatenate.
const SPM_SPACE: char = '\u{2581}';

/// Which proposer to use for speculative decoding.
#[derive(Debug, Clone, Copy)]
pub enum SpeculativeConfig {
    /// N-gram / prompt-lookup (0.15): match the request's own recent history.
    Ngram { ngram: usize, draft_len: usize },
    /// Draft-model (0.16): a second, smaller resident model predicts ahead. Requires
    /// `InferenceEngine::new`'s `draft_model` parameter to be `Some`.
    Draft { draft_len: usize },
}

/// Tunable engine behaviors. `Default` disables everything (single-shot prefill,
/// no context rolling, no speculation) — what tests and benches want.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineOptions {
    /// Max prompt tokens submitted per request per prefill step (0 = single-shot).
    pub max_prefill_chunk: usize,
    /// Context rolling on full: `Some(n_keep)` keeps generating past `n_ctx`,
    /// preserving the first `n_keep` tokens. Shiftable (non-recurrent) caches only.
    pub context_shift: Option<usize>,
    /// Speculative decoding for single-request decode steps, if enabled.
    pub speculative: Option<SpeculativeConfig>,
}

/// Main inference engine coordinating scheduler, model, and KV cache.
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    scheduler: Arc<crate::scheduler::Scheduler>,
    kv_cache: Arc<KVCacheManager>,
    next_request_id: AtomicU64,
    /// Per-request mutable state for output filtering and stop sequence detection.
    per_request_state: Arc<DashMap<u64, PerRequestState>>,
    /// Human-readable model identifier (basename of the loaded model file).
    model_name: String,
    /// Prometheus metrics (optional — None disables recording).
    metrics: Option<Arc<Metrics>>,
    /// Whether the loaded model supports KV-cache sequence copying (llama_memory_seq_cp).
    /// False for recurrent/hybrid models (Mamba, Qwen3.5, etc.); prefix caching is disabled.
    supports_prefix_cache: bool,
    /// Whether a finished sequence may be parked for its own slot to reuse. Weaker
    /// than `supports_prefix_cache`, which asks whether KV may be copied *between*
    /// sequences — see `Model::supports_slot_reuse`.
    supports_slot_reuse: bool,
    /// Serialise a sequence to the host-RAM prompt cache as soon as its prefill ends.
    ///
    /// Only for models whose KV cannot be rolled back (hybrid, recurrent). For them the
    /// resident-slot path is a dead end — reaching a past position means trimming across
    /// the whole generated reply, which recurrent state refuses beyond `--rs-rollback`
    /// snapshots. A serialised state has no such limit, and one taken at the end of
    /// prefill lands exactly on the boundary the next turn diverges from.
    checkpoint_after_prefill: bool,
    /// Text forms of the model's EOS and EOT tokens, used as base stop sequences.
    model_stop_tokens: Vec<String>,
    // See `EngineOptions` for the semantics of these two.
    max_prefill_chunk: usize,
    context_shift: Option<usize>,
    /// Resolved proposer (n-gram or draft-model) + draft_len, built from
    /// `EngineOptions::speculative` at construction time.
    speculative: Option<(Arc<dyn Proposer>, usize)>,
    /// Lifetime draft tokens proposed by speculation (for the acceptance metrics).
    spec_proposed: AtomicU64,
    /// Lifetime draft tokens the target model accepted.
    spec_accepted: AtomicU64,
}

impl InferenceEngine {
    /// `draft_model` is the second, smaller model to use when
    /// `options.speculative == Some(SpeculativeConfig::Draft { .. })`; ignored (and
    /// may be `None`) otherwise.
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
        model_name: String,
        metrics: Option<Arc<Metrics>>,
        options: EngineOptions,
        draft_model: Option<Arc<dyn Model>>,
    ) -> Self {
        let supports_prefix_cache = model.supports_seq_copy();
        // The scheduler decides to skip prefill; llama.cpp performs the copy. Telling
        // only the engine leaves the scheduler skipping prefill for cells nobody copied.
        scheduler.set_prefix_reuse(supports_prefix_cache);
        let supports_slot_reuse = model.supports_slot_reuse();
        scheduler.set_slot_reuse(supports_slot_reuse);
        if supports_prefix_cache {
            tracing::info!("prefix caching enabled (model supports seq_cp)");
        } else {
            // The reason is deliberately not spelled out: it is now either a
            // recurrent/hybrid model or a non-unified KV cache, and naming only the
            // first sent a previous investigation looking at the wrong thing.
            tracing::info!(
                slot_reuse = supports_slot_reuse,
                "cross-sequence prefix copying disabled (model cannot donate KV cells); \
                 slot reuse is unaffected"
            );
        }
        let model_stop_tokens = model.stop_tokens();
        if !model_stop_tokens.is_empty() {
            // Count at info, list at debug. Llama 3 declares 247 reserved special
            // tokens, so printing the list made starting the server emit a wall of
            // `<|reserved_special_token_N|>` that buried every other startup line.
            // The number is what an operator needs; the names are a debugging detail.
            tracing::info!(count = model_stop_tokens.len(), "model stop tokens loaded");
            tracing::debug!("model stop tokens: {:?}", model_stop_tokens);
        }
        let speculative = options.speculative.map(|cfg| match cfg {
            SpeculativeConfig::Ngram { ngram, draft_len } => {
                (Arc::new(NgramProposer { ngram }) as Arc<dyn Proposer>, draft_len)
            }
            SpeculativeConfig::Draft { draft_len } => {
                let dm = draft_model
                    .clone()
                    .expect("SpeculativeConfig::Draft requires InferenceEngine::new's draft_model to be Some");
                (
                    Arc::new(DraftModelProposer::new(dm)) as Arc<dyn Proposer>,
                    draft_len,
                )
            }
        });
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            per_request_state: Arc::new(DashMap::new()),
            model_name,
            metrics,
            supports_prefix_cache,
            supports_slot_reuse,
            // Dense models reach the same prefix by trimming, which is free; paying for
            // a serialised copy of every prompt would be a straight loss for them.
            checkpoint_after_prefill: !supports_prefix_cache,
            model_stop_tokens,
            max_prefill_chunk: options.max_prefill_chunk,
            context_shift: options.context_shift,
            speculative,
            spec_proposed: AtomicU64::new(0),
            spec_accepted: AtomicU64::new(0),
        }
    }

    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        self.model.tokenize(text)
    }

    /// Score one (query, document) pair for reranking. `Err` when the model has no
    /// relevance head — see `Model::rerank_score` for how that is detected.
    pub fn rerank_score(&self, tokens: &[i32]) -> anyhow::Result<f32> {
        self.model.rerank_score(tokens)
    }

    /// The vocabulary's separator token, used to join query and document.
    pub fn sep_token_id(&self) -> Option<i32> {
        self.model.sep_token_id()
    }

    /// The model's fill-in-the-middle tokens, or `None` if it has none — which is
    /// how `/infill` tells a code model from a chat model.
    pub fn fim_tokens(&self) -> Option<crate::engine::model::FimTokens> {
        self.model.fim_tokens()
    }

    /// Turn token ids back into text.
    ///
    /// Concatenates the raw piece bytes before decoding, rather than decoding each
    /// token separately: a multi-byte codepoint (emoji, CJK) is routinely split
    /// across two BPE tokens, and per-token decoding would turn each half into a
    /// replacement character. Same reason the streaming path buffers bytes.
    /// `U+2581` is mapped back to a space, matching the generation path.
    pub fn detokenize(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            bytes.extend_from_slice(&self.model.token_to_piece_bytes(t));
        }
        String::from_utf8_lossy(&bytes).replace(SPM_SPACE, " ")
    }

    /// The single piece text for one token, without the cross-token byte joining
    /// above — for `/tokenize`'s `with_pieces`, where per-token output is the point.
    /// Invalid UTF-8 (a token holding half a codepoint) is reported as raw bytes by
    /// the caller rather than lossily decoded here.
    pub fn token_piece_bytes(&self, token: i32) -> Vec<u8> {
        self.model.token_to_piece_bytes(token)
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

    pub fn build_prompt_tokens(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
    ) -> anyhow::Result<Vec<i32>> {
        self.model
            .build_prompt_tokens(messages, enable_thinking, tools)
    }

    /// Returns `true` when this model was loaded with a paired mmproj (vision
    /// projector), i.e. `tokenize_multimodal` can actually encode images.
    pub fn supports_vision(&self) -> bool {
        self.model.supports_vision()
    }

    pub fn tokenize_multimodal(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
        images: &[Vec<u8>],
    ) -> anyhow::Result<crate::engine::model::MultimodalChunks> {
        self.model
            .tokenize_multimodal(messages, enable_thinking, tools, images)
    }

    /// Which tool-call wire format the loaded model's own chat template natively
    /// speaks, if any (see [`crate::engine::model::NativeToolFormat`]), as opposed to
    /// needing fox's generic prompt-injected tool listing.
    pub fn native_tool_call_format(&self) -> Option<crate::engine::model::NativeToolFormat> {
        self.model.native_tool_call_format()
    }

    /// The model's reasoning (open, close) delimiters, resolved to the default
    /// `<think>`/`</think>` when the model uses the standard format.
    pub fn reasoning_delimiters(&self) -> (String, String) {
        self.model
            .reasoning_delimiters()
            .unwrap_or_else(|| ("<think>".to_string(), "</think>".to_string()))
    }

    pub fn submit_request(
        &self,
        req: InferenceRequest,
    ) -> Result<(), crate::scheduler::SubmitError> {
        self.scheduler.submit(req)
    }

    /// Record a rejected request in the `requests_rejected_total{reason}` metric.
    pub fn record_rejection(&self, reason: &str) {
        if let Some(m) = &self.metrics {
            m.requests_rejected_total.with_label_values(&[reason]).inc();
        }
    }

    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Serialise a sequence's KV to host memory (see `Model::state_seq_save`).
    pub fn state_seq_save(&self, seq_id: i32) -> anyhow::Result<Vec<u8>> {
        self.model.state_seq_save(seq_id)
    }

    /// Restore a previously saved sequence blob into `seq_id`.
    pub fn state_seq_load(&self, seq_id: i32, data: &[u8]) -> anyhow::Result<usize> {
        self.model.state_seq_load(seq_id, data)
    }

    /// Drop every cached sequence state. Called when the model unloads: the blobs
    /// encode this model's cell layout and are meaningless — and dangerous — to any
    /// other.
    pub fn clear_prompt_cache(&self) {
        self.scheduler.clear_prompt_cache();
    }

    /// Per-slot state for `GET /slots`.
    pub fn slots_snapshot(&self) -> Vec<crate::scheduler::SlotSnapshot> {
        self.scheduler.slots_snapshot()
    }

    /// Blocks actually held in the pool, and its capacity.
    ///
    /// Not the sum of the per-slot counts in `slots_snapshot`: a block shared by a
    /// prefix copy is counted once by *every* slot referencing it, so that sum cannot
    /// fall when sharing works and reading it as memory use overstates it by exactly
    /// the amount that was saved. This is the pool's own occupancy.
    pub fn kv_blocks(&self) -> (usize, usize) {
        (
            self.kv_cache.allocated_blocks(),
            self.kv_cache.total_blocks(),
        )
    }

    /// The loaded model's inspectable facts — architecture, dimensions, capabilities.
    /// Read from the model itself, never reconstructed from its filename, which is
    /// what `/props` and `/api/show` need to stop guessing.
    pub fn model_info(&self) -> crate::engine::model::ModelInfo {
        self.model.model_info()
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

    /// Lifetime speculative-decoding stats: `(drafts proposed, drafts accepted)`.
    /// Both zero when speculation is disabled or never triggered.
    pub fn spec_stats(&self) -> (u64, u64) {
        (
            self.spec_proposed.load(Ordering::Relaxed),
            self.spec_accepted.load(Ordering::Relaxed),
        )
    }
}
