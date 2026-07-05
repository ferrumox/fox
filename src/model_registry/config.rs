use std::path::PathBuf;
use std::sync::Arc;

use crate::metrics::Metrics;

/// GGML type IDs for KV cache element types.
pub mod kv_type {
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q8_0: u32 = 8;
}

pub struct RegistryConfig {
    pub models_dir: PathBuf,
    pub max_models: usize,
    pub max_batch_size: usize,
    /// Max prompt tokens submitted per request per prefill step (0 = single-shot).
    /// Chunking a long prompt lets it interleave with other requests' decode steps.
    pub max_prefill_chunk: usize,
    /// Context rolling: when a sequence fills `n_ctx`, discard its oldest KV window and
    /// shift the rest down so decode continues instead of stopping with `Length`.
    /// Only applied to shiftable (non-recurrent) caches.
    pub context_shift: bool,
    /// Tokens preserved at the front (BOS + system prompt) when context rolling fires.
    pub context_keep: usize,
    /// Enable n-gram / prompt-lookup speculative decoding for single-request decode steps.
    pub speculative: bool,
    /// Suffix length matched against history when speculating.
    pub spec_ngram: usize,
    /// Maximum draft tokens proposed per speculative step.
    pub spec_draft_len: usize,
    /// Per-sequence context length. `None` = auto-detect from the model's trained context.
    pub max_context_len: Option<u32>,
    pub block_size: usize,
    pub gpu_memory_bytes: usize,
    pub gpu_memory_fraction: f32,
    pub metrics: Option<Arc<Metrics>>,
    /// Seconds since last use before a model is evicted. 0 = never evict by time.
    pub keep_alive_secs: u64,
    /// Key cache element type. See `kv_type` constants.
    pub type_k: u32,
    /// Value cache element type. See `kv_type` constants.
    pub type_v: u32,
    /// Primary GPU index (0-based). Used when split_mode=NONE, or as the main GPU for splits.
    pub main_gpu: i32,
    /// How to distribute the model across GPUs: 0=none, 1=layer (default), 2=row.
    pub split_mode: u32,
    /// Normalized per-GPU VRAM proportions for tensor splitting (e.g. [0.75, 0.25]).
    /// Empty = llama.cpp decides proportionally to available VRAM.
    pub tensor_split: Vec<f32>,
    /// When true, MoE expert tensors are pinned to CPU RAM (via `tensor_buft_overrides`).
    /// Useful for MoE models (e.g. DeepSeek, Mixtral) where expert weights don't fit in VRAM.
    pub moe_offload_cpu: bool,
}
