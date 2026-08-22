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
    /// Maximum requests allowed to wait in the scheduler queue before new ones are
    /// rejected (0 = unbounded).
    pub max_queue_depth: usize,
    /// Max prompt tokens submitted per request per prefill step (0 = single-shot).
    /// Chunking a long prompt lets it interleave with other requests' decode steps.
    pub max_prefill_chunk: usize,
    /// Recurrent-state snapshots per sequence for cache rollback. See
    /// `ServeArgs::rs_rollback` for the memory cost; 0 disables prompt reuse on
    /// hybrid/recurrent models entirely.
    pub rs_rollback: u32,
    /// Context rolling: when a sequence fills `n_ctx`, discard its oldest KV window and
    /// shift the rest down so decode continues instead of stopping with `Length`.
    /// Only applied to shiftable (non-recurrent) caches.
    pub context_shift: bool,
    /// Tokens preserved at the front (BOS + system prompt) when context rolling fires.
    pub context_keep: usize,
    /// Keep a finished request's KV resident so a later prompt sharing a prefix with
    /// it can skip re-prefilling that much (`--kv-reuse`). False restores the
    /// pre-0.19 behaviour: every sequence cleared on completion, every prompt
    /// prefilled from token 0.
    /// Create the context with RANK pooling so `/rerank` can read the model's
    /// relevance head (`--reranking`). A reranker GGUF does not reliably declare its
    /// pooling type, so this cannot be auto-detected; llama-server takes a flag for
    /// the same reason. A model loaded this way is a reranker, not a generator.
    pub reranking: bool,
    /// Host-RAM budget in **bytes** for serialised sequence states (`--cache-ram`,
    /// given in MiB on the CLI). 0 disables the cache.
    pub cache_ram_bytes: usize,
    pub kv_reuse: bool,
    /// Minimum fraction of an incoming prompt that must already be resident in an idle
    /// slot before that slot's KV is inherited (`--slot-prompt-similarity`).
    pub slot_prompt_similarity: f32,
    /// Enable n-gram / prompt-lookup speculative decoding for single-request decode steps.
    pub speculative: bool,
    /// Suffix length matched against history when speculating.
    pub spec_ngram: usize,
    /// Maximum draft tokens proposed per speculative step.
    pub spec_draft_len: usize,
    /// Name/path of a second, smaller model to use as the speculative-decoding draft
    /// proposer instead of n-gram lookup. Only takes effect when `speculative` is
    /// also true; ignored (with a startup warning) otherwise. The draft and target
    /// must share the same tokenizer — checked at load time, fails loudly on
    /// mismatch. Loaded once alongside the target, for the process lifetime — not
    /// subject to LRU eviction/VRAM budgeting in 0.16 (see
    /// `docs/design/speculative-roadmap.md`).
    pub draft_model: Option<String>,
    /// Name/path of the mmproj (vision projector) GGUF paired with the main model,
    /// enabling image input via llama.cpp's `mtmd` library. Like `draft_model`, this
    /// is a single global setting — one mmproj active at a time, matched against
    /// whatever model is currently loaded. A mismatched pairing (mmproj for a
    /// different architecture) fails at load time rather than corrupting output.
    pub mmproj: Option<String>,
    /// Name/path of the multi-token-prediction head GGUF paired with the main model
    /// (`mtp-*.gguf`), enabling MTP speculative decoding. Like `draft_model` and
    /// `mmproj`, one global pairing against whichever model is loaded. Only takes
    /// effect when `speculative` is also true. Unlike `draft_model`, the head is not a
    /// standalone model: it is a trained NextN block that reads the target's hidden
    /// states, so a head belonging to another model is rejected at load time by width.
    pub mtp_model: Option<String>,
    /// Named LoRA adapters `(name, path, scale)` loaded alongside the primary
    /// model. A client selects one by naming it in the `model` field instead
    /// of the base model name — resolved the same way as `draft_model`/`mmproj`
    /// (one global pairing: all adapters here apply to whichever model is the
    /// primary one, not to arbitrary other loaded models).
    pub lora_modules: Vec<(String, PathBuf, f32)>,
    /// Stem name of the primary model — the one `lora_modules` adapters attach to,
    /// and what a LoRA-alias request (`model: "<adapter-name>"`) resolves to load
    /// instead of the adapter name itself. `None` when there's no model configured
    /// at startup (fully lazy mode, no `--model-path`, empty `models_dir`).
    pub primary_model: Option<String>,
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
    /// Transformer layers to offload to the GPU. `-1` (the default) means all of them;
    /// `0` keeps the model entirely on the CPU. Any value in between splits it, which is
    /// the only way to run a model whose weights do not fit in VRAM.
    pub n_gpu_layers: i32,
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
