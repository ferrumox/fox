use std::path::PathBuf;
use std::sync::Arc;

use crate::metrics::Metrics;
use crate::model_discovery::DiscoveredModel;

/// GGML type IDs for KV cache element types.
pub mod kv_type {
    pub const F16: u32 = 1;
    pub const Q4_0: u32 = 2;
    pub const Q8_0: u32 = 8;
    /// TurboQuant 3-bit KV (3.1 bpw, ~4.9x compression). Recommended sweet spot.
    /// Requires Flash Attention and head_dim divisible by 128.
    pub const TURBO3: u32 = 41;
    /// TurboQuant 4-bit KV (4.25 bpw, ~3.8x compression).
    pub const TURBO4: u32 = 42;
    /// TurboQuant 2-bit KV (2.1 bpw, ~6.4x compression).
    pub const TURBO2: u32 = 43;
}

pub struct RegistryConfig {
    pub models_dir: PathBuf,
    pub max_models: usize,
    pub max_batch_size: usize,
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
    /// Explicit path to a multimodal projector GGUF file for vision models.
    /// None = auto-detect *mmproj*.gguf in the same directory as the model.
    pub mmproj_path: Option<PathBuf>,
    /// Models discovered from well-known directories (HuggingFace, Ollama, LM Studio, etc.).
    pub discovered_models: Vec<DiscoveredModel>,
}
