use std::path::PathBuf;
use std::sync::Arc;

use crate::metrics::Metrics;

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
    /// KV cache element type: 1=F16 (default), 8=Q8_0, 2=Q4_0
    pub type_kv: u32,
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
