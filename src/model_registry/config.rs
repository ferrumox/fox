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
}
