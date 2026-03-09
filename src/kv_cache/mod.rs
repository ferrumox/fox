// KV-cache memory manager with O(1) block allocation.
// Blocks are logical units: block_size tokens per block.
// Total blocks computed from: gpu_memory_bytes * fraction / bytes_per_block
// bytes_per_block = block_size × num_layers × num_heads_kv × head_dim × 2 (K+V) × 2 (f16)

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use tracing::debug;

use crate::engine::model::ModelConfig;

/// Physical block ID in the KV cache pool.
pub type BlockId = usize;

/// Configuration for KV cache sizing.
/// Blocks = gpu_memory_bytes * fraction / bytes_per_block
/// bytes_per_block = block_size × num_layers × num_heads_kv × head_dim × 2 × 2
/// (×2 for K+V, ×2 for f16 = 2 bytes)
fn compute_total_blocks(
    gpu_memory_bytes: usize,
    gpu_memory_fraction: f32,
    block_size: usize,
    num_layers: usize,
    num_heads_kv: usize,
    head_dim: usize,
) -> usize {
    let bytes_per_block = block_size
        * num_layers
        * num_heads_kv
        * head_dim
        * 2 // K + V
        * 2; // f16 = 2 bytes
    let available = (gpu_memory_bytes as f64 * gpu_memory_fraction as f64) as usize;
    (available / bytes_per_block).max(1)
}

/// Thread-safe KV cache block manager.
/// Uses a free list for O(1) allocation and deallocation.
#[derive(Debug)]
pub struct KVCacheManager {
    /// Total number of blocks in the pool
    total_blocks: usize,
    /// Block size in tokens
    block_size: usize,
    /// Free block IDs (stack for O(1) pop)
    free_list: Mutex<Vec<BlockId>>,
    /// Number of currently allocated blocks (for memory_usage)
    allocated_count: AtomicUsize,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    ///
    /// # Arguments
    /// * `model_config` - Model architecture (num_layers, num_heads_kv, head_dim)
    /// * `gpu_memory_bytes` - Total GPU memory in bytes (or fallback for CPU)
    /// * `gpu_memory_fraction` - Fraction to use for KV cache (0.0-1.0)
    /// * `block_size` - Tokens per block
    pub fn new(
        model_config: &ModelConfig,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        block_size: usize,
    ) -> Self {
        let total_blocks = compute_total_blocks(
            gpu_memory_bytes,
            gpu_memory_fraction,
            block_size,
            model_config.num_layers,
            model_config.num_heads_kv,
            model_config.head_dim,
        );

        let free_list: Vec<BlockId> = (0..total_blocks).collect();
        debug!(
            total_blocks,
            block_size,
            gpu_memory_bytes,
            "KV cache manager initialized"
        );

        Self {
            total_blocks,
            block_size,
            free_list: Mutex::new(free_list),
            allocated_count: AtomicUsize::new(0),
        }
    }

    /// Check if n blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        let Ok(guard) = self.free_list.lock() else {
            return false;
        };
        guard.len() >= n
    }

    /// Allocate n blocks. Returns block IDs.
    pub fn allocate(&self, n: usize) -> Result<Vec<BlockId>> {
        let mut guard = self.free_list.lock().map_err(|e| anyhow::anyhow!("lock poisoned: {}", e))?;
        if guard.len() < n {
            anyhow::bail!("Insufficient KV cache blocks: need {}, have {}", n, guard.len());
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(guard.pop().expect("len checked"));
        }
        self.allocated_count.fetch_add(n, Ordering::Relaxed);
        Ok(ids)
    }

    /// Free blocks by their IDs.
    pub fn free_blocks(&self, ids: &[BlockId]) {
        let mut guard = match self.free_list.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for &id in ids {
            guard.push(id);
        }
        self.allocated_count.fetch_sub(ids.len(), Ordering::Relaxed);
    }

    /// Append one block for a request. Returns the block ID if allocated.
    pub fn append_block(&self, _req_id: u64) -> Option<BlockId> {
        let ids = self.allocate(1).ok()?;
        ids.into_iter().next()
    }

    /// Memory usage as fraction [0.0, 1.0].
    pub fn memory_usage(&self) -> f32 {
        let allocated = self.allocated_count.load(Ordering::Relaxed) as f32;
        let total = self.total_blocks as f32;
        if total == 0.0 {
            0.0
        } else {
            (allocated / total).min(1.0)
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }
}

unsafe impl Send for KVCacheManager {}
unsafe impl Sync for KVCacheManager {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::ModelConfig;

    fn test_model_config() -> ModelConfig {
        ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        }
    }

    #[test]
    fn test_allocate_free() {
        let config = test_model_config();
        let gpu_bytes = 8 * 1024 * 1024 * 1024; // 8 GB
        let mgr = KVCacheManager::new(&config, gpu_bytes, 0.85, 16);

        assert!(mgr.can_allocate(10));
        let ids = mgr.allocate(10).unwrap();
        assert_eq!(ids.len(), 10);

        mgr.free_blocks(&ids);
        assert!(mgr.can_allocate(10));
    }

    #[test]
    fn test_memory_usage() {
        let config = test_model_config();
        let mgr = KVCacheManager::new(&config, 1_000_000_000, 0.5, 16);

        let ids = mgr.allocate(1).unwrap();
        let usage = mgr.memory_usage();
        assert!(usage > 0.0 && usage <= 1.0);
        mgr.free_blocks(&ids);
    }
}
