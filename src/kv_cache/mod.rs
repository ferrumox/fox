// KV-cache memory manager with O(1) block allocation.
// Blocks are logical units: block_size tokens per block.
// Total blocks computed from: gpu_memory_bytes * fraction / bytes_per_block
// bytes_per_block = block_size × num_layers × num_heads_kv × head_dim × 2 (K+V) × 2 (f16)

use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use tracing::debug;

use crate::engine::model::ModelConfig;

/// Physical block ID in the KV cache pool.
pub type BlockId = usize;

// ---------------------------------------------------------------------------
// Block-level chain hashing
// ---------------------------------------------------------------------------

/// Process-stable random state for block hashing (same as the scheduler's).
/// A separate `OnceLock` ensures the seed is consistent across kv_cache and scheduler.
static BLOCK_HASH_STATE: OnceLock<ahash::RandomState> = OnceLock::new();

/// Compute the chain hash for one block.
///
/// # Chain hash design
/// `h_N = hash(h_{N-1} ‖ tokens_N)`
///
/// Starting with `parent_hash = 0` for the first block, each block's hash
/// captures the full prefix up to (and including) that block.  Two request
/// prompts that share their first K blocks will therefore produce identical
/// hashes at each of the first K block boundaries, enabling block-level prefix
/// cache matches even when the prompts diverge afterwards.
pub fn compute_block_hash(parent_hash: u64, tokens: &[i32]) -> u64 {
    let state = BLOCK_HASH_STATE.get_or_init(ahash::RandomState::new);
    let mut h = state.build_hasher();
    parent_hash.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// Compute the chain hashes for all *complete* blocks of `tokens`.
/// Returns one hash per full block (length = `tokens.len() / block_size`).
/// The partial trailing block (if any) is intentionally excluded because its
/// content can still change as more tokens are appended during decoding.
pub fn prompt_block_hashes(tokens: &[i32], block_size: usize) -> Vec<u64> {
    let full_blocks = tokens.len() / block_size;
    let mut h = 0u64;
    (0..full_blocks)
        .map(|i| {
            let start = i * block_size;
            let end = start + block_size;
            h = compute_block_hash(h, &tokens[start..end]);
            h
        })
        .collect()
}

/// Logical-to-physical page table for a single request.
///
/// `entries[logical_block_index] = physical_block_id`
///
/// This explicit struct replaces the flat `Vec<BlockId>` so the logical→physical
/// mapping is named and opaque to callers. It is also the integration point for
/// future copy-on-write: when a block is shared (ref_count > 1) the caller must
/// call `KVCacheManager::copy_on_write` before writing to it.
#[derive(Debug, Clone, Default)]
pub struct PageTable {
    pub entries: Vec<BlockId>,
}

impl PageTable {
    pub fn new(block_ids: Vec<BlockId>) -> Self {
        Self { entries: block_ids }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn block_ids(&self) -> &[BlockId] {
        &self.entries
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Extend the page table with newly allocated blocks.
    pub fn extend(&mut self, ids: impl IntoIterator<Item = BlockId>) {
        self.entries.extend(ids);
    }
}

/// Returns (numerator, denominator) for bytes-per-element of a KV cache type.
/// bytes/element = num/den (exact rational representation).
/// bytes/element = num/den (exact rational representation).
pub fn kv_type_bytes(t: u32) -> (u64, u64) {
    use crate::model_registry::kv_type;
    match t {
        kv_type::Q8_0 => (1, 1),   // 8 bits
        kv_type::Q4_0 => (1, 2),   // 4 bits
        kv_type::TURBO3 => (50, 128), // 3.125 bpw: 14 bits/elem × 128-block = 50 bytes
        kv_type::TURBO4 => (68, 128), // 4.25 bpw:  68 bytes / 128 elements
        kv_type::TURBO2 => (34, 128), // 2.125 bpw: 34 bytes / 128 elements
        _ => (2, 1), // F16 = 2 bytes
    }
}

/// Configuration for KV cache sizing.
/// Blocks = gpu_memory_bytes * fraction / bytes_per_block
/// bytes_per_block = block_size × num_layers × num_heads_kv × head_dim × (bytes_K + bytes_V)
#[allow(clippy::too_many_arguments)]
fn compute_total_blocks(
    gpu_memory_bytes: usize,
    gpu_memory_fraction: f32,
    block_size: usize,
    num_layers: usize,
    num_heads_kv: usize,
    head_dim: usize,
    type_k: u32,
    type_v: u32,
) -> usize {
    let elements = (block_size * num_layers * num_heads_kv * head_dim) as u64;
    let (k_num, k_den) = kv_type_bytes(type_k);
    let (v_num, v_den) = kv_type_bytes(type_v);
    // Compute bytes for K and V tensors separately, ceiling to avoid underestimating.
    let k_bytes = (elements * k_num).div_ceil(k_den);
    let v_bytes = (elements * v_num).div_ceil(v_den);
    let bytes_per_block = (k_bytes + v_bytes) as usize;
    let available = (gpu_memory_bytes as f64 * gpu_memory_fraction as f64) as usize;
    (available / bytes_per_block).max(1)
}

/// Thread-safe KV cache block manager.
///
/// Uses a free list for O(1) allocation and deallocation.
/// Maintains a per-block reference count to support copy-on-write semantics:
/// when a block is shared between requests (e.g. via prefix caching), its
/// ref_count > 1. A write to a shared block must first call `copy_on_write`
/// to obtain an exclusive copy.
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
    /// Per-block reference count. ref_count[id] == 0 means the block is free.
    /// ref_count[id] > 1 means the block is shared; CoW required before write.
    ref_count: Vec<AtomicUsize>,
}

impl KVCacheManager {
    /// Create a new KV cache manager.
    ///
    /// # Arguments
    /// * `model_config` - Model architecture (num_layers, num_heads_kv, head_dim)
    /// * `gpu_memory_bytes` - Total GPU memory in bytes (or fallback for CPU)
    /// * `gpu_memory_fraction` - Fraction to use for KV cache (0.0-1.0)
    /// * `block_size` - Tokens per block
    /// * `type_k` - Key cache element type (see `kv_type` constants)
    /// * `type_v` - Value cache element type (see `kv_type` constants)
    pub fn new(
        model_config: &ModelConfig,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        block_size: usize,
        type_k: u32,
        type_v: u32,
    ) -> Self {
        let total_blocks = compute_total_blocks(
            gpu_memory_bytes,
            gpu_memory_fraction,
            block_size,
            model_config.num_layers,
            model_config.num_heads_kv,
            model_config.head_dim,
            type_k,
            type_v,
        );

        let free_list: Vec<BlockId> = (0..total_blocks).collect();
        let ref_count: Vec<AtomicUsize> = (0..total_blocks).map(|_| AtomicUsize::new(0)).collect();

        debug!(
            total_blocks,
            block_size, gpu_memory_bytes, "KV cache manager initialized"
        );

        Self {
            total_blocks,
            block_size,
            free_list: Mutex::new(free_list),
            allocated_count: AtomicUsize::new(0),
            ref_count,
        }
    }

    /// Check if n blocks can be allocated.
    pub fn can_allocate(&self, n: usize) -> bool {
        let Ok(guard) = self.free_list.lock() else {
            return false;
        };
        guard.len() >= n
    }

    /// Allocate n blocks. Returns block IDs with ref_count set to 1.
    pub fn allocate(&self, n: usize) -> Result<Vec<BlockId>> {
        let mut guard = self
            .free_list
            .lock()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {}", e))?;
        if guard.len() < n {
            anyhow::bail!(
                "Insufficient KV cache blocks: need {}, have {}",
                n,
                guard.len()
            );
        }
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let id = guard.pop().expect("len checked");
            self.ref_count[id].store(1, Ordering::Relaxed);
            ids.push(id);
        }
        self.allocated_count.fetch_add(n, Ordering::Relaxed);
        Ok(ids)
    }

    /// Decrement the reference count for each block.
    /// A block is returned to the free list only when its ref_count reaches zero.
    pub fn free_blocks(&self, ids: &[BlockId]) {
        let mut guard = match self.free_list.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        let mut actually_freed = 0usize;
        for &id in ids {
            if id >= self.total_blocks {
                continue;
            }
            let prev = self.ref_count[id].fetch_sub(1, Ordering::AcqRel);
            if prev <= 1 {
                // ref_count hit zero (or was already zero — defensive): return to free list
                self.ref_count[id].store(0, Ordering::Relaxed);
                guard.push(id);
                actually_freed += 1;
            }
        }
        if actually_freed > 0 {
            self.allocated_count
                .fetch_sub(actually_freed, Ordering::Relaxed);
        }
    }

    /// Increment the reference count of a block (used when sharing a block between requests
    /// for prefix caching). The caller must eventually call `free_blocks` to release their hold.
    pub fn retain_block(&self, id: BlockId) {
        if id < self.total_blocks {
            self.ref_count[id].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Copy-on-write: if `block_id` is shared (ref_count > 1), allocate a new exclusive
    /// block and decrement the shared block's ref_count. Returns the new block ID.
    /// Returns `None` if the block is already exclusive (ref_count == 1).
    ///
    /// Note: This only updates our block-tracking layer. The caller is responsible for
    /// copying the actual KV data in llama.cpp via `model.copy_sequence_range`.
    pub fn copy_on_write(&self, block_id: BlockId) -> Option<BlockId> {
        if block_id >= self.total_blocks {
            return None;
        }
        let rc = self.ref_count[block_id].load(Ordering::Acquire);
        if rc <= 1 {
            return None; // already exclusive
        }
        // Allocate a new block
        let new_ids = self.allocate(1).ok()?;
        let new_id = new_ids[0];
        // Release our share of the old block
        let prev = self.ref_count[block_id].fetch_sub(1, Ordering::AcqRel);
        if prev <= 1 {
            // Unlikely: another thread raced and freed it; put it back to free list
            self.ref_count[block_id].store(0, Ordering::Relaxed);
            if let Ok(mut guard) = self.free_list.lock() {
                guard.push(block_id);
            }
            self.allocated_count.fetch_sub(1, Ordering::Relaxed);
        }
        Some(new_id)
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

    /// Returns `true` if `block_id` is currently shared (ref_count > 1).
    ///
    /// Used by the decode path to decide whether copy-on-write is needed before
    /// llama.cpp writes a new token into this block's KV slot.
    pub fn is_shared(&self, block_id: BlockId) -> bool {
        if block_id >= self.total_blocks {
            return false;
        }
        self.ref_count[block_id].load(Ordering::Acquire) > 1
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
        let mgr = KVCacheManager::new(&config, gpu_bytes, 0.85, 16, 1, 1);

        assert!(mgr.can_allocate(10));
        let ids = mgr.allocate(10).unwrap();
        assert_eq!(ids.len(), 10);

        mgr.free_blocks(&ids);
        assert!(mgr.can_allocate(10));
    }

    #[test]
    fn test_memory_usage() {
        let config = test_model_config();
        let mgr = KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1);

        let ids = mgr.allocate(1).unwrap();
        let usage = mgr.memory_usage();
        assert!(usage > 0.0 && usage <= 1.0);
        mgr.free_blocks(&ids);
    }

    #[test]
    fn test_ref_count_sharing() {
        let config = test_model_config();
        let mgr = KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1);

        let ids = mgr.allocate(1).unwrap();
        let id = ids[0];

        // Retain shares the block (ref_count = 2)
        mgr.retain_block(id);
        assert_eq!(mgr.ref_count[id].load(Ordering::Relaxed), 2);

        // Free once: ref_count = 1, block stays allocated
        mgr.free_blocks(&[id]);
        assert_eq!(mgr.ref_count[id].load(Ordering::Relaxed), 1);

        // Free again: ref_count = 0, block goes back to free list
        mgr.free_blocks(&[id]);
        assert_eq!(mgr.ref_count[id].load(Ordering::Relaxed), 0);
        assert!(mgr.can_allocate(1));
    }

    #[test]
    fn test_copy_on_write() {
        let config = test_model_config();
        let mgr = KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1);

        let ids = mgr.allocate(1).unwrap();
        let id = ids[0];

        // Exclusive block: CoW returns None
        assert!(mgr.copy_on_write(id).is_none());

        // Share the block
        mgr.retain_block(id);

        // CoW on shared block: returns a new block ID
        let new_id = mgr.copy_on_write(id).expect("should CoW a shared block");
        assert_ne!(new_id, id);

        // Old block ref_count drops to 1 (was 2)
        assert_eq!(mgr.ref_count[id].load(Ordering::Relaxed), 1);

        mgr.free_blocks(&[id]);
        mgr.free_blocks(&[new_id]);
    }

    #[test]
    fn test_page_table() {
        let ids = vec![0usize, 1, 2];
        let pt = PageTable::new(ids.clone());
        assert_eq!(pt.len(), 3);
        assert_eq!(pt.block_ids(), ids.as_slice());
        assert!(!pt.is_empty());

        let empty = PageTable::default();
        assert!(empty.is_empty());
    }
}
