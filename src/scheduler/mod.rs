// Continuous batching scheduler with LIFO preemption and prefix caching.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// When no KV blocks available, preempt LIFO (last admitted).

mod batch;
mod prefix_cache;
mod schedule;

#[allow(unused_imports)]
pub use batch::{
    InferenceRequest, RequestState, SamplingParams, ScheduledBatch, StopReason, Token,
};

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use tracing::info;

use crate::kv_cache::KVCacheManager;

use prefix_cache::PrefixCacheEntry;

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    pub(super) waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    pub(super) running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    pub(super) kv_cache: Arc<KVCacheManager>,
    /// Notified when a new request is submitted, waking the engine loop.
    work_notify: tokio::sync::Notify,
    /// Pool of available llama.cpp sequence IDs (0..max_batch_size).
    pub(super) seq_id_pool: std::sync::Mutex<Vec<i32>>,
    /// Prefix cache: completed-request KV data keyed by hash(prompt_tokens).
    /// LruCache provides O(1) access and automatic LRU ordering for future eviction.
    /// Lock ordering: always acquire `running_batch` BEFORE `prefix_cache`.
    pub(super) prefix_cache: std::sync::Mutex<lru::LruCache<u64, PrefixCacheEntry>>,
    /// Maximum number of entries held in the prefix cache simultaneously.
    pub(super) prefix_cache_max: usize,
    /// Lifetime hit counter (for metrics / logging).
    pub prefix_hits: AtomicU64,
    /// Lifetime miss counter (for metrics / logging).
    pub prefix_misses: AtomicU64,
}

impl Scheduler {
    pub fn new(kv_cache: Arc<KVCacheManager>, max_batch_size: usize) -> Self {
        let pool: Vec<i32> = (0..max_batch_size as i32).collect();
        // Reserve up to 1/4 of the batch size for prefix cache entries (minimum 1).
        let prefix_cache_max = (max_batch_size / 4).max(1);
        Self {
            waiting_queue: std::sync::Mutex::new(VecDeque::new()),
            running_batch: std::sync::Mutex::new(Vec::new()),
            kv_cache,
            work_notify: tokio::sync::Notify::new(),
            seq_id_pool: std::sync::Mutex::new(pool),
            prefix_cache: std::sync::Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(prefix_cache_max)
                    .expect("prefix_cache_max is max_batch_size/4 clamped to >= 1"),
            )),
            prefix_cache_max,
            prefix_hits: AtomicU64::new(0),
            prefix_misses: AtomicU64::new(0),
        }
    }

    /// Submit a request to the waiting queue.
    pub fn submit(&self, req: InferenceRequest) {
        let mut q = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("waiting_queue lock poisoned on submit: {}", e);
                e.into_inner()
            }
        };
        info!(request_id = req.id, "request admitted to waiting queue");
        q.push_back(req);
        drop(q);
        self.work_notify.notify_one();
    }

    /// Wait until at least one request is available to schedule.
    pub async fn wait_for_work(&self) {
        self.work_notify.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::ModelConfig;
    use crate::kv_cache::{KVCacheManager, PageTable};
    use prefix_cache::hash_tokens;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_scheduler_submit_and_schedule() {
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = InferenceRequest::new(1, vec![1, 2, 3], 10, SamplingParams::default(), tx);
        sched.submit(req);

        assert_eq!(sched.queue_depth(), 1);
        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![1]);
        assert_eq!(sched.queue_depth(), 0);
    }

    #[test]
    fn test_prefix_cache_insert_and_hit() {
        // block_size = 16; prompt must have ≥ 16 tokens so try_insert_prefix
        // has at least one complete block to cache.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // 18 tokens → 1 full block (16 tokens) + 2 leftover.
        let tokens: Vec<i32> = (1..=18).collect();

        // Simulate a finished request with 3 allocated blocks (1 prompt full + 2 gen).
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(42, tokens.clone(), 10, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.page_table = PageTable::new(vec![0, 1, 2]); // blocks 0-2
            req.state = RequestState::Finished;
            sched.running_batch.lock().unwrap().push(req);
        }

        // try_insert_prefix computes the hash internally; only block 0 should be cached.
        let inserted = sched.try_insert_prefix(42);
        assert!(inserted, "prefix should be inserted when cache has room");
        assert_eq!(sched.prefix_cache_size(), 1);

        // Submit a new request with the same 18-token prefix.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let req2 = InferenceRequest::new(99, tokens, 5, SamplingParams::default(), tx2);
        sched.submit(req2);

        // Return the cached seq_id to the pool (engine normally does this after KV copy).
        sched.return_prefix_seq_id(0);

        let batch = sched.schedule_step();
        assert!(
            batch.prefill.contains(&99),
            "request 99 should be admitted to prefill"
        );
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_prefix_cache_block_level_partial_match() {
        // Two requests share 16 tokens (1 full block) but diverge after that.
        // The first request completes and caches its full block.
        // The second request should get a partial prefix hit (16 tokens skipped).
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // Shared prefix: tokens 1-16
        let shared_prefix: Vec<i32> = (1..=16).collect();
        // Full prompt A: shared_prefix (= 16 tokens, 1 complete block)
        let tokens_a = shared_prefix.clone();
        // Full prompt B: same first 16 tokens + 4 different tokens
        let mut tokens_b = shared_prefix.clone();
        tokens_b.extend([100i32, 101, 102, 103]);

        // Finish request A (1 block cached)
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(1, tokens_a.clone(), 5, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.page_table = PageTable::new(vec![0, 1]); // block 0 = prompt, block 1 = gen
            req.state = RequestState::Finished;
            sched.running_batch.lock().unwrap().push(req);
        }
        let inserted = sched.try_insert_prefix(1);
        assert!(inserted, "request A prefix should be cached");
        sched.return_prefix_seq_id(0); // return cached seq_id to pool

        // Submit request B (diverges after the shared first block)
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let req2 = InferenceRequest::new(2, tokens_b, 5, SamplingParams::default(), tx2);
        sched.submit(req2);

        let batch = sched.schedule_step();
        assert!(
            batch.prefill.contains(&2),
            "request B should be admitted to prefill via partial block hit"
        );
        assert_eq!(
            sched.prefix_hits.load(Ordering::Relaxed),
            1,
            "exactly one prefix hit for the shared 16-token block"
        );

        // Verify skip_prefix_tokens was set to 16 (one full block)
        let running = sched.running_batch.lock().unwrap();
        let req_b = running
            .iter()
            .find(|r| r.id == 2)
            .expect("req B in running");
        assert_eq!(
            req_b.skip_prefix_tokens, 16,
            "should skip exactly one full block"
        );
    }

    #[test]
    fn test_hash_tokens_stable() {
        let a = hash_tokens(&[1, 2, 3]);
        let b = hash_tokens(&[1, 2, 3]);
        let c = hash_tokens(&[1, 2, 4]);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
