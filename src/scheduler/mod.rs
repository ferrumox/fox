// Continuous batching scheduler with LIFO preemption and prefix caching.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// When no KV blocks available, preempt LIFO (last admitted).

mod batch;

pub use batch::{
    InferenceRequest, RequestState, SamplingParams, ScheduledBatch, StopReason, Token,
};

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tracing::{debug, info};

use crate::kv_cache::{BlockId, KVCacheManager, PageTable};

// ---------------------------------------------------------------------------
// Prefix cache
// ---------------------------------------------------------------------------

/// One entry in the prefix cache. The seq_id still holds live KV data in llama.cpp
/// (we skipped calling `clear_sequence` when caching). The blocks are "owned" by this
/// entry and will be transferred to the first request that matches the hash.
struct PrefixCacheEntry {
    seq_id: i32,
    block_ids: Vec<BlockId>,
    token_count: usize,
}

/// Hash a slice of token IDs using the standard DefaultHasher.
pub fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    kv_cache: Arc<KVCacheManager>,
    /// Notified when a new request is submitted, waking the engine loop.
    work_notify: tokio::sync::Notify,
    /// Pool of available llama.cpp sequence IDs (0..max_batch_size).
    seq_id_pool: std::sync::Mutex<Vec<i32>>,
    /// Prefix cache: completed-request KV data keyed by hash(prompt_tokens).
    /// Lock ordering: always acquire `running_batch` BEFORE `prefix_cache`.
    prefix_cache: std::sync::Mutex<HashMap<u64, PrefixCacheEntry>>,
    /// Maximum number of entries held in the prefix cache simultaneously.
    prefix_cache_max: usize,
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
            prefix_cache: std::sync::Mutex::new(HashMap::new()),
            prefix_cache_max,
            prefix_hits: AtomicU64::new(0),
            prefix_misses: AtomicU64::new(0),
        }
    }

    /// Submit a request to the waiting queue.
    pub fn submit(&self, req: InferenceRequest) {
        let mut q = self.waiting_queue.lock().expect("scheduler queue lock");
        info!(request_id = req.id, "request admitted to waiting queue");
        q.push_back(req);
        drop(q);
        self.work_notify.notify_one();
    }

    /// Wait until at least one request is available to schedule.
    pub async fn wait_for_work(&self) {
        self.work_notify.notified().await;
    }

    /// Number of blocks needed for a request (prompt + max_new_tokens, in blocks).
    fn blocks_needed(&self, req: &InferenceRequest) -> usize {
        let total_tokens = req.prompt_tokens.len() + req.max_new_tokens;
        let block_size = self.kv_cache.block_size();
        total_tokens.div_ceil(block_size)
    }

    /// One scheduling step. Returns prefill and decode batches.
    ///
    /// 1. Evict Finished requests, free their KV blocks, return seq IDs to pool
    /// 2. Admit from waiting_queue — check prefix cache first, then normal allocation
    /// 3. If no blocks, LIFO preempt last admitted
    /// 4. Return prefill and decode id lists
    pub fn schedule_step(&self) -> ScheduledBatch {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return ScheduledBatch::default(),
        };
        let mut waiting = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(_) => return ScheduledBatch::default(),
        };
        let mut pool = match self.seq_id_pool.lock() {
            Ok(g) => g,
            Err(_) => return ScheduledBatch::default(),
        };
        let mut pcache = match self.prefix_cache.lock() {
            Ok(g) => g,
            Err(_) => return ScheduledBatch::default(),
        };

        // 1. Evict Finished and free blocks + return seq IDs.
        //    Requests whose kv_seq_id == -1 and page_table is empty were already transferred
        //    to the prefix cache by try_insert_prefix — skip them.
        let (finished, still_running): (Vec<_>, Vec<_>) = std::mem::take(&mut *running)
            .into_iter()
            .partition(|r| r.is_finished());

        for req in &finished {
            if !req.page_table.is_empty() {
                self.kv_cache.free_blocks(req.page_table.block_ids());
                debug!(
                    request_id = req.id,
                    blocks = req.page_table.len(),
                    "freed KV blocks for finished request"
                );
            }
            if req.kv_seq_id >= 0 {
                pool.push(req.kv_seq_id);
            }
        }
        *running = still_running;

        // 2. Admit from waiting_queue
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut preempted_seq_ids = Vec::new();

        'admit: while let Some(mut req) = waiting.pop_front() {
            if pool.is_empty() {
                waiting.push_front(req);
                break;
            }

            let token_hash = hash_tokens(&req.prompt_tokens);

            // --- Prefix cache hit path ---
            if let Some(hit) = pcache.remove(&token_hash) {
                // Only need blocks for the generation portion (prompt is already cached).
                let gen_blocks = {
                    let block_size = self.kv_cache.block_size();
                    req.max_new_tokens.div_ceil(block_size)
                };

                if self.kv_cache.can_allocate(gen_blocks) || gen_blocks == 0 {
                    let gen_ids = if gen_blocks > 0 {
                        match self.kv_cache.allocate(gen_blocks) {
                            Ok(ids) => ids,
                            Err(_) => {
                                // Allocation failed — put entry back and fall through to normal path
                                pcache.insert(token_hash, hit);
                                waiting.push_front(req);
                                break 'admit;
                            }
                        }
                    } else {
                        vec![]
                    };

                    let id = req.id;
                    // Start page_table with the cached prompt blocks, then append gen blocks.
                    req.page_table = PageTable::new(hit.block_ids);
                    req.page_table.extend(gen_ids);
                    req.kv_seq_id = pool.pop().expect("pool non-empty checked above");
                    req.skip_prefix_tokens = hit.token_count;
                    req.prefix_seq_id = Some(hit.seq_id);
                    req.state = batch::RequestState::Prefilling;
                    self.prefix_hits.fetch_add(1, Ordering::Relaxed);
                    info!(
                        request_id = id,
                        seq_id = req.kv_seq_id,
                        prefix_tokens = hit.token_count,
                        "prefix cache hit — skipping prefill of cached tokens"
                    );
                    running.push(req);
                    prefill.push(id);
                    continue 'admit;
                } else {
                    // No room for gen blocks — restore cache entry and try preemption.
                    pcache.insert(token_hash, hit);
                    waiting.push_front(req);
                    // Fall through to LIFO preemption below.
                }
            } else {
                // --- Normal admission path ---
                self.prefix_misses.fetch_add(1, Ordering::Relaxed);
                let needed = self.blocks_needed(&req);
                if self.kv_cache.can_allocate(needed) {
                    match self.kv_cache.allocate(needed) {
                        Ok(ids) => {
                            let id = req.id;
                            req.page_table = PageTable::new(ids);
                            req.kv_seq_id = pool.pop().expect("pool non-empty checked above");
                            req.state = batch::RequestState::Prefilling;
                            info!(
                                request_id = id,
                                seq_id = req.kv_seq_id,
                                "request admitted to batch"
                            );
                            running.push(req);
                            prefill.push(id);
                            continue 'admit;
                        }
                        Err(_) => {
                            waiting.push_front(req);
                            // Fall through to LIFO preemption.
                        }
                    }
                } else {
                    waiting.push_front(req);
                    // Fall through to LIFO preemption.
                }
            }

            // 3. No blocks available: LIFO preempt last admitted running request.
            if let Some(mut evicted) = running.pop() {
                evicted.state = batch::RequestState::Waiting;
                evicted.stop_reason = Some(batch::StopReason::Preempt);
                if !evicted.page_table.is_empty() {
                    self.kv_cache.free_blocks(evicted.page_table.block_ids());
                    evicted.page_table.clear();
                }
                if evicted.kv_seq_id >= 0 {
                    preempted_seq_ids.push(evicted.kv_seq_id);
                    pool.push(evicted.kv_seq_id);
                    evicted.kv_seq_id = -1;
                }
                info!(
                    request_id = evicted.id,
                    "LIFO preemption: evicted from batch"
                );
                waiting.push_front(evicted);
            }
            break;
        }

        // 4. Build decode list
        for req in running.iter() {
            if req.state == batch::RequestState::Decoding {
                decode.push(req.id);
            }
        }

        ScheduledBatch {
            prefill,
            decode,
            preempted_seq_ids,
        }
    }

    /// Update request state after a generated token.
    pub fn update_after_token(&self, req_id: u64, token_id: i32, from_prefill: bool) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.last_token = Some(token_id);
                req.generated_tokens += 1;
                req.generated_token_ids.push(token_id);
                if from_prefill && req.state == batch::RequestState::Prefilling {
                    req.state = batch::RequestState::Decoding;
                }
                break;
            }
        }
    }

    /// Mark request as Finished with the given stop reason.
    pub fn mark_finished(&self, req_id: u64, stop_reason: batch::StopReason) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.state = batch::RequestState::Finished;
                req.stop_reason = Some(stop_reason);
                break;
            }
        }
    }

    /// Try to cache the finished request's KV state for future prefix reuse.
    ///
    /// Atomically moves the request's `kv_seq_id` and `page_table` into the prefix cache
    /// (so `schedule_step` won't free them when evicting the Finished request).
    /// Returns `true` if the entry was inserted, `false` if the cache is full.
    ///
    /// Lock ordering: running_batch → prefix_cache (matches schedule_step order).
    pub fn try_insert_prefix(&self, req_id: u64, token_hash: u64) -> bool {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let mut pcache = match self.prefix_cache.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        if pcache.len() >= self.prefix_cache_max {
            return false;
        }

        for req in running.iter_mut() {
            if req.id == req_id && req.kv_seq_id >= 0 {
                let seq_id = req.kv_seq_id;
                let token_count = req.prompt_tokens.len();
                let block_ids = std::mem::take(&mut req.page_table).entries;
                // Zero out the request's ownership so schedule_step won't double-free.
                req.kv_seq_id = -1;
                pcache.insert(
                    token_hash,
                    PrefixCacheEntry {
                        seq_id,
                        block_ids,
                        token_count,
                    },
                );
                debug!(request_id = req_id, token_count, "cached prefix KV state");
                return true;
            }
        }
        false
    }

    /// Return a prefix seq_id back to the pool after the engine has copied and cleared it.
    pub fn return_prefix_seq_id(&self, seq_id: i32) {
        if let Ok(mut pool) = self.seq_id_pool.lock() {
            pool.push(seq_id);
        }
    }

    /// Get running requests by IDs.
    pub fn get_running(&self, ids: &[u64]) -> Vec<InferenceRequest> {
        let running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let id_set: std::collections::HashSet<_> = ids.iter().copied().collect();
        running
            .iter()
            .filter(|r| id_set.contains(&r.id))
            .cloned()
            .collect()
    }

    pub fn queue_depth(&self) -> usize {
        self.waiting_queue.lock().map(|q| q.len()).unwrap_or(0)
    }

    pub fn active_requests(&self) -> usize {
        self.running_batch.lock().map(|r| r.len()).unwrap_or(0)
    }

    pub fn prefix_cache_size(&self) -> usize {
        self.prefix_cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::ModelConfig;
    use crate::kv_cache::KVCacheManager;

    #[test]
    fn test_scheduler_submit_and_schedule() {
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 1_000_000_000, 0.5, 16));
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
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16));
        let sched = Scheduler::new(kv, 8);

        let tokens = vec![1i32, 2, 3, 4, 5];
        let hash = hash_tokens(&tokens);

        // Simulate inserting a prefix cache entry manually
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(42, tokens.clone(), 10, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.page_table = PageTable::new(vec![0, 1]);
            req.state = RequestState::Finished;
            sched.running_batch.lock().unwrap().push(req);
        }

        let inserted = sched.try_insert_prefix(42, hash);
        assert!(inserted, "prefix should be inserted when cache has room");
        assert_eq!(sched.prefix_cache_size(), 1);

        // Submit a new request with the same tokens and check it gets a prefix hit
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let req2 = InferenceRequest::new(99, tokens, 5, SamplingParams::default(), tx2);
        sched.submit(req2);

        // Manually push the seq_id back to pool (simulates what would happen after KV copy)
        sched.return_prefix_seq_id(0);

        let batch = sched.schedule_step();
        // The new request should be admitted to prefill
        assert!(
            batch.prefill.contains(&99),
            "request 99 should be in prefill"
        );
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);
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
