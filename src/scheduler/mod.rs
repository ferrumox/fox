// Continuous batching scheduler with prefix caching.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// Blocks are fully reserved at admission, so admission never preempts: a request
// that doesn't fit waits in the queue (FIFO) until running requests finish.

mod batch;
mod prefix_cache;
mod schedule;

#[allow(unused_imports)]
pub use batch::{
    InferenceRequest, RequestState, SamplingParams, ScheduledBatch, StopReason, Token,
    TokenLogprob, TopLogprob,
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
            n_embd: 4096,
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
            n_embd: 128,
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
        assert_eq!(
            inserted,
            Some(16),
            "prefix should be inserted when cache has room (1 block = 16 tokens)"
        );
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
            n_embd: 128,
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
        assert_eq!(
            inserted,
            Some(16),
            "request A prefix should be cached (1 block = 16 tokens)"
        );
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

    /// Prefix-cache eviction stress test — settles design-doc §7's open question
    /// ("does the prefix cache leak blocks/seq-ids under churn?") empirically.
    ///
    /// It drives real `schedule_step` + `try_insert_prefix` over hundreds of
    /// admit / finish / cache / hit / refuse-when-full cycles and asserts strict
    /// conservation after every step, under a **move** ownership model: each seq_id
    /// and each KV block is owned by exactly one of {pool, a running request, a
    /// prefix-cache entry} — never dropped, never duplicated. (The engine layers a
    /// KV-copy optimization on top via `return_prefix_seq_id`; the leak question is
    /// about the scheduler's bookkeeping, which the move model exercises exactly.)
    ///
    /// A leak would show up as either a seq_id that can't be accounted for
    /// (`seen.len() != TOTAL_SEQ`) or an allocated KV block that nothing references
    /// (`allocated_blocks() != reachable`), or a non-zero allocation after draining.
    #[test]
    fn stress_prefix_cache_no_leak() {
        use std::collections::HashSet;

        const TOTAL_SEQ: usize = 8; // = max_batch_size → seq_id pool {0..8}
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        // Plenty of blocks: this test targets prefix-cache churn, not block
        // starvation, so keep the loop live and deterministic.
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv.clone(), TOTAL_SEQ);
        assert_eq!(
            sched.prefix_cache_max, 2,
            "TOTAL_SEQ/4 → cache holds 2 entries"
        );

        // Deterministic prompt of `blocks` full 16-token blocks (+3 leftover tokens),
        // content keyed by `seed` so distinct seeds have distinct prefixes.
        let prompt = |seed: i32, blocks: usize| -> Vec<i32> {
            (0..(blocks * 16 + 3) as i32)
                .map(|i| seed * 1000 + i)
                .collect()
        };

        // Assert full conservation of seq_ids and blocks against the live state.
        let check_conservation = |label: &str| {
            let running = sched.running_batch.lock().unwrap();
            let pool = sched.seq_id_pool.lock().unwrap();
            let pcache = sched.prefix_cache.lock().unwrap();

            // Every seq_id lives in exactly one place, and all TOTAL_SEQ are present.
            let mut seen: HashSet<i32> = HashSet::new();
            for &s in pool.iter() {
                assert!(seen.insert(s), "{label}: seq {s} duplicated in pool");
            }
            for r in running.iter() {
                if r.kv_seq_id >= 0 {
                    assert!(
                        seen.insert(r.kv_seq_id),
                        "{label}: seq {} duplicated (running req {})",
                        r.kv_seq_id,
                        r.id
                    );
                }
            }
            for (_h, e) in pcache.iter() {
                assert!(
                    seen.insert(e.seq_id),
                    "{label}: seq {} duplicated (cache)",
                    e.seq_id
                );
            }
            assert_eq!(
                seen.len(),
                TOTAL_SEQ,
                "{label}: seq_ids not conserved (found {}, expected {TOTAL_SEQ})",
                seen.len()
            );

            // Every allocated block is reachable from a running request or a cache
            // entry — nothing dropped on the floor.
            let running_blocks: usize = running.iter().map(|r| r.page_table.len()).sum();
            let cache_blocks: usize = pcache.iter().map(|(_h, e)| e.block_ids.len()).sum();
            assert_eq!(
                kv.allocated_blocks(),
                running_blocks + cache_blocks,
                "{label}: KV block leak — {} allocated but only {} reachable",
                kv.allocated_blocks(),
                running_blocks + cache_blocks
            );

            // The refuse-when-full guard must keep the cache within bounds.
            assert!(
                pcache.len() <= sched.prefix_cache_max,
                "{label}: prefix cache exceeded its cap ({} > {})",
                pcache.len(),
                sched.prefix_cache_max
            );
        };

        let mut observed_hit = false;
        let mut observed_full_refuse = false;

        for iter in 0..400usize {
            // 1. Submit one request. 2/3 reuse a small shared-prompt set (so their
            //    prefixes hit once cached); 1/3 are unique (misses → fresh inserts).
            let seed = if iter % 3 == 0 {
                100_000 + iter as i32 // unique → miss
            } else {
                (iter % 3) as i32 // {1,2} shared → hit candidate
            };
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let id = iter as u64 + 1;
            let req = InferenceRequest::new(id, prompt(seed, 2), 8, SamplingParams::default(), tx);
            sched.submit(req);

            // 2. Schedule: cleans the previous iteration's finishes and admits.
            let hits_pre = sched.prefix_hits.load(Ordering::Relaxed);
            let _batch = sched.schedule_step();
            if sched.prefix_hits.load(Ordering::Relaxed) > hits_pre {
                observed_hit = true;
            }

            // 3. Conservation must hold after every step.
            check_conservation("mid-churn");

            // 4. Finish the oldest running request once we have a few in flight, and
            //    try to cache it. When the cache is full this exercises the
            //    refuse-when-full path (try_insert_prefix → false).
            let finish_id = {
                let mut running = sched.running_batch.lock().unwrap();
                if running.len() >= 3 {
                    let r = &mut running[0];
                    r.state = RequestState::Finished;
                    Some(r.id)
                } else {
                    None
                }
            };
            if let Some(id) = finish_id {
                let cache_full = sched.prefix_cache_size() >= sched.prefix_cache_max;
                let cached = sched.try_insert_prefix(id);
                if cache_full && cached.is_none() {
                    observed_full_refuse = true;
                }
            }
        }

        // The churn must have actually exercised the interesting paths, or the test
        // would be conserving trivially.
        assert!(
            observed_hit,
            "stress loop never produced a prefix-cache hit"
        );
        assert!(
            observed_full_refuse,
            "stress loop never hit the refuse-when-full path"
        );
        assert!(sched.prefix_hits.load(Ordering::Relaxed) > 0);

        // 5. Drain everything and prove nothing leaked: finish all running requests,
        //    let schedule_step reclaim them, then empty the cache (returning its
        //    seq_ids and freeing its blocks). Allocation must fall back to zero and
        //    the pool must be whole again.
        {
            let mut running = sched.running_batch.lock().unwrap();
            for r in running.iter_mut() {
                r.state = RequestState::Finished;
            }
        }
        sched.schedule_step();
        {
            let mut pcache = sched.prefix_cache.lock().unwrap();
            let mut pool = sched.seq_id_pool.lock().unwrap();
            while let Some((_h, e)) = pcache.pop_lru() {
                kv.free_blocks(&e.block_ids);
                pool.push(e.seq_id);
            }
        }
        assert_eq!(
            kv.allocated_blocks(),
            0,
            "KV blocks leaked after full churn + drain"
        );
        assert_eq!(
            sched.seq_id_pool.lock().unwrap().len(),
            TOTAL_SEQ,
            "seq_id pool not whole after drain"
        );
    }

    /// Chunked prefill state machine (S1): a request whose prompt is prefilled over
    /// several steps must stay `Prefilling` — and be re-emitted to the prefill batch —
    /// until its cursor reaches the prompt end and the first token is sampled, at which
    /// point it moves to the decode batch. The model (do_prefill) advances the cursor
    /// via `advance_prefill`; here we drive those transitions directly (no FFI).
    #[test]
    fn chunked_prefill_stays_prefilling_until_complete() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // 48-token prompt = 3 full blocks; chunked prefill would span several steps.
        let prompt: Vec<i32> = (0..48).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched.submit(InferenceRequest::new(
            1,
            prompt,
            8,
            SamplingParams::default(),
            tx,
        ));

        // Step 1: admitted → emitted to prefill, not decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "admitted request must prefill");
        assert!(b.decode.is_empty());

        // Model submitted a non-final chunk (16 of 48 tokens): still Prefilling.
        sched.advance_prefill(1, 16);

        // Step 2: incomplete prefill must be RE-EMITTED to prefill (the whole point —
        // it no longer completes in a single step) and never to decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "incomplete prefill must be re-emitted");
        assert!(
            b.decode.is_empty(),
            "must not decode before prefill completes"
        );

        // Final chunk reaches the prompt end and the first token is sampled →
        // update_after_token(from_prefill=true) transitions the request to Decoding.
        sched.advance_prefill(1, 48);
        sched.update_after_token(1, 42, true);

        // Step 3: now Decoding → emitted to decode, no longer to prefill.
        let b = sched.schedule_step();
        assert!(
            b.prefill.is_empty(),
            "completed prefill must not re-emit to prefill"
        );
        assert_eq!(b.decode, vec![1], "completed request must decode");
    }

    #[test]
    fn context_roll_reduces_logical_context_len() {
        // context_len() = prefilled + generated - rolled. A roll must shift the next
        // decode position down by exactly the discarded amount, and rolls accumulate.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // Put a request in the running batch that has "filled" 100 tokens of context.
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(7, (0..40).collect(), 200, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.prefilled_tokens = 40;
            req.generated_tokens = 60; // context_len = 100
            req.state = RequestState::Decoding;
            sched.running_batch.lock().unwrap().push(req);
        }

        let ctx_len = |s: &Scheduler| s.get_running(&[7])[0].context_len();
        assert_eq!(ctx_len(&sched), 100);

        // Roll away 30 oldest tokens → live length drops to 70 (next decode pos = 69).
        sched.record_context_roll(7, 30);
        assert_eq!(
            ctx_len(&sched),
            70,
            "rolled tokens subtract from context_len"
        );

        // Generation keeps advancing on top of the reduced length.
        sched.update_after_token(7, 123, false);
        assert_eq!(
            ctx_len(&sched),
            71,
            "generated token adds above the rolled window"
        );

        // A second roll accumulates with the first.
        sched.record_context_roll(7, 20);
        assert_eq!(ctx_len(&sched), 51, "rolls accumulate");
    }
    #[test]
    fn admission_never_preempts_running_requests() {
        // Blocks are fully reserved at admission, so a newcomer that doesn't fit must
        // WAIT — evicting the older running request would discard a generation whose
        // text the client already received (and the pair can evict each other forever:
        // the livelock this test originally exposed).
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        // A deliberately tiny pool: both requests cannot fit at once.
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        assert!(total >= 4, "pool too small to stage the scenario: {total}");
        let sched = Scheduler::new(kv, 4);

        // Request 1 reserves most of the pool and starts generating.
        let prompt1: Vec<i32> = (0..16).collect();
        let max_new1 = (total - 2) * 16 - prompt1.len();
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched.submit(InferenceRequest::new(
            1,
            prompt1.clone(),
            max_new1,
            SamplingParams::default(),
            tx1,
        ));
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "request 1 admitted");
        sched.set_prefilled_tokens(1, prompt1.len());
        for tok in [101, 102, 103] {
            sched.update_after_token(1, tok, tok == 101); // first token completes prefill
        }

        // Request 2 needs more blocks than remain. It must WAIT, not evict request 1.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched.submit(InferenceRequest::new(
            2,
            (0..16).collect(),
            32,
            SamplingParams::default(),
            tx2,
        ));
        let b = sched.schedule_step();
        assert!(
            b.preempted_seq_ids.is_empty(),
            "admission must never preempt a running request"
        );
        assert_eq!(
            b.decode,
            vec![1],
            "request 1 keeps running while request 2 waits"
        );
        assert_eq!(sched.queue_depth(), 1, "request 2 queued");
        {
            let running = sched.running_batch.lock().unwrap();
            let r1 = running.iter().find(|r| r.id == 1).expect("req 1 running");
            assert_eq!(r1.generated_tokens, 3, "generation state untouched");
        }

        // Once request 1 finishes, request 2 gets its turn.
        sched.mark_finished(1, StopReason::Eos);
        let b = sched.schedule_step();
        assert_eq!(
            b.prefill,
            vec![2],
            "request 2 admitted after request 1 finishes"
        );
    }

    #[test]
    fn oversized_request_is_rejected_not_queued_forever() {
        // A request that could never fit even into an EMPTY pool must not block the
        // queue head forever — it is dropped (channel closes) and the queue advances.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        let sched = Scheduler::new(kv, 4);

        // Oversized: needs more blocks than the entire pool.
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched.submit(InferenceRequest::new(
            1,
            (0..16).collect(),
            (total + 2) * 16,
            SamplingParams::default(),
            tx1,
        ));
        // A normal request queued behind it.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched.submit(InferenceRequest::new(
            2,
            (0..16).collect(),
            16,
            SamplingParams::default(),
            tx2,
        ));

        let b = sched.schedule_step();
        assert_eq!(
            b.prefill,
            vec![2],
            "the oversized request is dropped and the next one admitted"
        );
        assert_eq!(sched.queue_depth(), 0, "nothing left waiting");
        assert_eq!(sched.active_requests(), 1, "only the normal request runs");
    }
}
