// Continuous batching scheduler with LIFO preemption.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// When no KV blocks available, preempt LIFO (last admitted).

mod batch;

pub use batch::{InferenceRequest, SamplingParams, ScheduledBatch, StopReason, RequestState, Token};

use std::collections::VecDeque;
use std::sync::Arc;

use tracing::{debug, info};

use crate::kv_cache::KVCacheManager;

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    kv_cache: Arc<KVCacheManager>,
    /// Notified when a new request is submitted, waking the engine loop.
    work_notify: tokio::sync::Notify,
    /// Pool of available llama.cpp sequence IDs (0..max_batch_size).
    /// Popped when a request is admitted, pushed back when it finishes or is preempted.
    seq_id_pool: std::sync::Mutex<Vec<i32>>,
}

impl Scheduler {
    pub fn new(kv_cache: Arc<KVCacheManager>, max_batch_size: usize) -> Self {
        let pool: Vec<i32> = (0..max_batch_size as i32).collect();
        Self {
            waiting_queue: std::sync::Mutex::new(VecDeque::new()),
            running_batch: std::sync::Mutex::new(Vec::new()),
            kv_cache,
            work_notify: tokio::sync::Notify::new(),
            seq_id_pool: std::sync::Mutex::new(pool),
        }
    }

    /// Submit a request to the waiting queue.
    pub fn submit(&self, req: InferenceRequest) {
        let mut q = self
            .waiting_queue
            .lock()
            .expect("scheduler queue lock");
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
        (total_tokens + block_size - 1) / block_size
    }

    /// One scheduling step. Returns prefill and decode batches.
    ///
    /// 1. Evict Finished requests, free their KV blocks, return seq IDs to pool
    /// 2. Admit from waiting_queue if blocks and seq IDs are available
    /// 3. If no blocks, LIFO preempt last admitted (returns seq ID to pool)
    /// 4. Return prefill (need first token) and decode (need next token) ids
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

        // 1. Evict Finished and free blocks + return seq IDs
        let (finished, still_running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *running).into_iter().partition(|r| r.is_finished());

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

        // 2. Try to admit from waiting_queue
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        let mut preempted_seq_ids = Vec::new();

        while let Some(mut req) = waiting.pop_front() {
            // Need both a free KV block range and a free seq ID
            if pool.is_empty() {
                waiting.push_front(req);
                break;
            }
            let needed = self.blocks_needed(&req);
            if self.kv_cache.can_allocate(needed) {
                match self.kv_cache.allocate(needed) {
                    Ok(ids) => {
                        let id = req.id;
                        req.page_table = crate::kv_cache::PageTable::new(ids);
                        req.kv_seq_id = pool.pop().expect("pool non-empty checked above");
                        req.state = batch::RequestState::Prefilling;
                        info!(request_id = id, seq_id = req.kv_seq_id, "request admitted to batch");
                        running.push(req);
                        prefill.push(id);
                    }
                    Err(_) => {
                        waiting.push_front(req);
                        break;
                    }
                }
            } else {
                // 3. No blocks: LIFO preempt last admitted
                waiting.push_front(req);
                if let Some(mut evicted) = running.pop() {
                    evicted.state = batch::RequestState::Waiting;
                    evicted.stop_reason = Some(batch::StopReason::Preempt);
                    if !evicted.page_table.is_empty() {
                        self.kv_cache.free_blocks(evicted.page_table.block_ids());
                        evicted.page_table.clear();
                    }
                    if evicted.kv_seq_id >= 0 {
                        // Return to pool and signal engine to wipe the KV state for this seq_id.
                        // The engine will call model.clear_sequence() before the ID can be reused.
                        preempted_seq_ids.push(evicted.kv_seq_id);
                        pool.push(evicted.kv_seq_id);
                        evicted.kv_seq_id = -1;
                    }
                    info!(request_id = evicted.id, "LIFO preemption: evicted from batch");
                    waiting.push_front(evicted);
                }
                break;
            }
        }

        // 4. Build decode list (requests in Decoding state that need next token)
        for req in running.iter_mut() {
            if req.state == batch::RequestState::Prefilling {
                // After prefill we'll transition to Decoding
                continue;
            }
            if req.state == batch::RequestState::Decoding {
                decode.push(req.id);
            }
        }

        ScheduledBatch { prefill, decode, preempted_seq_ids }
    }

    /// Update request state after a generated token in a single lock acquisition.
    /// Handles prefill→decode transition, last_token, and generated_tokens counter.
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
}
