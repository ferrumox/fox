// Continuous batching scheduler with LIFO preemption.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// When no KV blocks available, preempt LIFO (last admitted).

mod batch;

pub use batch::{InferenceRequest, ScheduledBatch, StopReason, RequestState, Token};

use std::collections::VecDeque;
use std::sync::Arc;

use tracing::{debug, info};

use crate::kv_cache::KVCacheManager;

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    kv_cache: Arc<KVCacheManager>,
}

impl Scheduler {
    pub fn new(kv_cache: Arc<KVCacheManager>) -> Self {
        Self {
            waiting_queue: std::sync::Mutex::new(VecDeque::new()),
            running_batch: std::sync::Mutex::new(Vec::new()),
            kv_cache,
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
    }

    /// Number of blocks needed for a request (prompt + max_new_tokens, in blocks).
    fn blocks_needed(&self, req: &InferenceRequest) -> usize {
        let total_tokens = req.prompt_tokens.len() + req.max_new_tokens;
        let block_size = self.kv_cache.block_size();
        (total_tokens + block_size - 1) / block_size
    }

    /// One scheduling step. Returns prefill and decode batches.
    ///
    /// 1. Evict Finished requests, free their KV blocks
    /// 2. Admit from waiting_queue if blocks available
    /// 3. If no blocks, LIFO preempt last admitted
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

        // 1. Evict Finished and free blocks
        let (finished, still_running): (Vec<_>, Vec<_>) =
            std::mem::take(&mut *running).into_iter().partition(|r| r.is_finished());

        for req in &finished {
            if !req.kv_block_ids.is_empty() {
                self.kv_cache.free_blocks(&req.kv_block_ids);
                debug!(
                    request_id = req.id,
                    blocks = req.kv_block_ids.len(),
                    "freed KV blocks for finished request"
                );
            }
        }
        *running = still_running;

        // 2. Try to admit from waiting_queue
        let mut prefill = Vec::new();
        let mut decode = Vec::new();

        while let Some(mut req) = waiting.pop_front() {
            let needed = self.blocks_needed(&req);
            if self.kv_cache.can_allocate(needed) {
                match self.kv_cache.allocate(needed) {
                    Ok(ids) => {
                        let id = req.id;
                        req.kv_block_ids = ids;
                        req.state = batch::RequestState::Prefilling;
                        info!(request_id = id, "request admitted to batch");
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
                    if !evicted.kv_block_ids.is_empty() {
                        self.kv_cache.free_blocks(&evicted.kv_block_ids);
                        evicted.kv_block_ids.clear();
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

        ScheduledBatch { prefill, decode }
    }

    /// Transition request from Prefilling to Decoding after prefill step.
    pub fn mark_prefill_done(&self, req_id: u64) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id && req.state == batch::RequestState::Prefilling {
                req.state = batch::RequestState::Decoding;
                break;
            }
        }
    }

    /// Increment generated token count for a request.
    pub fn increment_generated(&self, req_id: u64) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.generated_tokens += 1;
                break;
            }
        }
    }

    /// Set the last sampled token for a request (used as input for next decode).
    pub fn set_last_token(&self, req_id: u64, token: i32) {
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        for req in running.iter_mut() {
            if req.id == req_id {
                req.last_token = Some(token);
                break;
            }
        }
    }

    /// Mark request as Finished and optionally extend kv_block_ids if we're doing incremental allocation.
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
        let sched = Scheduler::new(kv);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = InferenceRequest::new(1, vec![1, 2, 3], 10, tx);
        sched.submit(req);

        assert_eq!(sched.queue_depth(), 1);
        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![1]);
        assert_eq!(sched.queue_depth(), 0);
    }
}
