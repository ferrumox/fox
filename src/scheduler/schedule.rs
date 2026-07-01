use std::sync::atomic::Ordering;

use tracing::{debug, info};

use crate::kv_cache::{compute_block_hash, prompt_block_hashes, PageTable};

use super::batch;
use super::batch::{ScheduledBatch, StopReason};
use super::prefix_cache::PrefixCacheEntry;
use super::Scheduler;

impl Scheduler {
    /// Number of blocks needed for a request (prompt + max_new_tokens, in blocks).
    pub(super) fn blocks_needed(&self, req: &batch::InferenceRequest) -> usize {
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
        // Lock ordering (must be consistent across ALL callers to avoid deadlock):
        //   running_batch → waiting_queue → seq_id_pool → prefix_cache
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("running_batch lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };
        let mut waiting = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("waiting_queue lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };
        let mut pool = match self.seq_id_pool.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("seq_id_pool lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };
        let mut pcache = match self.prefix_cache.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("prefix_cache lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
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
        let block_size = self.kv_cache.block_size();

        'admit: while let Some(mut req) = waiting.pop_front() {
            if pool.is_empty() {
                waiting.push_front(req);
                break;
            }

            // --- Block-level prefix cache lookup ---
            //
            // Compute the chain hash for each complete block of the prompt.
            // Then search from the longest matching prefix down to 1 block.
            // This allows two requests that share a common prefix (e.g. the same
            // system prompt) to reuse each other's cached KV blocks even when the
            // rest of the prompt differs.
            let block_hashes = prompt_block_hashes(&req.prompt_tokens, block_size);
            let mut prefix_hit: Option<(usize, PrefixCacheEntry)> = None;
            for i in (0..block_hashes.len()).rev() {
                if let Some(entry) = pcache.pop(&block_hashes[i]) {
                    prefix_hit = Some((i + 1, entry)); // matched i+1 blocks
                    break;
                }
            }

            if let Some((matched_blocks, hit)) = prefix_hit {
                // --- Prefix cache hit path ---
                //
                // Allocate only the blocks needed beyond the cached prefix:
                // remaining prompt tokens (partial last block) + max_new_tokens.
                let cached_tokens = matched_blocks * block_size;
                let remaining_tokens = req.prompt_tokens.len().saturating_sub(cached_tokens);
                let new_blocks = (remaining_tokens + req.max_new_tokens).div_ceil(block_size);

                if new_blocks == 0 || self.kv_cache.can_allocate(new_blocks) {
                    let new_ids = if new_blocks > 0 {
                        match self.kv_cache.allocate(new_blocks) {
                            Ok(ids) => ids,
                            Err(_) => {
                                // Allocation failed — restore entry and fall through to preemption
                                pcache.put(block_hashes[matched_blocks - 1], hit);
                                waiting.push_front(req);
                                break 'admit;
                            }
                        }
                    } else {
                        vec![]
                    };

                    let id = req.id;
                    req.page_table = PageTable::new(hit.block_ids);
                    req.page_table.extend(new_ids);
                    req.kv_seq_id = hit.seq_id;
                    req.skip_prefix_tokens = cached_tokens;
                    req.prefix_seq_id = None;
                    req.stop_reason = None;
                    req.state = batch::RequestState::Prefilling;
                    self.prefix_hits.fetch_add(1, Ordering::Relaxed);
                    info!(
                        request_id = id,
                        seq_id = req.kv_seq_id,
                        matched_blocks,
                        cached_tokens,
                        "block prefix cache hit — skipping prefill of cached tokens"
                    );
                    running.push(req);
                    prefill.push(id);
                    continue 'admit;
                } else {
                    // No room for new blocks — restore entry and fall through to preemption.
                    pcache.put(block_hashes[matched_blocks - 1], hit);
                    waiting.push_front(req);
                    // Fall through to LIFO preemption below.
                }
            } else {
                // --- Normal admission path (no prefix match) ---
                self.prefix_misses.fetch_add(1, Ordering::Relaxed);
                let needed = self.blocks_needed(&req);
                if self.kv_cache.can_allocate(needed) {
                    match self.kv_cache.allocate(needed) {
                        Ok(ids) => {
                            let Some(seq_id) = pool.pop() else {
                                // Defensive: pool should be non-empty (checked at loop start).
                                self.kv_cache.free_blocks(&ids);
                                waiting.push_front(req);
                                break 'admit;
                            };
                            let id = req.id;
                            req.page_table = PageTable::new(ids);
                            req.kv_seq_id = seq_id;
                            req.stop_reason = None;
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

    /// Replace the physical block at `logical_idx` in the request's page table with `new_block_id`.
    ///
    /// Called by the engine's CoW path after `KVCacheManager::copy_on_write` has allocated a
    /// new exclusive block for a request that was sharing a block with the prefix cache.
    pub fn cow_update_page_table(&self, req_id: u64, logical_idx: usize, new_block_id: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    if let Some(entry) = req.page_table.entries.get_mut(logical_idx) {
                        *entry = new_block_id;
                    }
                    break;
                }
            }
        }
    }

    /// Record how many tokens were actually submitted to llama.cpp during prefill.
    /// Must be called once per request immediately after `run_prefill` returns.
    pub fn set_prefilled_tokens(&self, req_id: u64, count: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.prefilled_tokens = count;
                    break;
                }
            }
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
    pub fn mark_finished(&self, req_id: u64, stop_reason: StopReason) {
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

    /// Try to cache the finished request's KV state for future block-level prefix reuse.
    ///
    /// Only the *complete* block prefix of the prompt is stored: partial trailing blocks
    /// and all generation blocks are freed immediately.  The cache key is the chain hash
    /// of the last complete block, which encodes the full block prefix transitively.
    ///
    /// Returns `true` if an entry was inserted, `false` if the prompt has no complete
    /// blocks or the cache is at capacity.
    ///
    /// Lock ordering: running_batch → prefix_cache (matches schedule_step order).
    pub fn try_insert_prefix(&self, req_id: u64) -> bool {
        let block_size = self.kv_cache.block_size();

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

        // Do not cache if the inference seq_id pool is currently empty.
        // The finished request's seq_id has not yet been returned to the pool;
        // if we donate it to the prefix cache, the pool remains empty and the
        // next request will be unable to start (pool.is_empty() → not admitted).
        {
            let pool = match self.seq_id_pool.lock() {
                Ok(g) => g,
                Err(_) => return false,
            };
            if pool.is_empty() {
                return false;
            }
        }

        for req in running.iter_mut() {
            if req.id != req_id || req.kv_seq_id < 0 {
                continue;
            }

            let full_blocks = req.prompt_tokens.len() / block_size;
            if full_blocks == 0 {
                return false; // prompt too short to cache even one block
            }

            // Compute the chain hash for the full-block prefix.
            let final_hash = (0..full_blocks).fold(0u64, |h, i| {
                let start = i * block_size;
                let end = start + block_size;
                compute_block_hash(h, &req.prompt_tokens[start..end])
            });

            let seq_id = req.kv_seq_id;
            let all_blocks = std::mem::take(&mut req.page_table).entries;

            // Split: first `full_blocks` blocks go to cache; the rest are freed.
            let (cached_blocks, excess) = if all_blocks.len() >= full_blocks {
                let (c, e) = all_blocks.split_at(full_blocks);
                (c.to_vec(), e.to_vec())
            } else {
                // Fewer blocks than expected — free all and skip caching.
                self.kv_cache.free_blocks(&all_blocks);
                return false;
            };

            if !excess.is_empty() {
                self.kv_cache.free_blocks(&excess);
            }

            // Zero out seq ownership so schedule_step won't double-free.
            req.kv_seq_id = -1;

            pcache.put(
                final_hash,
                PrefixCacheEntry {
                    seq_id,
                    block_ids: cached_blocks,
                },
            );
            debug!(
                request_id = req_id,
                full_blocks,
                cached_tokens = full_blocks * block_size,
                "cached block prefix KV state"
            );
            return true;
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
    pub fn get_running(&self, ids: &[u64]) -> Vec<batch::InferenceRequest> {
        let running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return vec![],
        };
        let id_set: ahash::AHashSet<_> = ids.iter().copied().collect();
        running
            .iter()
            .filter(|r| id_set.contains(&r.id))
            .cloned()
            .collect()
    }

    /// Swap a decoding request out of the GPU KV cache into the `Swapped` state.
    ///
    /// The `page_table` is retained (the blocks remain allocated but are
    /// logically "on CPU" after the caller copies the raw KV tensors to a CPU
    /// buffer).  The `kv_seq_id` is kept so the engine can clear the llama.cpp
    /// sequence slot immediately after the caller copies the data out.
    ///
    /// Returns `true` if the request was found in `Decoding` state and
    /// transitioned to `Swapped`; `false` otherwise.
    ///
    /// # Implementation note
    /// The actual byte-level KV transfer (GPU → CPU memcpy) must be performed
    /// by the *caller* **before** calling this method, since the scheduler has
    /// no access to the model's tensor buffers.  See [`RequestState::Swapped`]
    /// for the current limitations.
    pub fn swap_out(&self, req_id: u64) -> bool {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id && req.state == batch::RequestState::Decoding {
                    req.state = batch::RequestState::Swapped;
                    tracing::debug!(request_id = req_id, "request swapped out to CPU");
                    return true;
                }
            }
        }
        false
    }

    /// Swap a previously swapped-out request back in to the GPU KV cache.
    ///
    /// Transitions the request from `Swapped` to `Decoding`.  The caller must
    /// have already copied the KV data from the CPU buffer back to the GPU
    /// **before** calling this method.
    ///
    /// Returns `true` if the request was found in `Swapped` state and
    /// transitioned to `Decoding`; `false` otherwise.
    pub fn swap_in(&self, req_id: u64) -> bool {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id && req.state == batch::RequestState::Swapped {
                    req.state = batch::RequestState::Decoding;
                    tracing::debug!(request_id = req_id, "request swapped back in to GPU");
                    return true;
                }
            }
        }
        false
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
