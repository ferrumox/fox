use std::sync::atomic::Ordering;

use tracing::{debug, info};

use crate::kv_cache::PageTable;

use super::batch;
use super::batch::{ScheduledBatch, StopReason};
use super::slots::common_prefix_len;
use super::Scheduler;

impl Scheduler {
    /// Number of blocks needed for a request (prompt + max_new_tokens, in blocks).
    pub(super) fn blocks_needed(&self, req: &batch::InferenceRequest) -> usize {
        let total_tokens = req.n_positions() + req.max_new_tokens;
        let block_size = self.kv_cache.block_size();
        total_tokens.div_ceil(block_size)
    }

    /// One scheduling step. Returns prefill and decode batches.
    ///
    /// 1. Retire Finished requests: park each one's sequence as a reusable idle slot
    ///    (KV kept, blocks kept), or release it outright when its KV must not be
    ///    reused.
    /// 2. Admit from waiting_queue, choosing each request's slot by longest-common-
    ///    prefix affinity so it inherits as much resident KV as possible.
    ///    Admission NEVER preempts a *running* request: blocks are fully reserved at
    ///    admission (prompt + max_new_tokens), so running requests never grow —
    ///    evicting an older running request for a newer waiting one is both unfair
    ///    and livelock-prone (the pair can evict each other forever). A request that
    ///    doesn't fit waits (FIFO). Reclaiming an *idle* slot is not preemption: its
    ///    request already finished and the client already has its output.
    /// 3. Return prefill and decode id lists, plus the KV trims/clears the engine
    ///    must apply before the next prefill.
    pub fn schedule_step(&self) -> ScheduledBatch {
        // Lock ordering (must be consistent across ALL callers to avoid deadlock):
        //   running_batch → waiting_queue → slots
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
        let mut slots = match self.slots.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("slots lock poisoned: {}", e);
                return ScheduledBatch::default();
            }
        };

        let mut kv_trims: Vec<(i32, usize)> = Vec::new();
        let mut kv_clears: Vec<i32> = Vec::new();
        let mut kv_saves: Vec<(i32, Vec<i32>)> = Vec::new();
        let mut kv_restores: Vec<(i32, Vec<u8>)> = Vec::new();
        // Held across the admission loop rather than re-locked per request. Acquired
        // after `slots`, matching the documented lock order.
        let mut pcache = self.prompt_cache.lock().ok();

        // 1. Retire Finished requests.
        //
        // `park_finished` (called from the engine on the completion path) has already
        // decided each one's fate and recorded it in `park_state`; here we only apply
        // it. A parked request handed its blocks to its slot and left `kv_seq_id = -1`,
        // so it is skipped below and its blocks are NOT freed — that is the whole
        // point: the sequence stays resident as a cache entry.
        let (finished, still_running): (Vec<_>, Vec<_>) = std::mem::take(&mut *running)
            .into_iter()
            .partition(|r| r.is_finished());

        for req in &finished {
            if req.kv_seq_id < 0 {
                continue; // already parked into its slot
            }
            let mut blocks = slots.release(req.kv_seq_id);
            blocks.extend_from_slice(req.page_table.block_ids());
            if !blocks.is_empty() {
                self.kv_cache.free_blocks(&blocks);
                debug!(
                    request_id = req.id,
                    blocks = blocks.len(),
                    "freed KV blocks for finished request"
                );
            }
            kv_clears.push(req.kv_seq_id);
        }
        *running = still_running;

        // 2. Admit from waiting_queue
        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        // Always empty since admission stopped preempting; retained so future
        // preemption sources (priority, growth) can reuse the engine-side clearing.
        let preempted_seq_ids = Vec::new();

        // Forked branches whose parent has not finished prefilling yet. Collected
        // rather than left at the queue head: they are not blocked on *capacity*, so
        // stalling everything behind them would be a self-inflicted head-of-line block.
        // Re-queued in order once the loop ends.
        let mut deferred: Vec<batch::InferenceRequest> = Vec::new();

        'admit: while let Some(mut req) = waiting.pop_front() {
            // A forked branch may only be admitted once its parent's prompt is in the
            // KV — there is nothing to copy before that.
            if let Some(parent_id) = req.fork_parent {
                match running.iter().find(|r| r.id == parent_id) {
                    Some(parent) if parent.state == batch::RequestState::Decoding => {
                        // Parent is prefilled: adopt its prompt wholesale, leaving one
                        // token to decode ([TAG_PROMPT_LOGITS]) so this branch produces
                        // logits of its own.
                        let parent_seq = parent.kv_seq_id;
                        let n_positions = req.n_positions();
                        // Multimodal is excluded: its positions come from image chunks
                        // while `effective_skip` counts `prompt_tokens`, which is empty,
                        // so the copy boundary and the resubmission boundary would not
                        // agree. LoRA too — a branch must not inherit KV computed under
                        // a different adapter.
                        // Same precondition as the affinity path below: an n>1 branch
                        // adopts its parent's KV through `seq_cp`, so a model that
                        // cannot donate cells must prefill it normally instead.
                        let forkable = req.multimodal.is_none()
                            && !req.skip_prefix_cache
                            && self.kv_reuse
                            && self.prefix_reuse_enabled(); // a fork copies: strict flag
                        if forkable && parent_seq >= 0 && n_positions > 0 {
                            req.fork_source = Some((parent_seq, n_positions - 1));
                        } else {
                            req.fork_parent = None; // unusable parent — prefill normally
                        }
                    }
                    // Still prefilling: try again next step.
                    Some(_) => {
                        deferred.push(req);
                        continue 'admit;
                    }
                    // Parent already finished, failed, or was never admitted. Fall back
                    // to an ordinary admission — slot affinity will usually still find
                    // the parent's parked KV, so this degrades to slower, not wrong.
                    None => req.fork_parent = None,
                }
            }

            // A request that could never fit, even into an empty pool, is rejected
            // synchronously by `Scheduler::submit()` before it ever reaches this queue
            // (0.16) — it's a static check (prompt + max_new_tokens vs. total pool size)
            // that doesn't need a scheduler turn. No corresponding check needed here.

            // Pick the slot whose resident KV this prompt can reuse most of.
            //
            // `skip_prefix_cache` (LoRA requests): KV computed under one adapter's
            // weights is invalid input for a different adapter (or none) at the same
            // positions, so such a request must start from a clean slot — see
            // docs/design/lora-support.md.
            // `prefix_reuse_enabled()` belongs here rather than further down, where a
            // first attempt at this fix put it: `allow_reuse` feeds BOTH reuse paths —
            // the slot table's LCP match and the copy-from-a-live-sibling fork below —
            // and gating only the first left the fork setting `prefix_seq_id`, which
            // still reached `llama_memory_seq_cp` and aborted. Reuse is only sound when
            // the model's KV cache can donate cells; deciding to skip prefill and
            // failing to copy afterwards does not degrade gracefully, it marks tokens
            // resident that were never written.
            // Two different permissions, because they need two different things from the
            // model. Inheriting the slot's own KV copies nothing and works on hybrids;
            // copying out of a live sibling needs `seq_cp` and does not.
            let allow_reuse = self.kv_reuse && !req.skip_prefix_cache && self.slot_reuse_enabled();
            let allow_copy = allow_reuse && self.prefix_reuse_enabled();
            let Some(choice) =
                slots.select(&req.prompt_tokens, self.slot_prompt_similarity, allow_reuse)
            else {
                // Every slot is Busy — wait for one to retire.
                waiting.push_front(req);
                break 'admit;
            };

            // Copy-from-a-live-sequence. `select` above only inherits *idle* slots, so
            // when N requests sharing a system prompt arrive together none of them can
            // reuse anything — each prefills and then holds its own copy. Measured: 6
            // concurrent clients behind a 672-token prompt held 264 blocks where ~54
            // would do. A busy sequence cannot be inherited, but it can be copied from:
            // `seq_cp` shares cells under `kv_unified` rather than duplicating them.
            //
            // The search runs over `running`, not over the slot table: `claim` clears a
            // slot's token list because while a request owns the sequence *it* is the
            // source of truth. Only a request that is already `Decoding` has its whole
            // prompt in the KV and is therefore copyable.
            if allow_copy && req.fork_source.is_none() && req.multimodal.is_none() {
                let best = running
                    .iter()
                    .filter(|r| r.kv_seq_id >= 0 && r.multimodal.is_none() && !r.skip_prefix_cache)
                    .map(|r| {
                        let lcp = common_prefix_len(&r.prompt_tokens, &req.prompt_tokens);
                        (r.kv_seq_id, lcp, r.state)
                    })
                    .filter(|&(_, lcp, _)| {
                        lcp > choice.lcp
                            && (lcp as f32 / req.prompt_tokens.len().max(1) as f32)
                                >= self.slot_prompt_similarity
                    })
                    .max_by_key(|&(_, lcp, _)| lcp);

                if let Some((donor_seq, lcp, state)) = best {
                    if state == batch::RequestState::Decoding {
                        req.fork_source = Some((donor_seq, lcp));
                    } else {
                        // Still prefilling: nothing to copy yet. Deferring keeps the
                        // queue moving — it is blocked on a sibling, not on capacity,
                        // so leaving it at the head would be a self-inflicted
                        // head-of-line block.
                        deferred.push(req);
                        continue 'admit;
                    }
                }
            }

            // The host-RAM cache is consulted only when it can beat what the live
            // slots already offer: restoring a state that covers less than `choice.lcp`
            // would be a memcpy that loses information. On a hit the chosen slot's KV
            // is replaced wholesale, so its own resident tokens no longer apply.
            //
            // EXCEPT when the slot's offer cannot actually be taken up. On a KV that
            // cannot be rolled back, reaching `choice.lcp` means trimming across the
            // whole generated reply, which the cache refuses — so the slot's LCP is a
            // promise the engine cannot keep, and a *tied* cache entry beats it because
            // a restore has no rollback to perform. Requiring the cache to strictly win
            // is what left Qwen3.5 with 20 slot hits, 20 refused trims and
            // `cached_tokens` 0 while a usable checkpoint sat in RAM unread.
            // Taking up an offer means rewinding the sequence from everything it holds
            // back to the divergence point. On a bounded-rollback cache that distance
            // has a hard limit, and it must be checked HERE: `trim_sequence` reports
            // success for an over-long rollback once the sequence's tail has been
            // invalidated, so the engine's after-the-fact guard never fires and the
            // request silently decodes from a state it never rewound to. Measured on
            // Qwen3.8-27B: slot holding 102 tokens, offer of 42, `n_rs_seq = 4` — a
            // 60-token rollback reported `trimmed=true` and every later request
            // returned a bare EOS. `usize::MAX` for attention caches makes this a no-op.
            let budget = self.rollback_budget();
            let slot_resident = slots.resident_len(choice.index);
            let slot_lcp = if slot_resident.saturating_sub(choice.lcp) > budget {
                debug!(
                    request_id = req.id,
                    seq_id = slots.seq_id_at(choice.index),
                    offered = choice.lcp,
                    slot_resident,
                    budget,
                    "declining slot offer — rewinding this far exceeds the cache's \
                     rollback budget; prefilling from scratch"
                );
                0
            } else {
                choice.lcp
            };

            let cache_floor = if self.prefix_reuse_enabled() {
                slot_lcp
            } else {
                slot_lcp.saturating_sub(1)
            };
            let mut lcp = slot_lcp;
            // What the sequence will actually hold when the trim runs. A restore
            // replaces the slot's KV wholesale, so on a hit this becomes the blob's
            // extent rather than the slot's.
            let mut resident_at_trim = slot_resident;
            // …and how far back it may then be rewound. A sequence that has been
            // decoding in place still has its per-token state snapshots, so it gets the
            // full budget. A *restored* one does not: `state_seq_load` writes the
            // recurrent state but not the snapshot history the rollback indexes into, so
            // rewinding even one token off a fresh blob reads a snapshot belonging to
            // whatever occupied those slots before. Measured on Qwen3.8-27B: restore a
            // 43-token checkpoint, trim by 1, and the reply is a bare EOS.
            let mut budget_at_trim = budget;

            // llama-server's [TAG_PROMPT_LOGITS] guard (server-context.cpp:3356-3361):
            // if the prompt is *entirely* resident there is nothing left to decode and
            // no logits would be produced, so the divergence point steps one token back.
            // That step-back is part of the rollback distance, so the budget has to be
            // checked against this, not against the raw LCP.
            let n_positions = req.n_positions();
            let divergence = |lcp: usize| {
                let n = lcp.min(n_positions);
                if n > 0 && n == n_positions {
                    n - 1
                } else {
                    n
                }
            };

            if allow_reuse {
                if let Some(hit) = pcache
                    .as_mut()
                    .and_then(|c| c.take_best(&req.prompt_tokens, cache_floor))
                {
                    // On a bounded cache a restored blob may not be rewound AT ALL, so
                    // the only sound checkpoint is one that lands exactly on the
                    // divergence point — which is the multi-turn case it exists for,
                    // where the next prompt extends the cached one. An unbounded
                    // (attention) cache keeps the old behaviour: it reaches any position
                    // by dropping cells, so a tied checkpoint is still worth restoring.
                    //
                    // Checked before the restore, not after: the blob is ~160 MB on a
                    // 27B, far too expensive to write into the sequence and then discard.
                    let restore_allowance = if budget == usize::MAX { usize::MAX } else { 0 };
                    if hit.resident.saturating_sub(divergence(hit.matched)) <= restore_allowance {
                        kv_restores.push((slots.seq_id_at(choice.index), hit.data));
                        slots.set_resident(
                            choice.index,
                            req.prompt_tokens[..hit.resident.min(req.prompt_tokens.len())].to_vec(),
                        );
                        lcp = hit.matched;
                        resident_at_trim = hit.resident;
                        budget_at_trim = 0;
                    } else {
                        debug!(
                            request_id = req.id,
                            matched = hit.matched,
                            resident = hit.resident,
                            "declining prompt-cache checkpoint — a restored blob has no \
                             snapshot history to rewind through"
                        );
                    }
                }
            }

            // Token-exact reuse, stepped back by the [TAG_PROMPT_LOGITS] guard above.
            let mut n_past = divergence(lcp);

            // The rollback the engine will be asked for, measured against the FINAL
            // divergence point — the `[TAG_PROMPT_LOGITS]` step-back above is part of
            // the distance, so this cannot be checked before it. Beyond the budget the
            // offer is dropped entirely: `n_past = 0` turns the queued trim into a full
            // clear and the prompt is prefilled from scratch. Slower, never wrong.
            if resident_at_trim.saturating_sub(n_past) > budget_at_trim {
                debug!(
                    request_id = req.id,
                    resident_at_trim,
                    n_past,
                    budget = budget_at_trim,
                    "declining reuse — the rollback it needs exceeds the cache's budget"
                );
                n_past = 0;
            }

            // Blocks are a budget, not addresses (see slots.rs): the request inherits
            // whatever the slot already holds and tops up only the difference.
            //
            // Resolve the donor's shared blocks BEFORE sizing the reservation. Charging
            // for the shared prefix and handing it back afterwards leaves the steady
            // state correct but the admission decision wrong: a burst can be turned away
            // for capacity it was never going to hold. Sizing for what the request will
            // actually own is what lets the sharing widen concurrency rather than only
            // shrink the pool.
            let block_size = self.kv_cache.block_size();
            let fork_share: Vec<crate::kv_cache::BlockId> = match req.fork_source {
                Some((parent_seq, fork_skip)) => {
                    let n_past = fork_skip.min(req.n_positions().saturating_sub(1));
                    // Whole blocks only. The block straddling the divergence point also
                    // covers positions this request will write, so it must stay private;
                    // this floor is what keeps a shared block write-free, which in turn
                    // is why `run_decode` needs no copy-on-write.
                    let want = n_past / block_size;
                    let got: Vec<_> = running
                        .iter()
                        .find(|r| r.kv_seq_id == parent_seq)
                        .map(|d| {
                            d.page_table
                                .block_ids()
                                .iter()
                                .take(want)
                                .copied()
                                .collect()
                        })
                        .unwrap_or_default();
                    if want > 0 && got.len() != want {
                        // The donor holds fewer blocks than its prefix implies. Reserving
                        // short here would under-budget a live request, so drop the copy
                        // and admit normally — slower, never wrong.
                        req.fork_source = None;
                        Vec::new()
                    } else {
                        got
                    }
                }
                None => Vec::new(),
            };

            let needed = self.blocks_needed(&req);
            let needed_own = needed.saturating_sub(fork_share.len());
            let have = slots.blocks_at(choice.index);
            let top_up = needed_own.saturating_sub(have);

            // Make room by reclaiming idle slots — LRU first, never the slot we just
            // chose, and never a Busy one. Not preemption; see SlotTable::reclaim_lru.
            while top_up > 0 && !self.kv_cache.can_allocate(top_up) {
                let Some((victim_seq, victim_tokens, victim_blocks)) =
                    slots.reclaim_lru(choice.index)
                else {
                    break;
                };
                // Serialise what it held to host RAM instead of throwing it away —
                // that is the whole point of the RAM cache: the conversation stays
                // reusable without continuing to occupy a GPU block.
                if pcache.as_ref().is_some_and(|c| c.enabled()) && !victim_tokens.is_empty() {
                    kv_saves.push((victim_seq, victim_tokens));
                }
                self.kv_cache.free_blocks(&victim_blocks);
                kv_clears.push(victim_seq);
                debug!(
                    seq_id = victim_seq,
                    blocks = victim_blocks.len(),
                    "reclaimed idle slot to make room"
                );
            }

            if top_up > 0 && !self.kv_cache.can_allocate(top_up) {
                // Still short — wait for capacity (FIFO head-of-line). Running
                // requests keep their reservations.
                waiting.push_front(req);
                break 'admit;
            }

            let new_ids = if top_up > 0 {
                match self.kv_cache.allocate(top_up) {
                    Ok(ids) => ids,
                    Err(_) => {
                        waiting.push_front(req);
                        break 'admit;
                    }
                }
            } else {
                Vec::new()
            };

            let id = req.id;
            let (seq_id, mut blocks) = slots.claim(choice.index, id);
            // Give back any surplus the previous occupant held beyond this request's
            // reservation, so a short prompt after a long one doesn't pin the pool.
            if blocks.len() > needed_own {
                let surplus = blocks.split_off(needed_own);
                self.kv_cache.free_blocks(&surplus);
            }
            blocks.extend(new_ids);

            // The shared prefix goes first, so the page table still reads in position
            // order; a reference is taken now that the request is certain to be admitted.
            if !fork_share.is_empty() {
                for &b in &fork_share {
                    self.kv_cache.retain_block(b);
                }
                debug!(
                    request_id = req.id,
                    shared_blocks = fork_share.len(),
                    own_blocks = blocks.len(),
                    "reserving only the blocks past the donor's shared prefix"
                );
                let mut merged = fork_share;
                merged.extend(blocks);
                blocks = merged;
            }
            req.page_table = PageTable::new(blocks);
            req.kv_seq_id = seq_id;
            // A forked branch overrides whatever the slot offered: copying the
            // parent's whole prompt beats any partial LCP match, and the two are
            // mutually exclusive — the copy overwrites positions 0..n.
            if let Some((parent_seq, fork_skip)) = req.fork_source {
                // [TAG_PROMPT_LOGITS] again: copying the *entire* prompt would leave
                // nothing to decode and no logits, so keep one token back.
                n_past = fork_skip.min(req.n_positions().saturating_sub(1));
                req.prefix_seq_id = Some(parent_seq);
                // The destination now holds the copied prefix; record it or a later
                // LCP match would be computed against tokens that are not resident.
                let resident = req.prompt_tokens[..n_past.min(req.prompt_tokens.len())].to_vec();
                slots.set_resident(choice.index, resident);
            } else {
                req.prefix_seq_id = None;
            }
            req.skip_prefix_tokens = n_past;
            req.prefill_pos = n_past;
            req.stop_reason = None;
            req.state = batch::RequestState::Prefilling;

            // Drop everything the slot holds past the divergence point before the
            // next prefill writes there (server-context.cpp:3392-3399). Stale cells
            // beyond n_past would collide with this request's own positions.
            kv_trims.push((seq_id, n_past));

            if n_past > 0 {
                self.prefix_hits.fetch_add(1, Ordering::Relaxed);
                info!(
                    request_id = id,
                    seq_id,
                    cached_tokens = n_past,
                    prompt_tokens = req.n_positions(),
                    "slot prefix hit — skipping prefill of resident tokens"
                );
            } else {
                self.prefix_misses.fetch_add(1, Ordering::Relaxed);
                info!(request_id = id, seq_id, "request admitted to batch");
            }
            running.push(req);
        }

        // Put deferred branches back at the front, in their original order, so they
        // are reconsidered next step ahead of anything that arrived meanwhile.
        for req in deferred.into_iter().rev() {
            waiting.push_front(req);
        }

        // 4. Build the prefill and decode lists from the running batch. A request stays
        //    `Prefilling` across steps until its prompt is fully chunked into the KV, so
        //    it is re-emitted to `prefill` each step (both freshly admitted and
        //    still-in-progress); `Decoding` requests generate one token per step.
        for req in running.iter() {
            match req.state {
                batch::RequestState::Prefilling => prefill.push(req.id),
                batch::RequestState::Decoding => decode.push(req.id),
                _ => {}
            }
        }

        ScheduledBatch {
            prefill,
            decode,
            preempted_seq_ids,
            kv_trims,
            kv_clears,
            kv_saves,
            kv_restores,
        }
    }

    /// Advance a request's prefill cursor after a chunk was submitted to the model.
    /// Called every prefill step; the request stays `Prefilling` (and is re-emitted
    /// to the prefill batch) until its final chunk is sampled by `handle_logits`.
    pub fn advance_prefill(&self, req_id: u64, new_prefill_pos: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.prefill_pos = new_prefill_pos;
                    break;
                }
            }
        }
    }

    /// Record that context rolling discarded `n_discard` of a request's oldest KV
    /// tokens. Reduces its logical context length (via `rolled_tokens`) so the next
    /// decode position matches the shifted KV cache.
    pub fn record_context_roll(&self, req_id: u64, n_discard: usize) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.id == req_id {
                    req.rolled_tokens += n_discard;
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

    /// The sequence id and prompt tokens of a request that has just finished prefill.
    ///
    /// Returns `None` once the request has generated anything, so a checkpoint can only
    /// ever capture the prompt boundary.
    pub fn prefilled_sequence(&self, req_id: u64) -> Option<(i32, Vec<i32>)> {
        let running = self.running_batch.lock().ok()?;
        running
            .iter()
            .find(|r| r.id == req_id && r.generated_tokens <= 1)
            .map(|r| (r.kv_seq_id, r.prompt_tokens.clone()))
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

    /// Undo the bookkeeping for a prompt-cache restore that failed at the model layer.
    ///
    /// The scheduler has already told the request how many tokens it may skip, on the
    /// assumption the restore would succeed. If it did not, the request would prefill
    /// on top of cells that were never written. Resetting it to prefill from token 0
    /// costs a re-prefill and nothing else — the alternative is silently wrong output.
    pub fn invalidate_restore(&self, seq_id: i32) {
        if let Ok(mut running) = self.running_batch.lock() {
            for req in running.iter_mut() {
                if req.kv_seq_id == seq_id {
                    req.skip_prefix_tokens = 0;
                    req.prefill_pos = 0;
                }
            }
        }
        if let Ok(mut slots) = self.slots.lock() {
            slots.forget_resident(seq_id);
        }
    }

    /// Park a finished request's sequence so its KV stays resident and reusable.
    ///
    /// This is the counterpart to llama-server keeping `slot.prompt.tokens` after a
    /// task completes (`server-context.cpp:489`). The whole logical sequence —
    /// **prompt *and* generated tokens** — becomes the slot's resident token list, so
    /// the next turn of a conversation (whose prompt contains the previous reply)
    /// matches well past where the previous prompt ended. The old block-hash cache
    /// discarded the generated tail, which is why multi-turn chat never hit it.
    ///
    /// The request keeps no blocks: they transfer to the slot, and `kv_seq_id` is set
    /// to `-1` so `schedule_step`'s retire pass knows not to free them.
    ///
    /// Returns `true` if the sequence was parked. `false` means the caller must clear
    /// the llama.cpp sequence itself — the KV is not safe to reuse:
    ///
    /// * **`--kv-reuse false`** — reuse disabled outright.
    /// * **context-rolled** (`rolled_tokens > 0`) — rolling discards the oldest KV
    ///   window and shifts the rest, so resident positions no longer line up with the
    ///   token list; a later LCP match would read the wrong cells.
    /// * **LoRA** (`skip_prefix_cache`) — KV computed under one adapter's weights is
    ///   invalid input for another (docs/design/lora-support.md).
    /// * **multimodal** — `prompt_tokens` is empty for these (positions come from
    ///   image chunks), so the token list can't describe what's resident.
    ///
    /// Lock ordering: running_batch → slots (matches schedule_step).
    pub fn park_finished(&self, req_id: u64) -> bool {
        if !self.kv_reuse {
            return false;
        }
        let mut running = match self.running_batch.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        let mut slots = match self.slots.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };

        let Some(req) = running.iter_mut().find(|r| r.id == req_id) else {
            return false;
        };
        if req.kv_seq_id < 0
            || req.rolled_tokens > 0
            || req.skip_prefix_cache
            || req.multimodal.is_some()
        {
            return false;
        }

        // What llama.cpp actually holds for this sequence: the prompt it prefilled
        // followed by every token it generated.
        let mut resident = req.prompt_tokens.clone();
        resident.extend_from_slice(&req.generated_token_ids);

        let seq_id = req.kv_seq_id;
        let blocks = std::mem::take(&mut req.page_table).entries;
        if !slots.park(seq_id, resident, blocks.clone()) {
            // Unknown seq_id (defensive) — hand the blocks back rather than leaking
            // them. `page_table` was already emptied above, so free the copy we hold.
            self.kv_cache.free_blocks(&blocks);
            return false;
        }

        // Zero out seq ownership so schedule_step's retire pass won't double-free.
        req.kv_seq_id = -1;
        debug!(
            request_id = req_id,
            seq_id,
            resident_tokens = req.prompt_tokens.len() + req.generated_token_ids.len(),
            "parked sequence as reusable idle slot"
        );
        true
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

    /// Number of slots currently holding reusable KV (the analogue of the old
    /// prefix cache's entry count).
    pub fn prefix_cache_size(&self) -> usize {
        self.slots
            .lock()
            .map(|s| s.count(|slot| slot.state == super::slots::SlotState::Idle))
            .unwrap_or(0)
    }
}
