use std::sync::Arc;

use anyhow::Result;

use crate::scheduler::{StopReason, Token};

use super::model::{InferenceRequestForModel, Logits};
use super::InferenceEngine;

impl InferenceEngine {
    /// Main inference loop.
    pub async fn run_loop(self: Arc<Self>) -> Result<()> {
        let engine = self.clone();
        // Delta trackers: AtomicU64 counters in the scheduler are monotonically increasing.
        // We increment the Prometheus IntCounters by the step delta each loop iteration.
        let mut last_prefix_hits: u64 = 0;
        let mut last_prefix_misses: u64 = 0;
        let mut last_spec_proposed: u64 = 0;
        let mut last_spec_accepted: u64 = 0;
        let mut last_bisection_retries: u64 = 0;

        loop {
            let batch = engine.scheduler.schedule_step();

            // Refresh gauges and propagate counter deltas every scheduling step.
            if let Some(m) = &engine.metrics {
                use std::sync::atomic::Ordering;
                m.kv_cache_usage_ratio
                    .set(engine.kv_cache.memory_usage() as f64);
                m.queue_depth.set(engine.scheduler.queue_depth() as i64);
                m.active_requests
                    .set(engine.scheduler.active_requests() as i64);

                let cur_hits = engine.scheduler.prefix_hits.load(Ordering::Relaxed);
                let cur_misses = engine.scheduler.prefix_misses.load(Ordering::Relaxed);
                let dh = cur_hits.saturating_sub(last_prefix_hits);
                let dm = cur_misses.saturating_sub(last_prefix_misses);
                if dh > 0 {
                    m.prefix_cache_hits_total.inc_by(dh);
                }
                if dm > 0 {
                    m.prefix_cache_misses_total.inc_by(dm);
                }
                last_prefix_hits = cur_hits;
                last_prefix_misses = cur_misses;

                let (cur_proposed, cur_accepted) = engine.spec_stats();
                let dp = cur_proposed.saturating_sub(last_spec_proposed);
                let da = cur_accepted.saturating_sub(last_spec_accepted);
                if dp > 0 {
                    m.spec_tokens_proposed_total.inc_by(dp);
                }
                if da > 0 {
                    m.spec_tokens_accepted_total.inc_by(da);
                }
                if cur_proposed > 0 {
                    m.spec_acceptance_ratio
                        .set(cur_accepted as f64 / cur_proposed as f64);
                }
                last_spec_proposed = cur_proposed;
                last_spec_accepted = cur_accepted;

                let cur_bisection_retries = engine.model.bisection_retry_count();
                let db = cur_bisection_retries.saturating_sub(last_bisection_retries);
                if db > 0 {
                    m.decode_bisection_retries_total.inc_by(db);
                }
                last_bisection_retries = cur_bisection_retries;
            }

            for seq_id in &batch.preempted_seq_ids {
                engine.model.clear_sequence(*seq_id);
            }

            // Apply the scheduler's KV bookkeeping BEFORE any prefill runs this step.
            // The scheduler has no model handle, so it records intent and we execute
            // it here. Ordering is load-bearing throughout:
            //
            //   saves → clears → restores → trims
            //
            // A save must read the sequence before the clear wipes it. A restore must
            // land after the clears (its destination may itself have just been
            // reclaimed) and before the trims, because the trim bounds the *restored*
            // state at the new request's divergence point. Getting this order wrong
            // corrupts context silently — garbage output, not a crash.
            for (seq_id, tokens) in &batch.kv_saves {
                match engine.model.state_seq_save(*seq_id) {
                    Ok(data) => {
                        tracing::debug!(
                            seq_id,
                            bytes = data.len(),
                            tokens = tokens.len(),
                            "serialised sequence to the host-RAM prompt cache"
                        );
                        engine.scheduler.store_prompt_state(tokens.clone(), data);
                    }
                    // A failed save costs a re-prefill later, nothing more — the
                    // sequence is being discarded either way.
                    Err(e) => tracing::debug!(seq_id, "prompt-cache save skipped: {e}"),
                }
            }
            for seq_id in &batch.kv_clears {
                engine.model.clear_sequence(*seq_id);
            }
            for (seq_id, data) in &batch.kv_restores {
                if let Err(e) = engine.model.state_seq_load(*seq_id, data) {
                    // The scheduler already told the request how much it may skip, so
                    // a failed restore would leave it reading cells that were never
                    // written. Wipe the sequence and let it re-prefill from scratch:
                    // the following trim then bounds an empty sequence, which is a
                    // no-op, and the request is merely slower rather than wrong.
                    tracing::warn!(seq_id, "prompt-cache restore failed: {e} — re-prefilling");
                    engine.model.clear_sequence(*seq_id);
                    engine.scheduler.invalidate_restore(*seq_id);
                }
            }
            for (seq_id, keep_from) in &batch.kv_trims {
                // A refused trim is the recurrent/hybrid cache saying the rollback fell
                // outside its snapshot window. Proceeding would leave the request
                // skipping a prefix that is no longer where it thinks it is, so take the
                // same escape hatch the prompt-cache restore failure uses: drop the
                // sequence and prefill it from scratch. Slower, never wrong.
                if !engine.model.trim_sequence(*seq_id, *keep_from) {
                    tracing::warn!(
                        seq_id,
                        keep_from,
                        "KV trim refused (rollback outside the cache's window) — re-prefilling"
                    );
                    engine.model.clear_sequence(*seq_id);
                    engine.scheduler.invalidate_restore(*seq_id);
                }
            }

            if batch.is_empty() {
                engine.scheduler.wait_for_work().await;
                continue;
            }

            let prefill_ids = batch.prefill.clone();
            let decode_ids = batch.decode.clone();

            if !prefill_ids.is_empty() {
                match engine.run_prefill(&prefill_ids).await {
                    Ok(prefill_results) => {
                        // Prefill yields one token per request; wrap each so handle_logits
                        // sees the same per-request token-list shape as decode.
                        let wrapped: Vec<(u64, Vec<Logits>)> = prefill_results
                            .into_iter()
                            .map(|(id, l)| (id, vec![l]))
                            .collect();
                        engine.handle_logits(&wrapped, true).await?;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "prefill failed (KV cache full?): {} — stopping {} request(s) with EngineError",
                            e,
                            prefill_ids.len()
                        );
                        // Send an explicit terminal token before the sequence is cleared and
                        // the request is marked finished — otherwise `response_tx` is only
                        // dropped later, which closes the channel with no message and the
                        // HTTP handler reports a fake empty 200 instead of an error.
                        let failed = engine.scheduler.get_running(&prefill_ids);
                        for req in &failed {
                            let _ = req.response_tx.send(Token {
                                id: req.id,
                                token_id: -1,
                                text: String::new(),
                                is_eos: true,
                                stop_reason: Some(StopReason::EngineError),
                                logprob: None,
                                cached_tokens: 0,
                            });
                            // Clear KV before the seq_id returns to the pool — a failed
                            // llama_decode leaves partial cells that poison the next occupant.
                            if req.kv_seq_id >= 0 {
                                engine.model.clear_sequence(req.kv_seq_id);
                            }
                        }
                        for req_id in &prefill_ids {
                            engine
                                .scheduler
                                .mark_finished(*req_id, StopReason::EngineError);
                            engine.model.free_grammar(*req_id);
                        }
                    }
                }
            }

            if !decode_ids.is_empty() {
                // Before decoding, roll any sequence whose KV window is full so it can
                // continue past n_ctx instead of failing (context shift).
                engine.roll_full_contexts(&decode_ids).await;
                let mut decode_result = engine.run_decode(&decode_ids).await;
                // Reactive context roll: bisection retry (batch.rs) already shrank the
                // batch as far as it can — if it still failed because exactly one
                // request has no KV slot even alone, and that request has old context
                // it can afford to discard, roll it and retry the whole batch once
                // more before giving up. See docs/design/reactive-context-rolling.md.
                if let Err(e) = &decode_result {
                    if let Some(true) = engine.try_reactive_roll(&decode_ids, e).await {
                        decode_result = engine.run_decode(&decode_ids).await;
                    }
                }
                match decode_result {
                    Ok(decode_results) => {
                        engine.handle_logits(&decode_results, false).await?;
                    }
                    Err(e) => {
                        // KV cache exhausted or llama_decode failure — stop all affected
                        // requests gracefully instead of crashing the engine loop.
                        tracing::warn!(
                            "decode failed (KV cache full?): {} — stopping {} request(s) with EngineError",
                            e,
                            decode_ids.len()
                        );
                        // Same explicit-terminal-token + poisoned-sequence guard as the
                        // prefill error path (see comment there).
                        let failed = engine.scheduler.get_running(&decode_ids);
                        for req in &failed {
                            let _ = req.response_tx.send(Token {
                                id: req.id,
                                token_id: -1,
                                text: String::new(),
                                is_eos: true,
                                stop_reason: Some(StopReason::EngineError),
                                logprob: None,
                                cached_tokens: 0,
                            });
                            if req.kv_seq_id >= 0 {
                                engine.model.clear_sequence(req.kv_seq_id);
                            }
                        }
                        for req_id in &decode_ids {
                            engine
                                .scheduler
                                .mark_finished(*req_id, StopReason::EngineError);
                            engine.model.free_grammar(*req_id);
                        }
                    }
                }
            }
        }
    }

    /// Roll the KV window of any decoding request that has filled the context.
    ///
    /// When context shift is enabled (`context_shift = Some(n_keep)`) and the model's
    /// KV cache is shiftable, a request whose `context_len` has reached `n_ctx` has its
    /// oldest half (after the preserved `n_keep`-token head) discarded and the survivors
    /// shifted down, then its logical length is reduced by the same amount so subsequent
    /// decode positions line up with the shifted KV. Recurrent/hybrid caches (not
    /// shiftable) are skipped — those requests hit the decode error path and stop with
    /// `Length`, the pre-context-shift behavior.
    async fn roll_full_contexts(&self, decode_ids: &[u64]) {
        let Some(n_keep_cfg) = self.context_shift else {
            return;
        };
        // Recurrent/hybrid caches can't shift positions — leave today's behavior.
        if !self.supports_prefix_cache {
            return;
        }
        let n_ctx = self.model.context_len() as usize;
        if n_ctx == 0 {
            return;
        }
        // Roll with headroom for the largest possible next step (a speculative verify
        // batch writes up to draft_len + 1 cells): triggering exactly AT n_ctx is too
        // late — the boundary-crossing step fails before the roll can fire.
        let reserve = self.speculative.as_ref().map(|(_, d)| d + 1).unwrap_or(1);
        let threshold = n_ctx.saturating_sub(reserve);
        for req in self.scheduler.get_running(decode_ids) {
            let ctx_len = req.context_len();
            if ctx_len < threshold || req.kv_seq_id < 0 {
                continue;
            }
            // Preserve the head; discard half of what remains (at least one token). Keep
            // at least one token beyond the head so the shifted tail is non-empty.
            let n_keep = n_keep_cfg.min(n_ctx.saturating_sub(1));
            let n_discard = (ctx_len.saturating_sub(n_keep) / 2).max(1);
            let seq_id = req.kv_seq_id;
            let model = self.model.clone();
            let res =
                tokio::task::spawn_blocking(move || model.roll_context(seq_id, n_keep, n_discard))
                    .await;
            match res {
                Ok(Ok(())) => {
                    self.scheduler.record_context_roll(req.id, n_discard);
                    tracing::info!(
                        request_id = req.id,
                        seq_id,
                        n_keep,
                        n_discard,
                        ctx_len,
                        n_ctx,
                        "rolled full context window to keep generating"
                    );
                }
                Ok(Err(e)) => tracing::warn!(
                    request_id = req.id,
                    "context roll failed: {e} — request will stop with Length"
                ),
                Err(e) => {
                    tracing::warn!(request_id = req.id, "context roll join error: {e}")
                }
            }
        }
    }

    /// If `err` failed a decode step, that request has old context worth
    /// discarding, and one more attempt at the full batch might succeed:
    /// attempt exactly one reactive context roll on the specific request that
    /// `batch.rs` reported as `KvCacheFullAtMinimum`, and let the caller retry.
    ///
    /// Returns `None` when `err` isn't that specific signal at all (a normal
    /// fatal decode error — unchanged behavior). Returns `Some(false)` when it
    /// is, but rolling doesn't apply (context-shift disabled, model can't
    /// shift, or the request doesn't have enough context left to make
    /// discarding meaningful) — the caller falls through to `EngineError`
    /// exactly as it did before this feature existed. Returns `Some(true)`
    /// only after a real, successful roll, which the caller should retry once.
    async fn try_reactive_roll(&self, decode_ids: &[u64], err: &anyhow::Error) -> Option<bool> {
        let req_id = err
            .downcast_ref::<super::model::KvCacheFullAtMinimum>()?
            .req_id;
        let n_keep_cfg = self.context_shift?;
        if !self.supports_prefix_cache {
            return Some(false);
        }
        let n_ctx = self.model.context_len() as usize;
        let running = self.scheduler.get_running(decode_ids);
        let Some(req) = running.iter().find(|r| r.id == req_id) else {
            return Some(false);
        };
        if req.kv_seq_id < 0 {
            return Some(false);
        }
        let Some((n_keep, n_discard)) = reactive_roll_amounts(n_keep_cfg, n_ctx, req.context_len())
        else {
            return Some(false);
        };
        let seq_id = req.kv_seq_id;
        let model = self.model.clone();
        let res =
            tokio::task::spawn_blocking(move || model.roll_context(seq_id, n_keep, n_discard))
                .await;
        match res {
            Ok(Ok(())) => {
                self.scheduler.record_context_roll(req_id, n_discard);
                tracing::info!(
                    request_id = req_id,
                    seq_id,
                    n_keep,
                    n_discard,
                    n_ctx,
                    "reactively rolled context after KV-cache-full at minimum batch size — retrying decode"
                );
                Some(true)
            }
            Ok(Err(e)) => {
                tracing::warn!(request_id = req_id, "reactive context roll failed: {e}");
                Some(false)
            }
            Err(e) => {
                tracing::warn!(request_id = req_id, "reactive context roll join error: {e}");
                Some(false)
            }
        }
    }

    pub(super) async fn run_prefill(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);
        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: r.prompt_tokens.clone(),
                last_token: r.last_token,
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
                kv_seq_id: r.kv_seq_id,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                top_k: r.sampling.top_k,
                repetition_penalty: r.sampling.repetition_penalty,
                frequency_penalty: r.sampling.frequency_penalty,
                presence_penalty: r.sampling.presence_penalty,
                repeat_last_n: r.sampling.repeat_last_n,
                top_n_sigma: r.sampling.top_n_sigma,
                min_keep: r.sampling.min_keep,
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: r.skip_prefix_tokens,
                prefix_seq_id: r.prefix_seq_id,
                prefill_pos: r.prefill_pos,
                grammar: r.sampling.grammar.clone(),
                min_p: r.sampling.min_p,
                min_tokens: r.sampling.min_tokens,
                logit_bias: r.sampling.logit_bias.clone(),
                multimodal: r.multimodal.clone(),
                lora: r.lora.clone(),
                needs_logits: r.sampling.logprobs.is_some(),
            })
            .collect();

        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        let max_chunk = self.max_prefill_chunk;
        let raw = tokio::task::spawn_blocking(move || {
            model.prefill_sync(&req_ids_vec, &model_requests, max_chunk)
        })
        .await
        .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

        // NOTE: `prefix_seq_id` (a source sequence to `seq_cp` from before prefill) is
        // currently never set — a request now inherits its predecessor's seq_id
        // directly instead of copying out of it, so no post-prefill cleanup is needed.
        // The field and its handler in `llama_cpp/batch.rs` stay for the shared-prefill
        // fork of `n>1`/`best_of`, where the source is a *live* sibling slot that must
        // NOT be cleared here. See docs/design/llama-server-gap-analysis.md §0.3.

        // Advance each request's prefill cursor. A request only carries `logits` (and a
        // non-zero `tokens_in_kv`) on its FINAL chunk; intermediate chunks just move the
        // cursor forward and stay `Prefilling`, so they are re-emitted next step. Only
        // completed requests reach `handle_logits` (which samples the first token and
        // transitions them to `Decoding`).
        let mut result = Vec::with_capacity(raw.len());
        for step in raw {
            self.scheduler
                .advance_prefill(step.req_id, step.prefill_pos);
            if step.tokens_in_kv > 0 {
                self.scheduler
                    .set_prefilled_tokens(step.req_id, step.tokens_in_kv);
            }
            if let Some(logits) = step.logits {
                result.push((step.req_id, logits));
            }
        }

        Ok(result)
    }

    pub(super) async fn run_decode(&self, req_ids: &[u64]) -> Result<Vec<(u64, Vec<Logits>)>> {
        // There is deliberately NO copy-on-write here, and that is load-bearing rather
        // than an omission.
        //
        // fox's blocks are an admission budget, not addresses — they are never handed to
        // llama.cpp (see scheduler/slots.rs). When a request copies a shared prefix, the
        // cells really are shared inside llama.cpp: `seq_cp` under `kv_unified` shares
        // them rather than duplicating the buffer. So privatising a block here would
        // allocate budget for memory that nobody occupies, re-inflating exactly the
        // over-count the sharing exists to remove — while copying no KV, because there is
        // no KV at this layer to copy.
        //
        // What makes that safe is the invariant enforced where the sharing is set up: only
        // WHOLE blocks below `n_past` are shared, so the block straddling the divergence
        // point stays private. A shared block therefore never receives a write, and CoW
        // has nothing to protect. If that floor-division ever becomes a `div_ceil`, this
        // reasoning collapses and two live sequences start writing the same budgeted
        // block.

        let requests = self.scheduler.get_running(req_ids);
        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: r.prompt_tokens.clone(),
                last_token: r.last_token,
                generated_tokens: r.generated_tokens,
                max_new_tokens: r.max_new_tokens,
                context_len: r.context_len(),
                kv_seq_id: r.kv_seq_id,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                top_k: r.sampling.top_k,
                repetition_penalty: r.sampling.repetition_penalty,
                frequency_penalty: r.sampling.frequency_penalty,
                presence_penalty: r.sampling.presence_penalty,
                repeat_last_n: r.sampling.repeat_last_n,
                top_n_sigma: r.sampling.top_n_sigma,
                min_keep: r.sampling.min_keep,
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: 0,
                prefix_seq_id: None,
                prefill_pos: r.prefill_pos,
                grammar: r.sampling.grammar.clone(),
                min_p: r.sampling.min_p,
                min_tokens: r.sampling.min_tokens,
                logit_bias: r.sampling.logit_bias.clone(),
                multimodal: r.multimodal.clone(),
                lora: r.lora.clone(),
                needs_logits: r.sampling.logprobs.is_some(),
            })
            .collect();
        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();

        // Speculative fast path: a single decoding request with no grammar, when enabled.
        // Speculation helps most at low concurrency; multi-request batches decode normally.
        if let (Some((proposer, draft_len)), [only_id]) = (&self.speculative, req_ids) {
            let no_grammar = model_requests
                .first()
                .map(|r| r.grammar.is_none())
                .unwrap_or(false);
            if no_grammar {
                let only_id = *only_id;
                let draft_len = *draft_len;
                let proposer = proposer.clone();
                let request = model_requests.into_iter().next().unwrap();
                // Proposing and verifying both run inside spawn_blocking: n-gram
                // proposing is cheap CPU work, but a draft-model proposer makes its
                // own llama.cpp FFI call, which must never run on the async executor.
                let (committed, proposed) = tokio::task::spawn_blocking(move || {
                    let mut seq = Vec::with_capacity(
                        request.prompt_tokens.len() + request.generated_token_ids.len(),
                    );
                    seq.extend_from_slice(&request.prompt_tokens);
                    seq.extend_from_slice(&request.generated_token_ids);
                    let drafts = proposer.propose(only_id, &seq, draft_len);
                    let proposed = drafts.len();
                    let committed = model.speculative_decode_sync(only_id, &request, drafts)?;
                    anyhow::Ok((committed, proposed))
                })
                .await
                .map_err(|e| anyhow::anyhow!("speculative decode spawn_blocking: {}", e))??;
                // Acceptance accounting: committed = accepted drafts + 1 bonus token.
                use std::sync::atomic::Ordering;
                self.spec_proposed
                    .fetch_add(proposed as u64, Ordering::Relaxed);
                self.spec_accepted
                    .fetch_add((committed.len() - 1) as u64, Ordering::Relaxed);
                return Ok(vec![(only_id, committed)]);
            }
        }

        // Normal batched decode: one token per request.
        let out =
            tokio::task::spawn_blocking(move || model.decode_sync(&req_ids_vec, &model_requests))
                .await
                .map_err(|e| anyhow::anyhow!("decode spawn_blocking: {}", e))??;
        Ok(out.into_iter().map(|(id, l)| (id, vec![l])).collect())
    }
}

/// Compute `(n_keep, n_discard)` for a reactive context roll, or `None` when
/// there isn't enough context beyond the preserved head to make discarding
/// meaningful (e.g. `n_ctx` itself is tiny, or the request's `ctx_len` barely
/// exceeds `n_keep`). Deliberately a separate, small function from
/// `roll_full_contexts`'s inline arithmetic rather than a shared abstraction:
/// the proactive trigger only ever calls its math once `ctx_len` has already
/// crossed a large threshold (`n_ctx - reserve`), where this edge case can't
/// arise, so reusing it here isn't necessary and forcing a shared helper into
/// that already-shipped, already-tested path isn't worth the risk.
fn reactive_roll_amounts(
    n_keep_cfg: usize,
    n_ctx: usize,
    ctx_len: usize,
) -> Option<(usize, usize)> {
    if n_ctx == 0 {
        return None;
    }
    let n_keep = n_keep_cfg.min(n_ctx.saturating_sub(1));
    if ctx_len <= n_keep + 1 {
        return None;
    }
    let n_discard = (ctx_len.saturating_sub(n_keep) / 2).max(1);
    Some((n_keep, n_discard))
}

#[cfg(test)]
mod tests {
    use super::reactive_roll_amounts;

    #[test]
    fn enough_context_yields_keep_and_discard() {
        let (n_keep, n_discard) = reactive_roll_amounts(0, 2048, 2000).unwrap();
        assert_eq!(n_keep, 0);
        assert_eq!(n_discard, 1000);
    }

    #[test]
    fn n_keep_cfg_is_clamped_below_n_ctx() {
        // n_keep_cfg (10_000) exceeds n_ctx (100) — clamp to n_ctx - 1 = 99.
        let (n_keep, n_discard) = reactive_roll_amounts(10_000, 100, 150).unwrap();
        assert_eq!(n_keep, 99);
        assert_eq!(n_discard, 25);
    }

    #[test]
    fn discard_is_at_least_one() {
        // ctx_len just barely above n_keep + 1 — halving would round to 0,
        // clamped up to 1.
        let (_, n_discard) = reactive_roll_amounts(100, 2048, 102).unwrap();
        assert_eq!(n_discard, 1);
    }

    #[test]
    fn zero_n_ctx_yields_none() {
        assert!(reactive_roll_amounts(0, 0, 100).is_none());
    }

    #[test]
    fn context_barely_over_keep_yields_none() {
        // ctx_len <= n_keep + 1: nothing meaningful left to discard.
        assert!(reactive_roll_amounts(100, 2048, 101).is_none());
        assert!(reactive_roll_amounts(100, 2048, 100).is_none());
    }
}
