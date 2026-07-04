use std::sync::Arc;

use anyhow::Result;

use crate::scheduler::StopReason;

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
            }

            for seq_id in &batch.preempted_seq_ids {
                engine.model.clear_sequence(*seq_id);
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
                        engine.handle_logits(&prefill_results, true).await?;
                    }
                    Err(e) => {
                        tracing::warn!(
                            "prefill failed (KV cache full?): {} — stopping {} request(s) with Length",
                            e,
                            prefill_ids.len()
                        );
                        for req_id in &prefill_ids {
                            engine.scheduler.mark_finished(*req_id, StopReason::Length);
                            engine.model.free_grammar(*req_id);
                        }
                    }
                }
            }

            if !decode_ids.is_empty() {
                // Before decoding, roll any sequence whose KV window is full so it can
                // continue past n_ctx instead of failing (context shift).
                engine.roll_full_contexts(&decode_ids).await;
                match engine.run_decode(&decode_ids).await {
                    Ok(decode_results) => {
                        engine.handle_logits(&decode_results, false).await?;
                    }
                    Err(e) => {
                        // KV cache exhausted or llama_decode failure — stop all affected
                        // requests gracefully instead of crashing the engine loop.
                        tracing::warn!(
                            "decode failed (KV cache full?): {} — stopping {} request(s) with Length",
                            e,
                            decode_ids.len()
                        );
                        for req_id in &decode_ids {
                            engine.scheduler.mark_finished(*req_id, StopReason::Length);
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
        for req in self.scheduler.get_running(decode_ids) {
            let ctx_len = req.context_len();
            if ctx_len < n_ctx || req.kv_seq_id < 0 {
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
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: r.skip_prefix_tokens,
                prefix_seq_id: r.prefix_seq_id,
                prefill_pos: r.prefill_pos,
                grammar: r.sampling.grammar.clone(),
            })
            .collect();

        let prefix_cleanup: Vec<i32> = model_requests
            .iter()
            .filter_map(|r| r.prefix_seq_id)
            .collect();

        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        let max_chunk = self.max_prefill_chunk;
        let raw = tokio::task::spawn_blocking(move || {
            model.prefill_sync(&req_ids_vec, &model_requests, max_chunk)
        })
        .await
        .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

        for prefix_seq_id in prefix_cleanup {
            self.model.clear_sequence(prefix_seq_id);
            self.scheduler.return_prefix_seq_id(prefix_seq_id);
        }

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

    pub(super) async fn run_decode(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        // Copy-on-write: if any block in a decoding request is shared (ref_count > 1),
        // allocate a new exclusive copy before llama.cpp writes to it.
        //
        // With the current prefix-caching scheme (blocks are transferred exclusively on
        // cache hit), shared blocks arise only if `retain_block` was called explicitly.
        // This guard makes the decode path safe for future scenarios where multiple active
        // requests share KV blocks.
        for req_id in req_ids {
            let requests = self.scheduler.get_running(&[*req_id]);
            let Some(req) = requests.first() else {
                continue;
            };
            for (logical_idx, &block_id) in req.page_table.entries.iter().enumerate() {
                if self.kv_cache.is_shared(block_id) {
                    if let Some(new_block_id) = self.kv_cache.copy_on_write(block_id) {
                        self.scheduler
                            .cow_update_page_table(*req_id, logical_idx, new_block_id);
                        tracing::debug!(
                            request_id = req_id,
                            logical_idx,
                            old_block = block_id,
                            new_block = new_block_id,
                            "CoW: privatised shared KV block before decode"
                        );
                    }
                }
            }
        }

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
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: 0,
                prefix_seq_id: None,
                prefill_pos: r.prefill_pos,
                grammar: r.sampling.grammar.clone(),
            })
            .collect();
        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        tokio::task::spawn_blocking(move || model.decode_sync(&req_ids_vec, &model_requests))
            .await
            .map_err(|e| anyhow::anyhow!("decode spawn_blocking: {}", e))?
    }
}
