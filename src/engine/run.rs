use std::sync::Arc;

use anyhow::Result;

use crate::scheduler::StopReason;

use super::model::{InferenceRequestForModel, Logits, VisionPrefillParams};
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
                        }
                    }
                }
            }

            if !decode_ids.is_empty() {
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
                        }
                    }
                }
            }
        }
    }

    pub(super) async fn run_prefill(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);

        // Partition into vision and text-only requests.
        let mut vision_ids: Vec<u64> = Vec::new();
        let mut vision_params: Vec<VisionPrefillParams> = Vec::new();
        let mut text_ids: Vec<u64> = Vec::new();
        let mut text_model_requests: Vec<InferenceRequestForModel> = Vec::new();

        for r in &requests {
            if r.vision_image.is_some() && r.vision_prompt.is_some() {
                vision_ids.push(r.id);
                // If this vision request got a false prefix cache hit (prompt_tokens
                // are dummy zeros that matched a prior entry), clean up the prefix
                // seq_id — the cached KV is invalid for vision prefill.
                if let Some(prefix_sid) = r.prefix_seq_id {
                    self.model.clear_sequence(prefix_sid);
                    self.scheduler.return_prefix_seq_id(prefix_sid);
                }
                // Clear any stale KV data on this sequence before vision prefill.
                self.model.clear_sequence(r.kv_seq_id);
                vision_params.push(VisionPrefillParams {
                    seq_id: r.kv_seq_id,
                    text_prompt: r.vision_prompt.clone().unwrap(),
                    image_bytes: r.vision_image.clone().unwrap(),
                    temperature: r.sampling.temperature,
                    top_p: r.sampling.top_p,
                    top_k: r.sampling.top_k,
                    repetition_penalty: r.sampling.repetition_penalty,
                    seed: r.sampling.seed,
                });
            } else {
                text_ids.push(r.id);
                text_model_requests.push(InferenceRequestForModel {
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
                    min_p: r.sampling.min_p,
                    repetition_penalty: r.sampling.repetition_penalty,
                    frequency_penalty: r.sampling.frequency_penalty,
                    presence_penalty: r.sampling.presence_penalty,
                    seed: r.sampling.seed,
                    generated_token_ids: r.generated_token_ids.clone(),
                    skip_prefix_tokens: r.skip_prefix_tokens,
                    prefix_seq_id: r.prefix_seq_id,
                });
            }
        }

        let mut all_results: Vec<(u64, Logits)> = Vec::new();

        // Vision requests: process sequentially via vision_prefill_sync.
        if !vision_ids.is_empty() {
            let model = self.model.clone();
            let vision_ids_clone = vision_ids.clone();
            let vision_results = tokio::task::spawn_blocking(move || {
                let mut results = Vec::new();
                for (i, id) in vision_ids_clone.iter().enumerate() {
                    match model.vision_prefill_sync(&vision_params[i]) {
                        Ok((n_past, logits)) => results.push((*id, n_past, logits)),
                        Err(e) => return Err(e),
                    }
                }
                Ok(results)
            })
            .await
            .map_err(|e| anyhow::anyhow!("vision prefill spawn_blocking: {}", e))??;

            for (id, n_past, logits) in vision_results {
                self.scheduler.set_prefilled_tokens(id, n_past);
                all_results.push((id, logits));
            }
        }

        // Text-only requests: existing batched prefill path.
        if !text_ids.is_empty() {
            let prefix_cleanup: Vec<i32> = text_model_requests
                .iter()
                .filter_map(|r| r.prefix_seq_id)
                .collect();

            let model = self.model.clone();
            let text_ids_clone = text_ids.clone();
            let raw = tokio::task::spawn_blocking(move || {
                model.prefill_sync(&text_ids_clone, &text_model_requests)
            })
            .await
            .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

            for prefix_seq_id in prefix_cleanup {
                self.model.clear_sequence(prefix_seq_id);
                self.scheduler.return_prefix_seq_id(prefix_seq_id);
            }

            for (id, logits, tokens_in_kv) in raw {
                if tokens_in_kv > 0 {
                    self.scheduler.set_prefilled_tokens(id, tokens_in_kv);
                }
                all_results.push((id, logits));
            }
        }

        Ok(all_results)
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
                min_p: r.sampling.min_p,
                repetition_penalty: r.sampling.repetition_penalty,
                frequency_penalty: r.sampling.frequency_penalty,
                presence_penalty: r.sampling.presence_penalty,
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: 0,
                prefix_seq_id: None,
            })
            .collect();
        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        tokio::task::spawn_blocking(move || model.decode_sync(&req_ids_vec, &model_requests))
            .await
            .map_err(|e| anyhow::anyhow!("decode spawn_blocking: {}", e))?
    }
}
