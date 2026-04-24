use std::sync::Arc;

use anyhow::Result;

use crate::scheduler::StopReason;

use super::model::{InferenceRequestForModel, Logits, VisionDecodeParams, VisionPreprocessParams};
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

            // Run prefill and decode concurrently. They operate on disjoint
            // request sets (determined by schedule_step above). The GPU work
            // serializes on the llama_context mutex, but vision CLIP preprocessing
            // and async bookkeeping overlap with decode.
            let prefill_engine = engine.clone();
            let decode_engine = engine.clone();

            let (prefill_result, decode_result) = tokio::join!(
                async {
                    if prefill_ids.is_empty() {
                        return Ok(());
                    }
                    match prefill_engine.run_prefill(&prefill_ids).await {
                        Ok(prefill_results) => {
                            prefill_engine.handle_logits(&prefill_results, true).await
                        }
                        Err(e) => {
                            tracing::warn!(
                                "prefill failed (KV cache full?): {} — stopping {} request(s) with Length",
                                e,
                                prefill_ids.len()
                            );
                            for req_id in &prefill_ids {
                                prefill_engine
                                    .scheduler
                                    .mark_finished(*req_id, StopReason::Length);
                            }
                            Ok(())
                        }
                    }
                },
                async {
                    if decode_ids.is_empty() {
                        return Ok(());
                    }
                    match decode_engine.run_decode(&decode_ids).await {
                        Ok(decode_results) => {
                            decode_engine.handle_logits(&decode_results, false).await
                        }
                        Err(e) => {
                            tracing::warn!(
                                "decode failed (KV cache full?): {} — stopping {} request(s) with Length",
                                e,
                                decode_ids.len()
                            );
                            for req_id in &decode_ids {
                                decode_engine
                                    .scheduler
                                    .mark_finished(*req_id, StopReason::Length);
                            }
                            Ok(())
                        }
                    }
                }
            );
            prefill_result?;
            decode_result?;
        }
    }

    pub(super) async fn run_prefill(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
        let requests = self.scheduler.get_running(req_ids);

        // Partition into vision and text-only requests.
        let mut vision_ids: Vec<u64> = Vec::new();
        let mut vision_seq_ids: Vec<i32> = Vec::new();
        let mut vision_preprocess_params: Vec<VisionPreprocessParams> = Vec::new();
        let mut vision_sampling: Vec<(f32, f32, u32, f32, Option<u64>)> = Vec::new();
        let mut text_ids: Vec<u64> = Vec::new();
        let mut text_model_requests: Vec<InferenceRequestForModel> = Vec::new();

        for r in &requests {
            if let (Some(vision_image), Some(vision_prompt)) = (&r.vision_image, &r.vision_prompt) {
                vision_ids.push(r.id);
                if let Some(prefix_sid) = r.prefix_seq_id {
                    self.model.clear_sequence(prefix_sid);
                    self.scheduler.return_prefix_seq_id(prefix_sid);
                }
                self.model.clear_sequence(r.kv_seq_id);
                vision_seq_ids.push(r.kv_seq_id);
                vision_preprocess_params.push(VisionPreprocessParams {
                    text_prompt: vision_prompt.clone(),
                    image_bytes: Arc::clone(vision_image),
                });
                vision_sampling.push((
                    r.sampling.temperature,
                    r.sampling.top_p,
                    r.sampling.top_k,
                    r.sampling.repetition_penalty,
                    r.sampling.seed,
                ));
            } else {
                text_ids.push(r.id);
                text_model_requests.push(InferenceRequestForModel {
                    id: r.id,
                    prompt_tokens: Arc::clone(&r.prompt_tokens),
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

        // Vision requests: parallel CLIP preprocess, then batched decode.
        if !vision_ids.is_empty() {
            // Phase 1: Preprocess in parallel (off inference thread).
            // Check CLIP embedding cache before encoding.
            let mut preprocess_handles = Vec::new();
            for (i, _id) in vision_ids.iter().enumerate() {
                let image_bytes = Arc::clone(&vision_preprocess_params[i].image_bytes);
                let image_hash = Self::hash_image_bytes(&image_bytes);
                let cached = self
                    .clip_cache
                    .lock()
                    .ok()
                    .and_then(|mut c| c.get(&image_hash).cloned());

                let model = self.model.clone();
                let params = VisionPreprocessParams {
                    text_prompt: vision_preprocess_params[i].text_prompt.clone(),
                    image_bytes: Arc::clone(&vision_preprocess_params[i].image_bytes),
                };

                preprocess_handles.push(tokio::task::spawn_blocking(move || {
                    let pp = model
                        .vision_preprocess_sync_with_cache(&params, cached)
                        .or_else(|_| model.vision_preprocess_sync(&params))?;
                    Ok::<_, anyhow::Error>((image_hash, pp))
                }));
            }

            let mut decode_params: Vec<VisionDecodeParams> = Vec::new();
            let mut decode_req_ids: Vec<u64> = Vec::new();
            for (i, handle) in preprocess_handles.into_iter().enumerate() {
                let (image_hash, pp) = handle
                    .await
                    .map_err(|e| anyhow::anyhow!("vision preprocess spawn_blocking: {}", e))??;

                // Store CLIP embeddings in cache for future reuse.
                if let Ok(mut cache) = self.clip_cache.lock() {
                    if let Some((_, embd)) = pp.image_embeddings.first() {
                        cache.put(image_hash, embd.clone());
                    }
                }

                let (temp, top_p, top_k, rep_pen, seed) = vision_sampling[i];
                decode_params.push(VisionDecodeParams {
                    seq_id: vision_seq_ids[i],
                    preprocessed: pp,
                    temperature: temp,
                    top_p,
                    top_k,
                    repetition_penalty: rep_pen,
                    seed,
                });
                decode_req_ids.push(vision_ids[i]);
            }

            // Phase 2: Batch decode — acquires locks once for all requests.
            let model = self.model.clone();
            let vision_results = tokio::task::spawn_blocking(move || {
                model.vision_decode_prefill_batch_sync(decode_params)
            })
            .await
            .map_err(|e| anyhow::anyhow!("vision decode spawn_blocking: {}", e))??;

            for (i, (n_past, logits)) in vision_results.into_iter().enumerate() {
                let id = decode_req_ids[i];
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
        // Fetch all requests in one call to avoid per-request lock acquisition.
        {
            let cow_requests = self.scheduler.get_running(req_ids);
            for req in &cow_requests {
                for (logical_idx, &block_id) in req.page_table.entries.iter().enumerate() {
                    if self.kv_cache.is_shared(block_id) {
                        if let Some(new_block_id) = self.kv_cache.copy_on_write(block_id) {
                            self.scheduler
                                .cow_update_page_table(req.id, logical_idx, new_block_id);
                            tracing::debug!(
                                request_id = req.id,
                                logical_idx,
                                old_block = block_id,
                                new_block = new_block_id,
                                "CoW: privatised shared KV block before decode"
                            );
                        }
                    }
                }
            }
        }

        let requests = self.scheduler.get_running(req_ids);
        let model_requests: Vec<InferenceRequestForModel> = requests
            .iter()
            .map(|r| InferenceRequestForModel {
                id: r.id,
                prompt_tokens: Arc::clone(&r.prompt_tokens),
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
