// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;
mod output_filter;

use output_filter::{apply_output_filter, check_stop_sequences, drain_valid_utf8, PerRequestState};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use tracing::debug;

use crate::kv_cache::KVCacheManager;
use crate::metrics::Metrics;
use crate::scheduler::{InferenceRequest, StopReason, Token};

use self::model::{InferenceRequestForModel, Logits, Model};

/// SentencePiece uses U+2581 (▁) for word boundaries. Replace with space so words don't concatenate.
const SPM_SPACE: char = '\u{2581}';

// ---------------------------------------------------------------------------
// InferenceEngine
// ---------------------------------------------------------------------------

/// Main inference engine coordinating scheduler, model, and KV cache.
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    scheduler: Arc<crate::scheduler::Scheduler>,
    kv_cache: Arc<KVCacheManager>,
    next_request_id: AtomicU64,
    /// Per-request mutable state for output filtering and stop sequence detection.
    per_request_state: Arc<Mutex<HashMap<u64, PerRequestState>>>,
    /// Human-readable model identifier (basename of the loaded model file).
    model_name: String,
    /// Prometheus metrics (optional — None disables recording).
    metrics: Option<Arc<Metrics>>,
    /// Whether the loaded model supports KV-cache sequence copying (llama_memory_seq_cp).
    /// False for recurrent/hybrid models (Mamba, Qwen3.5, etc.); prefix caching is disabled.
    supports_prefix_cache: bool,
    /// Text forms of the model's EOS and EOT tokens, used as base stop sequences.
    model_stop_tokens: Vec<String>,
}

impl InferenceEngine {
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
        model_name: String,
        metrics: Option<Arc<Metrics>>,
    ) -> Self {
        let supports_prefix_cache = model.supports_seq_copy();
        if supports_prefix_cache {
            tracing::info!("prefix caching enabled (model supports seq_cp)");
        } else {
            tracing::info!(
                "prefix caching disabled (model uses recurrent/hybrid memory — seq_cp unsupported)"
            );
        }
        let model_stop_tokens = model.stop_tokens();
        if !model_stop_tokens.is_empty() {
            tracing::info!("model stop tokens: {:?}", model_stop_tokens);
        }
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            per_request_state: Arc::new(Mutex::new(HashMap::new())),
            model_name,
            metrics,
            supports_prefix_cache,
            model_stop_tokens,
        }
    }

    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        self.model.tokenize(text)
    }

    pub fn embedding_dim(&self) -> usize {
        self.model.embedding_dim()
    }

    pub async fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let tokens = self.model.tokenize(text)?;
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || model.get_embeddings(&tokens))
            .await
            .map_err(|e| anyhow::anyhow!("embed spawn_blocking: {}", e))?
    }

    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> anyhow::Result<String> {
        self.model.apply_chat_template(messages)
    }

    pub fn submit_request(&self, req: InferenceRequest) {
        self.scheduler.submit(req);
    }

    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

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

    async fn run_prefill(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
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
                seed: r.sampling.seed,
                generated_token_ids: r.generated_token_ids.clone(),
                skip_prefix_tokens: r.skip_prefix_tokens,
                prefix_seq_id: r.prefix_seq_id,
            })
            .collect();

        let prefix_cleanup: Vec<i32> = model_requests
            .iter()
            .filter_map(|r| r.prefix_seq_id)
            .collect();

        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        let raw =
            tokio::task::spawn_blocking(move || model.prefill_sync(&req_ids_vec, &model_requests))
                .await
                .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

        for prefix_seq_id in prefix_cleanup {
            self.model.clear_sequence(prefix_seq_id);
            self.scheduler.return_prefix_seq_id(prefix_seq_id);
        }

        // Register how many tokens were actually placed in the KV for each request so
        // decode positions are consecutive (no gaps for recurrent/hybrid models).
        let result = raw
            .into_iter()
            .map(|(id, logits, tokens_in_kv)| {
                if tokens_in_kv > 0 {
                    self.scheduler.set_prefilled_tokens(id, tokens_in_kv);
                }
                (id, logits)
            })
            .collect();

        Ok(result)
    }

    async fn run_decode(&self, req_ids: &[u64]) -> Result<Vec<(u64, Logits)>> {
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

    async fn handle_logits(&self, results: &[(u64, Logits)], from_prefill: bool) -> Result<()> {
        let req_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let running = self.scheduler.get_running(&req_ids);
        let eos_token_id = self.model.eos_token_id();

        for (req_id, logits) in results {
            let req = running.iter().find(|r| r.id == *req_id);
            let Some(req) = req else {
                continue;
            };

            let token_id = logits.sampled_token;
            // Use is_eog_token() to catch ALL end-of-generation tokens, not just the
            // primary EOS.  Models like Qwen3.5 have multiple EOG tokens
            // (e.g. <|endoftext|>, <|im_end|>, <|fim_pad|>, …).
            let is_eos = self.model.is_eog_token(token_id);
            let _ = eos_token_id; // kept for metrics / future use
            let reached_max = req.generated_tokens + 1 >= req.max_new_tokens;

            // Detokenize to raw bytes (EOS tokens produce no bytes).
            // Bytes may represent an incomplete UTF-8 sequence (e.g. the first two bytes
            // of a 4-byte emoji split across BPE tokens).  They are accumulated in the
            // per-request state buffer below and only decoded when complete.
            let token_bytes: Vec<u8> = if is_eos {
                vec![]
            } else {
                self.model.token_to_piece_bytes(token_id)
            };

            // Apply output filtering AND stop sequence detection in a single lock scope.
            let (text, is_stop_hit) = {
                let mut state_map = match self.per_request_state.lock() {
                    Ok(g) => g,
                    Err(e) => {
                        // Lock poisoned — a thread panicked while holding this lock.
                        // This is a bug; log it and emit lossy text with no stop check as a
                        // best-effort recovery so the client still receives a response.
                        tracing::error!(
                            request_id = req_id,
                            "per_request_state lock poisoned: {}; emitting token without filtering",
                            e
                        );
                        let raw_text = String::from_utf8_lossy(&token_bytes).into_owned();
                        if req
                            .response_tx
                            .send(Token {
                                id: *req_id,
                                token_id,
                                text: raw_text,
                                is_eos,
                                stop_reason: None,
                            })
                            .is_err()
                        {
                            // Client disconnected while lock was poisoned — cancel.
                            if req.kv_seq_id >= 0 {
                                self.model.clear_sequence(req.kv_seq_id);
                            }
                            self.scheduler.mark_finished(*req_id, StopReason::Preempt);
                        }
                        continue;
                    }
                };
                // Clone stop tokens only on first token for this request (inside or_insert_with),
                // not on every subsequent token.
                let state = state_map.entry(*req_id).or_insert_with(|| PerRequestState {
                    show_thinking: req.sampling.show_thinking,
                    in_thinking: req.sampling.initial_in_thinking,
                    model_control_patterns: self.model_stop_tokens.clone(),
                    ..Default::default()
                });

                // Accumulate raw token bytes and drain complete UTF-8 codepoints.
                // This prevents "??" artifacts when multi-byte characters (e.g. emoji)
                // are split across BPE tokens and passed through from_utf8_lossy.
                state.utf8_buf.extend_from_slice(&token_bytes);
                let raw_text = drain_valid_utf8(&mut state.utf8_buf).replace(SPM_SPACE, " ");

                // Stage 1: thinking-block suppression + control-token holdback.
                // Returns (filtered_text, control_stop) where control_stop is true when a
                // complete control-token pattern was detected (e.g. multi-token <|im_end|>).
                let (filtered, control_stop) = apply_output_filter(state, &raw_text);

                // Stage 2: user-supplied stop strings checked on the rolling buffer.
                let (text, user_stop) = check_stop_sequences(state, filtered, &req.sampling.stop);

                (text, control_stop || user_stop)
            };

            let is_done = is_eos || reached_max || is_stop_hit;
            let stop_reason: Option<StopReason> = if is_done {
                Some(if is_stop_hit {
                    StopReason::StopSequence
                } else if is_eos {
                    StopReason::Eos
                } else {
                    StopReason::Length
                })
            } else {
                None
            };

            let send_ok = req
                .response_tx
                .send(Token {
                    id: *req_id,
                    token_id,
                    text,
                    is_eos,
                    stop_reason: stop_reason.clone(),
                })
                .is_ok();

            debug!(
                request_id = req_id,
                token_id, is_stop_hit, "token generated"
            );

            // Client disconnected: receiver was dropped. Cancel the request
            // immediately to free KV cache and scheduler slot.
            if !send_ok {
                if req.kv_seq_id >= 0 {
                    self.model.clear_sequence(req.kv_seq_id);
                }
                self.scheduler.mark_finished(*req_id, StopReason::Preempt);
                match self.per_request_state.lock() {
                    Ok(mut state) => { state.remove(req_id); }
                    Err(e) => tracing::error!("per_request_state lock poisoned on cleanup: {}", e),
                }
                continue;
            }

            // Record per-token metrics.
            if let Some(m) = &self.metrics {
                m.tokens_generated_total.inc();
            }

            if is_done {
                // Record per-request metrics.
                if let Some(m) = &self.metrics {
                    let reason_label = match &stop_reason {
                        Some(StopReason::Eos) => "stop",
                        Some(StopReason::Length) => "length",
                        Some(StopReason::StopSequence) => "stop",
                        Some(StopReason::Preempt) => "preempt",
                        None => "unknown",
                    };
                    m.requests_total.with_label_values(&[reason_label]).inc();
                    let elapsed = req.submitted_at.elapsed().as_secs_f64();
                    m.request_latency_seconds.observe(elapsed);
                }

                // Cache the KV state for potential prefix reuse on future identical prompts.
                // Disabled for models that don't support llama_memory_seq_cp (e.g. Mamba/hybrid).
                let should_clear = if self.supports_prefix_cache
                    && matches!(
                        stop_reason,
                        Some(StopReason::Eos)
                            | Some(StopReason::Length)
                            | Some(StopReason::StopSequence)
                    ) {
                    !self.scheduler.try_insert_prefix(*req_id)
                } else {
                    true
                };

                if should_clear && req.kv_seq_id >= 0 {
                    self.model.clear_sequence(req.kv_seq_id);
                }

                self.scheduler.mark_finished(*req_id, stop_reason.unwrap());

                if let Ok(mut state) = self.per_request_state.lock() {
                    state.remove(req_id);
                }
            } else {
                self.scheduler
                    .update_after_token(*req_id, token_id, from_prefill);
            }
        }

        Ok(())
    }

    pub fn kv_cache_usage(&self) -> f32 {
        self.kv_cache.memory_usage()
    }

    pub fn queue_depth(&self) -> usize {
        self.scheduler.queue_depth()
    }

    pub fn active_requests(&self) -> usize {
        self.scheduler.active_requests()
    }

    pub fn prefix_cache_hits(&self) -> u64 {
        self.scheduler.prefix_hits.load(Ordering::Relaxed)
    }

    pub fn prefix_cache_misses(&self) -> u64 {
        self.scheduler.prefix_misses.load(Ordering::Relaxed)
    }
}
