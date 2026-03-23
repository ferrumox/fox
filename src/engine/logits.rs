use anyhow::Result;
use tracing::debug;

use crate::scheduler::{StopReason, Token};

use super::model::Logits;
use super::output_filter::{
    apply_output_filter, check_stop_sequences, drain_valid_utf8, PerRequestState,
};
use super::{InferenceEngine, SPM_SPACE};

impl InferenceEngine {
    pub(super) async fn handle_logits(
        &self,
        results: &[(u64, Logits)],
        from_prefill: bool,
    ) -> Result<()> {
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
                    emit_think_open_tag: req.sampling.initial_in_thinking,
                    model_control_patterns: self.model_stop_tokens.clone(),
                    max_thinking_chars: req.sampling.max_thinking_chars,
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
                    Ok(mut state) => {
                        state.remove(req_id);
                    }
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
}
