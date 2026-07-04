use anyhow::Result;
use tracing::debug;

use crate::scheduler::{StopReason, Token, TokenLogprob, TopLogprob};

use super::model::Logits;
use super::output_filter::{
    apply_output_filter, check_stop_sequences, drain_valid_utf8, PerRequestState,
};
use super::{InferenceEngine, SPM_SPACE};

impl InferenceEngine {
    /// Compute the OpenAI-style log-probabilities for a sampled token from the full
    /// logits vector: the chosen token's logprob plus the `top_n` most-likely
    /// alternatives. Returns `None` when logits are unavailable (e.g. EOS with no
    /// vector). The distribution is the model's raw output (before any grammar mask).
    fn compute_token_logprob(
        &self,
        values: &[f32],
        token_id: i32,
        top_n: u8,
    ) -> Option<TokenLogprob> {
        let (chosen_lp, tops) = logprob_core(values, token_id, top_n)?;
        let piece = |id: i32| -> (String, Vec<u8>) {
            let bytes = self.model.token_to_piece_bytes(id);
            let text = String::from_utf8_lossy(&bytes).replace(SPM_SPACE, " ");
            (text, bytes)
        };
        let (token, bytes) = piece(token_id);
        Some(TokenLogprob {
            token,
            logprob: chosen_lp,
            bytes,
            top: tops
                .into_iter()
                .map(|(id, lp)| {
                    let (token, bytes) = piece(id as i32);
                    TopLogprob {
                        token,
                        logprob: lp,
                        bytes,
                    }
                })
                .collect(),
        })
    }

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

            // Apply output filtering AND stop sequence detection.
            // DashMap gives us per-entry locking so concurrent requests don't block each other.
            let (text, is_stop_hit) = {
                // Clone stop tokens only on first token for this request (inside or_insert_with),
                // not on every subsequent token.
                let mut state = self.per_request_state.entry(*req_id).or_insert_with(|| {
                    // Reasoning delimiters come from the model (default `<think>`/`</think>`,
                    // or the model's own markers, e.g. Gemma's `<|channel>`/`<channel|>`).
                    let (think_open, think_close) = self
                        .model
                        .reasoning_delimiters()
                        .unwrap_or_else(|| ("<think>".to_string(), "</think>".to_string()));
                    PerRequestState {
                        think_open,
                        think_close,
                        show_thinking: req.sampling.show_thinking,
                        in_thinking: req.sampling.initial_in_thinking,
                        emit_think_open_tag: req.sampling.initial_in_thinking,
                        model_control_patterns: self.model_stop_tokens.clone(),
                        max_thinking_chars: req.sampling.max_thinking_chars,
                        ..Default::default()
                    }
                });

                // Accumulate raw token bytes and drain complete UTF-8 codepoints.
                // This prevents "??" artifacts when multi-byte characters (e.g. emoji)
                // are split across BPE tokens and passed through from_utf8_lossy.
                state.utf8_buf.extend_from_slice(&token_bytes);
                let raw_text = drain_valid_utf8(&mut state.utf8_buf).replace(SPM_SPACE, " ");

                // Stage 1: thinking-block suppression + control-token holdback.
                // Returns (filtered_text, control_stop) where control_stop is true when a
                // complete control-token pattern was detected (e.g. multi-token <|im_end|>).
                let (filtered, control_stop) = apply_output_filter(&mut state, &raw_text);

                // Stage 2: user-supplied stop strings checked on the rolling buffer.
                let (text, user_stop) =
                    check_stop_sequences(&mut state, filtered, &req.sampling.stop);

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

            // Per-token logprobs, only when requested (and there is a real token piece).
            let logprob = if is_eos {
                None
            } else {
                req.sampling
                    .logprobs
                    .and_then(|top_n| self.compute_token_logprob(&logits.values, token_id, top_n))
            };

            let send_ok = req
                .response_tx
                .send(Token {
                    id: *req_id,
                    token_id,
                    text,
                    is_eos,
                    stop_reason: stop_reason.clone(),
                    logprob,
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
                self.per_request_state.remove(req_id);
                self.model.free_grammar(*req_id);
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

                self.per_request_state.remove(req_id);
                self.model.free_grammar(*req_id);
            } else {
                self.scheduler
                    .update_after_token(*req_id, token_id, from_prefill);
            }
        }

        Ok(())
    }
}

/// Numeric core of per-token logprobs (no model access, so it's unit-testable):
/// returns the sampled token's natural-log softmax value and the `top_n` highest
/// `(token_id, logprob)` pairs (descending). `None` if `values` is empty or `token_id`
/// is out of range.
fn logprob_core(values: &[f32], token_id: i32, top_n: u8) -> Option<(f32, Vec<(usize, f32)>)> {
    if values.is_empty() || token_id < 0 || token_id as usize >= values.len() {
        return None;
    }
    // log-softmax via a numerically-stable log-sum-exp.
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + values.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    let logprob_of = |id: usize| values[id] - lse;

    let top = if top_n == 0 {
        Vec::new()
    } else {
        let n = (top_n as usize).min(values.len());
        let mut idx: Vec<usize> = (0..values.len()).collect();
        let cmp = |a: &usize, b: &usize| {
            values[*b]
                .partial_cmp(&values[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        };
        idx.select_nth_unstable_by(n - 1, cmp);
        idx.truncate(n);
        idx.sort_by(cmp);
        idx.into_iter().map(|id| (id, logprob_of(id))).collect()
    };

    Some((logprob_of(token_id as usize), top))
}

#[cfg(test)]
mod tests {
    use super::logprob_core;

    #[test]
    fn logprob_core_out_of_range_or_empty_is_none() {
        assert!(logprob_core(&[], 0, 0).is_none());
        assert!(logprob_core(&[1.0, 2.0], 5, 0).is_none());
        assert!(logprob_core(&[1.0, 2.0], -1, 0).is_none());
    }

    #[test]
    fn logprob_core_matches_manual_log_softmax() {
        // Uniform logits over 4 tokens → each prob = 0.25 → logprob = ln(0.25).
        let (lp, _) = logprob_core(&[0.0, 0.0, 0.0, 0.0], 2, 0).unwrap();
        assert!((lp - 0.25f32.ln()).abs() < 1e-5, "got {lp}");

        // Probabilities always sum to 1: exp of every token's logprob sums to ~1.
        let vals = [2.0f32, -1.0, 0.5, 3.5, -2.0];
        let total: f32 = (0..vals.len())
            .map(|i| logprob_core(&vals, i as i32, 0).unwrap().0.exp())
            .sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax must sum to 1, got {total}"
        );
    }

    #[test]
    fn logprob_core_top_n_is_descending_and_capped() {
        let vals = [0.1f32, 5.0, 0.2, 9.0, 3.0];
        let (_, top) = logprob_core(&vals, 3, 3).unwrap();
        assert_eq!(top.len(), 3);
        // Highest logits first: token 3 (9.0), token 1 (5.0), token 4 (3.0).
        assert_eq!(top[0].0, 3);
        assert_eq!(top[1].0, 1);
        assert_eq!(top[2].0, 4);
        assert!(
            top[0].1 >= top[1].1 && top[1].1 >= top[2].1,
            "must be descending"
        );

        // top_n larger than the vocab is clamped, not a panic.
        let (_, all) = logprob_core(&vals, 0, 200).unwrap();
        assert_eq!(all.len(), vals.len());
    }
}
