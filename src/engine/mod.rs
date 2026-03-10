// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;

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

/// Special token patterns that mark end-of-turn or other control sequences.
/// These are detected in `check_stop_sequences` using a rolling buffer so that
/// patterns spanning multiple tokens (e.g. `<`, `|`, `im_end`, `|`, `>`) are
/// also caught.  Never emitted to the user.
const CONTROL_TOKEN_PATTERNS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|endofthought|>",
];

// ---------------------------------------------------------------------------
// Per-request output state
// ---------------------------------------------------------------------------

/// Per-request mutable state for output processing.
#[derive(Default)]
struct PerRequestState {
    /// True while we are inside a `<think>…</think>` block.
    in_thinking: bool,
    /// When true the `<think>…</think>` block is forwarded to the caller instead
    /// of being silently discarded.  Set from `SamplingParams::show_thinking`.
    show_thinking: bool,
    /// Text that has passed the thinking filter but is being held back until
    /// we know it is not the start of a control-token pattern (e.g. `<|im_end|>`
    /// arriving across several BPE tokens: `<`, `|`, `im_end`, `|`, `>`).
    pending_output: String,
    /// Rolling suffix of recently emitted text (length ≤ 2 × max_stop_len).
    /// Used to detect *user-supplied* stop strings that span multiple tokens.
    text_buffer: String,
}

// ---------------------------------------------------------------------------
// Pure helpers (no self, no lock)
// ---------------------------------------------------------------------------

/// Apply output filtering rules.
///
/// Returns `(text_to_pass_downstream, control_stop)`.
/// * `text_to_pass_downstream` — safe visible text (empty while thinking or holding back
///   a partial control-token prefix).
/// * `control_stop` — `true` when a complete control-token pattern (`<|im_end|>` etc.)
///   was detected, meaning generation should stop.
///
/// **Why two stages?**
/// Models like Qwen3.5 often generate `<|im_end|>` as 5–6 separate BPE tokens
/// (`<`, `|`, `im`, `_end`, `|`, `>`).  A single-token `contains("<|")` check
/// misses this.  We therefore buffer the output in `state.pending_output` and
/// only flush text that cannot be the start of a control pattern.
fn apply_output_filter(state: &mut PerRequestState, raw: &str) -> (String, bool) {
    if raw.is_empty() {
        return (String::new(), false);
    }

    // 1. Enter <think> block (usually a single special token like 248068 for Qwen3.5).
    if raw.contains("<think>") {
        state.in_thinking = true;
        if state.show_thinking {
            // Emit the <think> tag so the user can see when reasoning starts.
            return (raw.to_string(), false);
        }
        return (String::new(), false);
    }

    // 2. Exit <think> block; text *after* the closing tag goes to the pending buffer.
    if raw.contains("</think>") {
        state.in_thinking = false;
        if state.show_thinking {
            // Emit the closing tag, then flush whatever was pending.
            let after_tag = raw
                .find("</think>")
                .map(|i| i + "</think>".len())
                .unwrap_or(raw.len());
            let mut out = raw[..after_tag].to_string();
            // Any text after </think> in the same token also needs to be flushed.
            if after_tag < raw.len() {
                state.pending_output.push_str(&raw[after_tag..]);
                let (rest, stop) = flush_pending_output(&mut state.pending_output);
                out.push_str(&rest);
                if stop {
                    return (out, true);
                }
            }
            return (out, false);
        }
        // Normal mode: discard the tag, keep text after it.
        if let Some(idx) = raw.find("</think>") {
            let after = idx + "</think>".len();
            if after < raw.len() {
                state.pending_output.push_str(&raw[after..]);
            }
        }
        return flush_pending_output(&mut state.pending_output);
    }

    // 3. Inside a thinking block.
    if state.in_thinking {
        if state.show_thinking {
            // Emit thinking tokens directly (no holdback needed — control patterns
            // like <|im_end|> should not appear inside a thinking block).
            return (raw.to_string(), false);
        }
        return (String::new(), false);
    }

    // 4. Normal text: push through the pending buffer; hold back any partial
    //    control-token prefix (e.g. `<` that could be the start of `<|im_end|>`).
    state.pending_output.push_str(raw);
    flush_pending_output(&mut state.pending_output)
}

/// Flush as much of `pending` as is safe.
///
/// 1. If `pending` contains a complete control-token pattern, emit everything
///    *before* the pattern and signal stop (the pattern itself is discarded).
/// 2. Otherwise, hold back the longest suffix that is a strict prefix of any
///    control pattern (could be the start of `<|im_end|>` etc.).
fn flush_pending_output(pending: &mut String) -> (String, bool) {
    // Check for complete control-token patterns.
    for &pat in CONTROL_TOKEN_PATTERNS {
        if let Some(idx) = pending.find(pat) {
            let emit = pending[..idx].to_string();
            pending.clear();
            return (emit, true); // stop generation
        }
    }

    // Find the earliest `<` that could be the start of a control pattern.
    let holdback_start = find_holdback_start(pending);
    let emit = pending[..holdback_start].to_string();
    *pending = pending[holdback_start..].to_string();
    (emit, false)
}

/// Returns the byte offset of the first `<` in `text` from which a control-token
/// pattern *could* still begin (i.e. some pattern starts with the suffix
/// `text[offset..]`).  Returns `text.len()` when nothing needs to be held back.
fn find_holdback_start(text: &str) -> usize {
    for (i, c) in text.char_indices() {
        if c != '<' {
            continue;
        }
        let suffix = &text[i..];
        if CONTROL_TOKEN_PATTERNS.iter().any(|p| p.starts_with(suffix)) {
            return i;
        }
    }
    text.len()
}

/// Check whether the rolling text buffer (extended with `new_text`) ends with
/// any of the user-supplied stop strings.
///
/// Returns `(text_to_emit, was_stopped)`. When stopped, `text_to_emit` is
/// the prefix of `new_text` that appears *before* the stop string; the stop
/// string itself is NOT emitted (OpenAI spec behaviour).
///
/// Note: built-in control-token patterns (`<|im_end|>` etc.) are already handled
/// upstream in `apply_output_filter` / `flush_pending_output`.  This function
/// only deals with user-supplied stop strings.
fn check_stop_sequences(
    state: &mut PerRequestState,
    new_text: String,
    stop: &Option<Vec<String>>,
) -> (String, bool) {
    let stops = match stop.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => {
            // No stop strings — just maintain the buffer for future calls.
            state.text_buffer.push_str(&new_text);
            trim_text_buffer(&mut state.text_buffer, 0);
            return (new_text, false);
        }
    };

    let max_stop_len: usize = stops.iter().map(|s| s.len()).max().unwrap_or(0);

    // Extend the buffer with the new token text.
    state.text_buffer.push_str(&new_text);

    // Check every stop string.
    for stop_str in stops {
        if stop_str.is_empty() {
            continue;
        }
        if state.text_buffer.ends_with(stop_str.as_str()) {
            // Find how much of `new_text` to emit (the part before the stop string).
            let buf_len = state.text_buffer.len();
            let stop_start_in_buf = buf_len - stop_str.len();
            // `new_text` starts at `buf_len - new_text.len()` within the buffer.
            let text_start_in_buf = buf_len.saturating_sub(new_text.len());

            let emit = if stop_start_in_buf >= text_start_in_buf {
                let offset = stop_start_in_buf - text_start_in_buf;
                new_text[..offset].to_string()
            } else {
                // Stop string began in a previously-emitted token — emit nothing.
                String::new()
            };

            state.text_buffer.clear();
            return (emit, true);
        }
    }

    // No match — trim the buffer to avoid unbounded growth.
    trim_text_buffer(&mut state.text_buffer, max_stop_len);
    (new_text, false)
}

/// Keep only the trailing `max_stop_len` characters of the buffer (aligned to a char boundary).
fn trim_text_buffer(buf: &mut String, max_stop_len: usize) {
    let keep = (max_stop_len * 2).max(128);
    if buf.len() > keep {
        let trim_byte = buf.len() - keep;
        // Walk forward to the next valid char boundary.
        let trim_at = buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= trim_byte)
            .unwrap_or(trim_byte);
        *buf = buf[trim_at..].to_string();
    }
}

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
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            per_request_state: Arc::new(Mutex::new(HashMap::new())),
            model_name,
            metrics,
            supports_prefix_cache,
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

            // Detokenize (EOS tokens produce empty text to avoid leaking control tokens).
            let raw_text = if is_eos {
                String::new()
            } else {
                let mut t = self.model.token_to_piece(token_id).unwrap_or_default();
                t = t.replace(SPM_SPACE, " ");
                t
            };

            // Apply output filtering AND stop sequence detection in a single lock scope.
            let (text, is_stop_hit) = {
                let mut state_map = match self.per_request_state.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        // Lock poisoned — skip processing, emit raw text with no stop check.
                        if req.response_tx.send(Token {
                            id: *req_id,
                            token_id,
                            text: raw_text,
                            is_eos,
                            stop_reason: None,
                        }).is_err() {
                            // Client disconnected while lock was poisoned — cancel.
                            if req.kv_seq_id >= 0 {
                                self.model.clear_sequence(req.kv_seq_id);
                            }
                            self.scheduler.mark_finished(*req_id, StopReason::Preempt);
                        }
                        continue;
                    }
                };
                let state = state_map.entry(*req_id).or_insert_with(|| PerRequestState {
                    show_thinking: req.sampling.show_thinking,
                    ..Default::default()
                });

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

            let send_ok = req.response_tx.send(Token {
                id: *req_id,
                token_id,
                text,
                is_eos,
                stop_reason: stop_reason.clone(),
            }).is_ok();

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
                if let Ok(mut state) = self.per_request_state.lock() {
                    state.remove(req_id);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(v: &[&str]) -> Option<Vec<String>> {
        Some(v.iter().map(|s| s.to_string()).collect())
    }

    // Helper: unwrap the text from apply_output_filter (ignores the control_stop bool)
    fn aof(state: &mut PerRequestState, raw: &str) -> String {
        apply_output_filter(state, raw).0
    }

    // --- apply_output_filter ---

    #[test]
    fn test_filter_think_block() {
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "<think>"), "");
        assert!(s.in_thinking);
        assert_eq!(aof(&mut s, "internal thought"), "");
        assert_eq!(aof(&mut s, "</think> hello"), " hello");
        assert!(!s.in_thinking);
        assert_eq!(aof(&mut s, " world"), " world");
    }

    #[test]
    fn test_filter_passthrough() {
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "hello"), "hello");
        assert_eq!(aof(&mut s, " world"), " world");
    }

    #[test]
    fn test_filter_control_single_token_stopped() {
        // Single-token <|im_end|> (e.g. EOS token decoded to its text form) must stop.
        let mut s = PerRequestState::default();
        let (text, stop) = apply_output_filter(&mut s, "<|im_end|>");
        assert_eq!(text, "");
        assert!(stop, "<|im_end|> single token must trigger control stop");
    }

    #[test]
    fn test_filter_control_multi_token_im_end() {
        // Qwen3.5 emits <|im_end|> as 5 separate BPE tokens.
        let mut s = PerRequestState::default();
        for tok in &["<", "|", "im", "_end", "|"] {
            let (text, stop) = apply_output_filter(&mut s, tok);
            assert_eq!(text, "", "partial token '{tok}' must be held back");
            assert!(!stop, "no stop on partial token '{tok}'");
        }
        let (text, stop) = apply_output_filter(&mut s, ">");
        assert_eq!(text, "", "closing '>' must not leak");
        assert!(stop, "closing '>' completes <|im_end|> → must stop");
    }

    #[test]
    fn test_filter_holdback_released_on_non_pattern() {
        // A lone `<` is held back until the next token confirms it is not a control pattern.
        let mut s = PerRequestState::default();
        let (t1, _) = apply_output_filter(&mut s, "<");
        assert_eq!(t1, "", "< must be held back");
        // `x` cannot extend any control pattern starting with `<` → release both.
        let (t2, stop) = apply_output_filter(&mut s, "x");
        assert_eq!(t2, "<x", "< and x must be released together");
        assert!(!stop);
    }

    #[test]
    fn test_filter_text_before_control_token_emitted() {
        // Normal text followed by <|im_end|>: text emitted, pattern stops generation.
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "Hello!"), "Hello!");
        let (text, stop) = apply_output_filter(&mut s, "<|im_end|>");
        assert_eq!(text, "");
        assert!(stop);
    }

    // --- check_stop_sequences ---

    #[test]
    fn test_stop_no_stops_configured() {
        let mut s = PerRequestState::default();
        let (text, hit) = check_stop_sequences(&mut s, "hello world".to_string(), &None);
        assert_eq!(text, "hello world");
        assert!(!hit);
    }

    #[test]
    fn test_stop_exact_single_token() {
        let mut s = PerRequestState::default();
        let (text, hit) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["User:"]));
        assert_eq!(text, "", "stop string itself must not be emitted");
        assert!(hit);
    }

    #[test]
    fn test_stop_partial_current_token_emitted() {
        let mut s = PerRequestState::default();
        let (text, hit) =
            check_stop_sequences(&mut s, "Hello\nUser:".to_string(), &stops(&["\nUser:"]));
        assert_eq!(text, "Hello");
        assert!(hit);
    }

    #[test]
    fn test_stop_multi_token_span() {
        let mut s = PerRequestState::default();
        let (t1, h1) = check_stop_sequences(&mut s, "Hello\n".to_string(), &stops(&["\nUser:"]));
        assert_eq!(t1, "Hello\n");
        assert!(!h1);
        let (t2, h2) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["\nUser:"]));
        assert_eq!(t2, "");
        assert!(h2);
    }

    #[test]
    fn test_stop_multiple_candidates_first_wins() {
        let mut s = PerRequestState::default();
        let (text, hit) =
            check_stop_sequences(&mut s, "STOP".to_string(), &stops(&["STOP", "OTHER"]));
        assert_eq!(text, "");
        assert!(hit);
    }

    #[test]
    fn test_stop_no_match_passes_through() {
        let mut s = PerRequestState::default();
        let (t1, h1) = check_stop_sequences(&mut s, "hello ".to_string(), &stops(&["User:"]));
        assert_eq!(t1, "hello ");
        assert!(!h1);
        let (t2, h2) = check_stop_sequences(&mut s, "world".to_string(), &stops(&["User:"]));
        assert_eq!(t2, "world");
        assert!(!h2);
    }

    // --- trim_text_buffer ---

    #[test]
    fn test_trim_buffer_below_limit_unchanged() {
        let mut buf = "hello".to_string();
        trim_text_buffer(&mut buf, 10); // keep = 20, buf.len() = 5 < 20
        assert_eq!(buf, "hello");
    }

    #[test]
    fn test_trim_buffer_trims_to_trailing_content() {
        let mut buf = "a".repeat(200);
        trim_text_buffer(&mut buf, 10); // keep = 20
        assert!(buf.len() <= 200);
        assert!(buf.len() >= 20);
        // Verify suffix is preserved
        assert!(buf.chars().all(|c| c == 'a'));
    }
}
