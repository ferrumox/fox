// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
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

/// Known special tokens that should not appear in user-facing output.
const SPECIAL_TOKEN_PATTERNS: &[&str] = &[
    "<|endoftext|>",
    "<|im_end|>",
    "<|im_start|>",
    "<|endofthought|>",
];

// ---------------------------------------------------------------------------
// Per-request output state
// ---------------------------------------------------------------------------

/// Per-request mutable state for output processing.
/// Combines `<think>` block tracking with the rolling text buffer used for
/// multi-token stop sequence detection.
#[derive(Default)]
struct PerRequestState {
    /// True while we are inside a `<think>…</think>` block.
    in_thinking: bool,
    /// Rolling suffix of recently emitted text (length ≤ 2 × max_stop_len).
    /// Used to detect stop strings that span multiple tokens.
    text_buffer: String,
}

// ---------------------------------------------------------------------------
// Pure helpers (no self, no lock)
// ---------------------------------------------------------------------------

/// Apply output filtering rules using the per-request mutable state.
/// Returns the (possibly empty) visible text for this token.
fn apply_output_filter(state: &mut PerRequestState, raw: &str) -> String {
    if raw.is_empty() {
        return String::new();
    }

    // 1. Suppress known special tokens
    for &pattern in SPECIAL_TOKEN_PATTERNS {
        if raw == pattern || raw.contains(pattern) {
            return String::new();
        }
    }
    if raw.contains("<|") {
        return String::new();
    }

    // 2. Enter <think> block
    if raw.contains("<think>") {
        state.in_thinking = true;
        return String::new();
    }

    // 3. Exit <think> block
    if raw.contains("</think>") {
        state.in_thinking = false;
        if let Some(idx) = raw.find("</think>") {
            let after = idx + "</think>".len();
            if after < raw.len() {
                return raw[after..].to_string();
            }
        }
        return String::new();
    }

    // 4. Suppress tokens inside a thinking block
    if state.in_thinking {
        return String::new();
    }

    raw.to_string()
}

/// Check whether the rolling text buffer (extended with `new_text`) ends with
/// any of the user-supplied stop strings.
///
/// Returns `(text_to_emit, was_stopped)`. When stopped, `text_to_emit` is
/// the prefix of `new_text` that appears *before* the stop string; the stop
/// string itself is NOT emitted (OpenAI spec behaviour).
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
                            engine
                                .scheduler
                                .mark_finished(*req_id, StopReason::Length);
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
                            engine
                                .scheduler
                                .mark_finished(*req_id, StopReason::Length);
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
            let is_eos = token_id == eos_token_id;
            let reached_max = req.generated_tokens + 1 >= req.max_new_tokens;

            // Detokenize (EOS tokens produce empty text to avoid leaking control tokens).
            let raw_text = if is_eos {
                String::new()
            } else {
                let mut t = self.model.token_to_piece(token_id).unwrap_or_default();
                t = t.replace(SPM_SPACE, " ");
                t
            };

            // Apply output filtering AND stop sequence detection in a single lock scope
            // to avoid re-acquiring the lock for both operations.
            let (text, is_stop_hit) = {
                let mut state_map = match self.per_request_state.lock() {
                    Ok(g) => g,
                    Err(_) => {
                        // Lock poisoned — skip processing, emit raw text with no stop check.
                        let _ = req.response_tx.send(Token {
                            id: *req_id,
                            token_id,
                            text: raw_text,
                            is_eos,
                            stop_reason: None,
                        });
                        continue;
                    }
                };
                let state = state_map.entry(*req_id).or_default();

                let filtered = apply_output_filter(state, &raw_text);
                check_stop_sequences(state, filtered, &req.sampling.stop)
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

            let _ = req.response_tx.send(Token {
                id: *req_id,
                token_id,
                text,
                is_eos,
                stop_reason: stop_reason.clone(),
            });

            debug!(
                request_id = req_id,
                token_id, is_stop_hit, "token generated"
            );

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
                    let hash = hash_tokens(&req.prompt_tokens);
                    !self.scheduler.try_insert_prefix(*req_id, hash)
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

fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(v: &[&str]) -> Option<Vec<String>> {
        Some(v.iter().map(|s| s.to_string()).collect())
    }

    // --- apply_output_filter ---

    #[test]
    fn test_filter_special_tokens_suppressed() {
        let mut s = PerRequestState::default();
        assert_eq!(apply_output_filter(&mut s, "<|im_end|>"), "");
        assert_eq!(apply_output_filter(&mut s, "<|endoftext|>"), "");
        assert_eq!(apply_output_filter(&mut s, "hello <|im_start|>"), "");
    }

    #[test]
    fn test_filter_think_block() {
        let mut s = PerRequestState::default();
        // Enter thinking block
        assert_eq!(apply_output_filter(&mut s, "<think>"), "");
        assert!(s.in_thinking);
        // Inside: all suppressed
        assert_eq!(apply_output_filter(&mut s, "internal thought"), "");
        // Exit: text after </think> is emitted
        assert_eq!(apply_output_filter(&mut s, "</think> hello"), " hello");
        assert!(!s.in_thinking);
        // Normal text passes through
        assert_eq!(apply_output_filter(&mut s, " world"), " world");
    }

    #[test]
    fn test_filter_passthrough() {
        let mut s = PerRequestState::default();
        assert_eq!(apply_output_filter(&mut s, "hello"), "hello");
        assert_eq!(apply_output_filter(&mut s, " world"), " world");
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
        // Stop string matches exactly the current token
        let (text, hit) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["User:"]));
        assert_eq!(text, "", "stop string itself must not be emitted");
        assert!(hit);
    }

    #[test]
    fn test_stop_partial_current_token_emitted() {
        let mut s = PerRequestState::default();
        // "Hello\nUser:" — stop is "\nUser:", so "Hello" should be emitted
        let (text, hit) =
            check_stop_sequences(&mut s, "Hello\nUser:".to_string(), &stops(&["\nUser:"]));
        assert_eq!(text, "Hello");
        assert!(hit);
    }

    #[test]
    fn test_stop_multi_token_span() {
        let mut s = PerRequestState::default();
        // First token builds up buffer without triggering stop
        let (t1, h1) = check_stop_sequences(&mut s, "Hello\n".to_string(), &stops(&["\nUser:"]));
        assert_eq!(t1, "Hello\n");
        assert!(!h1);
        // Second token completes the stop string
        let (t2, h2) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["\nUser:"]));
        // The portion "User:" was already in the stop string that started in the previous token
        // so nothing from the current token should be emitted.
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
