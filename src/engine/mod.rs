// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use tracing::debug;

use crate::kv_cache::KVCacheManager;
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

/// Main inference engine coordinating scheduler, model, and KV cache.
pub struct InferenceEngine {
    model: Arc<dyn Model>,
    scheduler: Arc<crate::scheduler::Scheduler>,
    kv_cache: Arc<KVCacheManager>,
    next_request_id: AtomicU64,
    /// Per-request state for output filtering (e.g. <think> block suppression).
    output_filter_state: Arc<Mutex<HashMap<u64, OutputFilterState>>>,
}

#[derive(Default)]
struct OutputFilterState {
    in_thinking: bool,
    /// Last character sent (for inserting missing spaces between words)
    last_char: Option<char>,
}

impl InferenceEngine {
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
    ) -> Self {
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            output_filter_state: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Tokenize text using the model's vocabulary.
    pub fn tokenize(&self, text: &str) -> anyhow::Result<Vec<i32>> {
        self.model.tokenize(text)
    }

    /// Apply chat template to format messages for the model.
    pub fn apply_chat_template(&self, messages: &[(String, String)]) -> anyhow::Result<String> {
        self.model.apply_chat_template(messages)
    }

    /// Submit a request to the scheduler.
    pub fn submit_request(&self, req: InferenceRequest) {
        self.scheduler.submit(req);
    }

    /// Allocate a unique request ID.
    pub fn next_request_id(&self) -> u64 {
        self.next_request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Main inference loop.
    pub async fn run_loop(self: Arc<Self>) -> Result<()> {
        let engine = self.clone();
        loop {
            let batch = engine.scheduler.schedule_step();

            if batch.is_empty() {
                tokio::time::sleep(Duration::from_micros(100)).await;
                continue;
            }

            // Run prefill and decode in parallel when both are non-empty
            let prefill_ids = batch.prefill.clone();
            let decode_ids = batch.decode.clone();

            if !prefill_ids.is_empty() {
                let prefill_results = engine.run_prefill(&prefill_ids).await?;
                engine.handle_logits(&prefill_results, true).await?;
            }

            if !decode_ids.is_empty() {
                let decode_results = engine.run_decode(&decode_ids).await?;
                engine.handle_logits(&decode_results, false).await?;
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
            })
            .collect();
        let model = self.model.clone();
        let req_ids = req_ids.to_vec();
        let model_requests = model_requests.clone();
        tokio::task::spawn_blocking(move || model.prefill_sync(&req_ids, &model_requests))
            .await
            .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))?
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
            })
            .collect();
        let model = self.model.clone();
        let req_ids = req_ids.to_vec();
        let model_requests = model_requests.clone();
        tokio::task::spawn_blocking(move || model.decode_sync(&req_ids, &model_requests))
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

            // Detokenize for streaming; skip text for EOS (control tokens like <|im_end|>)
            let mut raw_text = if is_eos {
                String::new()
            } else {
                self.model
                    .token_to_piece(token_id)
                    .unwrap_or_default()
            };
            // SentencePiece: ▁ (U+2581) means word boundary → insert space
            raw_text = raw_text.replace(SPM_SPACE, " ");

            // Filter special tokens, <think> blocks, and other control tokens
            let text = self.filter_output_text(*req_id, raw_text);

            let _ = req.response_tx.send(Token {
                id: *req_id,
                token_id,
                text,
                is_eos,
            });

            debug!(request_id = req_id, token_id, "token generated");

            if is_eos || reached_max {
                let reason = if is_eos {
                    StopReason::Eos
                } else {
                    StopReason::Length
                };
                self.scheduler.mark_finished(*req_id, reason);
                // Clean up filter state
                if let Ok(mut state) = self.output_filter_state.lock() {
                    state.remove(req_id);
                }
            } else {
                if from_prefill {
                    self.scheduler.mark_prefill_done(*req_id);
                }
                self.scheduler.set_last_token(*req_id, token_id);
                self.scheduler.increment_generated(*req_id);
            }
        }

        Ok(())
    }

    /// Filter output text: remove special tokens, <think> blocks, and control sequences.
    fn filter_output_text(&self, req_id: u64, raw: String) -> String {
        if raw.is_empty() {
            return raw;
        }

        let mut state_map = match self.output_filter_state.lock() {
            Ok(g) => g,
            Err(_) => return raw,
        };
        let state = state_map.entry(req_id).or_default();

        // 1. Check for special tokens - always suppress
        for &pattern in SPECIAL_TOKEN_PATTERNS {
            if raw == pattern || raw.contains(pattern) {
                return String::new();
            }
        }
        // Suppress any token containing <|...|> (common special token pattern)
        if raw.contains("<|") && raw.contains("|>") {
            return String::new();
        }

        // 2. Check for <think> tag - enter thinking block
        if raw.contains("<think>") {
            state.in_thinking = true;
            return String::new();
        }

        // 3. Check for </think> tag - exit thinking block
        if raw.contains("</think>") {
            state.in_thinking = false;
            if let Some(idx) = raw.find("</think>") {
                let after = idx + "</think>".len();
                if after < raw.len() {
                    let text = raw[after..].to_string();
                    return Self::maybe_add_space_and_update_last(state, text);
                }
            }
            return String::new();
        }

        // 4. If inside thinking block, suppress
        if state.in_thinking {
            return String::new();
        }

        // 5. Strip any leading/trailing control fragments (e.g. stray "<" or ">")
        let text = if raw != raw.trim() && !raw.trim().is_empty() {
            raw.trim().to_string()
        } else {
            raw
        };

        Self::maybe_add_space_and_update_last(state, text)
    }

    /// Insert space between concatenated words and track last char for next token.
    fn maybe_add_space_and_update_last(state: &mut OutputFilterState, mut text: String) -> String {
        fn is_word_char(c: char) -> bool {
            c.is_alphanumeric() || c == '\'' || c == '-' || c == '\u{00e9}'
        }
        if text.is_empty() {
            return text;
        }
        // If prev token ended with word char and this token starts with word char (no space), add one
        if let Some(prev) = state.last_char {
            if is_word_char(prev) {
                let first = text.chars().next().unwrap();
                if is_word_char(first) && first != ' ' {
                    text.insert_str(0, " ");
                }
            }
        }
        state.last_char = text.chars().last();
        text
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
}
