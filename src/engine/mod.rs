// Inference engine - main loop coordinating scheduler, model, and KV cache.

mod ffi;
pub mod model;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
    /// Human-readable model identifier (basename of the loaded model file).
    model_name: String,
}

#[derive(Default)]
struct OutputFilterState {
    in_thinking: bool,
}

impl InferenceEngine {
    pub fn new(
        model: Arc<dyn Model>,
        scheduler: Arc<crate::scheduler::Scheduler>,
        kv_cache: Arc<KVCacheManager>,
        model_name: String,
    ) -> Self {
        Self {
            model,
            scheduler,
            kv_cache,
            next_request_id: AtomicU64::new(0),
            output_filter_state: Arc::new(Mutex::new(HashMap::new())),
            model_name,
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

    /// Human-readable name of the loaded model.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Main inference loop.
    pub async fn run_loop(self: Arc<Self>) -> Result<()> {
        let engine = self.clone();
        loop {
            let batch = engine.scheduler.schedule_step();

            // Wipe KV state for preempted sequences BEFORE their IDs can be reassigned.
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

        // Collect prefix seq_ids BEFORE moving model_requests into spawn_blocking.
        // After the blocking task returns we'll clear and return them to the pool.
        let prefix_cleanup: Vec<(i32,)> = model_requests
            .iter()
            .filter_map(|r| r.prefix_seq_id.map(|ps| (ps,)))
            .collect();

        let model = self.model.clone();
        let req_ids_vec = req_ids.to_vec();
        let result = tokio::task::spawn_blocking(move || {
            model.prefill_sync(&req_ids_vec, &model_requests)
        })
        .await
        .map_err(|e| anyhow::anyhow!("prefill spawn_blocking: {}", e))??;

        // Return each prefix seq_id to the pool now that the KV copy is done.
        // The copy itself happened inside do_prefill (before building the batch).
        for (prefix_seq_id,) in prefix_cleanup {
            self.model.clear_sequence(prefix_seq_id);
            self.scheduler.return_prefix_seq_id(prefix_seq_id);
        }

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
            let is_done = is_eos || reached_max;

            let stop_reason = if is_done {
                Some(if is_eos { StopReason::Eos } else { StopReason::Length })
            } else {
                None
            };

            let mut raw_text = if is_eos {
                String::new()
            } else {
                self.model
                    .token_to_piece(token_id)
                    .unwrap_or_default()
            };
            raw_text = raw_text.replace(SPM_SPACE, " ");

            let text = self.filter_output_text(*req_id, raw_text);

            let _ = req.response_tx.send(Token {
                id: *req_id,
                token_id,
                text,
                is_eos,
                stop_reason: stop_reason.clone(),
            });

            debug!(request_id = req_id, token_id, "token generated");

            if is_done {
                // Attempt to cache the KV state for this request before clearing it.
                // try_insert_prefix atomically moves seq_id + blocks into the cache
                // and zeroes them on the InferenceRequest so schedule_step won't free them.
                let should_clear = if matches!(stop_reason, Some(StopReason::Eos) | Some(StopReason::Length)) {
                    let hash = hash_tokens(&req.prompt_tokens);
                    let cached = self.scheduler.try_insert_prefix(*req_id, hash);
                    !cached // if not cached, still need to clear
                } else {
                    true
                };

                if should_clear && req.kv_seq_id >= 0 {
                    self.model.clear_sequence(req.kv_seq_id);
                }

                self.scheduler.mark_finished(*req_id, stop_reason.unwrap());

                if let Ok(mut state) = self.output_filter_state.lock() {
                    state.remove(req_id);
                }
            } else {
                self.scheduler.update_after_token(*req_id, token_id, from_prefill);
            }
        }

        Ok(())
    }

    /// Filter output text: remove special tokens, <think> blocks, and other control sequences.
    fn filter_output_text(&self, req_id: u64, raw: String) -> String {
        if raw.is_empty() {
            return raw;
        }

        let mut state_map = match self.output_filter_state.lock() {
            Ok(g) => g,
            Err(_) => return raw,
        };
        let state = state_map.entry(req_id).or_default();

        for &pattern in SPECIAL_TOKEN_PATTERNS {
            if raw == pattern || raw.contains(pattern) {
                return String::new();
            }
        }
        if raw.contains("<|") {
            return String::new();
        }

        if raw.contains("<think>") {
            state.in_thinking = true;
            return String::new();
        }

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

        if state.in_thinking {
            return String::new();
        }

        raw
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

/// Stable hash for a slice of token IDs (delegates to scheduler's helper).
fn hash_tokens(tokens: &[i32]) -> u64 {
    let mut h = DefaultHasher::new();
    tokens.hash(&mut h);
    h.finish()
}
