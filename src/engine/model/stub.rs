// StubModel — test-only Model implementation (no FFI).

use anyhow::Result;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{InferenceRequestForModel, Logits, Model, ModelConfig};

pub struct StubModel;

impl Model for StubModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        Ok(req_ids
            .iter()
            .zip(requests.iter())
            .map(|(&id, r)| {
                // Return a non-EOS token so the engine emits one text token.
                (id, Logits::new(vec![], 65), r.prompt_tokens.len())
            })
            .collect())
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        // Return EOS immediately — generation ends after one text token.
        Ok(req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], 2)))
            .collect())
    }

    fn model_config(&self) -> ModelConfig {
        ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            vocab_size: 256,
        }
    }

    fn eos_token_id(&self) -> i32 {
        2
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        token_id == 2
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(text.bytes().map(|b| b as i32).collect())
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        if token == 2 {
            Ok(String::new())
        } else {
            Ok("hi ".to_string())
        }
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        Ok(messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn clear_sequence(&self, _seq_id: i32) {}

    fn copy_sequence_range(&self, _src: i32, _dst: i32, _count: i32) {}

    fn supports_seq_copy(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> usize {
        4
    }

    fn get_embeddings(&self, _tokens: &[i32]) -> Result<Vec<f32>> {
        Ok(vec![0.1, 0.2, 0.3, 0.4])
    }

    fn stop_tokens(&self) -> Vec<String> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// ThinkingStubModel — simulates a reasoning model (supports_thinking = true)
// ---------------------------------------------------------------------------
//
// Token sequence emitted (after <think>\n is injected into the prompt):
//
//   prefill  → 300  "thought"    ← thinking content (already inside block)
//   decode 0 → 301  "</think>"   ← closes the thinking block
//   decode 1 → 302  "answer"     ← visible content
//   decode 2+ → 2   (EOS)
//
// Expected outputs by mode:
//   show_thinking=false, initial_in_thinking=true  →  content = "answer"
//   show_thinking=true,  initial_in_thinking=true  →  full = "<think>\nthought</think>answer"
//     → extract_thinking → thinking = "thought", content = "answer"
pub struct ThinkingStubModel {
    decode_step: AtomicUsize,
}

impl ThinkingStubModel {
    pub fn new() -> Self {
        Self {
            decode_step: AtomicUsize::new(0),
        }
    }
}

impl Model for ThinkingStubModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        Ok(req_ids
            .iter()
            .zip(requests.iter())
            .map(|(&id, r)| (id, Logits::new(vec![], 300), r.prompt_tokens.len()))
            .collect())
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        const DECODE_SEQ: &[i32] = &[301, 302, 2];
        let step = self.decode_step.fetch_add(1, Ordering::SeqCst);
        let token = DECODE_SEQ.get(step).copied().unwrap_or(2);
        Ok(req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], token)))
            .collect())
    }

    fn model_config(&self) -> ModelConfig {
        StubModel.model_config()
    }

    fn eos_token_id(&self) -> i32 {
        2
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        token_id == 2
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(text.bytes().map(|b| b as i32).collect())
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        Ok(match token {
            2 => String::new(),
            300 => "thought".to_string(),
            301 => "</think>".to_string(),
            302 => "answer".to_string(),
            _ => "hi ".to_string(),
        })
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        StubModel.apply_chat_template(messages)
    }

    fn supports_thinking(&self) -> bool {
        true
    }

    fn clear_sequence(&self, _seq_id: i32) {}

    fn copy_sequence_range(&self, _src: i32, _dst: i32, _count: i32) {}

    fn supports_seq_copy(&self) -> bool {
        true
    }

    fn embedding_dim(&self) -> usize {
        4
    }

    fn get_embeddings(&self, _tokens: &[i32]) -> Result<Vec<f32>> {
        Ok(vec![0.1, 0.2, 0.3, 0.4])
    }

    fn stop_tokens(&self) -> Vec<String> {
        vec![]
    }
}
