// StubModel — test-only Model implementation (no FFI).

use anyhow::Result;

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
