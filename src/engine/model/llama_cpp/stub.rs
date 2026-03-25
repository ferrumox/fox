// Stub implementation — compiled only when the `fox_stub` cfg flag is set.
// Provides a no-op LlamaCppModel that satisfies the Model trait without any FFI.

#[cfg(fox_stub)]
use anyhow::Result;

#[cfg(fox_stub)]
use crate::engine::model::{InferenceRequestForModel, Logits, Model, ModelConfig};

#[cfg(fox_stub)]
pub struct LlamaCppModel {
    pub(super) config: ModelConfig,
}

#[cfg(fox_stub)]
impl LlamaCppModel {
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
        max_context_len: Option<u32>,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        type_kv: u32,
        main_gpu: i32,
        split_mode: u32,
        tensor_split: &[f32],
        moe_offload_cpu: bool,
    ) -> Result<Self> {
        let _ = (
            model_path,
            max_batch_size,
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_kv,
            main_gpu,
            split_mode,
            tensor_split,
            moe_offload_cpu,
        );
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        };
        Ok(Self { config })
    }
}

#[cfg(fox_stub)]
unsafe impl Send for LlamaCppModel {}
#[cfg(fox_stub)]
unsafe impl Sync for LlamaCppModel {}

#[cfg(fox_stub)]
impl Model for LlamaCppModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        let results: Vec<_> = req_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let tokens_in_kv = requests.get(i).map(|r| r.prompt_tokens.len()).unwrap_or(0);
                (id, Logits::new(vec![], 2), tokens_in_kv)
            })
            .collect();
        Ok(results)
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        let results: Vec<_> = req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], 2)))
            .collect();
        Ok(results)
    }

    fn model_config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn eos_token_id(&self) -> i32 {
        2
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        token_id == 2
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Stub: return byte-level tokens
        let tokens: Vec<i32> = text.bytes().map(|b| b as i32).collect();
        Ok(if tokens.is_empty() { vec![0] } else { tokens })
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        let _ = token;
        Ok(String::new())
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        Ok(messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn clear_sequence(&self, _seq_id: i32) {}

    fn copy_sequence_range(&self, _src_seq_id: i32, _dst_seq_id: i32, _token_count: i32) {}

    fn supports_seq_copy(&self) -> bool {
        false
    }

    fn embedding_dim(&self) -> usize {
        self.config.num_heads * self.config.head_dim
    }

    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let _ = tokens;
        Ok(vec![0.0f32; self.embedding_dim()])
    }

    fn stop_tokens(&self) -> Vec<String> {
        vec![]
    }
}
