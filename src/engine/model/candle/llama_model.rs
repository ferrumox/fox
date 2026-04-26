//! `Model`-trait wrapper around [`LlamaArch`].
//!
//! Phase C.3.2 scope: single in-flight sequence per instance, greedy decoding
//! (the request's sampling parameters are accepted but ignored — proper
//! sampling integration happens in phase D when both backends share the same
//! sampler entry point). The KV cache lives inside the underlying
//! `quantized_llama::ModelWeights` and grows monotonically; `clear_sequence`
//! resets the position cursor but the cache itself is not flushed — calling
//! into a fresh prompt after `clear_sequence` is therefore *not* a clean
//! reset and is reserved for C.3.4.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use candle_core::Device;

use super::chat_template::apply_chat_template;
use super::gguf_metadata::{self, GgufMetadata};
use super::llama_arch::LlamaArch;
use super::tokenizer::{vocab_from_metadata, ByteBpeTokenizer, Vocab};
use crate::engine::model::{
    InferenceRequestForModel, Logits, Model, ModelConfig, RecommendedSampling,
};

pub struct CandleLlamaModel {
    arch: LlamaArch,
    tokenizer: ByteBpeTokenizer,
    vocab: Vocab,
    config: ModelConfig,
    context_len: u32,
    eos_token_id: i32,
    /// Absolute position of the next token to write into the KV cache.
    /// Bumped by `prefill_sync` (by prompt length) and `decode_sync` (by 1).
    cursor: Mutex<usize>,
}

impl CandleLlamaModel {
    /// Load a Llama-family GGUF and build the runtime around it.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let meta = gguf_metadata::load(path)
            .map_err(|e| anyhow!("failed to read GGUF metadata: {e}"))?;
        let vocab =
            vocab_from_metadata(&meta).map_err(|e| anyhow!("failed to load vocab: {e}"))?;
        let config = build_model_config(&meta, vocab.size())?;
        let context_len = meta
            .arch_uint("context_length")
            .map(|v| v as u32)
            .unwrap_or(4096);
        let eos = vocab.specials.eos.map(|v| v as i32).unwrap_or(-1);
        let arch = LlamaArch::from_gguf(path, device)
            .map_err(|e| anyhow!("failed to load weights: {e}"))?;
        let tokenizer = ByteBpeTokenizer::new(vocab.clone());
        Ok(Self {
            arch,
            tokenizer,
            vocab,
            config,
            context_len,
            eos_token_id: eos,
            cursor: Mutex::new(0),
        })
    }

    fn ensure_single_request(req_ids: &[u64]) -> Result<()> {
        if req_ids.len() > 1 {
            return Err(anyhow!(
                "candle backend phase C.3 only supports single-sequence prefill/decode \
                 (received batch of {}); multi-sequence support is C.3.4 work",
                req_ids.len()
            ));
        }
        Ok(())
    }

    fn forward_at(&self, tokens: &[i32], position: usize) -> Result<i32> {
        let logits = self
            .arch
            .forward(tokens, position)
            .map_err(|e| anyhow!("candle forward failed: {e}"))?;
        Ok(greedy_argmax(&logits))
    }
}

/// Translate a `GgufMetadata` block into the runtime `ModelConfig` shape.
/// Falls back to the standard derivation `head_dim = embedding_length /
/// head_count` when `attention.key_length` is absent.
fn build_model_config(meta: &GgufMetadata, vocab_size: usize) -> Result<ModelConfig> {
    let num_layers = meta
        .arch_uint("block_count")
        .ok_or_else(|| anyhow!("metadata missing '{}.block_count'", meta.architecture))?
        as usize;
    let num_heads = meta
        .arch_uint("attention.head_count")
        .ok_or_else(|| {
            anyhow!(
                "metadata missing '{}.attention.head_count'",
                meta.architecture
            )
        })? as usize;
    let num_heads_kv = meta
        .arch_uint("attention.head_count_kv")
        .map(|v| v as usize)
        .unwrap_or(num_heads);
    let embedding_length = meta
        .arch_uint("embedding_length")
        .ok_or_else(|| anyhow!("metadata missing '{}.embedding_length'", meta.architecture))?
        as usize;
    let head_dim = meta
        .arch_uint("attention.key_length")
        .map(|v| v as usize)
        .unwrap_or_else(|| {
            if num_heads > 0 {
                embedding_length / num_heads
            } else {
                0
            }
        });
    Ok(ModelConfig {
        num_layers,
        num_heads,
        num_heads_kv,
        head_dim,
        vocab_size,
    })
}

/// Greedy: pick the token id with the largest logit.
fn greedy_argmax(logits: &[f32]) -> i32 {
    let mut best_idx = 0i32;
    let mut best_score = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_score {
            best_score = v;
            best_idx = i as i32;
        }
    }
    best_idx
}

impl Model for CandleLlamaModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        Self::ensure_single_request(req_ids)?;
        let req = &requests[0];
        let id = req_ids[0];

        let mut cursor = self
            .cursor
            .lock()
            .map_err(|e| anyhow!("candle cursor mutex poisoned: {e}"))?;
        let position = *cursor;
        let tokens: Vec<i32> = req.prompt_tokens.clone();
        let sampled = self.forward_at(&tokens, position)?;
        *cursor = position + tokens.len();
        Ok(vec![(id, Logits::new(Vec::new(), sampled), tokens.len())])
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        Self::ensure_single_request(req_ids)?;
        let req = &requests[0];
        let id = req_ids[0];
        let last = req
            .last_token
            .ok_or_else(|| anyhow!("decode_sync requires last_token to be set"))?;

        let mut cursor = self
            .cursor
            .lock()
            .map_err(|e| anyhow!("candle cursor mutex poisoned: {e}"))?;
        let position = *cursor;
        let sampled = self.forward_at(&[last], position)?;
        *cursor = position + 1;
        Ok(vec![(id, Logits::new(Vec::new(), sampled))])
    }

    fn model_config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn eos_token_id(&self) -> i32 {
        self.eos_token_id
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        if token_id < 0 {
            return true;
        }
        let id = token_id as u32;
        matches!(self.vocab.specials.eos, Some(v) if v == id)
            || matches!(self.vocab.specials.eot, Some(v) if v == id)
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        Ok(self
            .tokenizer
            .encode(text)
            .into_iter()
            .map(|t| t as i32)
            .collect())
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        if token < 0 {
            return Ok(String::new());
        }
        Ok(self.tokenizer.decode(&[token as u32]))
    }

    fn token_to_piece_bytes(&self, token: i32) -> Vec<u8> {
        self.token_to_piece(token).unwrap_or_default().into_bytes()
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        if self.vocab.chat_template.is_some() {
            apply_chat_template(&self.vocab, messages, true)
                .map_err(|e| anyhow!("chat template render failed: {e}"))
        } else {
            // No template in the GGUF — fall back to a join the engine still
            // understands (matches the default fallback documented on the
            // `Model::apply_chat_template` trait method).
            Ok(messages
                .iter()
                .map(|(role, content)| format!("{role}: {content}"))
                .collect::<Vec<_>>()
                .join("\n"))
        }
    }

    fn context_len(&self) -> u32 {
        self.context_len
    }

    fn supports_thinking(&self) -> bool {
        false
    }

    fn uses_channel_thinking(&self) -> bool {
        false
    }

    fn recommended_sampling(&self) -> Option<RecommendedSampling> {
        None
    }

    fn clear_sequence(&self, _seq_id: i32) {
        if let Ok(mut c) = self.cursor.lock() {
            *c = 0;
        }
        // Note: the underlying ModelWeights KV cache is *not* cleared. C.3.4
        // will introduce an external KV cache and a real reset path.
    }

    fn copy_sequence_range(&self, _src_seq_id: i32, _dst_seq_id: i32, _token_count: i32) {
        // Single-sequence model — copying is a no-op.
    }

    fn supports_seq_copy(&self) -> bool {
        false
    }

    fn embedding_dim(&self) -> usize {
        self.config.head_dim * self.config.num_heads
    }

    fn get_embeddings(&self, _tokens: &[i32]) -> Result<Vec<f32>> {
        Err(anyhow!(
            "embeddings are not implemented for the candle backend in phase C.3"
        ))
    }

    fn stop_tokens(&self) -> Vec<String> {
        let mut out = Vec::new();
        for slot in [self.vocab.specials.eos, self.vocab.specials.eot]
            .into_iter()
            .flatten()
        {
            if let Some(s) = self.vocab.token(slot) {
                out.push(s.to_string());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_returns_index_of_max() {
        assert_eq!(greedy_argmax(&[0.1, 0.5, -0.2, 0.7, 0.3]), 3);
    }

    #[test]
    fn greedy_argmax_is_zero_on_empty_input() {
        assert_eq!(greedy_argmax(&[]), 0);
    }

    #[test]
    fn build_model_config_derives_head_dim_from_embedding_length() {
        use crate::engine::model::candle::gguf_metadata::GgufMetadata;
        let mut meta = GgufMetadata::default();
        meta.architecture = "llama".into();
        meta.uints.insert("llama.block_count".into(), 28);
        meta.uints.insert("llama.attention.head_count".into(), 24);
        meta.uints
            .insert("llama.attention.head_count_kv".into(), 8);
        meta.uints.insert("llama.embedding_length".into(), 3072);
        let cfg = build_model_config(&meta, 128_256).unwrap();
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.num_heads, 24);
        assert_eq!(cfg.num_heads_kv, 8);
        assert_eq!(cfg.head_dim, 128); // 3072 / 24
        assert_eq!(cfg.vocab_size, 128_256);
    }

    #[test]
    fn build_model_config_uses_explicit_key_length_when_present() {
        use crate::engine::model::candle::gguf_metadata::GgufMetadata;
        let mut meta = GgufMetadata::default();
        meta.architecture = "gemma4".into();
        meta.uints.insert("gemma4.block_count".into(), 30);
        meta.uints.insert("gemma4.attention.head_count".into(), 7);
        meta.uints.insert("gemma4.embedding_length".into(), 1536);
        meta.uints
            .insert("gemma4.attention.key_length".into(), 512);
        let cfg = build_model_config(&meta, 262_144).unwrap();
        assert_eq!(cfg.head_dim, 512);
    }

    #[test]
    fn build_model_config_errors_on_missing_required_keys() {
        use crate::engine::model::candle::gguf_metadata::GgufMetadata;
        let mut meta = GgufMetadata::default();
        meta.architecture = "llama".into();
        let err = build_model_config(&meta, 32_000).unwrap_err();
        assert!(err.to_string().contains("block_count"));
    }
}
