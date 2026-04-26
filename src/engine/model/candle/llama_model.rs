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
use dashmap::DashMap;

use super::chat_template::apply_chat_template;
use super::gguf_metadata::{self, GgufMetadata};
use super::llama_arch::LlamaArch;
use super::tokenizer::{vocab_from_metadata, ByteBpeTokenizer, Vocab};
use crate::engine::model::mirostat::{self, MirostatV2};
use crate::engine::model::sampling::{sample_token, SamplerParams};
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
    /// Mirostat v2 state per request id. Populated lazily on the first
    /// sample for a given request and dropped on `clear_sequence`.
    mirostat_states: DashMap<u64, MirostatV2>,
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
            mirostat_states: DashMap::new(),
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

    /// Run the forward pass and return the raw logits for the last token.
    fn forward_logits(&self, tokens: &[i32], position: usize) -> Result<Vec<f32>> {
        self.arch
            .forward(tokens, position)
            .map_err(|e| anyhow!("candle forward failed: {e}"))
    }
}

/// Apply the shared sampling pipeline (the same one llama.cpp goes through)
/// to the raw logits, using the per-request parameters carried in
/// `InferenceRequestForModel`. Dispatches to Mirostat v2 when
/// `mirostat_tau > 0` and falls back to the regular stochastic / greedy
/// pipeline otherwise. When `temperature <= 0` the regular path short-
/// circuits to greedy regardless of the other knobs, so the candle backend
/// still produces deterministic output for the default zero-temperature
/// requests our parity tests use.
fn sample_with_request(
    logits: &[f32],
    req: &InferenceRequestForModel,
    mirostat_states: &DashMap<u64, MirostatV2>,
) -> i32 {
    let token_count = req.generated_tokens + req.prompt_tokens.len();

    let bias_opt: Option<&_> = if req.logit_bias.is_empty() {
        None
    } else {
        Some(&req.logit_bias)
    };

    if req.mirostat_tau > 0.0 {
        let mut state = mirostat_states
            .entry(req.id)
            .or_insert_with(|| MirostatV2::new(req.mirostat_tau, req.mirostat_eta));
        return mirostat::sample(
            logits,
            state.value_mut(),
            req.seed,
            token_count,
            bias_opt,
        );
    }

    let counts: Option<&_> = if req.token_counts.is_empty() {
        None
    } else {
        Some(&req.token_counts)
    };
    let params = SamplerParams {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        repetition_penalty: req.repetition_penalty,
        token_counts: counts,
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        generated_ids: &req.generated_token_ids,
        seed: req.seed,
        token_count,
        logit_bias: bias_opt,
    };
    sample_token(logits, params)
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
        let logits = self.forward_logits(&tokens, position)?;
        let sampled = sample_with_request(&logits, req, &self.mirostat_states);
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
        let logits = self.forward_logits(&[last], position)?;
        let sampled = sample_with_request(&logits, req, &self.mirostat_states);
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
        // Reset the cursor to 0. The KV cache itself is *not* explicitly
        // cleared, but the next call to `prefill_sync` will pass
        // `position = 0` to `LlamaArch::forward`, and `quantized_llama`'s
        // attention layer treats `index_pos == 0` as an override of any
        // stored cache rather than a concatenation. Net effect: the next
        // conversation starts from a clean context.
        //
        // This works because we only ever serve ONE active sequence per
        // model instance (single-batch limitation; see prefill/decode_sync).
        // For concurrent multi-sequence support, the KV cache must be lifted
        // out of `ModelWeights` into a fox-managed pool — phase C.3.4 work.
        if let Ok(mut c) = self.cursor.lock() {
            *c = 0;
        }
        // Drop any per-request Mirostat state — reuse of the same model
        // instance for a new conversation should start from μ = 2τ again.
        self.mirostat_states.clear();
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

    /// Helper that builds a default-shaped request so the sampler tests stay
    /// concise. Mirrors `Default` semantics — `temperature: 0` selects greedy.
    fn fake_request(temperature: f32, seed: Option<u64>) -> InferenceRequestForModel {
        InferenceRequestForModel {
            id: 0,
            prompt_tokens: vec![1, 2, 3],
            last_token: None,
            generated_tokens: 0,
            max_new_tokens: 1,
            context_len: 32,
            kv_seq_id: 0,
            temperature,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            min_p: 0.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed,
            generated_token_ids: Vec::new(),
            token_counts: std::collections::HashMap::new(),
            mirostat_tau: 0.0,
            mirostat_eta: 0.1,
            logit_bias: std::collections::HashMap::new(),
            skip_prefix_tokens: 0,
            prefix_seq_id: None,
        }
    }

    #[test]
    fn sample_with_request_is_greedy_at_zero_temperature() {
        let logits = [0.1, 0.5, -0.2, 0.7, 0.3];
        let req = fake_request(0.0, None);
        let states = DashMap::new();
        assert_eq!(sample_with_request(&logits, &req, &states), 3);
    }

    #[test]
    fn sample_with_request_is_reproducible_with_a_seed() {
        let logits = [0.1, 0.4, 0.3, 0.2];
        let req = fake_request(1.0, Some(42));
        let states = DashMap::new();
        let a = sample_with_request(&logits, &req, &states);
        let b = sample_with_request(&logits, &req, &states);
        assert_eq!(a, b, "same seed must produce the same draw");
    }

    #[test]
    fn mirostat_path_takes_over_when_tau_is_set() {
        let logits = [0.5, 0.4, 0.3, 0.2, 0.1];
        let mut req = fake_request(1.0, Some(7));
        req.id = 99;
        req.mirostat_tau = 5.0;
        req.mirostat_eta = 0.1;
        let states: DashMap<u64, MirostatV2> = DashMap::new();
        let token = sample_with_request(&logits, &req, &states);
        assert!(token >= 0 && (token as usize) < logits.len());
        assert!(
            states.contains_key(&99),
            "mirostat state should be created on first sample"
        );
        let after = states.get(&99).unwrap().mu;
        assert!(after.is_finite());
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

    /// End-to-end check that the candle backend honours the request's
    /// sampling parameters: temperature 0 is greedy (deterministic), and
    /// temperature > 0 with a seed is reproducible. Loads ~2GB of weights —
    /// gated behind `#[ignore]` so the standard suite stays fast.
    #[test]
    #[ignore = "loads ~2GB Llama-3.2-3B; run explicitly with --ignored"]
    fn prefill_sync_routes_request_params_through_the_sampler() {
        use candle_core::Device;
        use std::path::PathBuf;

        let home = std::env::var("HOME").expect("HOME must be set");
        let path = PathBuf::from(home)
            .join(".cache/ferrumox/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping — fixture missing: {}", path.display());
            return;
        }

        let model = CandleLlamaModel::load(&path, Device::Cpu).expect("model loads");
        let prompt: Vec<i32> = vec![9906]; // "Hello"

        // Greedy: deterministic.
        let mut req = fake_request(0.0, None);
        req.prompt_tokens = prompt.clone();
        let r1 = model
            .prefill_sync(&[1], std::slice::from_ref(&req))
            .expect("greedy prefill");
        let greedy_a = r1[0].1.sampled_token;

        model.clear_sequence(0);
        let r2 = model
            .prefill_sync(&[2], std::slice::from_ref(&req))
            .expect("greedy prefill again");
        let greedy_b = r2[0].1.sampled_token;
        assert_eq!(greedy_a, greedy_b, "temperature 0 must be deterministic");

        // Stochastic with a seed: reproducible.
        let mut req_seeded = fake_request(1.5, Some(42));
        req_seeded.prompt_tokens = prompt.clone();
        req_seeded.top_k = 0; // no truncation, exercise the full distribution

        model.clear_sequence(0);
        let r3 = model
            .prefill_sync(&[3], std::slice::from_ref(&req_seeded))
            .expect("seeded prefill");
        let seeded_a = r3[0].1.sampled_token;

        model.clear_sequence(0);
        let r4 = model
            .prefill_sync(&[4], std::slice::from_ref(&req_seeded))
            .expect("seeded prefill again");
        let seeded_b = r4[0].1.sampled_token;
        assert_eq!(
            seeded_a, seeded_b,
            "same seed must produce the same sampled token"
        );

        // Sampled tokens must be valid vocab ids.
        for &tok in &[greedy_a, seeded_a] {
            assert!(tok >= 0, "token id is non-negative");
            assert!((tok as usize) < model.config.vocab_size, "token id in range");
        }

        eprintln!("greedy={greedy_a}, seeded(1.5,42)={seeded_a}");

        // Mirostat path: distinct from greedy, deterministic with same seed.
        let mut req_mirostat = fake_request(1.0, Some(99));
        req_mirostat.id = 100;
        req_mirostat.prompt_tokens = prompt.clone();
        req_mirostat.mirostat_tau = 5.0;
        req_mirostat.mirostat_eta = 0.1;

        model.clear_sequence(0);
        let r5 = model
            .prefill_sync(&[100], std::slice::from_ref(&req_mirostat))
            .expect("mirostat prefill");
        let mirostat_a = r5[0].1.sampled_token;

        model.clear_sequence(0); // also drops the mirostat state
        let r6 = model
            .prefill_sync(&[100], std::slice::from_ref(&req_mirostat))
            .expect("mirostat prefill again");
        let mirostat_b = r6[0].1.sampled_token;
        assert_eq!(
            mirostat_a, mirostat_b,
            "mirostat with the same seed must reproduce"
        );
        assert!(mirostat_a >= 0 && (mirostat_a as usize) < model.config.vocab_size);
        eprintln!("mirostat(τ=5,η=0.1,seed=99)={mirostat_a}");

        // Logit bias: force a specific token under greedy. Without bias the
        // model picks something semantic ("Question"); with a strong bias on
        // token 9906 ("Hello"), the sampler must pick 9906.
        let mut req_biased = fake_request(0.0, None);
        req_biased.id = 200;
        req_biased.prompt_tokens = prompt.clone();
        req_biased.logit_bias.insert(9906, 1000.0);

        model.clear_sequence(0);
        let r7 = model
            .prefill_sync(&[200], std::slice::from_ref(&req_biased))
            .expect("biased prefill");
        let biased_token = r7[0].1.sampled_token;
        assert_eq!(
            biased_token, 9906,
            "logit_bias=+1000 on token 9906 must force the sampler to pick it"
        );
        assert_ne!(
            greedy_a, 9906,
            "sanity: unbiased greedy should pick a different token"
        );
        eprintln!("logit_bias(token=9906,+1000)={biased_token}");
    }
}
