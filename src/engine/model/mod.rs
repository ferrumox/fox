// Model trait and shared types.
//
// Sub-modules:
//   sampling   — token sampling (temperature, top-k, top-p, repetition penalty)
//   llama_cpp  — LlamaCppModel implementation (real + fox_stub variant)
//   stub       — StubModel for tests / test-helpers feature

use anyhow::Result;

pub(crate) mod llama_cpp;
#[cfg(not(fox_stub))]
pub(crate) mod sampling;
#[cfg(any(test, feature = "test-helpers"))]
pub(crate) mod stub;

pub use llama_cpp::LlamaCppModel;
#[cfg(any(test, feature = "test-helpers"))]
pub use stub::{StubModel, ThinkingStubModel};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Sampling parameters recommended by the model's GGUF metadata.
/// Fields are `None` when the model doesn't specify a recommendation for that parameter.
#[derive(Debug, Clone)]
pub struct RecommendedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}

/// Model architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
}

/// Logits from a single decode step (vocab_size floats).
#[derive(Debug, Clone)]
pub struct Logits {
    pub values: Vec<f32>,
    pub sampled_token: i32,
}

impl Logits {
    pub fn new(values: Vec<f32>, sampled_token: i32) -> Self {
        Self {
            values,
            sampled_token,
        }
    }
}

/// Inference request (minimal view for model).
#[derive(Debug, Clone)]
pub struct InferenceRequestForModel {
    pub id: u64,
    pub prompt_tokens: Vec<i32>,
    pub last_token: Option<i32>,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub context_len: usize,
    /// Stable llama.cpp sequence ID assigned at admission — never changes for the lifetime of
    /// a request. Using the batch index here would cause seq_id collisions across decode steps.
    pub kv_seq_id: i32,
    /// Sampling temperature (0 = greedy, 1 = unscaled).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Top-K filter (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// RNG seed for reproducible sampling (None = random).
    pub seed: Option<u64>,
    /// Previously generated token IDs (for repetition penalty).
    pub generated_token_ids: Vec<i32>,
    /// Number of prompt tokens already in the KV cache from a prefix hit.
    /// `do_prefill` submits only `prompt_tokens[skip_prefix_tokens..]` starting at
    /// position `skip_prefix_tokens`.
    pub skip_prefix_tokens: usize,
    /// Sequence ID that holds the cached prefix KV data. When set, `do_prefill` calls
    /// `llama_memory_seq_cp` to transfer positions 0..skip_prefix_tokens before adding
    /// the remaining tokens to the batch.
    pub prefix_seq_id: Option<i32>,
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Backend model trait.
pub trait Model: Send + Sync {
    /// Sync prefill (called by engine from spawn_blocking).
    /// Returns `(req_id, logits, tokens_submitted)` — `tokens_submitted` is how many tokens
    /// were actually placed in the KV cache for each request (may differ from
    /// `prompt_tokens.len()` when effective_skip > 0).
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>>;

    /// Sync decode step (called by engine from spawn_blocking).
    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    fn model_config(&self) -> ModelConfig;

    fn eos_token_id(&self) -> i32;

    /// Returns true if `token_id` is ANY end-of-generation token for this model
    /// (e.g. `<|im_end|>`, `<|endoftext|>`, etc.).  More reliable than comparing
    /// with `eos_token_id()` alone because models like Qwen3.5 have multiple EOG tokens.
    fn is_eog_token(&self, token_id: i32) -> bool;

    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    fn token_to_piece(&self, token: i32) -> Result<String>;

    /// Returns the raw bytes produced by `llama_token_to_piece` without UTF-8
    /// validation or lossy replacement.  Used by the engine to accumulate
    /// per-request byte buffers so that multi-token UTF-8 sequences (e.g. emoji
    /// split across BPE tokens) are decoded correctly.
    ///
    /// The default implementation encodes the `token_to_piece` String back to
    /// bytes, which is safe for stub/mock models that already return valid UTF-8.
    /// `LlamaCppModel` overrides this to return the actual raw C bytes.
    fn token_to_piece_bytes(&self, token: i32) -> Vec<u8> {
        self.token_to_piece(token).unwrap_or_default().into_bytes()
    }

    /// Apply chat template to messages. Returns formatted prompt for tokenization.
    /// Fallback: simple "role: content\n" concatenation if template unavailable.
    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String>;

    /// Effective per-sequence context length (tokens) this model was configured with.
    /// For `LlamaCppModel` this is the value used in `llama_init_from_model`.
    fn context_len(&self) -> u32 {
        4096
    }

    /// Returns `true` when the model has native thinking support — i.e. `<think>` is a
    /// single special token in the vocabulary (Qwen3, DeepSeek-R1, etc.).
    /// Models without native thinking always return `false`.
    fn supports_thinking(&self) -> bool {
        false
    }

    /// Return sampling parameters recommended by the model's GGUF metadata, if any.
    /// Returns `None` when the model file contains no sampling hints.
    fn recommended_sampling(&self) -> Option<RecommendedSampling> {
        None
    }

    /// Remove all KV cache / recurrent state for the given sequence ID.
    /// Must be called before a seq_id is reused for a new request; otherwise the new request
    /// will inherit stale positions from the previous occupant and llama_decode will fail.
    fn clear_sequence(&self, seq_id: i32);

    /// Copy `token_count` tokens worth of KV cache from `src_seq_id` to `dst_seq_id`
    /// (positions 0..token_count). Used by prefix caching: before prefilling a request whose
    /// prompt matches a completed one, we copy the KV data so only the non-cached suffix
    /// needs to be computed.
    fn copy_sequence_range(&self, src_seq_id: i32, dst_seq_id: i32, token_count: i32);

    /// Returns true if the loaded model's memory backend supports sequence copying
    /// (`llama_memory_seq_cp`).  Standard transformer (attention-only) models return true;
    /// recurrent / hybrid models (Mamba, Qwen3.5, etc.) return false.
    /// Prefix caching must be disabled when this returns false.
    fn supports_seq_copy(&self) -> bool;

    /// Return the embedding dimension (n_embd) for the model.
    fn embedding_dim(&self) -> usize;

    /// Run a forward pass in embedding mode and return the sequence embedding vector.
    /// Uses sequence slot 0; caller must not have an active inference request on slot 0.
    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>>;

    /// Return the text forms of the model's EOS and EOT tokens.
    /// Used as base stop sequences so generation halts on model-native terminators
    /// even when the token ID is not caught by `is_eog_token`.
    fn stop_tokens(&self) -> Vec<String>;
}
