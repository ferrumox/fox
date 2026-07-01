// ModelInfo — the single, inspectable snapshot of a loaded model's facts.
//
// Built once from a loaded model (metadata + llama.cpp API), it is the basis of
// `fox probe` and, over the P1/P2 rework, will become the source of truth that
// downstream code (KV manager, sampling, output filter, API) reads from instead
// of re-deriving the same numbers with disagreeing formulas.
//
// This struct is plain data and compiles in every build; backends populate it
// (see `Model::model_info`, overridden by `LlamaCppModel` to report GGUF truth).

use super::RecommendedSampling;

/// A snapshot of everything fox knows about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// GGUF `general.architecture`, verbatim (e.g. "llama", "gemma3"), or "unknown".
    pub arch_name: String,

    // dimensions — each read from the model, never reconstructed by formula.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    pub n_ctx_train: u32,
    pub effective_ctx: u32,
    pub vocab_size: usize,

    // capabilities / identity
    pub eos_token_id: i32,
    pub has_chat_template: bool,
    pub supports_thinking: bool,
    pub supports_seq_copy: bool,
    pub stop_token_count: usize,
    pub recommended_sampling: Option<RecommendedSampling>,
}

impl ModelInfo {
    /// Human-readable list of internal contradictions between the model's
    /// metadata-derived facts and the formulas fox uses elsewhere in the code.
    ///
    /// An empty result means the model is coherent with fox's assumptions. A
    /// non-empty result names exactly which hardcoded formula would mis-handle
    /// this model — the whack-a-mole surface the rework is closing.
    pub fn contradictions(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.n_head == 0 {
            return out;
        }

        // head_dim is read from `<arch>.attention.key_length`; the legacy
        // fallback formula n_embd/n_head is wrong for Gemma (256), MLA, etc.
        let formula_head_dim = self.n_embd / self.n_head;
        if formula_head_dim != self.head_dim {
            out.push(format!(
                "head_dim = {} (metadata); n_embd/n_head = {}/{} = {} — the fallback formula would mis-size the KV cache",
                self.head_dim, self.n_embd, self.n_head, formula_head_dim
            ));
        }

        // The embeddings path reconstructs n_embd as n_head*head_dim; wrong
        // whenever head_dim != n_embd/n_head (exactly the Gemma/MLA class).
        let reconstructed_embd = self.n_head * self.head_dim;
        if reconstructed_embd != self.n_embd {
            out.push(format!(
                "embedding_dim: n_head*head_dim = {}*{} = {} ≠ n_embd = {} — the num_heads*head_dim reconstruction breaks embeddings for this model",
                self.n_head, self.head_dim, reconstructed_embd, self.n_embd
            ));
        }

        out
    }
}
