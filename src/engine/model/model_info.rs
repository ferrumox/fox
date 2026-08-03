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

/// Which memory model a model's KV/state footprint follows. Not a sizing
/// formula (fox never computes byte sizes from this) — purely observability
/// for `fox probe`/load-time logs, and a signal that context-size decisions
/// come from real allocation, not a per-token formula, for `Latent`/
/// `Recurrent` models. See `docs/design/mla-recurrent-kv-sizing.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvMemoryClass {
    /// Plain positional attention — KV bytes scale with `n_head_kv * head_dim
    /// * n_layer * tokens` (dense, GQA, MoE, Gemma-style SWA/shared-KV, etc).
    Standard,
    /// Multi-head Latent Attention (DeepSeek-V2/V3): KV is compressed into a
    /// small latent vector, far smaller than the `Standard` formula assumes.
    Latent,
    /// Recurrent/hybrid (Mamba, RWKV, Jamba, LFM2): state size is
    /// approximately constant per sequence, not a function of context length
    /// at all — the `Standard` formula's inputs aren't even meaningful here.
    Recurrent,
}

impl std::fmt::Display for KvMemoryClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvMemoryClass::Standard => write!(f, "standard"),
            KvMemoryClass::Latent => write!(f, "latent (MLA)"),
            KvMemoryClass::Recurrent => write!(f, "recurrent"),
        }
    }
}

/// Classify a model's KV memory model from facts already available on
/// `ModelInfo` — never a new heuristic. `supports_seq_copy` already comes
/// from llama.cpp's own `llama_memory_can_shift` (true only for standard,
/// shiftable KV caches), so `!supports_seq_copy` is authoritative for
/// `Recurrent`, checked first. `arch_name` is the model's own declared
/// `general.architecture` GGUF value; matching it against llama.cpp's own MLA
/// architecture identifiers (`vendor/llama.cpp/src/llama-arch.cpp`) is a
/// fact lookup, not content-sniffing — DeepSeek-V3 reuses the `deepseek2` tag,
/// there is no separate `deepseek3`.
pub fn classify_kv_memory(arch_name: &str, supports_seq_copy: bool) -> KvMemoryClass {
    if !supports_seq_copy {
        return KvMemoryClass::Recurrent;
    }
    match arch_name {
        "deepseek2" | "deepseek2-ocr" => KvMemoryClass::Latent,
        _ => KvMemoryClass::Standard,
    }
}

/// A snapshot of everything fox knows about a loaded model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// GGUF `general.architecture`, verbatim (e.g. "llama", "gemma3"), or "unknown".
    pub arch_name: String,

    /// Active compute backend (e.g. "Vulkan0 — AMD Radeon 890M", or "CPU").
    pub backend: String,

    // dimensions — each read from the model, never reconstructed by formula.
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_layer: usize,
    pub n_ctx_train: u32,
    pub effective_ctx: u32,
    pub vocab_size: usize,
    /// Exact parameter count, from `llama_model_n_params`. `0` when the backend
    /// cannot report it (the stub). Never derived by formula: a GQA- and
    /// tied-embedding-aware estimate still lands ~30% high, which would replace
    /// "unknown" with a confident wrong number.
    pub n_params: u64,

    // capabilities / identity
    pub eos_token_id: i32,
    pub has_chat_template: bool,
    pub supports_thinking: bool,
    pub supports_seq_copy: bool,
    pub kv_memory_class: KvMemoryClass,
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

        // n_embd differs from head geometry for the Gemma/MLA class. fox reads
        // n_embd directly (ModelConfig.n_embd); this flags models where any code
        // still assuming n_head*head_dim == n_embd would be wrong.
        let geometry_embd = self.n_head * self.head_dim;
        if geometry_embd != self.n_embd {
            out.push(format!(
                "n_embd = {} ≠ n_head*head_dim = {}*{} = {} — embedding size must be read from metadata, not reconstructed from head geometry",
                self.n_embd, self.n_head, self.head_dim, geometry_embd
            ));
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recurrent_takes_precedence_over_arch_name() {
        // supports_seq_copy=false is authoritative — checked before arch_name,
        // so even a (hypothetical, none exists today) recurrent deepseek2
        // variant would classify as Recurrent, not Latent.
        assert_eq!(
            classify_kv_memory("deepseek2", false),
            KvMemoryClass::Recurrent
        );
        assert_eq!(classify_kv_memory("mamba", false), KvMemoryClass::Recurrent);
        assert_eq!(
            classify_kv_memory("unknown", false),
            KvMemoryClass::Recurrent
        );
    }

    #[test]
    fn deepseek2_is_latent() {
        assert_eq!(classify_kv_memory("deepseek2", true), KvMemoryClass::Latent);
    }

    #[test]
    fn deepseek2_ocr_is_latent() {
        assert_eq!(
            classify_kv_memory("deepseek2-ocr", true),
            KvMemoryClass::Latent
        );
    }

    #[test]
    fn dense_arch_is_standard() {
        assert_eq!(classify_kv_memory("llama", true), KvMemoryClass::Standard);
        assert_eq!(classify_kv_memory("qwen2", true), KvMemoryClass::Standard);
        assert_eq!(classify_kv_memory("gemma3", true), KvMemoryClass::Standard);
    }

    #[test]
    fn unknown_arch_with_seq_copy_is_standard() {
        assert_eq!(classify_kv_memory("unknown", true), KvMemoryClass::Standard);
    }
}
