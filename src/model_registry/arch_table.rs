//! Declarative table of known model architectures and the backend that
//! handles each one best.
//!
//! [`recommend_backend`] is intentionally additive: when a profile matches no
//! row it falls through to [`BackendHint::Either`] so unfamiliar GGUFs can
//! still be loaded — they only lose the architecture-specific advice.

use super::inspect::{Modality, ModelProfile};

/// Which backend should run a given profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendHint {
    /// Backed by the bundled llama.cpp FFI.
    LlamaCpp,
    /// Backed by the in-process candle implementation.
    Candle,
    /// No strong preference — the router will pick by priority.
    Either,
}

/// Severity of a diagnostic note attached to a profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticLevel {
    Info,
    Warn,
    Block,
}

/// A single line of human-readable feedback derived from a profile.
#[derive(Debug, Clone)]
pub struct DiagnosticNote {
    pub level: DiagnosticLevel,
    pub message: String,
    pub hint: Option<String>,
}

impl DiagnosticNote {
    fn info(message: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Info,
            message: message.into(),
            hint: None,
        }
    }

    fn warn(message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Warn,
            message: message.into(),
            hint: Some(hint.into()),
        }
    }

    fn block(message: impl Into<String>, hint: impl Into<String>) -> Self {
        Self {
            level: DiagnosticLevel::Block,
            message: message.into(),
            hint: Some(hint.into()),
        }
    }
}

/// Pick the backend best suited for `profile` and produce diagnostic notes
/// explaining the decision. The notes are always returned, even when the
/// chosen backend is `Either`, so the operator can see why a model is
/// considered unusual.
pub fn recommend_backend(profile: &ModelProfile) -> (BackendHint, Vec<DiagnosticNote>) {
    let mut notes = Vec::new();
    let lower = profile.architecture.to_lowercase();

    // Strong overrides — these win regardless of architecture-table entry.
    if let Some(modality) = profile.quirks.multimodal {
        let kind = match modality {
            Modality::Vision => "vision",
            Modality::Audio => "audio",
        };
        notes.push(DiagnosticNote::warn(
            format!("model declares {kind} inputs"),
            "multimodal models require the candle backend; \
             pass --backend candle when serving",
        ));
        return (BackendHint::Candle, notes);
    }

    if profile.quirks.hybrid_memory {
        notes.push(DiagnosticNote::warn(
            "architecture uses recurrent or hybrid memory",
            "prefix caching is unavailable on this model under llama.cpp; \
             the candle backend can serve it without losing prefix reuse",
        ));
        return (BackendHint::Candle, notes);
    }

    // Architecture-specific preferences.
    let entry = lookup_arch(&lower);
    let mut hint = entry.preferred_backend;

    if profile.quirks.nonstandard_head_dim {
        notes.push(DiagnosticNote::info(format!(
            "non-standard attention head dimension ({})",
            profile.head_dim
        )));
        if matches!(hint, BackendHint::Either | BackendHint::LlamaCpp) && profile.head_dim >= 256 {
            notes.push(DiagnosticNote::warn(
                "head_dim ≥ 256 is unusual",
                "head dimensions of 256 or larger run more reliably on the \
                 candle backend; pass --backend candle if generation fails",
            ));
            hint = BackendHint::Candle;
        }
    }

    if let Some(experts) = profile.quirks.moe_experts {
        notes.push(DiagnosticNote::info(format!(
            "mixture-of-experts model with {experts} experts"
        )));
    }

    if profile.head_count_kv != profile.head_count {
        notes.push(DiagnosticNote::info(format!(
            "grouped-query attention: {} kv heads for {} query heads",
            profile.head_count_kv, profile.head_count
        )));
    }

    if profile.context_length == 0 {
        notes.push(DiagnosticNote::warn(
            "metadata does not declare a context length",
            "the runtime will fall back to the configured default; \
             verify with `fox inspect` before serving",
        ));
    }

    if profile.vocab_size == 0 {
        notes.push(DiagnosticNote::block(
            "vocabulary size is unknown",
            "the GGUF file is missing both the tokenizer array and an \
             explicit vocab_size key — the model cannot be loaded as-is",
        ));
    }

    if matches!(entry.preferred_backend, BackendHint::Either) && entry.is_unknown {
        notes.push(DiagnosticNote::info(format!(
            "architecture '{}' is not in the known table; \
             defaulting to the configured backend priority",
            profile.architecture
        )));
    }

    (hint, notes)
}

/// Description of a known architecture for [`recommend_backend`].
struct ArchEntry {
    preferred_backend: BackendHint,
    /// `true` when the entry was synthesised because no row matched.
    is_unknown: bool,
}

fn lookup_arch(lower_arch: &str) -> ArchEntry {
    // Order matters: the most specific patterns must run before the generic
    // family names they share a prefix with (e.g. "qwen3" before "qwen").
    const TABLE: &[(&str, BackendHint)] = &[
        ("gemma4", BackendHint::Candle),
        ("gemma3", BackendHint::Candle),
        ("gemma2", BackendHint::LlamaCpp),
        ("gemma", BackendHint::LlamaCpp),
        ("qwen3.5", BackendHint::Candle),
        ("qwen3", BackendHint::Candle),
        ("qwen2", BackendHint::LlamaCpp),
        ("qwen", BackendHint::LlamaCpp),
        ("llama", BackendHint::LlamaCpp),
        ("mistral", BackendHint::LlamaCpp),
        ("mixtral", BackendHint::LlamaCpp),
        ("phi3", BackendHint::LlamaCpp),
        ("phi2", BackendHint::LlamaCpp),
        ("phi", BackendHint::LlamaCpp),
        ("falcon", BackendHint::LlamaCpp),
        ("deepseek", BackendHint::LlamaCpp),
        ("stablelm", BackendHint::LlamaCpp),
        ("starcoder", BackendHint::LlamaCpp),
    ];

    for (pattern, backend) in TABLE {
        if lower_arch.contains(pattern) {
            return ArchEntry {
                preferred_backend: *backend,
                is_unknown: false,
            };
        }
    }

    ArchEntry {
        preferred_backend: BackendHint::Either,
        is_unknown: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::inspect::{ArchQuirks, Modality, ModelProfile};

    fn baseline(arch: &str) -> ModelProfile {
        ModelProfile {
            architecture: arch.to_string(),
            embedding_length: 4096,
            head_count: 32,
            head_count_kv: 32,
            head_dim: 128,
            context_length: 8192,
            vocab_size: 32000,
            ff_length: 11008,
            gguf_version: 3,
            tensor_count: 0,
            quirks: ArchQuirks::default(),
        }
    }

    #[test]
    fn llama_prefers_llama_cpp_with_no_warnings() {
        let (hint, notes) = recommend_backend(&baseline("llama"));
        assert_eq!(hint, BackendHint::LlamaCpp);
        assert!(
            notes.iter().all(|n| n.level == DiagnosticLevel::Info),
            "notes: {notes:?}"
        );
    }

    #[test]
    fn gemma4_prefers_candle() {
        let mut p = baseline("gemma4");
        p.head_dim = 512;
        p.quirks.nonstandard_head_dim = true;
        let (hint, _) = recommend_backend(&p);
        assert_eq!(hint, BackendHint::Candle);
    }

    #[test]
    fn vision_modality_overrides_arch_preference() {
        let mut p = baseline("llama");
        p.quirks.multimodal = Some(Modality::Vision);
        let (hint, notes) = recommend_backend(&p);
        assert_eq!(hint, BackendHint::Candle);
        assert!(notes.iter().any(|n| n.message.contains("vision")));
    }

    #[test]
    fn hybrid_memory_overrides_arch_preference() {
        let mut p = baseline("mamba");
        p.quirks.hybrid_memory = true;
        let (hint, notes) = recommend_backend(&p);
        assert_eq!(hint, BackendHint::Candle);
        assert!(notes
            .iter()
            .any(|n| n.message.contains("recurrent or hybrid")));
    }

    #[test]
    fn unknown_arch_returns_either_with_info_note() {
        let p = baseline("acme-secret-architecture");
        let (hint, notes) = recommend_backend(&p);
        assert_eq!(hint, BackendHint::Either);
        assert!(notes
            .iter()
            .any(|n| n.message.contains("not in the known table")));
    }

    #[test]
    fn missing_vocab_blocks_loading() {
        let mut p = baseline("llama");
        p.vocab_size = 0;
        let (_, notes) = recommend_backend(&p);
        assert!(notes
            .iter()
            .any(|n| n.level == DiagnosticLevel::Block && n.message.contains("vocabulary")));
    }

    #[test]
    fn grouped_query_attention_emits_info_note() {
        let mut p = baseline("llama");
        p.head_count_kv = 8;
        let (_, notes) = recommend_backend(&p);
        assert!(notes
            .iter()
            .any(|n| n.message.contains("grouped-query attention")));
    }

    #[test]
    fn moe_expert_count_is_reported() {
        let mut p = baseline("mixtral");
        p.quirks.moe_experts = Some(8);
        let (_, notes) = recommend_backend(&p);
        assert!(notes
            .iter()
            .any(|n| n.message.contains("mixture-of-experts")));
    }

    #[test]
    fn large_head_dim_on_unknown_arch_promotes_candle() {
        let mut p = baseline("acme");
        p.head_dim = 384;
        p.quirks.nonstandard_head_dim = true;
        let (hint, _) = recommend_backend(&p);
        assert_eq!(hint, BackendHint::Candle);
    }
}
