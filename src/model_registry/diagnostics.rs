//! Pretty-print a `ModelProfile` together with the backend recommendation
//! and any diagnostic notes produced by the architecture table.
//!
//! The output is plain text by design — used both by `fox inspect` (printed
//! to a TTY) and by the loader (logged on startup).

use std::fmt::Write;

use super::arch_table::{BackendHint, DiagnosticLevel, DiagnosticNote};
use super::inspect::{Modality, ModelProfile};

/// Bundle of profile + recommendation that downstream code formats.
#[derive(Debug, Clone)]
pub struct ModelDiagnostic {
    pub profile: ModelProfile,
    pub hint: BackendHint,
    pub notes: Vec<DiagnosticNote>,
}

/// Render a diagnostic as a human-readable, multi-section block.
pub fn render_diagnostic(d: &ModelDiagnostic) -> String {
    let mut out = String::with_capacity(512);

    writeln!(&mut out, "Architecture").ok();
    write_kv(&mut out, "  name", &d.profile.architecture);
    write_kv(
        &mut out,
        "  gguf version",
        &d.profile.gguf_version.to_string(),
    );
    write_kv(&mut out, "  tensors", &d.profile.tensor_count.to_string());

    writeln!(&mut out, "\nAttention").ok();
    write_kv(
        &mut out,
        "  embedding length",
        &d.profile.embedding_length.to_string(),
    );
    write_kv(&mut out, "  query heads", &d.profile.head_count.to_string());
    write_kv(&mut out, "  kv heads", &d.profile.head_count_kv.to_string());
    write_kv(&mut out, "  head dim", &d.profile.head_dim.to_string());
    write_kv(
        &mut out,
        "  feed-forward length",
        &d.profile.ff_length.to_string(),
    );

    writeln!(&mut out, "\nVocabulary & context").ok();
    write_kv(&mut out, "  vocab size", &d.profile.vocab_size.to_string());
    write_kv(
        &mut out,
        "  context length",
        &d.profile.context_length.to_string(),
    );

    writeln!(&mut out, "\nQuirks").ok();
    write_kv(
        &mut out,
        "  hybrid memory",
        yes_no(d.profile.quirks.hybrid_memory),
    );
    write_kv(
        &mut out,
        "  non-standard head dim",
        yes_no(d.profile.quirks.nonstandard_head_dim),
    );
    write_kv(
        &mut out,
        "  experts",
        &d.profile
            .quirks
            .moe_experts
            .map(|n| n.to_string())
            .unwrap_or_else(|| "—".to_string()),
    );
    write_kv(
        &mut out,
        "  multimodal",
        match d.profile.quirks.multimodal {
            Some(Modality::Vision) => "vision",
            Some(Modality::Audio) => "audio",
            None => "—",
        },
    );

    writeln!(&mut out, "\nRecommendation").ok();
    write_kv(&mut out, "  backend", backend_label(d.hint));

    if !d.notes.is_empty() {
        writeln!(&mut out, "\nNotes").ok();
        for note in &d.notes {
            writeln!(&mut out, "  [{}] {}", level_tag(note.level), note.message).ok();
            if let Some(hint) = &note.hint {
                writeln!(&mut out, "        → {hint}").ok();
            }
        }
    }

    out
}

fn write_kv(out: &mut String, key: &str, value: &str) {
    writeln!(out, "{key:<28} {value}").ok();
}

fn yes_no(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn backend_label(hint: BackendHint) -> &'static str {
    match hint {
        BackendHint::LlamaCpp => "llama-cpp",
        BackendHint::Candle => "candle",
        BackendHint::Either => "either (use --backend to override)",
    }
}

fn level_tag(level: DiagnosticLevel) -> &'static str {
    match level {
        DiagnosticLevel::Info => "info",
        DiagnosticLevel::Warn => "warn",
        DiagnosticLevel::Block => "block",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::inspect::{ArchQuirks, ModelProfile};

    fn sample(arch: &str) -> ModelProfile {
        ModelProfile {
            architecture: arch.to_string(),
            embedding_length: 4096,
            head_count: 32,
            head_count_kv: 8,
            head_dim: 128,
            context_length: 8192,
            vocab_size: 32000,
            ff_length: 14336,
            gguf_version: 3,
            tensor_count: 291,
            quirks: ArchQuirks::default(),
        }
    }

    #[test]
    fn render_includes_all_sections() {
        let d = ModelDiagnostic {
            profile: sample("llama"),
            hint: BackendHint::LlamaCpp,
            notes: vec![DiagnosticNote {
                level: DiagnosticLevel::Info,
                message: "grouped-query attention: 8 kv heads for 32 query heads".to_string(),
                hint: None,
            }],
        };
        let rendered = render_diagnostic(&d);
        for section in [
            "Architecture",
            "Attention",
            "Vocabulary & context",
            "Quirks",
            "Recommendation",
            "Notes",
        ] {
            assert!(
                rendered.contains(section),
                "section '{section}' missing from output:\n{rendered}"
            );
        }
        assert!(rendered.contains("backend"));
        assert!(rendered.contains("llama-cpp"));
        assert!(rendered.contains("grouped-query attention"));
    }

    #[test]
    fn notes_section_omitted_when_empty() {
        let d = ModelDiagnostic {
            profile: sample("llama"),
            hint: BackendHint::Either,
            notes: vec![],
        };
        let rendered = render_diagnostic(&d);
        assert!(!rendered.contains("Notes"));
    }
}
