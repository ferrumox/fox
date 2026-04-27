//! Polymorphic dispatcher over the candle-transformers quantised model
//! implementations the backend reuses.
//!
//! Each variant wraps one of the `quantized_*::ModelWeights` types from
//! `candle-transformers`. They all expose the same `from_gguf` /
//! `forward(&mut self, &Tensor, usize) -> Result<Tensor>` shape, so
//! [`CandleRunner`] is a thin enum that picks the right one based on the
//! architecture string read from the GGUF metadata.
//!
//! Architectures not yet covered fall back to `LlamaCppBackend` via the
//! router — see [`CandleRunner::supports_arch`].

use std::fs::File;
use std::path::Path;

use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_transformers::models::{
    quantized_gemma3, quantized_llama, quantized_qwen3, quantized_qwen3_moe,
};

/// Concrete inner runner. The variants intentionally mirror the model
/// catalogue in `candle-transformers/src/models/quantized_*` — adding a new
/// architecture means adding one variant here, one branch in
/// [`CandleRunner::from_gguf`] and one branch in [`CandleRunner::forward`].
pub enum CandleRunner {
    Llama(quantized_llama::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Qwen3Moe(quantized_qwen3_moe::GGUFQWenMoE),
    Gemma3(quantized_gemma3::ModelWeights),
}

#[derive(Debug)]
pub enum RunnerError {
    Io(std::io::Error),
    Candle(String),
    UnsupportedArch(String),
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error opening GGUF file: {e}"),
            Self::Candle(msg) => write!(f, "candle error: {msg}"),
            Self::UnsupportedArch(arch) => write!(
                f,
                "candle backend has no quantised runner for architecture '{arch}'"
            ),
        }
    }
}

impl std::error::Error for RunnerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl CandleRunner {
    /// Returns the runner kind that handles `architecture` (lowercase, as
    /// returned by `inspect::probe`). `None` means "decline" — the router
    /// falls through to llama.cpp.
    ///
    /// Kept as a const-style table so [`CandleBackend::supports`] can ask
    /// the same question without instantiating any weights.
    ///
    /// **Strict on purpose.** Each candle-transformers loader looks for the
    /// exact arch string it was written for and rejects anything else. We
    /// only map architectures that have been validated end-to-end against a
    /// real GGUF on disk:
    /// * `llama*` (Llama 1/2/3.x) → `quantized_llama`
    /// * `qwen3` (strict) → `quantized_qwen3`
    /// * `qwen3moe` → `quantized_qwen3_moe`
    /// * `gemma3` (strict) → `quantized_gemma3`
    ///
    /// Anything else — `qwen2*`, `qwen35`, `gemma2`, `gemma4`, `mistral`,
    /// `phi`, … — returns `None` and the router routes through llama.cpp,
    /// where they already work. Native implementations for `qwen35` (KV-norm)
    /// and `gemma4` (head_dim=512) are scheduled for C.5.b.
    pub fn kind_for_arch(architecture: &str) -> Option<RunnerKind> {
        let lower = architecture.to_lowercase();
        if lower.starts_with("qwen3_moe") || lower == "qwen3moe" {
            return Some(RunnerKind::Qwen3Moe);
        }
        if lower == "qwen3" {
            return Some(RunnerKind::Qwen3);
        }
        if lower == "gemma3" {
            return Some(RunnerKind::Gemma3);
        }
        if lower == "llama" || lower.starts_with("llama-") || lower.starts_with("llama_") {
            return Some(RunnerKind::Llama);
        }
        None
    }

    /// Open `path` and instantiate the appropriate runner for `architecture`.
    pub fn from_gguf(
        path: &Path,
        architecture: &str,
        device: &Device,
    ) -> Result<Self, RunnerError> {
        let kind = Self::kind_for_arch(architecture)
            .ok_or_else(|| RunnerError::UnsupportedArch(architecture.to_string()))?;

        let mut file = File::open(path).map_err(RunnerError::Io)?;
        let mut content =
            gguf_file::Content::read(&mut file).map_err(|e| RunnerError::Candle(e.to_string()))?;

        // candle-transformers' quantised models look up keys under the exact
        // architecture string they were written for (e.g. `qwen3.*`). Some
        // GGUFs in the wild use a slightly different arch (`qwen35.*` for
        // Qwen 3.5, `gemma3-2b.*` for some Gemma 3 quants) — rewrite the
        // prefix in-place so the loader finds the metadata where it expects.
        let target = kind.canonical_arch();
        if architecture.to_lowercase() != target {
            rewrite_arch_prefix(&mut content, &architecture.to_lowercase(), target);
        }

        let runner = match kind {
            RunnerKind::Llama => Self::Llama(
                quantized_llama::ModelWeights::from_gguf(content, &mut file, device)
                    .map_err(|e| RunnerError::Candle(e.to_string()))?,
            ),
            RunnerKind::Qwen3 => Self::Qwen3(
                quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)
                    .map_err(|e| RunnerError::Candle(e.to_string()))?,
            ),
            RunnerKind::Qwen3Moe => Self::Qwen3Moe(
                quantized_qwen3_moe::GGUFQWenMoE::from_gguf(
                    content,
                    &mut file,
                    device,
                    DType::F32,
                )
                .map_err(|e| RunnerError::Candle(e.to_string()))?,
            ),
            RunnerKind::Gemma3 => Self::Gemma3(
                quantized_gemma3::ModelWeights::from_gguf(content, &mut file, device)
                    .map_err(|e| RunnerError::Candle(e.to_string()))?,
            ),
        };
        Ok(runner)
    }

    /// Forward a token batch and return logits for the last position.
    pub fn forward(&mut self, input: &Tensor, position: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(input, position),
            Self::Qwen3(m) => m.forward(input, position),
            Self::Qwen3Moe(m) => m.forward(input, position),
            Self::Gemma3(m) => m.forward(input, position),
        }
    }

    /// Identifier for logs / metrics.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Llama(_) => "llama",
            Self::Qwen3(_) => "qwen3",
            Self::Qwen3Moe(_) => "qwen3-moe",
            Self::Gemma3(_) => "gemma3",
        }
    }
}

/// Lightweight tag for the `kind_for_arch` table. Lets [`CandleBackend`]
/// answer capability questions without touching the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunnerKind {
    Llama,
    Qwen3,
    Qwen3Moe,
    Gemma3,
}

impl RunnerKind {
    /// The architecture string each candle-transformers loader expects to
    /// find in the GGUF metadata. Used to rewrite keys when a GGUF reports
    /// a closely-related arch (e.g. `qwen35`, `gemma3-2b`) that the loader
    /// would otherwise reject for missing `<arch>.attention.head_count`.
    pub fn canonical_arch(&self) -> &'static str {
        match self {
            RunnerKind::Llama => "llama",
            RunnerKind::Qwen3 => "qwen3",
            RunnerKind::Qwen3Moe => "qwen3moe",
            RunnerKind::Gemma3 => "gemma3",
        }
    }
}

/// Rewrite every `<from>.*` metadata key as `<to>.*` and update the
/// `general.architecture` value to `to`. Idempotent and key-collision-free
/// because we remove the source key before inserting the destination one.
fn rewrite_arch_prefix(content: &mut gguf_file::Content, from: &str, to: &str) {
    let from_prefix = format!("{from}.");
    let to_prefix = format!("{to}.");
    let keys: Vec<String> = content
        .metadata
        .keys()
        .filter(|k| k.starts_with(&from_prefix))
        .cloned()
        .collect();
    for key in keys {
        if let Some(value) = content.metadata.remove(&key) {
            let new_key = format!("{to_prefix}{}", &key[from_prefix.len()..]);
            content.metadata.insert(new_key, value);
        }
    }
    if content.metadata.contains_key("general.architecture") {
        content.metadata.insert(
            "general.architecture".to_string(),
            gguf_file::Value::String(to.to_string()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_llama_family_to_llama_runner() {
        assert_eq!(CandleRunner::kind_for_arch("llama"), Some(RunnerKind::Llama));
        assert_eq!(
            CandleRunner::kind_for_arch("LlaMa"),
            Some(RunnerKind::Llama)
        );
    }

    #[test]
    fn maps_qwen3_strictly() {
        assert_eq!(
            CandleRunner::kind_for_arch("qwen3"),
            Some(RunnerKind::Qwen3)
        );
    }

    #[test]
    fn declines_qwen2_and_qwen_3_5_until_native_implementations() {
        // quantized_qwen3 looks up `qwen3.attention.key_length` and tensor
        // names that neither qwen2 nor qwen35 ship with — sending them to
        // llama.cpp is the only way they actually load.
        for arch in ["qwen", "qwen2", "qwen2.5", "qwen35", "qwen3.5"] {
            assert!(
                CandleRunner::kind_for_arch(arch).is_none(),
                "expected '{arch}' to be declined"
            );
        }
    }

    #[test]
    fn maps_qwen3_moe_to_dedicated_runner() {
        assert_eq!(
            CandleRunner::kind_for_arch("qwen3_moe"),
            Some(RunnerKind::Qwen3Moe)
        );
        assert_eq!(
            CandleRunner::kind_for_arch("qwen3moe"),
            Some(RunnerKind::Qwen3Moe)
        );
    }

    #[test]
    fn maps_gemma3_strictly() {
        assert_eq!(
            CandleRunner::kind_for_arch("gemma3"),
            Some(RunnerKind::Gemma3)
        );
        // Gemma 3 quants in the wild sometimes use `gemma3-2b` etc., but
        // candle's loader rejects them — keep the table strict.
        assert!(CandleRunner::kind_for_arch("gemma3-2b").is_none());
    }

    #[test]
    fn declines_gemma2_until_a_dedicated_runner_lands() {
        // gemma2 does NOT match gemma3 — the candle backend declines it so
        // the router falls back to llama.cpp where gemma2 already works.
        assert!(CandleRunner::kind_for_arch("gemma2").is_none());
        assert!(CandleRunner::kind_for_arch("gemma").is_none());
    }

    #[test]
    fn declines_gemma4_pending_native_implementation() {
        // gemma4 has head_dim=512 which neither quantized_gemma3 nor
        // quantized_llama handle correctly — schedule a from-scratch
        // implementation for C.5.b.
        assert!(CandleRunner::kind_for_arch("gemma4").is_none());
    }

    #[test]
    fn declines_unknown_architectures() {
        assert!(CandleRunner::kind_for_arch("acme-secret-arch").is_none());
        assert!(CandleRunner::kind_for_arch("").is_none());
    }
}
