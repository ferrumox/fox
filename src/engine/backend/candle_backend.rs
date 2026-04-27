//! [`InferenceBackend`] adapter for the in-process candle implementation.
//!
//! Phase C.3.2 status: the backend serves the **Llama family natively** (Llama
//! 1/2/3) on CPU, single-sequence-per-instance. Other architectures fall
//! through to llama.cpp via the router. CUDA, multi-sequence batching and
//! the Gemma 4 / Qwen 3.5 native architectures arrive in C.3.4 / C.5.

use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use candle_core::Device;

use crate::engine::model::candle::arch_runner::CandleRunner;
use crate::engine::model::candle::llama_model::CandleLlamaModel;
use crate::model_registry::{ModelProfile, RegistryConfig};

use super::{ids, BackendInstance, Compatibility, InferenceBackend};

/// Stateless wrapper. The real state lives inside the `CandleLlamaModel`
/// produced by [`Self::instantiate`].
pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for CandleBackend {
    fn id(&self) -> &'static str {
        ids::CANDLE
    }

    fn supports(&self, profile: &ModelProfile) -> Compatibility {
        // Quirks are evaluated before architecture lookup so a multimodal
        // Llama (e.g. LLaVA-Llama-3) is correctly declined even though its
        // architecture string starts with "llama".
        if profile.quirks.multimodal.is_some() {
            return Compatibility::Unsupported(
                "candle backend does not yet handle multimodal inputs (C.5 work)".to_string(),
            );
        }
        // Delegate the capability question to the runner table — adding a
        // new architecture means touching one place (`arch_runner.rs`) and
        // both the loader and the router pick it up automatically.
        if CandleRunner::kind_for_arch(&profile.architecture).is_some() {
            Compatibility::Native
        } else {
            Compatibility::Unsupported(format!(
                "candle backend has no quantised runner for architecture '{}'; \
                 the router will fall back to llama-cpp",
                profile.architecture
            ))
        }
    }

    fn instantiate(
        &self,
        path: &Path,
        profile: Option<&ModelProfile>,
        _cfg: &RegistryConfig,
    ) -> Result<BackendInstance> {
        if let Some(p) = profile {
            if !matches!(self.supports(p), Compatibility::Native | Compatibility::Workable) {
                return Err(anyhow!(
                    "candle backend declined to instantiate '{}'; the router \
                     should have picked another backend",
                    p.architecture
                ));
            }
        }

        // CPU device for the C.3.2 milestone — deterministic, no CUDA setup
        // required. CUDA selection lands in C.3.4 once the multi-sequence KV
        // cache lives in fox rather than inside ModelWeights.
        let device = Device::Cpu;
        let model = CandleLlamaModel::load(path, device).map_err(|e| {
            anyhow!(
                "candle backend failed to load model: {e}. \
                 If you saw this for a non-Llama architecture, force \
                 --backend llama-cpp; that path stays fully supported."
            )
        })?;

        Ok(BackendInstance {
            model: Arc::new(model),
            effective_context_len: None,
            // KV-cache element types are llama.cpp specific; candle ignores
            // them. Reporting f16 (1) keeps downstream consumers (metrics,
            // logging) reasonable.
            effective_type_k: 1,
            effective_type_v: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::{ArchQuirks, Modality, ModelProfile};

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
    fn reports_id_as_candle() {
        assert_eq!(CandleBackend::new().id(), ids::CANDLE);
    }

    #[test]
    fn supports_known_arch_families_natively() {
        let backend = CandleBackend::new();
        for arch in ["llama", "qwen3", "gemma3"] {
            assert!(
                matches!(backend.supports(&baseline(arch)), Compatibility::Native),
                "expected Native for '{arch}'"
            );
        }
    }

    #[test]
    fn declines_unsupported_architectures_with_explanation() {
        let backend = CandleBackend::new();
        for arch in [
            "gemma4", "gemma2", "mistral", "qwen", "qwen2", "qwen35", "acme-unknown",
        ] {
            match backend.supports(&baseline(arch)) {
                Compatibility::Unsupported(reason) => {
                    assert!(
                        reason.contains("no quantised runner"),
                        "expected runner-table explanation for '{arch}', got: {reason}"
                    );
                }
                other => panic!("expected Unsupported for '{arch}', got {other:?}"),
            }
        }
    }

    #[test]
    fn declines_multimodal_with_dedicated_message() {
        let backend = CandleBackend::new();
        let mut p = baseline("llama");
        p.quirks.multimodal = Some(Modality::Vision);
        match backend.supports(&p) {
            Compatibility::Unsupported(reason) => {
                assert!(reason.contains("multimodal"));
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
