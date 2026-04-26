//! [`InferenceBackend`] adapter for the in-process candle implementation.
//!
//! At this rollout stage (C.1) the backend is *registered* on the router but
//! declines every model. That keeps the router's decision tree exercised by
//! real flows without fooling users into thinking candle can yet serve their
//! prompts.
//!
//! Subsequent rollout phases (C.2–C.5) replace the placeholder bodies with
//! real capability checks and a working `instantiate`.

use std::path::Path;

use anyhow::{anyhow, Result};

use crate::engine::model::candle::gguf_loader;
use crate::model_registry::{ModelProfile, RegistryConfig};

use super::{ids, BackendInstance, Compatibility, InferenceBackend};

/// Stub indicating where future capability logic will live. Distinct from
/// `LlamaCppBackend` so binaries can opt into candle without paying for it.
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

const NOT_READY_MESSAGE: &str =
    "candle backend is registered but its model loaders are not yet implemented; \
     fall back to --backend llama-cpp until phase C.4 lands";

impl InferenceBackend for CandleBackend {
    fn id(&self) -> &'static str {
        ids::CANDLE
    }

    fn supports(&self, _profile: &ModelProfile) -> Compatibility {
        Compatibility::Unsupported(NOT_READY_MESSAGE.to_string())
    }

    fn instantiate(
        &self,
        path: &Path,
        _profile: Option<&ModelProfile>,
        _cfg: &RegistryConfig,
    ) -> Result<BackendInstance> {
        // Even though we cannot yet load weights, we *can* validate that the
        // file is a parseable GGUF — that exercises the new tensor-index
        // reader on every real `--backend candle` invocation and produces a
        // helpful error before the user-facing failure.
        match gguf_loader::load_index(path) {
            Ok(idx) => {
                tracing::info!(
                    tensor_count = idx.len(),
                    alignment = idx.alignment,
                    "candle tensor index parsed (no weight loading yet)"
                );
            }
            Err(err) => {
                tracing::warn!(
                    "candle tensor index reader rejected the file: {}",
                    err
                );
            }
        }
        Err(anyhow!(NOT_READY_MESSAGE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_registry::{ArchQuirks, ModelProfile};

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
    fn declines_every_architecture_during_rollout() {
        let backend = CandleBackend::new();
        for arch in ["llama", "gemma4", "qwen35", "mistral", "acme-unknown"] {
            let verdict = backend.supports(&baseline(arch));
            match verdict {
                Compatibility::Unsupported(reason) => {
                    assert!(
                        reason.contains("not yet implemented"),
                        "expected pending message, got: {reason}"
                    );
                }
                other => panic!("expected Unsupported during rollout, got {other:?}"),
            }
        }
    }
}
