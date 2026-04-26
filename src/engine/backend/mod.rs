//! Indirection layer above the [`Model`] trait.
//!
//! `Model` is the contract every loaded model implements: prefill, decode,
//! tokenise, etc. `InferenceBackend` is the layer above it — a stateless
//! factory plus capability advertisement. The router walks the registered
//! backends, looks at each model's [`ModelProfile`] and asks "who handles
//! this best?" before instantiating.
//!
//! Adding a new backend (candle, mistralrs, …) means writing a struct that
//! implements `InferenceBackend` and registering it on the router.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::model::Model;
use crate::model_registry::{ModelProfile, RegistryConfig};

#[cfg(feature = "backend-candle")]
pub mod candle_backend;
pub mod llama_cpp_backend;
pub mod router;

#[cfg(feature = "backend-candle")]
pub use candle_backend::CandleBackend;
pub use llama_cpp_backend::LlamaCppBackend;
pub use router::BackendRouter;

/// Identifier strings for the built-in backends. Kept as constants so that
/// CLI flags, config files and the router agree on spelling.
pub mod ids {
    pub const LLAMA_CPP: &str = "llama-cpp";
    pub const CANDLE: &str = "candle";
}

/// Result of [`InferenceBackend::supports`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Compatibility {
    /// The backend implements this architecture first-class.
    Native,
    /// The backend can probably load it but the user should expect rough
    /// edges (suboptimal performance, missing optimisations, etc.).
    Workable,
    /// The backend declines this profile. The string is a short, human
    /// reason that can surface in error messages and logs.
    ///
    /// `#[allow(dead_code)]` because no shipped backend currently returns
    /// this variant — `LlamaCppBackend` accepts every architecture it sees.
    /// Future backends (candle, …) will use it when an architecture is
    /// outside their supported set.
    #[allow(dead_code)]
    Unsupported(String),
}

impl Compatibility {
    pub fn is_runnable(&self) -> bool {
        matches!(self, Self::Native | Self::Workable)
    }
}

/// Outcome of [`InferenceBackend::instantiate`].
///
/// Some backends (notably `LlamaCppBackend`) may degrade configuration on
/// load — for instance the OOM cascade reduces context length and KV cache
/// precision. The instance therefore reports the *effective* values used so
/// the registry can record the truth instead of the requested values.
pub struct BackendInstance {
    pub model: Arc<dyn Model>,
    pub effective_context_len: Option<u32>,
    pub effective_type_k: u32,
    pub effective_type_v: u32,
}

/// Stateless factory + capability descriptor for a model backend.
pub trait InferenceBackend: Send + Sync {
    /// Stable string identifier (matches CLI flag values, e.g. `"llama-cpp"`).
    fn id(&self) -> &'static str;

    /// Inspect the profile and tell the router how good a fit this backend is.
    /// Pure function — no I/O, no allocations of significant size.
    fn supports(&self, profile: &ModelProfile) -> Compatibility;

    /// Load the model from disk and produce a [`BackendInstance`].
    /// Implementations are expected to perform their own degradation
    /// strategies (OOM recovery, etc.) and report the values that finally
    /// stuck via the returned struct.
    fn instantiate(
        &self,
        path: &Path,
        profile: Option<&ModelProfile>,
        cfg: &RegistryConfig,
    ) -> Result<BackendInstance>;
}
