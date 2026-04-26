//! Choose which [`InferenceBackend`] should run a given model.
//!
//! The router is the single decision point that ties together CLI overrides,
//! the architecture-table recommendation and the per-backend `supports`
//! capability check. Once a backend is picked, the loader hands control to
//! [`InferenceBackend::instantiate`].

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::model_registry::{recommend_backend, BackendHint, ModelProfile};

use super::{ids, InferenceBackend};

/// Registry of available backends, in registration order. Order matters only
/// when no preference can be derived from the user override or from the
/// architecture table — in that case the router picks the first registered
/// backend that reports a runnable [`Compatibility`].
pub struct BackendRouter {
    backends: Vec<Arc<dyn InferenceBackend>>,
}

impl BackendRouter {
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    pub fn register(&mut self, backend: Arc<dyn InferenceBackend>) {
        self.backends.push(backend);
    }

    pub fn registered_ids(&self) -> Vec<&'static str> {
        self.backends.iter().map(|b| b.id()).collect()
    }

    fn find_by_id(&self, id: &str) -> Option<Arc<dyn InferenceBackend>> {
        self.backends
            .iter()
            .find(|b| b.id().eq_ignore_ascii_case(id))
            .cloned()
    }

    /// Pick a backend for `profile`. The decision tree:
    ///
    /// 1. If `override_id` is set, use that backend (error if unknown).
    /// 2. If `profile` is provided, ask the architecture table for a hint.
    ///    When the hint matches a registered backend that reports the model
    ///    as runnable, use it.
    /// 3. Walk `priority` in order and pick the first registered backend
    ///    that reports the model as runnable.
    /// 4. As a last resort, pick the first registered backend that reports
    ///    the model as runnable, or — when no profile is available — the
    ///    first registered backend.
    pub fn pick(
        &self,
        profile: Option<&ModelProfile>,
        override_id: Option<&str>,
        priority: &[String],
    ) -> Result<Arc<dyn InferenceBackend>> {
        if self.backends.is_empty() {
            return Err(anyhow!("no inference backends are registered"));
        }

        if let Some(id) = override_id {
            return self.find_by_id(id).ok_or_else(|| {
                anyhow!(
                    "unknown backend '{id}'; registered: {}",
                    self.registered_ids().join(", ")
                )
            });
        }

        if let Some(profile) = profile {
            if let Some(picked) = self.pick_from_profile(profile) {
                return Ok(picked);
            }
        }

        for id in priority {
            if let Some(backend) = self.find_by_id(id) {
                if profile_runnable(&backend, profile) {
                    return Ok(backend);
                }
            }
        }

        for backend in &self.backends {
            if profile_runnable(backend, profile) {
                return Ok(backend.clone());
            }
        }

        Err(anyhow!(
            "no registered backend can handle this model; tried: {}",
            self.registered_ids().join(", ")
        ))
    }

    fn pick_from_profile(&self, profile: &ModelProfile) -> Option<Arc<dyn InferenceBackend>> {
        let (hint, _) = recommend_backend(profile);
        let target_id = match hint {
            BackendHint::LlamaCpp => Some(ids::LLAMA_CPP),
            BackendHint::Candle => Some(ids::CANDLE),
            BackendHint::Either => None,
        }?;

        let backend = self.find_by_id(target_id)?;
        if backend.supports(profile).is_runnable() {
            Some(backend)
        } else {
            None
        }
    }
}

impl Default for BackendRouter {
    fn default() -> Self {
        Self::new()
    }
}

fn profile_runnable(backend: &Arc<dyn InferenceBackend>, profile: Option<&ModelProfile>) -> bool {
    match profile {
        Some(p) => backend.supports(p).is_runnable(),
        None => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::backend::ids;
    use crate::engine::backend::{BackendInstance, Compatibility, InferenceBackend};
    use crate::engine::model::{Model, StubModel};
    use crate::model_registry::{ArchQuirks, ModelProfile, RegistryConfig};
    use std::path::Path;

    /// Test double that reports a fixed compatibility and instantiates a
    /// `StubModel` regardless of inputs.
    struct FakeBackend {
        id: &'static str,
        verdict: Compatibility,
    }

    impl InferenceBackend for FakeBackend {
        fn id(&self) -> &'static str {
            self.id
        }
        fn supports(&self, _profile: &ModelProfile) -> Compatibility {
            self.verdict.clone()
        }
        fn instantiate(
            &self,
            _path: &Path,
            _profile: Option<&ModelProfile>,
            _cfg: &RegistryConfig,
        ) -> Result<BackendInstance> {
            Ok(BackendInstance {
                model: Arc::new(StubModel) as Arc<dyn Model>,
                effective_context_len: None,
                effective_type_k: 1,
                effective_type_v: 1,
            })
        }
    }

    fn router_with(specs: &[(&'static str, Compatibility)]) -> BackendRouter {
        let mut r = BackendRouter::new();
        for (id, verdict) in specs {
            r.register(Arc::new(FakeBackend {
                id,
                verdict: verdict.clone(),
            }));
        }
        r
    }

    fn profile(arch: &str) -> ModelProfile {
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

    /// `Arc<dyn InferenceBackend>` is not `Debug`, which prevents `Result::unwrap`
    /// from compiling. These helpers extract the success/error value via `match`.
    fn ok(result: Result<Arc<dyn InferenceBackend>>) -> Arc<dyn InferenceBackend> {
        match result {
            Ok(b) => b,
            Err(e) => panic!("expected Ok backend, got error: {e}"),
        }
    }
    fn err(result: Result<Arc<dyn InferenceBackend>>) -> anyhow::Error {
        match result {
            Ok(_) => panic!("expected error, got Ok backend"),
            Err(e) => e,
        }
    }

    #[test]
    fn override_takes_precedence_over_profile_hint() {
        let r = router_with(&[
            (ids::LLAMA_CPP, Compatibility::Native),
            (ids::CANDLE, Compatibility::Native),
        ]);
        let p = profile("gemma4"); // recommend_backend → Candle
        let chosen = ok(r.pick(Some(&p), Some(ids::LLAMA_CPP), &[]));
        assert_eq!(chosen.id(), ids::LLAMA_CPP);
    }

    #[test]
    fn unknown_override_returns_error() {
        let r = router_with(&[(ids::LLAMA_CPP, Compatibility::Native)]);
        let e = err(r.pick(None, Some("gibberish"), &[]));
        assert!(e.to_string().contains("unknown backend"));
    }

    #[test]
    fn profile_hint_drives_selection_when_no_override() {
        let r = router_with(&[
            (ids::LLAMA_CPP, Compatibility::Native),
            (ids::CANDLE, Compatibility::Native),
        ]);
        let mut p = profile("gemma4");
        p.head_dim = 512;
        p.quirks.nonstandard_head_dim = true;
        let chosen = ok(r.pick(Some(&p), None, &[]));
        assert_eq!(chosen.id(), ids::CANDLE);
    }

    #[test]
    fn falls_through_to_priority_when_hint_is_either() {
        let r = router_with(&[
            (ids::CANDLE, Compatibility::Native),
            (ids::LLAMA_CPP, Compatibility::Native),
        ]);
        let p = profile("acme-arch"); // unknown → Either
        let chosen = ok(r.pick(Some(&p), None, &[ids::LLAMA_CPP.to_string()]));
        assert_eq!(chosen.id(), ids::LLAMA_CPP);
    }

    #[test]
    fn skips_backends_that_decline_the_profile() {
        let r = router_with(&[
            (ids::CANDLE, Compatibility::Unsupported("nope".into())),
            (ids::LLAMA_CPP, Compatibility::Native),
        ]);
        let mut p = profile("gemma4");
        p.head_dim = 512;
        p.quirks.nonstandard_head_dim = true;
        let chosen = ok(r.pick(Some(&p), None, &[]));
        assert_eq!(chosen.id(), ids::LLAMA_CPP);
    }

    #[test]
    fn empty_router_errors() {
        let r = BackendRouter::new();
        let e = err(r.pick(None, None, &[]));
        assert!(e.to_string().contains("no inference backends"));
    }

    #[test]
    fn no_profile_returns_first_registered_backend() {
        let r = router_with(&[
            (ids::CANDLE, Compatibility::Native),
            (ids::LLAMA_CPP, Compatibility::Native),
        ]);
        let chosen = ok(r.pick(None, None, &[]));
        assert_eq!(chosen.id(), ids::CANDLE);
    }
}
