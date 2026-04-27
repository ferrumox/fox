//! [`InferenceBackend`] wrapper around the bundled llama.cpp FFI loader.
//!
//! Owns the OOM degradation cascade that used to live in `loader.rs`: if the
//! requested context length or KV-cache precision cannot fit, the wrapper
//! retries with progressively smaller settings until one succeeds (or all
//! fail).

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::model::error::LoadError;
use crate::engine::model::LlamaCppModel;
use crate::model_registry::{kv_type, FlashAttnMode, ModelProfile, RegistryConfig};

use super::{ids, BackendInstance, Compatibility, InferenceBackend};

/// Sequence of `(context_len, kv_type)` pairs tried after the initial load
/// fails with an OOM-shaped error. Each entry is strictly less expensive than
/// the previous one. Order is intentional — the runtime stops at the first
/// success and only walks further on repeated failure.
const OOM_RETRY_CONFIGS: &[(u32, u32)] = &[
    (4096, kv_type::F16),
    (2048, kv_type::F16),
    (2048, kv_type::Q8_0),
    (1024, kv_type::Q8_0),
];

/// Stateless wrapper. The real state lives inside the `LlamaCppModel`
/// produced by [`Self::instantiate`].
pub struct LlamaCppBackend;

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for LlamaCppBackend {
    fn id(&self) -> &'static str {
        ids::LLAMA_CPP
    }

    fn supports(&self, profile: &ModelProfile) -> Compatibility {
        // llama.cpp covers the broadest catalogue of architectures and is the
        // safe fallback. We mark a handful of well-known sore spots as
        // "Workable" rather than "Native" so the router prefers a more
        // specialised backend when one is registered.
        if profile.quirks.multimodal.is_some() {
            return Compatibility::Workable;
        }
        if profile.quirks.hybrid_memory {
            return Compatibility::Workable;
        }
        if profile.quirks.nonstandard_head_dim && profile.head_dim >= 256 {
            return Compatibility::Workable;
        }
        Compatibility::Native
    }

    fn instantiate(
        &self,
        path: &Path,
        _profile: Option<&ModelProfile>,
        cfg: &RegistryConfig,
    ) -> Result<BackendInstance> {
        match try_load(path, cfg.max_context_len, cfg.type_k, cfg.type_v, cfg) {
            Ok(model) => Ok(BackendInstance {
                model: Arc::new(model),
                effective_context_len: cfg.max_context_len,
                effective_type_k: cfg.type_k,
                effective_type_v: cfg.type_v,
            }),
            Err(first_err) => {
                let kind = LoadError::classify(&first_err);
                if kind.is_oom() {
                    tracing::warn!(
                        "initial llama.cpp load failed ({kind}), retrying with reduced settings: {}",
                        first_err
                    );
                    run_oom_cascade(path, cfg, first_err)
                } else {
                    Err(first_err)
                }
            }
        }
    }
}

fn try_load(
    path: &Path,
    context_len: Option<u32>,
    type_k: u32,
    type_v: u32,
    cfg: &RegistryConfig,
) -> Result<LlamaCppModel> {
    LlamaCppModel::load(
        path,
        cfg.max_batch_size,
        context_len,
        cfg.gpu_memory_bytes,
        cfg.gpu_memory_fraction,
        type_k,
        type_v,
        cfg.main_gpu,
        cfg.split_mode,
        &cfg.tensor_split,
        cfg.moe_offload_cpu,
        cfg.flash_attn,
    )
}

fn run_oom_cascade(
    path: &Path,
    cfg: &RegistryConfig,
    first_err: anyhow::Error,
) -> Result<BackendInstance> {
    for &(retry_ctx, retry_kv) in OOM_RETRY_CONFIGS {
        match try_load(path, Some(retry_ctx), retry_kv, retry_kv, &cascade_cfg(cfg)) {
            Ok(model) => {
                let kv_label = match retry_kv {
                    kv_type::Q8_0 => "q8_0",
                    _ => "f16",
                };
                tracing::warn!(
                    context = retry_ctx,
                    type_kv = kv_label,
                    "llama.cpp loaded with reduced settings (OOM recovery)"
                );
                return Ok(BackendInstance {
                    model: Arc::new(model),
                    effective_context_len: Some(retry_ctx),
                    effective_type_k: retry_kv,
                    effective_type_v: retry_kv,
                });
            }
            Err(_) => continue,
        }
    }
    Err(first_err)
}

/// During OOM recovery we force `FlashAttnMode::Auto` so llama.cpp can pick
/// whatever pairing is cheapest with the smaller context. Other fields stay
/// untouched — the cascade only mutates context length and KV precision.
fn cascade_cfg(cfg: &RegistryConfig) -> RegistryConfig {
    RegistryConfig {
        flash_attn: FlashAttnMode::Auto,
        models_dir: cfg.models_dir.clone(),
        max_models: cfg.max_models,
        max_batch_size: cfg.max_batch_size,
        max_context_len: cfg.max_context_len,
        block_size: cfg.block_size,
        gpu_memory_bytes: cfg.gpu_memory_bytes,
        gpu_memory_fraction: cfg.gpu_memory_fraction,
        metrics: cfg.metrics.clone(),
        keep_alive_secs: cfg.keep_alive_secs,
        type_k: cfg.type_k,
        type_v: cfg.type_v,
        main_gpu: cfg.main_gpu,
        split_mode: cfg.split_mode,
        tensor_split: cfg.tensor_split.clone(),
        moe_offload_cpu: cfg.moe_offload_cpu,
        backend_override: cfg.backend_override.clone(),
        backend_priority: cfg.backend_priority.clone(),
    }
}

