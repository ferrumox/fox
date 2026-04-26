use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::backend::{BackendRouter, InferenceBackend, LlamaCppBackend};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::Scheduler;

use super::arch_table::{recommend_backend, DiagnosticLevel};
use super::config::RegistryConfig;
use super::entry::EngineEntry;
use super::inspect::{self, ModelProfile};

pub(super) async fn load_model(
    name: &str,
    path: &Path,
    cfg: &RegistryConfig,
) -> Result<EngineEntry> {
    let path = path.to_path_buf();
    let name = name.to_string();

    let block_size = cfg.block_size;
    let max_batch_size = cfg.max_batch_size;
    let gpu_memory_bytes = cfg.gpu_memory_bytes;
    let gpu_memory_fraction = cfg.gpu_memory_fraction;
    let metrics = cfg.metrics.clone();

    warn_if_unlikely_to_fit_in_vram(&path, &name, gpu_memory_bytes);

    tracing::info!(model = %name, path = ?path, "loading model");

    let profile = inspect_and_log(&path, &name);

    let backend = pick_backend(profile.as_ref(), cfg)?;
    tracing::info!(model = %name, backend = backend.id(), "selected backend");

    let instance = {
        let path_for_load = path.clone();
        let cfg_for_load = clone_cfg(cfg);
        let profile_for_load = profile.clone();
        let backend = backend.clone();
        tokio::task::spawn_blocking(move || {
            backend.instantiate(&path_for_load, profile_for_load.as_ref(), &cfg_for_load)
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??
    };

    let model = instance.model;
    let model_config = model.model_config();
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        gpu_memory_fraction,
        block_size,
        instance.effective_type_k,
        instance.effective_type_v,
    ));

    let scheduler = Arc::new(Scheduler::new(kv_cache.clone(), max_batch_size));
    let engine = Arc::new(InferenceEngine::new(
        model, scheduler, kv_cache, name, metrics,
    ));

    let loop_handle = {
        let e = engine.clone();
        tokio::spawn(async move {
            if let Err(err) = e.run_loop().await {
                tracing::error!("engine loop error: {err}");
            }
        })
    };

    tracing::info!(
        model = %engine.model_name(),
        thinking = engine.supports_thinking(),
        channel_thinking = engine.uses_channel_thinking(),
        context = engine.context_len(),
        effective_context_hint = ?instance.effective_context_len,
        "model ready"
    );

    Ok(EngineEntry {
        engine,
        loop_handle,
    })
}

/// Heuristic VRAM check: file size × 1.8 covers weights + overhead. Warn the
/// user up-front instead of letting the loader fail with an opaque CUDA error.
fn warn_if_unlikely_to_fit_in_vram(path: &Path, name: &str, gpu_memory_bytes: usize) {
    let Ok(meta) = std::fs::metadata(path) else {
        return;
    };
    let estimated_bytes = (meta.len() as f64 * 1.8) as usize;
    if estimated_bytes > gpu_memory_bytes {
        let est_gib = estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let avail_gib = gpu_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        tracing::warn!(
            model = %name,
            estimated_gib = format!("{est_gib:.1}"),
            available_gib = format!("{avail_gib:.1}"),
            "model may not fit in VRAM — consider a smaller quantization, \
             --max-context-len to reduce KV cache, or closing other GPU processes"
        );
    }
}

/// Run the GGUF probe and emit human-readable diagnostics. Returns the parsed
/// profile when available so the backend router can use it for routing; on
/// failure (file missing, malformed header, etc.) returns `None` and lets the
/// backend produce its own error.
fn inspect_and_log(path: &Path, name: &str) -> Option<ModelProfile> {
    match inspect::probe(path) {
        Ok(profile) => {
            let (hint, notes) = recommend_backend(&profile);
            tracing::info!(
                model = %name,
                architecture = %profile.architecture,
                head_dim = profile.head_dim,
                kv_heads = profile.head_count_kv,
                experts = ?profile.quirks.moe_experts,
                multimodal = ?profile.quirks.multimodal,
                recommended_backend = ?hint,
                "model profile"
            );
            for note in &notes {
                match note.level {
                    DiagnosticLevel::Info => {
                        tracing::info!(model = %name, "{}", note.message);
                    }
                    DiagnosticLevel::Warn => {
                        if let Some(hint) = &note.hint {
                            tracing::warn!(model = %name, "{} — {}", note.message, hint);
                        } else {
                            tracing::warn!(model = %name, "{}", note.message);
                        }
                    }
                    DiagnosticLevel::Block => {
                        if let Some(hint) = &note.hint {
                            tracing::error!(model = %name, "{} — {}", note.message, hint);
                        } else {
                            tracing::error!(model = %name, "{}", note.message);
                        }
                    }
                }
            }
            Some(profile)
        }
        Err(err) => {
            tracing::warn!(
                model = %name,
                "could not pre-inspect GGUF metadata ({}); proceeding with load",
                err
            );
            None
        }
    }
}

/// Build the router with every backend compiled into this binary, then ask it
/// to pick one for the supplied profile. Order of registration is the tie
/// breaker used by `BackendRouter::pick` when neither the override nor the
/// architecture table can decide.
fn pick_backend(
    profile: Option<&ModelProfile>,
    cfg: &RegistryConfig,
) -> Result<Arc<dyn InferenceBackend>> {
    let mut router = BackendRouter::new();
    router.register(Arc::new(LlamaCppBackend::new()));
    #[cfg(feature = "backend-candle")]
    router.register(Arc::new(crate::engine::backend::CandleBackend::new()));

    router.pick(
        profile,
        cfg.backend_override.as_deref(),
        &cfg.backend_priority,
    )
}

/// `RegistryConfig` is not `Clone` because it owns an `Option<Arc<Metrics>>`
/// shared with the rest of the program. We clone field-by-field so the loader
/// can hand a self-contained copy to `spawn_blocking`.
fn clone_cfg(cfg: &RegistryConfig) -> RegistryConfig {
    RegistryConfig {
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
        flash_attn: cfg.flash_attn,
        backend_override: cfg.backend_override.clone(),
        backend_priority: cfg.backend_priority.clone(),
    }
}
