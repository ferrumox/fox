use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::Scheduler;

use super::config::{FlashAttnMode, RegistryConfig};
use super::entry::EngineEntry;

/// OOM recovery configurations tried in sequence until one succeeds.
/// Earlier attempts use larger context + higher precision; later attempts degrade gracefully.
const OOM_RETRY_CONFIGS: &[(Option<u32>, u32)] = &[
    (Some(4096), super::kv_type::F16),
    (Some(2048), super::kv_type::F16),
    (Some(2048), super::kv_type::Q8_0),
    (Some(1024), super::kv_type::Q8_0),
];

fn is_oom_error(err: &anyhow::Error) -> bool {
    let msg = format!("{err:#}");
    msg.contains("out of memory")
        || msg.contains("failed to allocate")
        || msg.contains("CUDA error")
        || msg.contains("llama_init_from_model failed")
}

pub(super) async fn load_model(
    name: &str,
    path: &Path,
    cfg: &RegistryConfig,
) -> Result<EngineEntry> {
    let path = path.to_path_buf();
    let name = name.to_string();
    let max_batch_size = cfg.max_batch_size;
    let max_context_len = cfg.max_context_len;
    let gpu_memory_bytes = cfg.gpu_memory_bytes;
    let gpu_memory_fraction = cfg.gpu_memory_fraction;
    let block_size = cfg.block_size;
    let metrics = cfg.metrics.clone();
    let type_k = cfg.type_k;
    let type_v = cfg.type_v;
    let main_gpu = cfg.main_gpu;
    let split_mode = cfg.split_mode;
    let tensor_split = cfg.tensor_split.clone();
    let moe_offload_cpu = cfg.moe_offload_cpu;
    let flash_attn = cfg.flash_attn;

    // Estimate VRAM requirement before attempting to load.
    // Heuristic: file_size × 1.8 covers weights + overhead. Warn early so the
    // user gets actionable advice instead of a cryptic load failure.
    if let Ok(meta) = std::fs::metadata(&path) {
        let estimated_bytes = (meta.len() as f64 * 1.8) as usize;
        let available_bytes = gpu_memory_bytes;
        if estimated_bytes > available_bytes {
            let est_gib = estimated_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            let avail_gib = available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::warn!(
                model = %name,
                estimated_gib = format!("{est_gib:.1}"),
                available_gib = format!("{avail_gib:.1}"),
                "model may not fit in VRAM — consider a smaller quantization, \
                 --max-context-len to reduce KV cache, or closing other GPU processes"
            );
        }
    }

    tracing::info!(model = %name, path = ?path, "loading model");

    // Try the user-requested configuration first, then degrade via OOM_RETRY_CONFIGS.
    let (model, effective_ctx, effective_type_k, effective_type_v) = {
        let path_c = path.clone();
        let ts = tensor_split.clone();
        let name_c = name.clone();

        let first_result = tokio::task::spawn_blocking({
            let p = path_c.clone();
            let ts2 = ts.clone();
            move || {
                LlamaCppModel::load(
                    &p, max_batch_size, max_context_len, gpu_memory_bytes,
                    gpu_memory_fraction, type_k, type_v, main_gpu, split_mode,
                    &ts2, moe_offload_cpu, flash_attn,
                )
            }
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?;

        match first_result {
            Ok(m) => (m, max_context_len, type_k, type_v),
            Err(first_err) if is_oom_error(&first_err) => {
                tracing::warn!(
                    model = %name_c,
                    "initial load failed (OOM), attempting recovery with reduced settings"
                );
                let mut recovered = None;
                for &(retry_ctx, retry_kv) in OOM_RETRY_CONFIGS {
                    let p2 = path_c.clone();
                    let ts3 = ts.clone();
                    let attempt = tokio::task::spawn_blocking(move || {
                        LlamaCppModel::load(
                            &p2, max_batch_size, Some(retry_ctx.unwrap()), gpu_memory_bytes,
                            gpu_memory_fraction, retry_kv, retry_kv, main_gpu, split_mode,
                            &ts3, moe_offload_cpu, FlashAttnMode::Auto,
                        )
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?;

                    match attempt {
                        Ok(m) => {
                            let kv_name = match retry_kv {
                                super::kv_type::Q8_0 => "q8_0",
                                _ => "f16",
                            };
                            tracing::warn!(
                                model = %name_c,
                                context = retry_ctx.unwrap(),
                                type_kv = kv_name,
                                "loaded with reduced settings (OOM recovery)"
                            );
                            recovered = Some((m, retry_ctx, retry_kv, retry_kv));
                            break;
                        }
                        Err(_) => continue,
                    }
                }
                match recovered {
                    Some((m, ctx, tk, tv)) => (m, ctx, tk, tv),
                    None => return Err(first_err),
                }
            }
            Err(e) => return Err(e),
        }
    };

    let model_config = model.model_config();
    let model: Arc<dyn Model> = Arc::new(model);
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        gpu_memory_fraction,
        block_size,
        effective_type_k,
        effective_type_v,
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

    let supports_thinking = engine.supports_thinking();
    let uses_channel = engine.uses_channel_thinking();
    let _ = effective_ctx; // used as hint; actual context comes from engine.context_len()
    tracing::info!(
        model = %engine.model_name(),
        thinking = supports_thinking,
        channel_thinking = uses_channel,
        context = engine.context_len(),
        "model ready"
    );

    Ok(EngineEntry {
        engine,
        loop_handle,
    })
}
