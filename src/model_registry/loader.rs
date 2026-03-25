use std::path::Path;
use std::sync::Arc;

use anyhow::Result;

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::Scheduler;

use super::config::RegistryConfig;
use super::entry::EngineEntry;

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
    let type_kv = cfg.type_kv;
    let main_gpu = cfg.main_gpu;
    let split_mode = cfg.split_mode;
    let tensor_split = cfg.tensor_split.clone();
    let moe_offload_cpu = cfg.moe_offload_cpu;

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

    let model = tokio::task::spawn_blocking(move || {
        LlamaCppModel::load(
            &path,
            max_batch_size,
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_kv,
            main_gpu,
            split_mode,
            &tensor_split,
            moe_offload_cpu,
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;

    let model_config = model.model_config();
    let model: Arc<dyn Model> = Arc::new(model);
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        gpu_memory_fraction,
        block_size,
        type_kv,
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
    tracing::info!(
        model = %engine.model_name(),
        thinking = supports_thinking,
        "model ready"
    );

    Ok(EngineEntry {
        engine,
        loop_handle,
    })
}
