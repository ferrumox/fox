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

    tracing::info!(model = %name, path = ?path, "loading model");

    let model = tokio::task::spawn_blocking(move || {
        LlamaCppModel::load(
            &path,
            max_batch_size,
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_kv,
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
