use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::Scheduler;

use super::config::kv_type;
use super::config::RegistryConfig;
use super::entry::EngineEntry;

fn detect_mmproj(model_path: &Path) -> Option<PathBuf> {
    let dir = model_path.parent()?;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.filter_map(|e| e.ok()) {
        let p = entry.path();
        let name = p.file_name()?.to_str()?.to_lowercase();
        if name.contains("mmproj") && name.ends_with(".gguf") {
            return Some(p);
        }
    }
    None
}

struct OomFallback {
    context_len: Option<u32>,
    type_k: u32,
    type_v: u32,
    label: String,
}

fn kv_type_name(t: u32) -> &'static str {
    match t {
        kv_type::F16 => "f16",
        kv_type::Q8_0 => "q8_0",
        kv_type::Q4_0 => "q4_0",
        kv_type::TURBO3 => "turbo3",
        kv_type::TURBO4 => "turbo4",
        kv_type::TURBO2 => "turbo2",
        _ => "unknown",
    }
}

fn build_oom_fallbacks(original_ctx: Option<u32>, type_k: u32, type_v: u32) -> Vec<OomFallback> {
    let mut fallbacks = Vec::new();

    let start = original_ctx.unwrap_or(65536);
    let mut ctx = start / 2;
    while ctx >= 1024 {
        fallbacks.push(OomFallback {
            context_len: Some(ctx),
            type_k,
            type_v,
            label: format!("context={ctx} kv={}", kv_type_name(type_k)),
        });
        ctx /= 2;
    }

    if type_k != kv_type::Q8_0 || type_v != kv_type::Q8_0 {
        for ctx in [2048, 1024] {
            fallbacks.push(OomFallback {
                context_len: Some(ctx),
                type_k: kv_type::Q8_0,
                type_v: kv_type::Q8_0,
                label: format!("context={ctx} kv=q8_0"),
            });
        }
    }

    fallbacks
}

fn is_retryable_oom(err: &anyhow::Error) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("llama_init_from_model failed")
        || msg.contains("out of memory")
        || msg.contains("failed to allocate")
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
    let mmproj_path = cfg.mmproj_path.clone().or_else(|| detect_mmproj(&path));
    let flash_attn = cfg.flash_attn;

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

    if let Some(ref mmproj) = mmproj_path {
        tracing::info!(model = %name, mmproj = ?mmproj, "multimodal projector detected");
    }

    let (model, degraded, effective_type_k, effective_type_v) = {
        let p = path.clone();
        let ts = tensor_split.clone();
        let name_for_log = name.clone();
        let mmproj = mmproj_path.clone();
        tokio::task::spawn_blocking(move || {
            let name = name_for_log;

            match LlamaCppModel::load(
                &p, max_batch_size, max_context_len, gpu_memory_bytes, gpu_memory_fraction,
                type_k, type_v, main_gpu, split_mode, &ts, moe_offload_cpu,
                mmproj.as_deref(), flash_attn,
            ) {
                Ok(model) => return Ok((model, None, type_k, type_v)),
                Err(e) if is_retryable_oom(&e) => {
                    tracing::warn!(
                        model = %name,
                        error = %e,
                        "OOM on initial load, trying reduced settings"
                    );
                }
                Err(e) => return Err(e),
            }

            let fallbacks = build_oom_fallbacks(max_context_len, type_k, type_v);
            let mut last_err = None;
            for fb in &fallbacks {
                tracing::info!(model = %name, fallback = %fb.label, "OOM recovery: retrying");
                match LlamaCppModel::load(
                    &p, max_batch_size, fb.context_len, gpu_memory_bytes, gpu_memory_fraction,
                    fb.type_k, fb.type_v, main_gpu, split_mode, &ts, moe_offload_cpu,
                    mmproj.as_deref(), flash_attn,
                ) {
                    Ok(model) => {
                        tracing::warn!(
                            model = %name,
                            settings = %fb.label,
                            "loaded with reduced settings (OOM recovery)"
                        );
                        return Ok((model, Some(fb.label.clone()), fb.type_k, fb.type_v));
                    }
                    Err(e) if is_retryable_oom(&e) => {
                        last_err = Some(e);
                        continue;
                    }
                    Err(e) => return Err(e),
                }
            }

            Err(last_err.unwrap_or_else(|| anyhow::anyhow!("OOM recovery exhausted all fallbacks")))
        })
        .await
        .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??
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
    let supports_vision = engine.supports_vision();
    if let Some(ref d) = degraded {
        tracing::warn!(
            model = %engine.model_name(),
            thinking = supports_thinking,
            vision = supports_vision,
            degraded = %d,
            "model ready (degraded)"
        );
    } else {
        tracing::info!(
            model = %engine.model_name(),
            thinking = supports_thinking,
            vision = supports_vision,
            "model ready"
        );
    }

    Ok(EngineEntry {
        engine,
        loop_handle,
        degraded,
    })
}
