use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::{EngineOptions, InferenceEngine, SpeculativeConfig};
use crate::kv_cache::KVCacheManager;
use crate::scheduler::Scheduler;

use super::config::RegistryConfig;
use super::entry::EngineEntry;

/// Load the draft model for draft-model speculation: same load parameters as the
/// target (GPU/quantization config), but `max_batch_size = 1` since it only ever
/// decodes one dedicated sequence, never batches with the target's requests.
/// Fails loudly (via the same `ModelLoadFailed` path as any other load error) if the
/// draft's tokenizer doesn't match the target's — a draft token id from a mismatched
/// tokenizer is meaningless input to the target's verify batch.
async fn load_draft_model(
    stem: &str,
    path: PathBuf,
    cfg: &RegistryConfig,
    target: &Arc<dyn Model>,
) -> Result<Arc<dyn Model>> {
    let max_context_len = cfg.max_context_len;
    let gpu_memory_bytes = cfg.gpu_memory_bytes;
    let gpu_memory_fraction = cfg.gpu_memory_fraction;
    let type_k = cfg.type_k;
    let type_v = cfg.type_v;
    let main_gpu = cfg.main_gpu;
    let split_mode = cfg.split_mode;
    let tensor_split = cfg.tensor_split.clone();
    let moe_offload_cpu = cfg.moe_offload_cpu;

    tracing::info!(draft_model = %stem, path = ?path, "loading draft model");
    let draft = tokio::task::spawn_blocking(move || {
        LlamaCppModel::load(
            &path,
            1, // max_batch_size — the draft never batches, one dedicated sequence
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_k,
            type_v,
            main_gpu,
            split_mode,
            &tensor_split,
            moe_offload_cpu,
            None,  // mmproj_path — draft models are text-only speculation proposers
            &[],   // lora_modules — adapters apply to the primary model, not the draft
            false, // reranking — a draft model only ever proposes tokens
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))?
    .with_context(|| format!("failed to load draft model '{stem}'"))?;

    if draft.vocab_fingerprint() != target.vocab_fingerprint() {
        anyhow::bail!(
            "draft model '{stem}' does not share a tokenizer with the target model \
             (vocab fingerprint mismatch) — draft and target must use the same tokenizer"
        );
    }

    tracing::info!(draft_model = %stem, "draft model ready");
    Ok(Arc::new(draft))
}

pub(super) async fn load_model(
    name: &str,
    path: &Path,
    cfg: &RegistryConfig,
    draft: Option<(String, PathBuf)>,
    mmproj: Option<PathBuf>,
    lora_modules: Vec<(String, PathBuf, f32)>,
) -> Result<EngineEntry> {
    let path = path.to_path_buf();
    let name = name.to_string();
    let max_batch_size = cfg.max_batch_size;
    let max_queue_depth = cfg.max_queue_depth;
    let max_prefill_chunk = cfg.max_prefill_chunk;
    // None disables context rolling; Some(n_keep) enables it, preserving n_keep head tokens.
    let context_shift = cfg.context_shift.then_some(cfg.context_keep);
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

    let reranking = cfg.reranking;
    let model = tokio::task::spawn_blocking(move || {
        LlamaCppModel::load(
            &path,
            max_batch_size,
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_k,
            type_v,
            main_gpu,
            split_mode,
            &tensor_split,
            moe_offload_cpu,
            mmproj.as_deref(),
            &lora_modules,
            reranking,
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;

    // Size the paged block pool from the backend's ACTUAL KV capacity
    // (llama_n_ctx), so the pool can never claim room llama.cpp didn't allocate.
    let kv_tokens = model.kv_cache_capacity();
    let model: Arc<dyn Model> = Arc::new(model);
    tracing::info!(model = %name, backend = %model.active_backend(), "model ready");
    let kv_cache = Arc::new(KVCacheManager::from_kv_tokens(kv_tokens, block_size));

    // Draft-model speculation (0.16): load the draft eagerly, alongside the target,
    // and check the tokenizer matches. Loaded once for the process lifetime — no
    // eviction pairing/VRAM budgeting (see `docs/design/speculative-roadmap.md`).
    let draft_model: Option<Arc<dyn Model>> = match draft {
        Some((draft_stem, draft_path)) => {
            Some(load_draft_model(&draft_stem, draft_path, cfg, &model).await?)
        }
        None => None,
    };

    // None disables speculation; `Ngram` is the 0.15 default, `Draft` when a draft
    // model was configured (and successfully loaded above).
    let speculative = if !cfg.speculative {
        None
    } else if draft_model.is_some() {
        Some(SpeculativeConfig::Draft {
            draft_len: cfg.spec_draft_len,
        })
    } else {
        Some(SpeculativeConfig::Ngram {
            ngram: cfg.spec_ngram,
            draft_len: cfg.spec_draft_len,
        })
    };

    let scheduler = Arc::new(
        Scheduler::with_max_queue_depth(kv_cache.clone(), max_batch_size, max_queue_depth)
            .with_kv_reuse(cfg.kv_reuse, cfg.slot_prompt_similarity)
            .with_prompt_cache(cfg.cache_ram_bytes),
    );
    let engine = Arc::new(InferenceEngine::new(
        model,
        scheduler,
        kv_cache,
        name,
        metrics,
        EngineOptions {
            max_prefill_chunk,
            context_shift,
            speculative,
        },
        draft_model,
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
