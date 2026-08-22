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
            // n_gpu_layers — always all of them, regardless of what the target model was
            // given. A draft is small, and it is only worth having if it is much faster
            // than the target; offloading part of it to the CPU would defeat speculation
            // rather than economise on VRAM.
            -1,
            main_gpu,
            split_mode,
            &tensor_split,
            moe_offload_cpu,
            None,  // mmproj_path — draft models are text-only speculation proposers
            &[],   // lora_modules — adapters apply to the primary model, not the draft
            false, // reranking — a draft model only ever proposes tokens
            0,     // rs_rollback — the draft holds no reusable prefix of its own
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
    mtp: Option<PathBuf>,
    lora_modules: Vec<(String, PathBuf, f32)>,
) -> Result<EngineEntry> {
    let path = path.to_path_buf();
    let name = name.to_string();
    let max_batch_size = cfg.max_batch_size;
    let rs_rollback = cfg.rs_rollback;
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
    let n_gpu_layers = cfg.n_gpu_layers;
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
            n_gpu_layers,
            main_gpu,
            split_mode,
            &tensor_split,
            moe_offload_cpu,
            mmproj.as_deref(),
            &lora_modules,
            reranking,
            rs_rollback,
        )
    })
    .await
    .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;

    // Attach the paired MTP head, if one was configured. Done after load() rather than
    // inside it because the head's context has to be linked to the target's, which does
    // not exist until load() returns — and because load() already carries 15 arguments.
    #[cfg(fox_mtp)]
    let model = {
        let mut model = model;
        if let Some(mtp_path) = mtp {
            if cfg.speculative {
                let n_draft = cfg.spec_draft_len as i32;
                model = tokio::task::spawn_blocking(move || {
                    model.enable_mtp(&mtp_path, n_draft).map(|()| model)
                })
                .await
                .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {e}"))??;
            } else {
                tracing::warn!(
                    "--mtp-model was set without --speculative true — the MTP head is ignored"
                );
            }
        }
        model
    };
    // Without the MTP shim there is nothing to attach it to; say so rather than
    // silently ignoring a path the operator asked for.
    #[cfg(not(fox_mtp))]
    if mtp.is_some() {
        tracing::warn!("this build has no MTP support (FOX_NO_MTP) — --mtp-model is ignored");
    }

    // Size the paged block pool from the backend's ACTUAL KV capacity
    // (llama_n_ctx), so the pool can never claim room llama.cpp didn't allocate.
    let kv_tokens = model.kv_cache_capacity();
    let model: Arc<dyn Model> = Arc::new(model);
    let active_backend = model.active_backend().to_string();
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
    } else if model.has_mtp() {
        // The model's own trained head beats both other proposers: unlike n-gram it does
        // not need the text to repeat, and unlike a draft model it costs no second set of
        // weights. Checked on the loaded model, not on the config, so a head that failed
        // to attach falls back rather than drafting nothing every step.
        Some(SpeculativeConfig::Mtp {
            draft_len: cfg.spec_draft_len,
        })
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

    // A model that cannot roll its KV back reaches a past prefix only by restoring a
    // serialised state, so for it the host-RAM prompt cache is not a tuning option — it
    // is the mechanism. Left at 0 (the flag's default) such a model reuses nothing at
    // all, which is what Qwen3.5 did: 20 slot hits, 20 refused trims, cached_tokens 0.
    // An explicit --cache-ram always wins; this only fills in the 0.
    const IMPLICIT_CACHE_RAM_BYTES: usize = 2048 * 1024 * 1024;
    let cache_ram_bytes = if cfg.cache_ram_bytes == 0 && !model.supports_seq_copy() {
        tracing::info!(
            mb = IMPLICIT_CACHE_RAM_BYTES / (1024 * 1024),
            "host-RAM prompt cache enabled implicitly: this model's KV cannot be rolled \
             back, so a checkpoint is the only way it can reuse a prompt"
        );
        IMPLICIT_CACHE_RAM_BYTES
    } else {
        cfg.cache_ram_bytes
    };

    let scheduler = Arc::new(
        Scheduler::with_max_queue_depth(kv_cache.clone(), max_batch_size, max_queue_depth)
            .with_kv_reuse(cfg.kv_reuse, cfg.slot_prompt_similarity)
            .with_prompt_cache(cache_ram_bytes),
    );
    let engine = Arc::new(InferenceEngine::new(
        model,
        scheduler,
        kv_cache,
        name,
        metrics.clone(),
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

    // One "model ready" per model. There used to be two — one after the weights
    // loaded and one after the engine was built — which read as two models loading.
    let supports_thinking = engine.supports_thinking();
    tracing::info!(
        model = %engine.model_name(),
        backend = %active_backend,
        thinking = supports_thinking,
        "model ready"
    );

    Ok(EngineEntry {
        engine,
        loop_handle,
        metrics,
    })
}
