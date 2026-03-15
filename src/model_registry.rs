// ModelRegistry: load models on demand, keep up to N in memory (LRU eviction).

use dashmap::DashMap;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;

use crate::cli::list_models;
use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::metrics::Metrics;
use crate::scheduler::Scheduler;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

pub struct RegistryConfig {
    pub models_dir: PathBuf,
    pub max_models: usize,
    pub max_batch_size: usize,
    pub max_context_len: u32,
    pub block_size: usize,
    pub gpu_memory_bytes: usize,
    pub gpu_memory_fraction: f32,
    pub metrics: Option<Arc<Metrics>>,
}

// ---------------------------------------------------------------------------
// EngineEntry
// ---------------------------------------------------------------------------

pub struct EngineEntry {
    pub engine: Arc<InferenceEngine>,
    /// Aborted when this entry is dropped (LRU eviction or explicit unload).
    loop_handle: tokio::task::JoinHandle<()>,
}

impl Drop for EngineEntry {
    fn drop(&mut self) {
        self.loop_handle.abort();
    }
}

// ---------------------------------------------------------------------------
// ModelRegistry
// ---------------------------------------------------------------------------

pub struct ModelRegistry {
    engines: DashMap<String, Arc<EngineEntry>>,
    lru: Mutex<lru::LruCache<String, ()>>,
    config: RegistryConfig,
    aliases: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new(config: RegistryConfig, aliases: HashMap<String, String>) -> Self {
        let cap = NonZeroUsize::new(config.max_models.max(1)).unwrap();
        Self {
            engines: DashMap::new(),
            lru: Mutex::new(lru::LruCache::new(cap)),
            config,
            aliases,
        }
    }

    /// Resolve `name`, returning the cached engine if already loaded or loading
    /// it from disk otherwise. Evicts the LRU model if capacity is exceeded.
    pub async fn get_or_load(&self, name: &str) -> Result<Arc<EngineEntry>> {
        let (stem, path) = self.resolve_model_name(name)?;

        // Already loaded — promote to MRU and return.
        if let Some(entry) = self.engines.get(&stem) {
            if let Ok(mut lru) = self.lru.lock() {
                lru.get(&stem); // promotes
            }
            return Ok(entry.clone());
        }

        // Evict if we are at capacity before loading.
        self.evict_lru_if_needed();

        // Load the model (FFI is blocking, so we use spawn_blocking inside).
        let entry = Arc::new(load_model(&stem, &path, &self.config).await?);
        self.engines.insert(stem.clone(), entry.clone());
        if let Ok(mut lru) = self.lru.lock() {
            lru.put(stem, ());
        }

        Ok(entry)
    }

    /// Returns all currently-loaded (name, entry) pairs.
    pub fn loaded(&self) -> Vec<(String, Arc<EngineEntry>)> {
        self.engines
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }

    /// Explicitly unload a model by stem name. Returns `true` if it was loaded.
    pub fn unload(&self, name: &str) -> bool {
        let removed = self.engines.remove(name).is_some();
        if removed {
            if let Ok(mut lru) = self.lru.lock() {
                lru.pop(name);
            }
        }
        removed
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    fn evict_lru_if_needed(&self) {
        while self.engines.len() >= self.config.max_models {
            let lru_key = {
                let mut lru = self.lru.lock().unwrap();
                lru.pop_lru().map(|(k, _)| k)
            };
            match lru_key {
                Some(name) => {
                    self.engines.remove(&name);
                    tracing::info!("evicted model '{}' from registry (LRU)", name);
                }
                None => break,
            }
        }
    }

    /// Resolve a user-supplied name to `(stem, PathBuf)` using the following priority:
    /// 1. Alias lookup
    /// 2. Exact stem match (case-insensitive)
    /// 3. Stem starts with the name
    /// 4. Stem contains the name
    fn resolve_model_name(&self, name: &str) -> Result<(String, PathBuf)> {
        let resolved = self
            .aliases
            .get(name)
            .map(String::as_str)
            .unwrap_or(name);

        let entries = list_models(&self.config.models_dir)?;
        let lower = resolved.to_lowercase();

        // 1. Exact match (case-insensitive)
        for (path, _) in &entries {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem.eq_ignore_ascii_case(resolved) {
                    return Ok((stem.to_string(), path.clone()));
                }
            }
        }

        // 2. Starts with
        for (path, _) in &entries {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem.to_lowercase().starts_with(&lower) {
                    return Ok((stem.to_string(), path.clone()));
                }
            }
        }

        // 3. Contains
        for (path, _) in &entries {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem.to_lowercase().contains(&lower) {
                    return Ok((stem.to_string(), path.clone()));
                }
            }
        }

        anyhow::bail!(
            "model '{}' not found in {}",
            name,
            self.config.models_dir.display()
        )
    }
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

async fn load_model(name: &str, path: &Path, cfg: &RegistryConfig) -> Result<EngineEntry> {
    let path = path.to_path_buf();
    let name = name.to_string();
    let max_batch_size = cfg.max_batch_size;
    let max_context_len = cfg.max_context_len;
    let gpu_memory_bytes = cfg.gpu_memory_bytes;
    let gpu_memory_fraction = cfg.gpu_memory_fraction;
    let block_size = cfg.block_size;
    let metrics = cfg.metrics.clone();

    tracing::info!("loading model '{}' from {:?}", name, path);

    let model = tokio::task::spawn_blocking(move || {
        LlamaCppModel::load(&path, max_batch_size, max_context_len)
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
    ));
    let scheduler = Arc::new(Scheduler::new(kv_cache.clone(), max_batch_size));
    let engine = Arc::new(InferenceEngine::new(model, scheduler, kv_cache, name, metrics));

    let loop_handle = {
        let e = engine.clone();
        tokio::spawn(async move {
            if let Err(err) = e.run_loop().await {
                tracing::error!("engine loop error: {err}");
            }
        })
    };

    Ok(EngineEntry { engine, loop_handle })
}
