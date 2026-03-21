// ModelRegistry: load models on demand, keep up to N in memory (LRU + time-based eviction).

use dashmap::DashMap;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    /// Per-sequence context length. `None` = auto-detect from the model's trained context.
    pub max_context_len: Option<u32>,
    pub block_size: usize,
    pub gpu_memory_bytes: usize,
    pub gpu_memory_fraction: f32,
    pub metrics: Option<Arc<Metrics>>,
    /// Seconds since last use before a model is evicted. 0 = never evict by time.
    pub keep_alive_secs: u64,
    /// KV cache element type: 1=F16 (default), 8=Q8_0, 2=Q4_0
    pub type_kv: u32,
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
    last_used: DashMap<String, Instant>,
    config: RegistryConfig,
    aliases: HashMap<String, String>,
}

impl ModelRegistry {
    pub fn new(config: RegistryConfig, aliases: HashMap<String, String>) -> Self {
        let cap = NonZeroUsize::new(config.max_models.max(1))
            .expect("max(1) guarantees a non-zero value");
        Self {
            engines: DashMap::new(),
            lru: Mutex::new(lru::LruCache::new(cap)),
            last_used: DashMap::new(),
            config,
            aliases,
        }
    }

    /// Spawn a background task that evicts models idle longer than `keep_alive_secs`.
    /// Uses a weak reference so the task stops automatically when the registry is dropped.
    pub fn start_eviction_task(self: Arc<Self>) {
        if self.config.keep_alive_secs == 0 {
            return;
        }
        let weak = Arc::downgrade(&self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            interval.tick().await; // skip the immediate first tick
            loop {
                interval.tick().await;
                match weak.upgrade() {
                    Some(registry) => registry.evict_expired(),
                    None => break, // Registry dropped — stop task.
                }
            }
        });
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
            self.last_used.insert(stem, Instant::now());
            return Ok(entry.clone());
        }

        // Evict if we are at capacity before loading.
        self.evict_lru_if_needed();

        // Load the model (FFI is blocking, so we use spawn_blocking inside).
        let entry = Arc::new(load_model(&stem, &path, &self.config).await?);
        self.engines.insert(stem.clone(), entry.clone());
        self.last_used.insert(stem.clone(), Instant::now());
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
            self.last_used.remove(name);
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
                let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
                lru.pop_lru().map(|(k, _)| k)
            };
            match lru_key {
                Some(name) => {
                    self.engines.remove(&name);
                    self.last_used.remove(&name);
                    tracing::info!("evicted model '{}' from registry (LRU)", name);
                }
                None => break,
            }
        }
    }

    fn evict_expired(&self) {
        if self.config.keep_alive_secs == 0 {
            return;
        }
        let now = Instant::now();
        let keep_alive = Duration::from_secs(self.config.keep_alive_secs);
        let expired: Vec<String> = self
            .last_used
            .iter()
            .filter(|e| now.duration_since(*e.value()) >= keep_alive)
            .map(|e| e.key().clone())
            .collect();
        for name in expired {
            self.unload(&name);
            tracing::info!("evicted model '{}' (keep-alive expired)", name);
        }
    }

    /// Resolve a user-supplied name to `(stem, PathBuf)` using the following priority:
    /// 1. Alias lookup
    /// 2. Exact stem match (case-insensitive)
    /// 3. Stem starts with the name
    /// 4. Stem contains the name
    fn resolve_model_name(&self, name: &str) -> Result<(String, PathBuf)> {
        let resolved = self.aliases.get(name).map(String::as_str).unwrap_or(name);

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
    let type_kv = cfg.type_kv;

    tracing::info!("loading model '{}' from {:?}", name, path);

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

    Ok(EngineEntry {
        engine,
        loop_handle,
    })
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "test-helpers"))]
impl EngineEntry {
    /// Build a test `EngineEntry` backed by `StubModel` (no FFI).
    /// Must be called inside a Tokio runtime (i.e. inside `#[tokio::test]`).
    pub fn for_test(name: &str) -> Arc<Self> {
        use crate::engine::model::StubModel;
        use crate::kv_cache::KVCacheManager;
        use crate::scheduler::Scheduler;

        let model: Arc<dyn crate::engine::model::Model> = Arc::new(StubModel);
        let cfg = model.model_config();
        // Small KV cache: 4 MiB, fraction 0.9, block_size 16
        let kv = Arc::new(KVCacheManager::new(&cfg, 4 * 1024 * 1024, 0.9, 16, 1));
        let sched = Arc::new(Scheduler::new(kv.clone(), 4));
        let engine = Arc::new(crate::engine::InferenceEngine::new(
            model,
            sched,
            kv,
            name.to_string(),
            None,
        ));
        let loop_handle = {
            let e = engine.clone();
            tokio::spawn(async move {
                let _ = e.run_loop().await;
            })
        };
        Arc::new(Self {
            engine,
            loop_handle,
        })
    }
}

#[cfg(any(test, feature = "test-helpers"))]
impl ModelRegistry {
    /// Inject a pre-built engine entry without touching the filesystem (for tests).
    pub fn preload_for_test(&self, name: impl Into<String>, entry: Arc<EngineEntry>) {
        let name = name.into();
        self.engines.insert(name.clone(), entry);
        self.last_used
            .insert(name.clone(), std::time::Instant::now());
        if let Ok(mut lru) = self.lru.lock() {
            lru.put(name, ());
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn minimal_cfg(
        dir: &std::path::Path,
        max_models: usize,
        keep_alive_secs: u64,
    ) -> RegistryConfig {
        RegistryConfig {
            models_dir: dir.to_path_buf(),
            max_models,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs,
            type_kv: 1,
        }
    }

    // -- unload ---------------------------------------------------------------

    #[tokio::test]
    async fn test_unload_removes_loaded_model() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 4, 0),
            HashMap::new(),
        ));
        let entry = EngineEntry::for_test("model-a");
        registry.preload_for_test("model-a", entry);

        assert_eq!(registry.loaded().len(), 1);
        assert!(registry.unload("model-a"));
        assert_eq!(registry.loaded().len(), 0);
    }

    #[tokio::test]
    async fn test_unload_nonexistent_returns_false() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 4, 0),
            HashMap::new(),
        ));
        assert!(!registry.unload("does-not-exist"));
    }

    // -- evict_lru_if_needed --------------------------------------------------

    #[tokio::test]
    async fn test_lru_eviction_when_at_capacity() {
        let dir = tempfile::tempdir().unwrap();
        // max_models = 1 → loading a second model evicts the first
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 1, 0),
            HashMap::new(),
        ));

        let entry_a = EngineEntry::for_test("model-a");
        let entry_b = EngineEntry::for_test("model-b");
        registry.preload_for_test("model-a", entry_a);

        assert_eq!(registry.loaded().len(), 1);

        // Manually trigger LRU eviction then insert model-b
        registry.evict_lru_if_needed();
        registry.preload_for_test("model-b", entry_b);

        // After eviction, only model-b should remain
        let loaded_names: Vec<String> = registry.loaded().into_iter().map(|(n, _)| n).collect();
        assert!(!loaded_names.contains(&"model-a".to_string()));
        assert!(loaded_names.contains(&"model-b".to_string()));
    }

    // -- evict_expired --------------------------------------------------------

    #[tokio::test]
    async fn test_evict_expired_removes_stale_model() {
        use std::time::{Duration, Instant};

        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 4, 60),
            HashMap::new(),
        ));

        let entry = EngineEntry::for_test("old-model");
        registry.preload_for_test("old-model", entry);

        // Back-date last_used by more than keep_alive_secs
        registry.last_used.insert(
            "old-model".to_string(),
            Instant::now() - Duration::from_secs(120),
        );

        registry.evict_expired();

        assert_eq!(registry.loaded().len(), 0);
    }

    #[tokio::test]
    async fn test_evict_expired_keeps_fresh_model() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 4, 60),
            HashMap::new(),
        ));

        let entry = EngineEntry::for_test("fresh");
        registry.preload_for_test("fresh", entry);
        // last_used is set to now by preload_for_test — still fresh

        registry.evict_expired();

        assert_eq!(registry.loaded().len(), 1);
    }

    #[tokio::test]
    async fn test_evict_expired_noop_when_keep_alive_zero() {
        use std::time::{Duration, Instant};

        let dir = tempfile::tempdir().unwrap();
        // keep_alive_secs = 0 → never evict by time
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 4, 0),
            HashMap::new(),
        ));

        let entry = EngineEntry::for_test("forever");
        registry.preload_for_test("forever", entry);
        registry.last_used.insert(
            "forever".to_string(),
            Instant::now() - Duration::from_secs(9999),
        );

        registry.evict_expired();

        // Model should still be there — keep_alive_secs = 0 disables time eviction
        assert_eq!(registry.loaded().len(), 1);
    }

    // -- resolve_model_name ---------------------------------------------------

    #[test]
    fn test_resolve_exact_match() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Llama-3.2-3B.gguf"), b"").unwrap();

        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), HashMap::new());
        let (stem, _path) = registry.resolve_model_name("Llama-3.2-3B").unwrap();
        assert_eq!(stem, "Llama-3.2-3B");
    }

    #[test]
    fn test_resolve_case_insensitive() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Llama-3.2-3B.gguf"), b"").unwrap();

        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), HashMap::new());
        let (stem, _) = registry.resolve_model_name("llama-3.2-3b").unwrap();
        assert_eq!(stem, "Llama-3.2-3B");
    }

    #[test]
    fn test_resolve_starts_with() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Llama-3.2-3B-Instruct.gguf"), b"").unwrap();

        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), HashMap::new());
        let (stem, _) = registry.resolve_model_name("llama-3.2").unwrap();
        assert_eq!(stem, "Llama-3.2-3B-Instruct");
    }

    #[test]
    fn test_resolve_contains() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"), b"").unwrap();

        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), HashMap::new());
        let (stem, _) = registry.resolve_model_name("Q4_K_M").unwrap();
        assert_eq!(stem, "Llama-3.2-3B-Instruct-Q4_K_M");
    }

    #[test]
    fn test_resolve_alias() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("Llama-3.2-3B.gguf"), b"").unwrap();

        let mut aliases = HashMap::new();
        aliases.insert("llama3".to_string(), "Llama-3.2-3B".to_string());
        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), aliases);
        let (stem, _) = registry.resolve_model_name("llama3").unwrap();
        assert_eq!(stem, "Llama-3.2-3B");
    }

    #[test]
    fn test_resolve_not_found_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 0), HashMap::new());
        assert!(registry.resolve_model_name("doesnt-exist").is_err());
    }
}
