// ModelRegistry: load models on demand, keep up to N in memory (LRU + time-based eviction).

mod config;
mod entry;
mod loader;

pub use config::{kv_type, RegistryConfig};
pub use entry::EngineEntry;

use dashmap::DashMap;
use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;

use crate::cli::list_models;
use crate::model_discovery::resolve_discovered;

use self::loader::load_model;

pub struct ModelRegistry {
    engines: DashMap<String, Arc<EngineEntry>>,
    lru: Mutex<lru::LruCache<String, ()>>,
    pub(crate) last_used: DashMap<String, Instant>,
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

    pub(crate) fn evict_lru_if_needed(&self) {
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

    pub(crate) fn evict_expired(&self) {
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
    /// 0. Direct file path (absolute or relative) that exists on disk
    /// 1. Alias lookup
    /// 2. Exact stem match (case-insensitive)
    /// 3. Stem starts with the name
    /// 4. Stem contains the name
    pub(crate) fn resolve_model_name(&self, name: &str) -> Result<(String, PathBuf)> {
        // Step 0: if `name` is already a path to an existing file, use it directly.
        // This handles FOX_MODEL_PATH pointing to a model in a subdirectory or
        // outside of models_dir entirely.
        let as_path = std::path::PathBuf::from(name);
        if as_path.is_file() {
            let stem = as_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(name)
                .to_string();
            return Ok((stem, as_path));
        }

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

        // 5. Fallback: check discovered models from well-known directories
        if !self.config.discovered_models.is_empty() {
            if let Some((disc_name, disc_path)) =
                resolve_discovered(resolved, &self.config.discovered_models)
            {
                let stem = disc_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or(&disc_name)
                    .to_string();
                tracing::info!(
                    name = %disc_name,
                    path = %disc_path.display(),
                    "resolved model from discovered sources"
                );
                return Ok((stem, disc_path));
            }
        }

        anyhow::bail!(
            "model '{}' not found in {} or discovered sources",
            name,
            self.config.models_dir.display()
        )
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
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            discovered_models: vec![],
        }
    }

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

    #[tokio::test]
    async fn test_lru_eviction_when_at_capacity() {
        let dir = tempfile::tempdir().unwrap();
        let registry = Arc::new(ModelRegistry::new(
            minimal_cfg(dir.path(), 1, 0),
            HashMap::new(),
        ));

        let entry_a = EngineEntry::for_test("model-a");
        let entry_b = EngineEntry::for_test("model-b");
        registry.preload_for_test("model-a", entry_a);

        assert_eq!(registry.loaded().len(), 1);

        registry.evict_lru_if_needed();
        registry.preload_for_test("model-b", entry_b);

        let loaded_names: Vec<String> = registry.loaded().into_iter().map(|(n, _)| n).collect();
        assert!(!loaded_names.contains(&"model-a".to_string()));
        assert!(loaded_names.contains(&"model-b".to_string()));
    }

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

        registry.evict_expired();

        assert_eq!(registry.loaded().len(), 1);
    }

    #[tokio::test]
    async fn test_evict_expired_noop_when_keep_alive_zero() {
        use std::time::{Duration, Instant};

        let dir = tempfile::tempdir().unwrap();
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

        assert_eq!(registry.loaded().len(), 1);
    }

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
