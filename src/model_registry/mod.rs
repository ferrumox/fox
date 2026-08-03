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

use self::loader::load_model;

pub struct ModelRegistry {
    engines: DashMap<String, Arc<EngineEntry>>,
    lru: Mutex<lru::LruCache<String, ()>>,
    pub(crate) last_used: DashMap<String, Instant>,
    /// Per-model keep-alive override set by a request's `keep_alive` field. Absent
    /// means "use the server-wide `--keep-alive-secs`". `None` inside the map means
    /// "never evict on a timer" (Ollama's negative values).
    pub(crate) keep_alive_override: DashMap<String, Option<Duration>>,
    /// Runtime scale overrides for `--lora-modules` adapters, set via
    /// `POST /lora-adapters`. Absent means "use the scale it was configured with".
    ///
    /// Lives here rather than on the model because the model's adapter map is
    /// immutable after load, and because only the *default* needs to change: the
    /// per-request `LoraSelection` already carries its own scale down to
    /// `llama_set_adapters_lora`, so overriding what gets copied into it is the whole
    /// mechanism.
    pub(crate) lora_scale_override: DashMap<String, f32>,
    /// Where each model fox has loaded actually came from, keyed by stem.
    ///
    /// `resolve_model_name` otherwise only knows about files inside `models_dir`, so a
    /// model pre-loaded with `--model-path` pointing anywhere else resolved to a 404
    /// on the request path even while it was resident and serving — its own `/health`
    /// reported it by a name nothing would accept.
    pub(crate) known_paths: DashMap<String, PathBuf>,
    config: RegistryConfig,
    aliases: HashMap<String, String>,
    /// Serializes model loading so two concurrent requests for the same cold model
    /// don't both load it (double VRAM, and the second insert would drop the first
    /// entry, aborting its in-flight generations).
    load_lock: tokio::sync::Mutex<()>,
}

impl ModelRegistry {
    pub fn new(config: RegistryConfig, aliases: HashMap<String, String>) -> Self {
        let cap = NonZeroUsize::new(config.max_models.max(1))
            .expect("max(1) guarantees a non-zero value");
        Self {
            engines: DashMap::new(),
            lru: Mutex::new(lru::LruCache::new(cap)),
            last_used: DashMap::new(),
            keep_alive_override: DashMap::new(),
            lora_scale_override: DashMap::new(),
            known_paths: DashMap::new(),
            config,
            aliases,
            load_lock: tokio::sync::Mutex::new(()),
        }
    }

    /// Spawn a background task that evicts models idle longer than `keep_alive_secs`.
    /// Uses a weak reference so the task stops automatically when the registry is dropped.
    pub fn start_eviction_task(self: Arc<Self>) {
        // Runs even when `keep_alive_secs == 0` (server default: never evict on a
        // timer), because a request's own `keep_alive` can now set a TTL for one
        // model. `evict_expired` decides per model; bailing out here would make the
        // per-request field silently inert again, which is the bug being fixed.
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

        // Single-flight: serialize loads, then re-check — another request may have
        // loaded this model while we waited for the lock.
        let _load_guard = self.load_lock.lock().await;
        if let Some(entry) = self.engines.get(&stem) {
            self.last_used.insert(stem, Instant::now());
            return Ok(entry.clone());
        }

        // Evict if we are at capacity before loading.
        self.evict_lru_if_needed();

        // Resolve the draft model's (stem, path) here — where `&self` is available —
        // rather than threading `&ModelRegistry` into the free `load_model` function.
        // Only takes effect alongside `--speculative true`; a draft model configured
        // without it would silently load an unused second model, so warn and skip.
        let draft = match &self.config.draft_model {
            Some(name) if self.config.speculative => Some(self.resolve_model_name(name)?),
            Some(_) => {
                tracing::warn!(
                    "--draft-model is set but --speculative is false — ignoring, no draft model will be loaded"
                );
                None
            }
            None => None,
        };

        // Resolve the paired mmproj (vision projector), if configured — same
        // resolver as the main model and draft, just against `models_dir`.
        let mmproj = match &self.config.mmproj {
            Some(name) => Some(self.resolve_model_name(name)?.1),
            None => None,
        };

        // Resolve each configured LoRA adapter's path the same way (direct file
        // path or a `models_dir` match) — the operator-chosen adapter `name` and
        // `scale` pass through unchanged.
        let mut lora_modules = Vec::with_capacity(self.config.lora_modules.len());
        for (name, path_or_name, scale) in &self.config.lora_modules {
            let resolved_path = self.resolve_model_name(&path_or_name.to_string_lossy())?.1;
            lora_modules.push((name.clone(), resolved_path, *scale));
        }

        // Load the model (FFI is blocking, so we use spawn_blocking inside).
        let entry =
            Arc::new(load_model(&stem, &path, &self.config, draft, mmproj, lora_modules).await?);
        self.engines.insert(stem.clone(), entry.clone());
        // Remember where it came from, so it stays reachable by name even when its
        // file lives outside `models_dir`. Deliberately NOT removed on unload: the
        // path is still the right answer for that stem, and forgetting it would make
        // an evicted `--model-path` model unreachable again — the exact bug this
        // fixes, resurfacing after the first keep-alive expiry.
        self.known_paths.insert(stem.clone(), path.clone());
        self.last_used.insert(stem.clone(), Instant::now());
        if let Ok(mut lru) = self.lru.lock() {
            lru.put(stem, ());
        }

        Ok(entry)
    }

    /// Whether `name` matches a configured LoRA adapter's name (`--lora-modules`),
    /// as opposed to a real model/alias. Used by `load_model_or_respond` to give a
    /// clean 404 only when `name` is neither.
    pub(crate) fn is_lora_alias(&self, name: &str) -> bool {
        self.config.lora_modules.iter().any(|(n, _, _)| n == name)
    }

    /// Resolve a client-supplied `model` name for a request. If `name` matches a
    /// configured LoRA adapter's name (`--lora-modules`), loads/reuses the
    /// *primary* model's engine (same cache as `get_or_load`, same single-flight
    /// and LRU behavior — this is a resolution-layer concern, not a new caching
    /// concept: `EngineEntry` itself carries no name/alias metadata, several
    /// names can share one already-loaded engine) and returns the adapter
    /// selection alongside it. Otherwise behaves exactly like `get_or_load` with
    /// no selection. See `docs/design/lora-support.md`.
    pub async fn resolve_for_request(
        &self,
        name: &str,
    ) -> Result<(Arc<EngineEntry>, Option<crate::scheduler::LoraSelection>)> {
        if let Some((_, _, configured)) =
            self.config.lora_modules.iter().find(|(n, _, _)| n == name)
        {
            let scale = &self
                .lora_scale_override
                .get(name)
                .map(|v| *v.value())
                .unwrap_or(*configured);
            let primary = self.config.primary_model.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "'{name}' matches a configured LoRA adapter, but no primary model is \
                     configured to apply it to (pass --model-path, or put a model in models_dir)"
                )
            })?;
            let entry = self.get_or_load(primary).await?;
            return Ok((
                entry,
                Some(crate::scheduler::LoraSelection {
                    name: name.to_string(),
                    scale: *scale,
                }),
            ));
        }
        Ok((self.get_or_load(name).await?, None))
    }

    /// Returns all currently-loaded (name, entry) pairs.
    /// Adapters loaded via `--lora-modules`, with their **current** scales — the
    /// runtime override if one was set, else the configured default.
    pub fn lora_adapters(&self) -> Vec<(String, std::path::PathBuf, f32)> {
        self.config
            .lora_modules
            .iter()
            .map(|(name, path, configured)| {
                let scale = self
                    .lora_scale_override
                    .get(name)
                    .map(|v| *v.value())
                    .unwrap_or(*configured);
                (name.clone(), path.clone(), scale)
            })
            .collect()
    }

    /// Set an adapter's scale for subsequent requests. `Err` naming the adapter when
    /// it was never loaded — silently accepting an unknown name would let a caller
    /// believe a scale change took effect when nothing exists to apply it to.
    pub fn set_lora_scale(&self, name: &str, scale: f32) -> Result<()> {
        if !self.config.lora_modules.iter().any(|(n, _, _)| n == name) {
            anyhow::bail!("LoRA adapter '{name}' is not loaded (see --lora-modules)");
        }
        self.lora_scale_override.insert(name.to_string(), scale);
        Ok(())
    }

    /// Read-only view of the server's registry configuration, for `/props`.
    pub fn config(&self) -> &RegistryConfig {
        &self.config
    }

    pub fn loaded(&self) -> Vec<(String, Arc<EngineEntry>)> {
        self.engines
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }

    /// Explicitly unload a model by stem name. Returns `true` if it was loaded.
    pub fn unload(&self, name: &str) -> bool {
        let evicted = self.engines.remove(name);
        if let Some((_, entry)) = &evicted {
            self.last_used.remove(name);
            // Drop the per-request TTL too, or a reloaded model would silently
            // inherit the keep_alive of whatever request last touched the old one.
            self.keep_alive_override.remove(name);
            // The host-RAM prompt cache holds blobs that encode THIS model's cell
            // layout; restoring one into a different model would corrupt it. The
            // cache dies with the scheduler anyway, but anything still holding an
            // `Arc` to the engine would otherwise keep stale blobs alive.
            entry.engine.clear_prompt_cache();
            if let Ok(mut lru) = self.lru.lock() {
                lru.pop(name);
            }
        }
        evicted.is_some()
    }

    pub(crate) fn evict_lru_if_needed(&self) {
        // Never evict a model with requests in flight or queued — dropping its entry
        // aborts the engine loop and kills those generations. A busy candidate is
        // re-inserted as MRU; if everyone is busy we temporarily exceed max_models.
        let mut attempts = self.engines.len();
        while self.engines.len() >= self.config.max_models && attempts > 0 {
            attempts -= 1;
            let lru_key = {
                let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
                lru.pop_lru().map(|(k, _)| k)
            };
            let Some(name) = lru_key else { break };
            if self.is_busy(&name) {
                let mut lru = self.lru.lock().unwrap_or_else(|e| e.into_inner());
                lru.put(name, ());
                continue;
            }
            self.engines.remove(&name);
            self.last_used.remove(&name);
            tracing::info!("evicted model '{}' from registry (LRU)", name);
        }
    }

    /// Whether a model currently has requests in flight or waiting.
    fn is_busy(&self, name: &str) -> bool {
        self.engines
            .get(name)
            .map(|e| e.engine.active_requests() > 0 || e.engine.queue_depth() > 0)
            .unwrap_or(false)
    }

    /// Record a per-request `keep_alive` for `model`, overriding the server default
    /// until another request changes it. `None` pins the model against timed eviction.
    ///
    /// This is the whole point of honouring the field: the value was previously parsed
    /// and thrown away, so a client asking to keep a model warm — or to drop it
    /// promptly — had no way to tell that nothing happened.
    pub(crate) fn set_keep_alive(&self, model: &str, ttl: Option<Duration>) {
        self.keep_alive_override.insert(model.to_string(), ttl);
    }

    /// Effective idle TTL for `model`: its override if one was set, else the
    /// server-wide default. `None` = never evict on a timer.
    fn effective_keep_alive(&self, model: &str) -> Option<Duration> {
        match self.keep_alive_override.get(model) {
            Some(o) => *o.value(),
            None => (self.config.keep_alive_secs > 0)
                .then(|| Duration::from_secs(self.config.keep_alive_secs)),
        }
    }

    pub(crate) fn evict_expired(&self) {
        let now = Instant::now();
        let expired: Vec<String> = self
            .last_used
            .iter()
            .filter(|e| {
                self.effective_keep_alive(e.key())
                    .is_some_and(|ttl| now.duration_since(*e.value()) >= ttl)
            })
            .map(|e| e.key().clone())
            .collect();
        for name in expired {
            // last_used marks request START, so a long generation looks idle — never
            // evict a busy model (it would be killed mid-generation).
            if self.is_busy(&name) {
                continue;
            }
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

        // Step 1.5: a model fox has already loaded is servable under its stem no
        // matter where its file lives. Checked before the directory scan, which only
        // ever sees `models_dir` — that gap is what made a `--model-path` model
        // outside it unreachable by name.
        if let Some(entry) = self
            .known_paths
            .iter()
            .find(|e| e.key().eq_ignore_ascii_case(resolved))
        {
            return Ok((entry.key().clone(), entry.value().clone()));
        }

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

#[cfg(any(test, feature = "test-helpers"))]
impl ModelRegistry {
    /// Register where a model came from, so `resolve_model_name` can find it by stem
    /// even though it lives outside `models_dir`. Used by the startup pre-load path
    /// and by tests.
    pub fn remember_path(&self, stem: impl Into<String>, path: PathBuf) {
        self.known_paths.insert(stem.into(), path);
    }

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

    /// A model outside `models_dir` must still resolve by its stem once fox knows it.
    ///
    /// Regression: `--model-path` pointing anywhere else produced a server whose own
    /// `/health` advertised a name that every request path then 404'd on.
    #[test]
    fn a_model_outside_models_dir_resolves_by_stem() {
        let dir = tempfile::tempdir().unwrap();
        let elsewhere = tempfile::tempdir().unwrap();
        let model = elsewhere.path().join("Far-Away-Model.gguf");
        std::fs::write(&model, b"").unwrap();

        let reg = ModelRegistry::new(minimal_cfg(dir.path(), 1, 0), HashMap::new());

        // Before fox knows about it, only the literal path works.
        assert!(reg.resolve_model_name("Far-Away-Model").is_err());
        assert!(reg.resolve_model_name(model.to_str().unwrap()).is_ok());

        reg.remember_path("Far-Away-Model", model.clone());
        let (stem, path) = reg
            .resolve_model_name("Far-Away-Model")
            .expect("a loaded model must be reachable by name");
        assert_eq!(stem, "Far-Away-Model");
        assert_eq!(path, model);
        // Case-insensitive, like the models_dir lookup it sits beside.
        assert!(reg.resolve_model_name("far-away-model").is_ok());
    }

    fn minimal_cfg(
        dir: &std::path::Path,
        max_models: usize,
        keep_alive_secs: u64,
    ) -> RegistryConfig {
        RegistryConfig {
            models_dir: dir.to_path_buf(),
            max_models,
            max_batch_size: 4,
            max_queue_depth: 0,
            max_prefill_chunk: 0,
            rs_rollback: 0,
            context_shift: false,
            context_keep: 0,
            reranking: false,
            cache_ram_bytes: 0,
            kv_reuse: true,
            slot_prompt_similarity: crate::scheduler::DEFAULT_SLOT_PROMPT_SIMILARITY,
            speculative: false,
            spec_ngram: 2,
            spec_draft_len: 4,
            draft_model: None,
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
            mmproj: None,
            lora_modules: Vec::new(),
            primary_model: None,
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
    #[tokio::test]
    async fn eviction_skips_models_with_requests_in_flight() {
        use crate::scheduler::{InferenceRequest, SamplingParams};

        let dir = tempfile::tempdir().unwrap();
        // keep_alive = 1s so entries expire immediately for the test.
        let registry = ModelRegistry::new(minimal_cfg(dir.path(), 4, 1), HashMap::new());

        let entry = EngineEntry::for_test("busy");
        // Abort the engine loop FIRST so a submitted request stays queued forever —
        // a deterministic "busy" state (no race with the stub finishing it).
        entry.loop_handle.abort();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let _ = entry.engine.submit_request(InferenceRequest::new(
            1,
            vec![1, 2, 3],
            4,
            SamplingParams::default(),
            tx,
        ));
        assert!(entry.engine.queue_depth() > 0, "request must be queued");

        registry.preload_for_test("busy", entry.clone());
        registry
            .last_used
            .insert("busy".into(), Instant::now() - Duration::from_secs(3600));

        // Keep-alive sweep must SKIP the busy model instead of killing its work.
        registry.evict_expired();
        assert!(
            registry.engines.contains_key("busy"),
            "a model with queued requests must never be keep-alive evicted"
        );

        // Same for the LRU path: at capacity 1, loading pressure must not evict it.
        let registry2 = ModelRegistry::new(minimal_cfg(dir.path(), 1, 0), HashMap::new());
        registry2.preload_for_test("busy", entry);
        registry2.evict_lru_if_needed();
        assert!(
            registry2.engines.contains_key("busy"),
            "a busy model must never be LRU evicted"
        );
    }

    #[test]
    fn test_is_lora_alias() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = minimal_cfg(dir.path(), 4, 0);
        cfg.lora_modules = vec![("finetune".to_string(), PathBuf::from("adapter.gguf"), 1.0)];
        let registry = ModelRegistry::new(cfg, HashMap::new());
        assert!(registry.is_lora_alias("finetune"));
        assert!(!registry.is_lora_alias("finetune-typo"));
        assert!(!registry.is_lora_alias("base"));
    }

    #[tokio::test]
    async fn test_resolve_for_request_lora_alias_resolves_to_primary_model() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("base.gguf"), b"").unwrap();
        let mut cfg = minimal_cfg(dir.path(), 4, 0);
        cfg.primary_model = Some("base".to_string());
        cfg.lora_modules = vec![("finetune".to_string(), PathBuf::from("adapter.gguf"), 0.7)];
        let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        registry.preload_for_test("base", EngineEntry::for_test("base"));

        let (_entry, lora) = registry.resolve_for_request("finetune").await.unwrap();
        let selection = lora.expect("expected a LoRA selection for an adapter alias");
        assert_eq!(selection.name, "finetune");
        assert_eq!(selection.scale, 0.7);
    }

    #[tokio::test]
    async fn test_resolve_for_request_plain_model_name_has_no_lora_selection() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("base.gguf"), b"").unwrap();
        let mut cfg = minimal_cfg(dir.path(), 4, 0);
        cfg.lora_modules = vec![("finetune".to_string(), PathBuf::from("adapter.gguf"), 0.7)];
        let registry = Arc::new(ModelRegistry::new(cfg, HashMap::new()));
        registry.preload_for_test("base", EngineEntry::for_test("base"));

        let (_entry, lora) = registry.resolve_for_request("base").await.unwrap();
        assert!(
            lora.is_none(),
            "a plain model name must not pick up a LoRA selection"
        );
    }

    #[tokio::test]
    async fn test_resolve_for_request_lora_alias_without_primary_model_errors() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = minimal_cfg(dir.path(), 4, 0);
        // primary_model left None on purpose.
        cfg.lora_modules = vec![("finetune".to_string(), PathBuf::from("adapter.gguf"), 1.0)];
        let registry = ModelRegistry::new(cfg, HashMap::new());

        let result = registry.resolve_for_request("finetune").await;
        let Err(err) = result else {
            panic!("a LoRA alias with no primary model configured must error, not panic");
        };
        assert!(err.to_string().contains("no primary model"));
    }
}
