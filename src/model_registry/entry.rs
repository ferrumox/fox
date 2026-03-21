use std::sync::Arc;

use crate::engine::InferenceEngine;

pub struct EngineEntry {
    pub engine: Arc<InferenceEngine>,
    /// Aborted when this entry is dropped (LRU eviction or explicit unload).
    pub(super) loop_handle: tokio::task::JoinHandle<()>,
}

impl Drop for EngineEntry {
    fn drop(&mut self) {
        self.loop_handle.abort();
    }
}

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
