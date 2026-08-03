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
    /// Abort the engine loop so no queued/running request is ever processed —
    /// gives integration tests a deterministic "stuck" entry (no race with the stub
    /// instantly finishing a request) to test queue-full/busy behavior against.
    pub fn abort_loop_for_test(&self) {
        self.loop_handle.abort();
    }

    /// Build a test `EngineEntry` backed by `StubModel` (no FFI).
    /// Must be called inside a Tokio runtime (i.e. inside `#[tokio::test]`).
    pub fn for_test(name: &str) -> Arc<Self> {
        use crate::engine::model::StubModel;
        Self::for_test_with_model(name, Arc::new(StubModel), None, 0)
    }

    /// Like `for_test`, with a capped scheduler queue (`max_queue_depth`) — for testing
    /// backpressure/429 behavior. 0 = unbounded (same as `for_test`).
    pub fn for_test_with_queue_depth(name: &str, max_queue_depth: usize) -> Arc<Self> {
        use crate::engine::model::StubModel;
        Self::for_test_with_model(name, Arc::new(StubModel), None, max_queue_depth)
    }

    /// Build a test `EngineEntry` with speculative decoding enabled (StubModel emits two
    /// tokens per speculative step, exercising the engine's multi-token commit path).
    pub fn for_test_speculative(name: &str) -> Arc<Self> {
        use crate::engine::model::StubModel;
        Self::for_test_with_model(
            name,
            Arc::new(StubModel),
            Some(crate::engine::SpeculativeConfig::Ngram {
                ngram: 2,
                draft_len: 4,
            }),
            0,
        )
    }

    /// Build a test `EngineEntry` backed by `ThinkingStubModel` (no FFI).
    /// The engine reports `supports_thinking() = true` and emits the sequence:
    /// "thought" → "</think>" → "answer" → EOS.
    pub fn for_test_thinking(name: &str) -> Arc<Self> {
        use crate::engine::model::ThinkingStubModel;
        Self::for_test_with_model(name, Arc::new(ThinkingStubModel::new()), None, 0)
    }

    fn for_test_with_model(
        name: &str,
        model: Arc<dyn crate::engine::model::Model>,
        speculative: Option<crate::engine::SpeculativeConfig>,
        max_queue_depth: usize,
    ) -> Arc<Self> {
        use crate::kv_cache::KVCacheManager;
        use crate::scheduler::Scheduler;

        let cfg = model.model_config();
        // Small KV cache: 4 MiB, fraction 0.9, block_size 16
        let kv = Arc::new(KVCacheManager::new(&cfg, 4 * 1024 * 1024, 0.9, 16, 1, 1));
        let sched = Arc::new(Scheduler::with_max_queue_depth(
            kv.clone(),
            4,
            max_queue_depth,
        ));
        let engine = Arc::new(crate::engine::InferenceEngine::new(
            model,
            sched,
            kv,
            name.to_string(),
            None,
            crate::engine::EngineOptions {
                speculative,
                ..Default::default()
            },
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
