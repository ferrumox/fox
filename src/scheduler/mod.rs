// Continuous batching scheduler with prefix caching.
// Flow: Waiting -> Prefilling -> Decoding -> Finished
// Blocks are fully reserved at admission, so admission never preempts: a request
// that doesn't fit waits in the queue (FIFO) until running requests finish.

mod batch;
mod prompt_cache;
mod schedule;
mod slots;

#[allow(unused_imports)]
pub use batch::{
    InferenceRequest, LoraSelection, RequestState, SamplingParams, ScheduledBatch, StopReason,
    Token, TokenLogprob, TopLogprob,
};

use std::collections::VecDeque;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use tracing::info;

use crate::kv_cache::KVCacheManager;

use prompt_cache::PromptCache;
pub use slots::SlotSnapshot;
use slots::SlotTable;

/// Default LCP-similarity floor below which an idle slot's KV is not worth
/// inheriting. Mirrors llama-server's `-sps/--slot-prompt-similarity` default.
pub const DEFAULT_SLOT_PROMPT_SIMILARITY: f32 = 0.1;

/// Scheduler managing waiting queue and running batch.
pub struct Scheduler {
    pub(super) waiting_queue: std::sync::Mutex<VecDeque<InferenceRequest>>,
    pub(super) running_batch: std::sync::Mutex<Vec<InferenceRequest>>,
    pub(super) kv_cache: Arc<KVCacheManager>,
    /// Notified when a new request is submitted, waking the engine loop.
    work_notify: tokio::sync::Notify,
    /// The llama.cpp sequence slots, one per concurrent request, each remembering
    /// the tokens resident in its KV. This *is* fox's prefix cache: it replaces the
    /// old `seq_id_pool` + block-hash `LruCache` pair (see `slots.rs` for why).
    ///
    /// Slot index == `seq_id`, matching `llama-server`'s `slot.id = i`. IDs are no
    /// longer required to be handed out densely: `split_equal`'s
    /// consecutive-ascending constraint only applies to a non-unified cache
    /// (`n_stream > 1`), and fox sets `kv_unified = true` since `1c36faf`, so
    /// `llama-kv-cache.cpp:725` picks `split_simple`, which has no ordering
    /// requirement. See `docs/design/llama-server-gap-analysis.md` §0.1. (The batch
    /// is still *emitted* in ascending seq_id order — see `do_decode_batch`.)
    ///
    /// Lock ordering: always acquire `running_batch` → `waiting_queue` → `slots`.
    pub(crate) slots: std::sync::Mutex<SlotTable>,
    /// Minimum fraction of an incoming prompt that must already be resident in an
    /// idle slot before that slot's KV is inherited instead of starting fresh.
    pub(super) slot_prompt_similarity: f32,
    /// Host-RAM store of serialised sequence states (`--cache-ram`). Complements the
    /// slot table: slots keep a conversation warm by holding GPU blocks, this keeps one
    /// warm without holding any. Empty and inert when the budget is 0 (the default).
    ///
    /// Lock ordering: acquired after `slots`, never while holding the model lock.
    pub(crate) prompt_cache: std::sync::Mutex<PromptCache>,
    /// False when the loaded model's KV cache cannot donate cells — see
    /// `set_prefix_reuse`.
    prefix_reuse: std::sync::atomic::AtomicBool,
    /// False only when a model cannot even reuse the KV a sequence already holds.
    /// Separate from `prefix_reuse` on purpose: see `set_slot_reuse`.
    slot_reuse: std::sync::atomic::AtomicBool,
    /// Master switch for KV reuse (`--kv-reuse`). When false, finished sequences are
    /// always cleared and every prompt is prefilled from token 0 — the pre-0.19
    /// behaviour, kept as an escape hatch and as the A/B baseline arm.
    pub(super) kv_reuse: bool,
    /// Lifetime hit counter (for metrics / logging).
    pub prefix_hits: AtomicU64,
    /// Lifetime miss counter (for metrics / logging).
    pub prefix_misses: AtomicU64,
    /// Maximum requests allowed to wait in `waiting_queue` before `submit()` rejects
    /// new ones with `SubmitError::QueueFull`. 0 = unbounded.
    max_queue_depth: usize,
}

/// Why `Scheduler::submit()` refused a request. Both variants are synchronous,
/// pre-queue rejections — the request never touches `waiting_queue` or gets a
/// response channel that could otherwise hang or silently close.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitError {
    /// `waiting_queue` is already at `max_queue_depth`.
    QueueFull { depth: usize, max: usize },
    /// The request needs more blocks than the KV pool will ever have, even empty.
    TooLarge {
        needed_blocks: usize,
        total_blocks: usize,
    },
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::QueueFull { depth, max } => {
                write!(f, "queue full ({depth}/{max} requests waiting)")
            }
            SubmitError::TooLarge {
                needed_blocks,
                total_blocks,
            } => write!(
                f,
                "request needs {needed_blocks} KV blocks but the pool only has {total_blocks}"
            ),
        }
    }
}

impl Scheduler {
    /// Whether slot-affinity prefix reuse may skip prefill.
    pub fn prefix_reuse_enabled(&self) -> bool {
        self.prefix_reuse.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn slot-affinity prefix reuse off for a model that cannot donate KV cells.
    ///
    /// Reuse is decided in the scheduler but *performed* in llama.cpp, and the two used
    /// to disagree: the engine knew `supports_seq_copy()` was false while the scheduler
    /// went on skipping prefill, leaving `llama_memory_seq_cp` to abort the process at
    /// `llama-kv-cache.cpp:518`. Observed under `FOX_KV_UNIFIED=0`; the same disagreement
    /// applies to recurrent/hybrid models, where the pre-existing guard in
    /// `batch.rs` tested `llama_memory_can_shift()` — which returns true for them.
    pub fn set_prefix_reuse(&self, enabled: bool) {
        self.prefix_reuse
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    /// Whether a request may inherit the KV its chosen slot already holds.
    pub fn slot_reuse_enabled(&self) -> bool {
        self.slot_reuse.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Turn off reuse of a slot's *own* resident KV.
    ///
    /// This is the weaker of the two capabilities and almost every model has it —
    /// nothing is copied, the request simply keeps what its sequence already holds and
    /// prefills the rest. It is set apart from [`Self::set_prefix_reuse`] because
    /// gating both on "can this model donate cells" silently disabled prompt reuse for
    /// every hybrid architecture (Qwen3.5, Qwen3-Next, Falcon-H1, Jamba), which
    /// `llama-server` performs on the same models without difficulty.
    pub fn set_slot_reuse(&self, enabled: bool) {
        self.slot_reuse
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn new(kv_cache: Arc<KVCacheManager>, max_batch_size: usize) -> Self {
        Self::with_max_queue_depth(kv_cache, max_batch_size, 0)
    }

    /// Like `new`, with an explicit queue-depth cap (0 = unbounded).
    pub fn with_max_queue_depth(
        kv_cache: Arc<KVCacheManager>,
        max_batch_size: usize,
        max_queue_depth: usize,
    ) -> Self {
        Self {
            waiting_queue: std::sync::Mutex::new(VecDeque::new()),
            running_batch: std::sync::Mutex::new(Vec::new()),
            kv_cache,
            work_notify: tokio::sync::Notify::new(),
            slots: std::sync::Mutex::new(SlotTable::new(max_batch_size)),
            // Default on; Engine::new turns it off for models whose KV cache cannot
            // donate cells. Kept here rather than passed to `new()` because both
            // constructors are called before the model is known.
            prefix_reuse: std::sync::atomic::AtomicBool::new(true),
            slot_reuse: std::sync::atomic::AtomicBool::new(true),
            prompt_cache: std::sync::Mutex::new(PromptCache::new(0)),
            slot_prompt_similarity: DEFAULT_SLOT_PROMPT_SIMILARITY,
            kv_reuse: true,
            prefix_hits: AtomicU64::new(0),
            prefix_misses: AtomicU64::new(0),
            max_queue_depth,
        }
    }

    /// Set the host-RAM prompt-cache budget in bytes (`--cache-ram`, 0 = disabled).
    pub fn with_prompt_cache(self, limit_bytes: usize) -> Self {
        if let Ok(mut c) = self.prompt_cache.lock() {
            *c = PromptCache::new(limit_bytes);
        }
        self
    }

    /// Override the KV-reuse policy (`--kv-reuse` / `--slot-prompt-similarity`).
    /// Chained onto a constructor so the many test and bench call sites that only
    /// care about the defaults stay untouched.
    pub fn with_kv_reuse(mut self, kv_reuse: bool, slot_prompt_similarity: f32) -> Self {
        self.kv_reuse = kv_reuse;
        self.slot_prompt_similarity = slot_prompt_similarity.clamp(0.0, 1.0);
        self
    }

    /// Submit a request to the waiting queue. Rejects synchronously (before the
    /// request ever enters the queue or gets a response channel) when the queue is
    /// full or the request could never fit in the KV pool even when empty — both
    /// checks are statically knowable at submission time, so there is no need to
    /// wait for a scheduler turn to reject them.
    pub fn submit(&self, req: InferenceRequest) -> Result<(), SubmitError> {
        let needed = self.blocks_needed(&req);
        let total = self.kv_cache.total_blocks();
        if needed > total {
            return Err(SubmitError::TooLarge {
                needed_blocks: needed,
                total_blocks: total,
            });
        }

        let mut q = match self.waiting_queue.lock() {
            Ok(g) => g,
            Err(e) => {
                tracing::error!("waiting_queue lock poisoned on submit: {}", e);
                e.into_inner()
            }
        };
        if self.max_queue_depth > 0 && q.len() >= self.max_queue_depth {
            return Err(SubmitError::QueueFull {
                depth: q.len(),
                max: self.max_queue_depth,
            });
        }
        info!(request_id = req.id, "request admitted to waiting queue");
        q.push_back(req);
        drop(q);
        self.work_notify.notify_one();
        Ok(())
    }

    /// Store a serialised sequence state in the host-RAM cache. Called by the engine
    /// after it has serialised a sequence the scheduler asked it to save.
    pub fn store_prompt_state(&self, tokens: Vec<i32>, data: Vec<u8>) {
        if let Ok(mut c) = self.prompt_cache.lock() {
            c.store(tokens, data);
        }
    }

    /// Drop every cached sequence state (model unload).
    pub fn clear_prompt_cache(&self) {
        if let Ok(mut c) = self.prompt_cache.lock() {
            c.clear();
        }
    }

    /// `(entries, bytes, hits, misses)` for metrics and `/props`.
    pub fn prompt_cache_stats(&self) -> (usize, usize, u64, u64) {
        self.prompt_cache
            .lock()
            .map(|c| {
                let (h, m) = c.stats();
                (c.len(), c.bytes(), h, m)
            })
            .unwrap_or((0, 0, 0, 0))
    }

    /// Per-slot state for `GET /slots`. Empty on a poisoned lock rather than
    /// panicking — introspection must never take the server down.
    pub fn slots_snapshot(&self) -> Vec<SlotSnapshot> {
        self.slots.lock().map(|s| s.snapshot()).unwrap_or_default()
    }

    /// Wait until at least one request is available to schedule.
    pub async fn wait_for_work(&self) {
        self.work_notify.notified().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::ModelConfig;
    use crate::kv_cache::KVCacheManager;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_scheduler_submit_and_schedule() {
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            n_embd: 4096,
            vocab_size: 32000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 1_000_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let req = InferenceRequest::new(1, vec![1, 2, 3], 10, SamplingParams::default(), tx);
        sched.submit(req).unwrap();

        assert_eq!(sched.queue_depth(), 1);
        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![1]);
        assert_eq!(sched.queue_depth(), 0);
    }

    fn test_kv(block_size: usize) -> Arc<KVCacheManager> {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        Arc::new(KVCacheManager::new(
            &config,
            500_000_000,
            0.5,
            block_size,
            1,
            1,
        ))
    }

    /// Run a request to completion and park its sequence, the way the engine does:
    /// admit → generate `gen` tokens → finish → park.
    fn run_and_park(sched: &Scheduler, id: u64, prompt: Vec<i32>, gen: &[i32]) {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                id,
                prompt,
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        for (i, &t) in gen.iter().enumerate() {
            sched.update_after_token(id, t, i == 0);
        }
        sched.mark_finished(id, StopReason::Eos);
        assert!(sched.park_finished(id), "request {id} should have parked");
    }

    /// A parked sequence is matched token-exactly, not rounded down to a block
    /// boundary — and the generated tail is part of what's reusable.
    #[test]
    fn parked_sequence_is_reused_token_exactly() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        // 18 tokens: deliberately NOT a multiple of block_size (16). The old
        // block-hash cache would have reused only 16 of them.
        let tokens: Vec<i32> = (1..=18).collect();
        run_and_park(&sched, 42, tokens.clone(), &[777]);
        assert_eq!(sched.prefix_cache_size(), 1, "one slot holds reusable KV");
        sched.schedule_step(); // retire the finished request

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                99,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(batch.prefill.contains(&99));
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 99).expect("99 running");
        // All 18 prompt tokens are resident, but [TAG_PROMPT_LOGITS] steps one back
        // so there is always a token left to decode and produce logits from.
        assert_eq!(req.skip_prefix_tokens, 17);
        assert_eq!(req.prefill_pos, 17, "prefill resumes at the skip boundary");
        assert_eq!(req.kv_seq_id, 0, "inherits the parked sequence in place");
        assert!(
            batch.kv_trims.contains(&(0, 17)),
            "everything past the divergence point must be trimmed before prefill"
        );
    }

    /// Two prompts sharing a prefix: the newcomer reuses exactly the shared span.
    #[test]
    fn partial_prefix_match_reuses_only_the_shared_span() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let shared: Vec<i32> = (1..=16).collect();
        run_and_park(&sched, 1, shared.clone(), &[777]);
        sched.schedule_step();

        // Diverges right after the shared span (777 != 100).
        let mut tokens_b = shared.clone();
        tokens_b.extend([100i32, 101, 102, 103]);

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                tokens_b,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(batch.prefill.contains(&2));
        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 1);

        let running = sched.running_batch.lock().unwrap();
        let req_b = running.iter().find(|r| r.id == 2).expect("req B running");
        assert_eq!(
            req_b.skip_prefix_tokens, 16,
            "reuse stops exactly where the prompts diverge"
        );
    }

    /// The generated reply is reusable too — this is the multi-turn chat case the
    /// old block-hash cache could never serve, because it discarded generation.
    #[test]
    fn generated_tokens_are_reusable_by_the_next_turn() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let turn1: Vec<i32> = (1..=20).collect();
        let reply = [900, 901, 902];
        run_and_park(&sched, 1, turn1.clone(), &reply);
        sched.schedule_step();

        // Turn 2's prompt = turn 1 + the assistant's reply + the new user message.
        let mut turn2 = turn1.clone();
        turn2.extend_from_slice(&reply);
        turn2.extend([500i32, 501]);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                turn2,
                5,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 2).expect("turn 2 running");
        assert_eq!(
            req.skip_prefix_tokens,
            turn1.len() + reply.len(),
            "the reply must be reused, not just the previous prompt"
        );
    }

    /// A context-rolled request's resident positions no longer line up with its token
    /// list, so its KV must never become a reuse candidate.
    #[test]
    fn rolled_and_lora_requests_are_never_parked() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                (1..=20).collect(),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        sched.record_context_roll(1, 4);
        sched.mark_finished(1, StopReason::Eos);
        assert!(!sched.park_finished(1), "rolled request must not park");

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let lora_req =
            InferenceRequest::new(2, (1..=20).collect(), 8, SamplingParams::default(), tx2)
                .with_lora(LoraSelection {
                    name: "adapter".into(),
                    scale: 1.0,
                });
        sched.submit(lora_req).unwrap();
        sched.schedule_step();
        sched.mark_finished(2, StopReason::Eos);
        assert!(!sched.park_finished(2), "LoRA request must not park");
    }

    /// Under block pressure, a parked (idle) slot is reclaimed to admit a new request
    /// — and a *running* one never is. Parking KV instead of freeing it is only safe
    /// because of this: without reclaim, warm slots would pin the pool forever.
    #[test]
    fn idle_slot_is_reclaimed_under_block_pressure() {
        use slots::SlotState;

        // 6 blocks total; each request below needs 2 (20 prompt + 8 new = 28 tokens).
        let kv = Arc::new(KVCacheManager::from_kv_tokens(16 * 6, 16));
        let sched = Scheduler::new(kv.clone(), 4);
        assert_eq!(kv.total_blocks(), 6);

        let prompt = |seed: i32| -> Vec<i32> { (0..20).map(|i| seed * 1000 + i).collect() };

        // Park one request, then fill the pool with two live ones.
        run_and_park(&sched, 1, prompt(1), &[777]);
        sched.schedule_step();
        assert_eq!(sched.prefix_cache_size(), 1, "slot 0 parked");

        for id in [2u64, 3] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    prompt(id as i32),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
            sched.schedule_step();
        }
        assert_eq!(kv.allocated_blocks(), 6, "pool is now full");

        // A fourth, unrelated request can only fit if the idle slot gives up its blocks.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                4,
                prompt(4),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(
            batch.prefill.contains(&4),
            "request 4 should have been admitted by reclaiming the idle slot"
        );
        assert!(
            batch.kv_clears.contains(&0),
            "the reclaimed sequence's KV must be wiped before reuse"
        );
        assert_eq!(
            sched.prefix_cache_size(),
            0,
            "the parked slot was reclaimed"
        );

        // The two live requests were never touched — reclaim is not preemption.
        let running = sched.running_batch.lock().unwrap();
        for id in [2u64, 3] {
            assert!(
                running.iter().any(|r| r.id == id && !r.is_finished()),
                "running request {id} must survive reclamation"
            );
        }
        let slots = sched.slots.lock().unwrap();
        assert_eq!(
            slots.count(|s| matches!(s.state, SlotState::Busy(_))),
            3,
            "requests 2, 3 and 4 hold slots"
        );
    }

    /// The RAM cache must capture a reclaimed sequence and offer it back, so a
    /// conversation survives losing its GPU blocks.
    #[test]
    fn reclaimed_sequence_is_offered_to_the_ram_cache() {
        // 6 blocks; each request below needs 2.
        let kv = Arc::new(KVCacheManager::from_kv_tokens(16 * 6, 16));
        let sched = Scheduler::new(kv.clone(), 4).with_prompt_cache(1024 * 1024);
        let prompt = |seed: i32| -> Vec<i32> { (0..20).map(|i| seed * 1000 + i).collect() };

        run_and_park(&sched, 1, prompt(1), &[777]);
        sched.schedule_step();
        for id in [2u64, 3] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    prompt(id as i32),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
            sched.schedule_step();
        }
        assert_eq!(kv.allocated_blocks(), 6, "pool full");

        // A fourth request forces the idle slot to be reclaimed.
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                4,
                prompt(4),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        // The scheduler must have asked the engine to serialise it first — losing the
        // blocks is fine, losing the conversation is the thing the cache prevents.
        assert_eq!(
            batch.kv_saves.len(),
            1,
            "reclaim must request a save: {batch:?}"
        );
        let (saved_seq, saved_tokens) = &batch.kv_saves[0];
        assert_eq!(*saved_seq, 0, "slot 0 was the idle victim");
        assert_eq!(
            saved_tokens.len(),
            21,
            "the whole resident sequence — prompt + generated — must be saved"
        );
        assert!(
            batch.kv_clears.contains(saved_seq),
            "the save must be paired with the clear that follows it"
        );
    }

    /// With the cache disabled nothing is serialised, so the default configuration
    /// carries none of its cost.
    #[test]
    fn no_saves_are_requested_when_the_ram_cache_is_off() {
        let kv = Arc::new(KVCacheManager::from_kv_tokens(16 * 6, 16));
        let sched = Scheduler::new(kv.clone(), 4); // cache defaults to 0 bytes
        let prompt = |seed: i32| -> Vec<i32> { (0..20).map(|i| seed * 1000 + i).collect() };

        run_and_park(&sched, 1, prompt(1), &[777]);
        sched.schedule_step();
        for id in [2u64, 3] {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    prompt(id as i32),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
            sched.schedule_step();
        }
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                4,
                prompt(4),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();
        assert!(batch.kv_saves.is_empty(), "cache off must cost nothing");
    }

    /// A stored state is handed back to a later prompt that extends it, and only when
    /// it beats what the live slots already cover.
    #[test]
    fn stored_state_is_restored_for_an_extending_prompt() {
        let kv = Arc::new(KVCacheManager::from_kv_tokens(16 * 64, 16));
        let sched = Scheduler::new(kv, 4).with_prompt_cache(1024 * 1024);
        let base: Vec<i32> = (0..20).collect();

        // Pretend the engine serialised this conversation earlier.
        sched.store_prompt_state(base.clone(), vec![1u8; 128]);

        // A prompt that extends it must get the state back.
        let mut longer = base.clone();
        longer.extend([900, 901]);
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                9,
                longer,
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert_eq!(batch.kv_restores.len(), 1, "cached state must be restored");
        assert_eq!(batch.kv_restores[0].1, vec![1u8; 128]);

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 9).unwrap();
        assert_eq!(
            req.skip_prefix_tokens, 20,
            "the restored prefix must be skipped, not re-prefilled"
        );
    }

    /// A forked branch waits for its parent's prefill, then copies it instead of
    /// re-prefilling the same prompt.
    #[test]
    fn forked_branch_waits_then_copies_the_parents_prefill() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);
        let prompt: Vec<i32> = (1..=40).collect();

        let (tx0, _rx0) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                prompt.clone(),
                8,
                SamplingParams::default(),
                tx0,
            ))
            .unwrap();
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(
                InferenceRequest::new(2, prompt.clone(), 8, SamplingParams::default(), tx1)
                    .with_fork_parent(1),
            )
            .unwrap();

        // Step 1: the parent is admitted; the branch is held back because there is
        // nothing to copy yet, and — crucially — does not block the queue.
        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![1], "only the parent starts: {batch:?}");
        assert_eq!(
            sched.queue_depth(),
            1,
            "the branch is re-queued, not dropped"
        );

        // Parent finishes prefilling.
        sched.update_after_token(1, 99, true);

        // Step 2: the branch is admitted and adopts the parent's prompt.
        sched.schedule_step();
        let running = sched.running_batch.lock().unwrap();
        let branch = running.iter().find(|r| r.id == 2).expect("branch admitted");
        let parent_seq = running.iter().find(|r| r.id == 1).unwrap().kv_seq_id;
        assert_eq!(
            branch.prefix_seq_id,
            Some(parent_seq),
            "the branch must copy from its parent's sequence"
        );
        assert_eq!(
            branch.skip_prefix_tokens,
            prompt.len() - 1,
            "the whole prompt is copied bar one token, so logits still get produced"
        );
        assert_ne!(
            branch.kv_seq_id, parent_seq,
            "branches need their own sequence"
        );
    }

    /// The burst case from `scripts/ab_shared_prefix.sh`: every client arrives before
    /// the scheduler has run even once. Measured on a real server this produced
    /// `cached_tokens = 0` for all eight, i.e. eight full prefills of the same prompt.
    /// This reproduces the arrival pattern to find out where the donor path is lost.
    #[test]
    fn simultaneous_burst_behind_one_prompt_reuses_it() {
        let kv = test_kv(64);
        let sched = Scheduler::new(kv, 8);
        let shared: Vec<i32> = (1..=60).collect();

        // All eight submitted before any schedule_step — no sequence exists yet.
        let mut _keep = Vec::new();
        for i in 0..8u64 {
            let mut prompt = shared.clone();
            prompt.extend([i as i32 + 500, i as i32 + 900]);
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            _keep.push(rx);
            sched
                .submit(InferenceRequest::new(
                    i + 1,
                    prompt,
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
        }

        let first = sched.schedule_step();
        assert_eq!(
            first.prefill.len(),
            1,
            "only the first can prefill; the rest have a donor that is not ready yet: {first:?}"
        );

        // Let the first reach Decoding, then drain the deferred arrivals.
        sched.update_after_token(first.prefill[0], 99, true);
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let donor = running.iter().find(|r| r.id == 1).unwrap().kv_seq_id;
        let copied = running
            .iter()
            .filter(|r| r.id != 1 && r.prefix_seq_id == Some(donor))
            .count();
        let admitted = running.len();
        assert!(
            copied + 1 == admitted && admitted > 1,
            "every later arrival must copy the live prefix: {copied} of {admitted} did"
        );
    }

    /// A pool too small to admit the burst naively, but ample once the shared prefix
    /// is charged once. Sizing the reservation after the fact left the steady state
    /// correct and the admission decision wrong: capacity the request was never going
    /// to hold could still turn it away.
    #[test]
    fn admission_reserves_only_the_blocks_past_the_shared_prefix() {
        // 34 blocks: naively 9 per request (128 prompt + 2 + 8 generated, 16/block),
        // so only 3 of 6 fit. Sharing the 8 whole prefix blocks drops each later
        // arrival to 1 of its own, and all 6 fit with room to spare.
        let kv = Arc::new(KVCacheManager::from_kv_tokens(34 * 16, 16));
        let sched = Scheduler::new(kv.clone(), 8);
        let shared: Vec<i32> = (1..=128).collect();
        assert!(
            kv.total_blocks() < 6 * 9,
            "the pool must be too small for the naive reservation, else this proves nothing"
        );

        let mut _keep = Vec::new();
        for i in 0..6u64 {
            let mut prompt = shared.clone();
            prompt.extend([i as i32 + 500, i as i32 + 900]);
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            _keep.push(rx);
            sched
                .submit(InferenceRequest::new(
                    i + 1,
                    prompt,
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
        }

        for _ in 0..8 {
            let batch = sched.schedule_step();
            for id in &batch.prefill {
                sched.update_after_token(*id, 99, true);
            }
        }

        let admitted = sched.running_batch.lock().unwrap().len();
        assert_eq!(
            admitted,
            6,
            "all six must be admitted; the pool only looks full if each is charged for \
             a prefix it shares ({} blocks of {})",
            kv.allocated_blocks(),
            kv.total_blocks()
        );
    }

    /// The budget must reflect the sharing, not just the compute. Before this, eight
    /// clients behind one system prompt held 264 blocks where ~54 were in use: each
    /// skipped the prefill but still reserved its own blocks for the shared positions.
    #[test]
    fn shared_prefix_is_charged_once_and_fully_returned() {
        let kv = test_kv(16);
        let block_size = kv.block_size();
        let sched = Scheduler::new(kv.clone(), 8);
        let shared: Vec<i32> = (1..=128).collect();

        let mut _keep = Vec::new();
        for i in 0..8u64 {
            let mut prompt = shared.clone();
            prompt.extend([i as i32 + 500, i as i32 + 900]);
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            _keep.push(rx);
            sched
                .submit(InferenceRequest::new(
                    i + 1,
                    prompt,
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .unwrap();
        }

        // Each arrival is deferred until its donor is decoding, so drain step by step:
        // prefill whoever was admitted, then let the next wave copy from them.
        for _ in 0..8 {
            let batch = sched.schedule_step();
            for id in &batch.prefill {
                sched.update_after_token(*id, 99, true);
            }
        }

        let per_request = (128usize + 2 + 8).div_ceil(block_size);
        let allocated = kv.allocated_blocks();
        let n = sched.running_batch.lock().unwrap().len();
        assert!(n > 1, "the burst must actually be admitted, got {n}");

        // Each sharer still needs private blocks for the positions past the shared
        // prefix; what it must NOT need is its own copy of the prefix itself.
        let naive = per_request * n;
        assert!(
            allocated < naive,
            "shared prefix charged {n} times: {allocated} blocks vs {naive} naive"
        );
        let shared_blocks = 128 / block_size;
        assert!(
            allocated <= naive - shared_blocks * (n - 1),
            "every sharer past the first must avoid re-charging {shared_blocks} prefix blocks: \
             {allocated} blocks, naive {naive}"
        );

        // Ref-counting is only correct if it also unwinds: a block referenced by four
        // sequences must return to the pool exactly once, not stay pinned forever.
        for i in 0..8u64 {
            sched.mark_finished(i + 1, StopReason::EngineError);
        }
        sched.schedule_step();
        assert_eq!(
            kv.allocated_blocks(),
            0,
            "shared blocks must be fully returned once every referent is gone"
        );
    }

    /// Concurrent requests behind one system prompt must share its KV, not each hold
    /// a copy. This is the case `select` cannot serve: when they arrive together no
    /// slot is idle, so nothing is inheritable — but a *live* sequence can be copied.
    #[test]
    fn concurrent_shared_prefix_copies_from_a_live_sequence() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);
        let shared: Vec<i32> = (1..=60).collect();
        let with_tail = |t: i32| {
            let mut v = shared.clone();
            v.extend([t, t + 1000]);
            v
        };

        // First arrival prefills the shared prompt.
        let (tx0, _rx0) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                with_tail(1),
                8,
                SamplingParams::default(),
                tx0,
            ))
            .unwrap();
        sched.schedule_step();

        // A second, *concurrent* request: the first is still Prefilling, so there is
        // nothing to copy yet and it must be deferred rather than block the queue.
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                with_tail(2),
                8,
                SamplingParams::default(),
                tx1,
            ))
            .unwrap();
        let batch = sched.schedule_step();
        assert!(
            !batch.prefill.contains(&2),
            "nothing to copy yet: {batch:?}"
        );
        assert_eq!(sched.queue_depth(), 1, "deferred, not dropped");

        // Once the first is decoding, its prompt is resident and copyable.
        sched.update_after_token(1, 99, true);
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let donor_seq = running.iter().find(|r| r.id == 1).unwrap().kv_seq_id;
        let second = running.iter().find(|r| r.id == 2).expect("admitted");
        assert_eq!(
            second.prefix_seq_id,
            Some(donor_seq),
            "must copy the shared prefix from the LIVE sequence"
        );
        assert_eq!(
            second.skip_prefix_tokens, 60,
            "the whole shared prefix is skipped, not re-prefilled"
        );
        assert_ne!(
            second.kv_seq_id, donor_seq,
            "each request keeps its own sequence"
        );
    }

    /// A branch whose parent never materialises must still run — slower, not stuck.
    #[test]
    fn forked_branch_falls_back_when_the_parent_is_gone() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);
        let prompt: Vec<i32> = (1..=40).collect();

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(
                InferenceRequest::new(7, prompt.clone(), 8, SamplingParams::default(), tx)
                    .with_fork_parent(999), // never submitted
            )
            .unwrap();

        let batch = sched.schedule_step();
        assert_eq!(batch.prefill, vec![7], "must be admitted, not stranded");
        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 7).unwrap();
        assert_eq!(req.prefix_seq_id, None, "nothing to copy from");
        assert_eq!(req.skip_prefix_tokens, 0, "so it prefills in full");
    }

    /// `--kv-reuse false` restores the pre-0.19 behaviour end to end.
    #[test]
    fn kv_reuse_disabled_never_parks_or_hits() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8).with_kv_reuse(false, DEFAULT_SLOT_PROMPT_SIMILARITY);

        let tokens: Vec<i32> = (1..=18).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                tokens.clone(),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();
        sched.mark_finished(1, StopReason::Eos);
        assert!(!sched.park_finished(1));
        sched.schedule_step();

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        sched.schedule_step();

        assert_eq!(sched.prefix_hits.load(Ordering::Relaxed), 0);
        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 2).unwrap();
        assert_eq!(req.skip_prefix_tokens, 0, "must prefill from token 0");
    }

    /// Prefix-cache eviction stress test — settles design-doc §7's open question
    /// ("does the prefix cache leak blocks/seq-ids under churn?") empirically.
    ///
    /// It drives real `schedule_step` + `try_insert_prefix` over hundreds of
    /// admit / finish / cache / hit / refuse-when-full cycles and asserts strict
    /// conservation after every step, under a **move** ownership model: each seq_id
    /// and each KV block is owned by exactly one of {pool, a running request, a
    /// prefix-cache entry} — never dropped, never duplicated. (The engine layers a
    /// KV-copy optimization on top via `return_prefix_seq_id`; the leak question is
    /// about the scheduler's bookkeeping, which the move model exercises exactly.)
    ///
    /// A leak would show up as either a seq_id that can't be accounted for
    /// (`seen.len() != TOTAL_SEQ`) or an allocated KV block that nothing references
    /// (`allocated_blocks() != reachable`), or a non-zero allocation after draining.
    #[test]
    fn stress_slot_reuse_no_leak() {
        use slots::SlotState;
        use std::collections::HashSet;

        const TOTAL_SEQ: usize = 8; // = max_batch_size → slots {0..8}
                                    // Plenty of blocks: this test targets slot churn, not block starvation, so
                                    // keep the loop live and deterministic.
        let kv = test_kv(16);
        let sched = Scheduler::new(kv.clone(), TOTAL_SEQ);

        // Deterministic prompt of `blocks` full 16-token blocks (+3 leftover tokens),
        // content keyed by `seed` so distinct seeds have distinct prefixes.
        let prompt = |seed: i32, blocks: usize| -> Vec<i32> {
            (0..(blocks * 16 + 3) as i32)
                .map(|i| seed * 1000 + i)
                .collect()
        };

        // Assert full conservation of sequences and blocks against the live state.
        let check_conservation = |label: &str| {
            let running = sched.running_batch.lock().unwrap();
            let slots = sched.slots.lock().unwrap();

            // Every seq_id appears exactly once, in exactly one state, and all
            // TOTAL_SEQ are accounted for.
            let mut seen: HashSet<i32> = HashSet::new();
            for slot in slots.iter() {
                assert!(
                    seen.insert(slot.seq_id),
                    "{label}: seq {} duplicated in the slot table",
                    slot.seq_id
                );
            }
            assert_eq!(
                seen.len(),
                TOTAL_SEQ,
                "{label}: seq_ids not conserved (found {}, expected {TOTAL_SEQ})",
                seen.len()
            );

            // A Busy slot must correspond to exactly one live request holding that
            // seq_id, and no two running requests may share one.
            let mut running_seqs: HashSet<i32> = HashSet::new();
            for r in running.iter() {
                if r.kv_seq_id >= 0 {
                    assert!(
                        running_seqs.insert(r.kv_seq_id),
                        "{label}: seq {} claimed by two running requests",
                        r.kv_seq_id
                    );
                }
            }
            for slot in slots.iter() {
                if let SlotState::Busy(req_id) = slot.state {
                    assert!(
                        running
                            .iter()
                            .any(|r| r.id == req_id && r.kv_seq_id == slot.seq_id),
                        "{label}: slot {} is Busy({req_id}) with no matching running request",
                        slot.seq_id
                    );
                } else {
                    assert!(
                        !running_seqs.contains(&slot.seq_id),
                        "{label}: slot {} is not Busy but a running request holds it",
                        slot.seq_id
                    );
                }
            }

            // Every allocated block is reachable from a running request or a slot —
            // nothing dropped on the floor, nothing counted twice.
            let running_blocks: usize = running.iter().map(|r| r.page_table.len()).sum();
            let slot_blocks = slots.charged_blocks();
            assert_eq!(
                kv.allocated_blocks(),
                running_blocks + slot_blocks,
                "{label}: KV block leak — {} allocated but only {} reachable",
                kv.allocated_blocks(),
                running_blocks + slot_blocks
            );

            // Reusable KV can never exceed the number of sequences that exist.
            assert!(
                slots.count(|s| s.state == SlotState::Idle) <= slots.len(),
                "{label}: more idle slots than the table holds"
            );
        };

        let mut observed_hit = false;

        for iter in 0..400usize {
            // 1. Submit one request. 2/3 reuse a small shared-prompt set (so their
            //    prefixes hit once parked); 1/3 are unique (misses → fresh slots).
            let seed = if iter % 3 == 0 {
                100_000 + iter as i32 // unique → miss
            } else {
                (iter % 3) as i32 // {1,2} shared → hit candidate
            };
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let id = iter as u64 + 1;
            let req = InferenceRequest::new(id, prompt(seed, 2), 8, SamplingParams::default(), tx);
            sched.submit(req).unwrap();

            // 2. Schedule: retires the previous iteration's finishes and admits.
            let hits_pre = sched.prefix_hits.load(Ordering::Relaxed);
            let _batch = sched.schedule_step();
            if sched.prefix_hits.load(Ordering::Relaxed) > hits_pre {
                observed_hit = true;
            }

            // 3. Conservation must hold after every step.
            check_conservation("mid-churn");

            // 4. Finish the oldest running request once a few are in flight, parking
            //    it so its KV becomes reusable — the churn this test exists to drive.
            let finish_id = {
                let mut running = sched.running_batch.lock().unwrap();
                if running.len() >= 3 {
                    let r = &mut running[0];
                    r.state = RequestState::Finished;
                    Some(r.id)
                } else {
                    None
                }
            };
            if let Some(id) = finish_id {
                sched.park_finished(id);
            }
            check_conservation("post-park");
        }

        // The churn must have actually exercised the interesting paths, or the test
        // would be conserving trivially.
        // (Block starvation and the reclaim path are deliberately NOT exercised here —
        // this pool is oversized on purpose so the loop stays live and deterministic.
        // `idle_slot_is_reclaimed_under_block_pressure` covers reclaim directly.)
        assert!(observed_hit, "stress loop never produced a prefix hit");
        assert!(sched.prefix_hits.load(Ordering::Relaxed) > 0);

        // 5. Drain everything and prove nothing leaked: finish all running requests,
        //    let schedule_step retire them, then release every parked slot. Allocation
        //    must fall back to zero with every slot Free.
        {
            let mut running = sched.running_batch.lock().unwrap();
            for r in running.iter_mut() {
                r.state = RequestState::Finished;
            }
        }
        sched.schedule_step();
        {
            let mut slots = sched.slots.lock().unwrap();
            let seq_ids: Vec<i32> = slots.iter().map(|s| s.seq_id).collect();
            for seq_id in seq_ids {
                let blocks = slots.release(seq_id);
                kv.free_blocks(&blocks);
            }
        }
        assert_eq!(
            kv.allocated_blocks(),
            0,
            "KV blocks leaked after full churn + drain"
        );
        let slots = sched.slots.lock().unwrap();
        assert_eq!(slots.len(), TOTAL_SEQ, "slot table lost sequences");
        assert!(
            slots.iter().all(|s| s.state == SlotState::Free),
            "not every slot returned to Free after drain"
        );
    }

    /// Chunked prefill state machine (S1): a request whose prompt is prefilled over
    /// several steps must stay `Prefilling` — and be re-emitted to the prefill batch —
    /// until its cursor reaches the prompt end and the first token is sampled, at which
    /// point it moves to the decode batch. The model (do_prefill) advances the cursor
    /// via `advance_prefill`; here we drive those transitions directly (no FFI).
    #[test]
    fn chunked_prefill_stays_prefilling_until_complete() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // 48-token prompt = 3 full blocks; chunked prefill would span several steps.
        let prompt: Vec<i32> = (0..48).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                prompt,
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();

        // Step 1: admitted → emitted to prefill, not decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "admitted request must prefill");
        assert!(b.decode.is_empty());

        // Model submitted a non-final chunk (16 of 48 tokens): still Prefilling.
        sched.advance_prefill(1, 16);

        // Step 2: incomplete prefill must be RE-EMITTED to prefill (the whole point —
        // it no longer completes in a single step) and never to decode.
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "incomplete prefill must be re-emitted");
        assert!(
            b.decode.is_empty(),
            "must not decode before prefill completes"
        );

        // Final chunk reaches the prompt end and the first token is sampled →
        // update_after_token(from_prefill=true) transitions the request to Decoding.
        sched.advance_prefill(1, 48);
        sched.update_after_token(1, 42, true);

        // Step 3: now Decoding → emitted to decode, no longer to prefill.
        let b = sched.schedule_step();
        assert!(
            b.prefill.is_empty(),
            "completed prefill must not re-emit to prefill"
        );
        assert_eq!(b.decode, vec![1], "completed request must decode");
    }

    #[test]
    fn context_roll_reduces_logical_context_len() {
        // context_len() = prefilled + generated - rolled. A roll must shift the next
        // decode position down by exactly the discarded amount, and rolls accumulate.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 8);

        // Put a request in the running batch that has "filled" 100 tokens of context.
        {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            let mut req =
                InferenceRequest::new(7, (0..40).collect(), 200, SamplingParams::default(), tx);
            req.kv_seq_id = 0;
            req.prefilled_tokens = 40;
            req.generated_tokens = 60; // context_len = 100
            req.state = RequestState::Decoding;
            sched.running_batch.lock().unwrap().push(req);
        }

        let ctx_len = |s: &Scheduler| s.get_running(&[7])[0].context_len();
        assert_eq!(ctx_len(&sched), 100);

        // Roll away 30 oldest tokens → live length drops to 70 (next decode pos = 69).
        sched.record_context_roll(7, 30);
        assert_eq!(
            ctx_len(&sched),
            70,
            "rolled tokens subtract from context_len"
        );

        // Generation keeps advancing on top of the reduced length.
        sched.update_after_token(7, 123, false);
        assert_eq!(
            ctx_len(&sched),
            71,
            "generated token adds above the rolled window"
        );

        // A second roll accumulates with the first.
        sched.record_context_roll(7, 20);
        assert_eq!(ctx_len(&sched), 51, "rolls accumulate");
    }
    #[test]
    fn admission_never_preempts_running_requests() {
        // Blocks are fully reserved at admission, so a newcomer that doesn't fit must
        // WAIT — evicting the older running request would discard a generation whose
        // text the client already received (and the pair can evict each other forever:
        // the livelock this test originally exposed).
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        // A deliberately tiny pool: both requests cannot fit at once.
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        assert!(total >= 4, "pool too small to stage the scenario: {total}");
        let sched = Scheduler::new(kv, 4);

        // Request 1 reserves most of the pool and starts generating.
        let prompt1: Vec<i32> = (0..16).collect();
        let max_new1 = (total - 2) * 16 - prompt1.len();
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                prompt1.clone(),
                max_new1,
                SamplingParams::default(),
                tx1,
            ))
            .unwrap();
        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![1], "request 1 admitted");
        sched.set_prefilled_tokens(1, prompt1.len());
        for tok in [101, 102, 103] {
            sched.update_after_token(1, tok, tok == 101); // first token completes prefill
        }

        // Request 2 needs more blocks than remain. It must WAIT, not evict request 1.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                32,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        let b = sched.schedule_step();
        assert!(
            b.preempted_seq_ids.is_empty(),
            "admission must never preempt a running request"
        );
        assert_eq!(
            b.decode,
            vec![1],
            "request 1 keeps running while request 2 waits"
        );
        assert_eq!(sched.queue_depth(), 1, "request 2 queued");
        {
            let running = sched.running_batch.lock().unwrap();
            let r1 = running.iter().find(|r| r.id == 1).expect("req 1 running");
            assert_eq!(r1.generated_tokens, 3, "generation state untouched");
        }

        // Once request 1 finishes, request 2 gets its turn.
        sched.mark_finished(1, StopReason::Eos);
        let b = sched.schedule_step();
        assert_eq!(
            b.prefill,
            vec![2],
            "request 2 admitted after request 1 finishes"
        );
    }

    #[test]
    fn oversized_request_is_rejected_synchronously_by_submit() {
        // A request that could never fit even into an EMPTY pool is rejected by
        // `submit()` itself — before it ever enters the queue — so it can never
        // block the queue head. This used to be a schedule_step()-time check; 0.16
        // moved it to submit() since it's statically knowable at submission time.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 200_000, 0.5, 16, 1, 1));
        let total = kv.total_blocks();
        let sched = Scheduler::new(kv, 4);

        // Oversized: needs more blocks than the entire pool.
        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        let err = sched
            .submit(InferenceRequest::new(
                1,
                (0..16).collect(),
                (total + 2) * 16,
                SamplingParams::default(),
                tx1,
            ))
            .expect_err("oversized request must be rejected by submit(), not queued");
        assert!(matches!(err, SubmitError::TooLarge { .. }));
        assert_eq!(
            sched.queue_depth(),
            0,
            "rejected request never entered the queue"
        );

        // A normal request submits and schedules fine.
        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                16,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();

        let b = sched.schedule_step();
        assert_eq!(b.prefill, vec![2], "the normal request is admitted");
        assert_eq!(sched.queue_depth(), 0, "nothing left waiting");
        assert_eq!(sched.active_requests(), 1, "only the normal request runs");
    }

    #[test]
    fn submit_rejects_when_queue_full() {
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        // max_batch_size=1 keeps the seq_id pool tiny so nothing gets admitted off
        // the queue between the two submits; max_queue_depth=1 caps the queue itself.
        let sched = Scheduler::with_max_queue_depth(kv, 1, 1);

        let (tx1, _rx1) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                1,
                (0..16).collect(),
                8,
                SamplingParams::default(),
                tx1,
            ))
            .expect("first submit fills the queue to its cap, should succeed");
        assert_eq!(sched.queue_depth(), 1);

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        let err = sched
            .submit(InferenceRequest::new(
                2,
                (0..16).collect(),
                8,
                SamplingParams::default(),
                tx2,
            ))
            .expect_err("second submit must be rejected — queue is at max_queue_depth");
        assert!(matches!(err, SubmitError::QueueFull { depth: 1, max: 1 }));
        assert_eq!(
            sched.queue_depth(),
            1,
            "rejected request never entered the queue"
        );
    }

    #[test]
    fn submit_unbounded_by_default() {
        // max_queue_depth=0 (the default via `Scheduler::new`) means unbounded — this
        // guards against accidentally changing the default and silently capping every
        // existing deployment's queue.
        let config = ModelConfig {
            num_layers: 2,
            num_heads: 2,
            num_heads_kv: 2,
            head_dim: 64,
            n_embd: 128,
            vocab_size: 1000,
        };
        let kv = Arc::new(KVCacheManager::new(&config, 500_000_000, 0.5, 16, 1, 1));
        let sched = Scheduler::new(kv, 1);
        for id in 1..=50u64 {
            let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
            sched
                .submit(InferenceRequest::new(
                    id,
                    (0..16).collect(),
                    8,
                    SamplingParams::default(),
                    tx,
                ))
                .expect("unbounded queue must accept many more requests than the batch size");
        }
        assert_eq!(sched.queue_depth(), 50);
    }

    /// A model that cannot donate KV between sequences must still reuse its own slot.
    ///
    /// This is the hybrid case (Qwen3.5, Qwen3-Next, Falcon-H1, Jamba): `seq_cp` on
    /// recurrent state is not a partial operation, so cross-sequence copying is off —
    /// but inheriting the KV a sequence already holds copies nothing and stays legal.
    /// Gating both on one flag silently cost fox all prompt reuse on those models while
    /// `llama-server` kept it, so the split is pinned here.
    #[test]
    fn slot_reuse_survives_when_cross_sequence_copying_is_off() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);
        sched.set_prefix_reuse(false); // no seq_cp — the hybrid case
        sched.set_slot_reuse(true); // but the slot's own KV is still inheritable

        let tokens: Vec<i32> = (1..=18).collect();
        run_and_park(&sched, 42, tokens.clone(), &[777]);
        sched.schedule_step();

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                99,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 99).expect("99 running");
        assert_eq!(
            req.skip_prefix_tokens, 17,
            "the slot's resident KV must still be inherited without seq_cp"
        );
        assert!(
            req.prefix_seq_id.is_none(),
            "nothing may be copied out of another sequence when seq_cp is unavailable"
        );
    }

    /// With slot reuse off too, nothing is reused and the prompt is prefilled whole.
    #[test]
    fn no_reuse_at_all_when_both_capabilities_are_off() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 8);
        sched.set_prefix_reuse(false);
        sched.set_slot_reuse(false);

        let tokens: Vec<i32> = (1..=18).collect();
        run_and_park(&sched, 42, tokens.clone(), &[777]);
        sched.schedule_step();

        let (tx2, _rx2) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                99,
                tokens,
                5,
                SamplingParams::default(),
                tx2,
            ))
            .unwrap();
        sched.schedule_step();

        let running = sched.running_batch.lock().unwrap();
        let req = running.iter().find(|r| r.id == 99).expect("99 running");
        assert_eq!(req.skip_prefix_tokens, 0, "no reuse of any kind");
        assert!(req.prefix_seq_id.is_none());
    }

    /// A checkpoint may only ever capture the prompt boundary.
    ///
    /// `prefilled_sequence` is what the engine asks after prefill ends, and the whole
    /// point of checkpointing *there* rather than at eviction is that the blob then
    /// covers exactly the prompt. If it kept answering once generation was under way,
    /// the saved state would include the reply — reproducing the problem the checkpoint
    /// exists to avoid, since the next turn would then have to roll back past it.
    #[test]
    fn prefilled_sequence_stops_answering_once_generation_starts() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 4);
        let tokens: Vec<i32> = (1..=20).collect();
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                7,
                tokens.clone(),
                8,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        sched.schedule_step();

        // First token: prefill has just ended, this is the boundary worth saving.
        sched.update_after_token(7, 101, true);
        let got = sched
            .prefilled_sequence(7)
            .expect("boundary is checkpointable");
        assert_eq!(got.1, tokens, "the blob must be keyed by the prompt alone");

        // Second token: the sequence now holds generated KV too.
        sched.update_after_token(7, 102, false);
        assert!(
            sched.prefilled_sequence(7).is_none(),
            "past the prompt boundary a checkpoint would capture the reply as well"
        );
    }

    /// When the KV cannot be rolled back, a *tied* cache entry beats the live slot.
    ///
    /// The slot advertises an LCP it cannot deliver on such a model: reaching it means
    /// trimming across the generated reply, which a recurrent cache refuses. Requiring
    /// the cache to win *strictly* is what left Qwen3.5 with 20 slot hits, 20 refused
    /// trims and `cached_tokens` 0 while a usable checkpoint sat in RAM unread.
    ///
    /// The tie is the whole point, so it has to be built: a parked slot holding the
    /// tokens (which is what makes `choice.lcp` non-zero) *and* a checkpoint covering
    /// the same ones. An earlier version of this test stored only the checkpoint, so
    /// `choice.lcp` was 0, there was no tie, and it passed with the fix reverted.
    #[test]
    fn a_tied_checkpoint_wins_when_the_kv_cannot_roll_back() {
        let kv = test_kv(16);
        let sched = Scheduler::new(kv, 4).with_prompt_cache(4096);
        sched.set_slot_reuse(true);

        let tokens: Vec<i32> = (1..=18).collect();
        // Park a sequence holding exactly these tokens: now a slot offers them.
        run_and_park(&sched, 42, tokens.clone(), &[777]);
        sched.schedule_step();
        // And a checkpoint covering the same prompt, as the engine stores after prefill.
        sched.store_prompt_state(tokens.clone(), vec![0xAB; 64]);

        // Hybrid/recurrent: the slot's offer cannot actually be taken up.
        sched.set_prefix_reuse(false);

        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        sched
            .submit(InferenceRequest::new(
                9,
                tokens,
                5,
                SamplingParams::default(),
                tx,
            ))
            .unwrap();
        let batch = sched.schedule_step();

        assert!(
            !batch.kv_restores.is_empty(),
            "a checkpoint tying the slot must still be restored when the KV cannot roll back"
        );
    }
}
