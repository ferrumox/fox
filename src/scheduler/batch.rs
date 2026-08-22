// Batch types and InferenceRequest for the scheduler.

use crate::kv_cache::PageTable;
use crate::seq::SeqId;
use tokio::sync::mpsc;

/// A LoRA adapter selection: which named adapter (`--lora-modules`) to apply for
/// this request, and at what scale. Set when a client names an adapter alias
/// (instead of the base model) in the `model` field — resolved by
/// `ModelRegistry::resolve_for_request`. `do_prefill`/`do_decode` group requests
/// by this value (comparing `name` — `f32` doesn't implement `Eq`, so equality is
/// name-based; two requests naming the same adapter always share its one
/// configured scale) and call `llama_set_adapters_lora` once per group, since the
/// adapter set is a property of the whole `llama_context`, not of a sequence —
/// see `docs/design/lora-support.md`.
#[derive(Debug, Clone, PartialEq)]
pub struct LoraSelection {
    pub name: String,
    pub scale: f32,
}

/// All sampling hyper-parameters for a single inference request.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Softmax temperature (0 = greedy, 1 = unscaled).
    pub temperature: f32,
    /// Top-p nucleus threshold (1.0 = disabled).
    pub top_p: f32,
    /// Top-K filter (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty applied to already-generated tokens (1.0 = disabled).
    pub repetition_penalty: f32,
    /// OpenAI-style frequency penalty (additive, scales with token count; 0 = disabled).
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty (additive, applied once per seen token; 0 = disabled).
    pub presence_penalty: f32,
    /// How far back the three penalties above look, in generated tokens
    /// (llama.cpp `repeat_last_n`): `-1` = whole history, `0` = disabled, `n` = last `n`.
    /// Defaults to `-1`, which is fox's historical behaviour.
    pub repeat_last_n: i32,
    /// Optional RNG seed for reproducible sampling.
    pub seed: Option<u64>,
    /// Stop generation when the output ends with any of these strings.
    /// The stop string itself is NOT included in the output (OpenAI spec behavior).
    pub stop: Option<Vec<String>>,
    /// When true, emit `<think>…</think>` reasoning tokens to the output stream
    /// instead of silently discarding them.  Useful for debugging or transparency.
    pub show_thinking: bool,
    /// When true, the engine initialises the per-request output filter with
    /// `in_thinking = true`, indicating that the prompt already ends with an
    /// open `<think>` tag and the first generated tokens are reasoning content.
    /// Set together with `show_thinking` when the caller has appended `<think>`
    /// to the prompt to force the model into thinking mode (Qwen3-style).
    pub initial_in_thinking: bool,
    /// Maximum characters of `<think>…` content before the block is force-closed
    /// and the model is allowed to start its response.  Prevents infinite thinking
    /// loops where the model never generates `</think>`.
    /// 0 = no limit.  Default: 8192 (roughly 2 000–3 000 tokens).
    pub max_thinking_chars: usize,
    /// GBNF grammar constraining generation (None = unconstrained). When set, the model
    /// masks every token the grammar forbids at the current position before fox's Rust
    /// sampler picks within the allowed set. Held behind `Arc` so cloning a request's
    /// sampling params (done per step) doesn't re-copy the grammar text.
    pub grammar: Option<std::sync::Arc<str>>,
    /// Emit per-token log-probabilities when `Some(n)`: the sampled token's logprob plus
    /// the top-`n` alternatives (`n = 0` → just the sampled token). `None` = off.
    pub logprobs: Option<u8>,
    /// Min-P sampling: drop tokens below `min_p × max_prob` (0.0 = disabled).
    pub min_p: f32,
    /// Suppress end-of-generation tokens until at least this many tokens are generated
    /// (0 = disabled).
    pub min_tokens: usize,
    /// Top-nσ: keep only tokens whose logit is within `n` standard deviations of the
    /// top logit (0 = disabled). Temperature-invariant, unlike top-p.
    pub top_n_sigma: f32,
    /// Floor on how few candidates any truncation step may leave (0 = just the top one).
    pub min_keep: usize,
    /// Additive per-token logit bias (OpenAI `logit_bias`). `Arc` so per-step request
    /// clones stay cheap.
    pub logit_bias: Option<std::sync::Arc<std::collections::HashMap<i32, f32>>>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: -1,
            seed: None,
            stop: None,
            show_thinking: false,
            initial_in_thinking: false,
            max_thinking_chars: 8192,
            grammar: None,
            logprobs: None,
            min_p: 0.0,
            min_tokens: 0,
            top_n_sigma: 0.0,
            min_keep: 0,
            logit_bias: None,
        }
    }
}

/// Streaming token payload.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u64,
    pub token_id: i32,
    pub text: String,
    pub is_eos: bool,
    /// Set on the final token of a request; indicates why generation stopped.
    pub stop_reason: Option<StopReason>,
    /// Per-token log-probabilities, populated only when the request asked for them
    /// (`SamplingParams.logprobs`). `None` on EOS/empty tokens and unconstrained requests.
    pub logprob: Option<TokenLogprob>,
    /// How many of this request's prompt tokens were already resident in the KV cache
    /// and so were never re-prefilled — the request's `skip_prefix_tokens`.
    ///
    /// Carried on every token (it is a `u32`, and the alternative is a first-token-only
    /// convention every consumer has to remember). Surfaces as OpenAI's
    /// `usage.prompt_tokens_details.cached_tokens`, which is how a client sees the KV
    /// reuse benefit at all.
    pub cached_tokens: u32,
}

/// Log-probability detail for one generated token (OpenAI `logprobs`).
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    /// The sampled token's piece text (lossy UTF-8 of `bytes`).
    pub token: String,
    /// Natural-log probability of the sampled token under the model's distribution.
    pub logprob: f32,
    /// Raw bytes of the sampled token's piece.
    pub bytes: Vec<u8>,
    /// The most-likely alternatives at this position (highest logprob first).
    pub top: Vec<TopLogprob>,
}

/// One alternative token and its log-probability (OpenAI `top_logprobs`).
#[derive(Debug, Clone)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

/// Stop reason when generation ends.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    Eos,
    Length,
    Preempt,
    /// A user-supplied stop string was found in the output.
    StopSequence,
    /// Prefill/decode failed (e.g. `llama_decode` error, KV cache exhaustion mid-step).
    /// Distinct from `Length` so clients and metrics can tell a real engine failure
    /// apart from a normal max-tokens stop.
    EngineError,
}

/// Request state machine.
///
/// Normal flow: `Waiting → Prefilling → Decoding → Finished`
/// On preemption: `Decoding → Waiting` (KV blocks freed, re-prefill required)
/// With CPU↔GPU swap (future): `Decoding → Swapped → Decoding` (KV blocks
/// persisted in CPU RAM; re-prefill not required on resume).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    /// In the waiting queue, not yet scheduled.
    Waiting,
    /// KV blocks allocated; prefill batch is being processed this step.
    Prefilling,
    /// Prefill complete; generating tokens one step at a time.
    Decoding,
    /// KV blocks have been swapped to CPU RAM.
    ///
    /// The request retains its `page_table` (now pointing to CPU-resident
    /// blocks) and its `generated_token_ids`.  On swap-in, the KV data is
    /// copied back to GPU and the state transitions to `Decoding`.
    ///
    /// Note: this state is currently a placeholder.  Actual CPU↔GPU KV
    /// transfer requires byte-level tensor access that llama.cpp does not yet
    /// expose through its public API.  The infrastructure (state, page_table
    /// retention) is in place; the transfer implementation will be added once
    /// the underlying API is available.
    Swapped,
    /// Generation complete (EOS, Length, StopSequence, or Preempt).
    Finished,
}

/// A single inference request in the scheduler.
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    pub id: u64,
    pub prompt_tokens: Vec<i32>,
    pub last_token: Option<i32>,
    pub generated_tokens: usize,
    /// Token IDs produced so far (used for repetition penalty).
    pub generated_token_ids: Vec<i32>,
    pub max_new_tokens: usize,
    pub state: RequestState,
    /// Logical-to-physical page table for this request's KV cache blocks.
    pub page_table: PageTable,
    pub response_tx: mpsc::UnboundedSender<Token>,
    pub stop_reason: Option<StopReason>,
    pub sampling: SamplingParams,
    /// The llama.cpp sequence this request occupies, issued by the slot table when
    /// it was admitted. `None` until then, and again once the request parks its KV
    /// and hands the slot back — a variant rather than the `-1` this used to carry,
    /// so "no sequence yet" cannot be missed by a comparison.
    pub kv_seq_id: Option<SeqId>,
    /// Number of prompt tokens already present in the KV cache from a prefix cache hit.
    /// The prefill step will skip these tokens (they don't need to be computed again).
    pub skip_prefix_tokens: usize,
    /// Sequence ID holding the cached prefix KV data in llama.cpp.
    /// Set during admission when a prefix cache hit is found.
    /// The engine copies KV from this seq_id before prefill, then clears and returns it to pool.
    pub prefix_seq_id: Option<SeqId>,
    /// Timestamp when the request was submitted (for latency metrics).
    pub submitted_at: std::time::Instant,
    /// Total prompt tokens in the sequence's KV once prefill completes (donated
    /// prefix + submitted = `prompt_tokens.len()`). The decode position derives from
    /// this, so it must equal the KV's true length — never the submitted count.
    pub prefilled_tokens: usize,
    /// Absolute prompt position already placed in the KV cache while this request is
    /// still `Prefilling`. Prefill submits up to `max_prefill_chunk` tokens per step
    /// and advances this cursor until it reaches `prompt_tokens.len()`, at which point
    /// the request is sampled and transitions to `Decoding`. Starts at the effective
    /// skip (0, or the prefix-hit boundary); reset to 0 on preemption (KV freed).
    pub prefill_pos: usize,
    /// Total tokens discarded from the KV window by context rolling (shift-on-full).
    /// Subtracted from the raw token count in [`Self::context_len`] so decode positions
    /// track the shifted KV. Reset to 0 on preemption (the KV — and any rolls — are gone).
    pub rolled_tokens: usize,
    /// Tokenized multimodal (text+image) prompt, set via `with_multimodal` when the
    /// request carries images. `prompt_tokens` stays empty for such requests —
    /// every place that sizes KV/blocks from prompt length must read
    /// [`Self::n_positions`] instead so it accounts for image tokens too.
    pub multimodal: Option<crate::engine::model::MultimodalChunks>,
    /// LoRA adapter to apply for this request, set via `with_lora` when the
    /// client selected a named adapter alias instead of the base model.
    pub lora: Option<LoraSelection>,
    /// When true, this request never hits or donates to the prefix cache. Set
    /// unconditionally by `with_lora` — KV computed under one adapter's weights
    /// is invalid input for a different adapter (or no adapter) at the same
    /// token positions, so cross-adapter reuse would silently corrupt
    /// generation. Multimodal requests get this for free from an empty
    /// `prompt_tokens` instead; LoRA requests have real text tokens, so this
    /// flag is the explicit mechanism for them.
    pub skip_prefix_cache: bool,
    /// For `n>1`/`best_of`: the sibling branch whose prefill this one waits for and
    /// then copies, instead of re-prefilling the identical prompt itself.
    ///
    /// `None` for branch 0 and for every ordinary request. Admission holds a forked
    /// branch back until its parent is decoding — the parent's KV has to exist before
    /// it can be copied — and falls back to a normal full prefill if the parent is
    /// gone, so a failed or finished parent costs speed, never correctness.
    pub fork_parent: Option<u64>,
    /// Resolved at admission from `fork_parent`: `(parent_seq_id, tokens_to_copy)`.
    /// Separate from `fork_parent` because the parent's `seq_id` is only known once it
    /// is actually decoding, and because clearing `fork_parent` is how the fallback
    /// path says "prefill normally".
    pub fork_source: Option<(SeqId, usize)>,
}

impl InferenceRequest {
    pub fn new(
        id: u64,
        prompt_tokens: Vec<i32>,
        max_new_tokens: usize,
        sampling: SamplingParams,
        response_tx: mpsc::UnboundedSender<Token>,
    ) -> Self {
        Self {
            id,
            prompt_tokens,
            last_token: None,
            generated_tokens: 0,
            generated_token_ids: Vec::new(),
            max_new_tokens,
            state: RequestState::Waiting,
            page_table: PageTable::default(),
            response_tx,
            stop_reason: None,
            sampling,
            kv_seq_id: None,
            skip_prefix_tokens: 0,
            prefix_seq_id: None,
            submitted_at: std::time::Instant::now(),
            prefilled_tokens: 0,
            prefill_pos: 0,
            rolled_tokens: 0,
            multimodal: None,
            lora: None,
            skip_prefix_cache: false,
            fork_parent: None,
            fork_source: None,
        }
    }

    /// Attach a tokenized multimodal prompt (image content already encoded by
    /// `mtmd_tokenize`). `prompt_tokens` should be left empty by the caller — the
    /// position count comes from the chunks via [`Self::n_positions`] instead.
    pub fn with_multimodal(mut self, chunks: crate::engine::model::MultimodalChunks) -> Self {
        self.multimodal = Some(chunks);
        self
    }

    /// Mark this request as a forked branch of `parent`: it shares the prompt and
    /// will copy the parent's prefilled KV rather than recomputing it.
    pub fn with_fork_parent(mut self, parent: u64) -> Self {
        self.fork_parent = Some(parent);
        self
    }

    /// Select a LoRA adapter for this request. Also sets `skip_prefix_cache` —
    /// see that field's doc comment for why.
    pub fn with_lora(mut self, selection: LoraSelection) -> Self {
        self.lora = Some(selection);
        self.skip_prefix_cache = true;
        self
    }

    /// Total KV positions this request's prompt occupies — `prompt_tokens.len()`
    /// for a plain text prompt, or the multimodal chunk count (image tokens can
    /// outnumber `prompt_tokens.len()`, which is always 0 for such a request).
    pub fn n_positions(&self) -> usize {
        self.multimodal
            .as_ref()
            .map(|m| m.n_positions())
            .unwrap_or(self.prompt_tokens.len())
    }

    /// Total tokens currently in the KV cache (prefilled + generated).
    /// Uses `prefilled_tokens` (set after prefill) so the decode position is always
    /// consecutive with the last prefill position, even when `effective_skip` caused
    /// fewer tokens to be submitted than `prompt_tokens.len()`.
    pub fn context_len(&self) -> usize {
        let raw = if self.prefilled_tokens > 0 {
            self.prefilled_tokens + self.generated_tokens
        } else {
            // Fallback before prefill completes (prefilled_tokens not yet set).
            self.n_positions() + self.generated_tokens
        };
        // Context rolling discards the oldest KV window; the live sequence length — and
        // thus the next decode position — is reduced by everything rolled out so far.
        raw.saturating_sub(self.rolled_tokens)
    }

    /// Whether this request has reached a terminal state.
    pub fn is_finished(&self) -> bool {
        self.state == RequestState::Finished
    }
}

/// Output of schedule_step: which requests to prefill vs decode,
/// plus the sequence IDs that were just preempted (so the engine can wipe their KV state
/// before those IDs are reassigned to new requests).
#[derive(Debug, Default)]
pub struct ScheduledBatch {
    pub prefill: Vec<u64>,
    pub decode: Vec<u64>,
    /// Sequence IDs whose KV cache must be cleared (request was preempted this step).
    pub preempted_seq_ids: Vec<SeqId>,
    /// `(seq_id, keep_from)` pairs: drop every KV position `>= keep_from` for that
    /// sequence before the next prefill writes there.
    ///
    /// Emitted when an admitted request inherits an idle slot's KV: only the common
    /// prefix is valid, and anything the previous occupant left beyond the divergence
    /// point would collide with this request's own positions. The scheduler has no
    /// model handle, so it records the intent here and `run_loop` applies it —
    /// **before** `run_prefill`, never after. Mirrors llama-server's
    /// `common_context_seq_rm(ctx, slot.id, p0, -1)` (`server-context.cpp:3392-3399`).
    pub kv_trims: Vec<(SeqId, usize)>,
    /// Sequence IDs whose KV must be wiped entirely: a finished request whose state
    /// was not safe to reuse, or an idle slot reclaimed to free blocks.
    pub kv_clears: Vec<SeqId>,
    /// `(seq_id, tokens)` to serialise into the host-RAM prompt cache **before** the
    /// matching `kv_clears` entry wipes it. Order matters: the engine saves first,
    /// then clears, or the state is gone before it can be copied.
    pub kv_saves: Vec<(SeqId, Vec<i32>)>,
    /// `(seq_id, state_blob)` to restore into a sequence before it is prefilled.
    /// Carries the blob by value: it has just been removed from the cache, so there
    /// is exactly one copy in flight and no second reference to keep consistent.
    pub kv_restores: Vec<(SeqId, Vec<u8>)>,
}

impl ScheduledBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}
