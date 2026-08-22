// Model trait and shared types.
//
// Sub-modules:
//   sampling   — token sampling (temperature, top-k, top-p, repetition penalty)
//   chat_template — Jinja chat-template compilation/rendering (llama.cpp-free)
//   llama_cpp  — LlamaCppModel implementation (real + fox_stub variant)
//   stub       — StubModel for tests / test-helpers feature

use anyhow::{anyhow, Result};

use crate::seq::SeqId;

pub(crate) mod chat_template;
pub(crate) mod llama_cpp;
pub(crate) mod model_info;
// Not gated on `fox_stub` despite its only caller (`llama_cpp::batch`) being
// gated: this module is pure math over `&[f32]` plus `rand`, with no llama.cpp
// dependency at all. It *was* gated, which meant `make ci` — which runs with
// FOX_SKIP_LLAMA=1 — never compiled it and never ran any of its ~60 tests. That
// is how a NaN bug in `sample_greedy` survived. `dead_code` is expected in a stub
// build: the tests are the point, not the callers.
#[cfg_attr(fox_stub, allow(dead_code))]
pub(crate) mod sampling;
#[cfg(any(test, feature = "test-helpers"))]
pub(crate) mod stub;

pub use llama_cpp::LlamaCppModel;
pub use model_info::ModelInfo;
#[cfg(any(test, feature = "test-helpers"))]
pub use stub::{StubModel, ThinkingStubModel};

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Which tool-call wire format the model's OWN chat template natively speaks, if any.
/// `Hermes`/Qwen tool-use templates instruct `<tool_call>...</tool_call>`; Mistral's
/// template instructs the `[TOOL_CALLS]` marker. Llama3 is deliberately absent here:
/// real-world GGUF chat templates for Llama3 models routinely strip the tool-calling
/// block entirely (verified against a cached `llama-3.2-1b-instruct` GGUF, whose
/// baked-in template has no tool-call convention at all), so there is no reliable
/// template signal to detect it by — it is explicit-opt-in only via
/// `--tool-call-parser llama3`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeToolFormat {
    Hermes,
    Mistral,
}

/// Literal spliced into a rendered chat prompt in place of an image content block,
/// consumed by `mtmd_tokenize` to split the prompt into text/image chunks (mirrors
/// llama.cpp server's own approach: flatten content parts to one marker-laced
/// string *before* the chat template runs, rather than teaching every Jinja
/// template about images). Fox-owned rather than `mtmd_default_marker()`'s
/// `<__media__>` only so the constant lives next to the code that emits and
/// consumes it instead of behind an FFI call.
pub const MEDIA_MARKER: &str = "<__fox_media__>";

/// Owned handle to a tokenized multimodal (text+image) prompt — mtmd's
/// `mtmd_input_chunks`. Defined unconditionally (not `#[cfg(not(fox_stub))]`) so
/// `Model::tokenize_multimodal`'s signature doesn't need a stub-only variant;
/// only the real (non-stub) `LlamaCppModel` backend ever constructs one, via
/// `from_raw` (private to that module). `Clone` is cheap (refcount bump) and
/// safe to move across the scheduler/engine boundary — the underlying chunks
/// are freed exactly once, when the last clone drops.
#[derive(Debug)]
pub struct MultimodalChunks {
    inner: std::sync::Arc<RawChunksPtr>,
}

// The field is only read in `#[cfg(not(fox_stub))]` code (Drop, n_positions) —
// a stub build never constructs one, so the field is legitimately unused there.
#[derive(Debug)]
#[cfg_attr(fox_stub, allow(dead_code))]
struct RawChunksPtr(*mut std::ffi::c_void);
unsafe impl Send for RawChunksPtr {}
unsafe impl Sync for RawChunksPtr {}

impl Drop for RawChunksPtr {
    fn drop(&mut self) {
        #[cfg(not(fox_stub))]
        unsafe {
            crate::engine::ffi::mtmd_input_chunks_free(self.0.cast());
        }
    }
}

impl Clone for MultimodalChunks {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl MultimodalChunks {
    /// # Safety
    /// `ptr` must be a valid, uniquely-owned `mtmd_input_chunks*` (e.g. freshly
    /// returned by `mtmd_input_chunks_init` + a successful `mtmd_tokenize`).
    #[cfg(not(fox_stub))]
    pub(crate) unsafe fn from_raw(ptr: *mut std::ffi::c_void) -> Self {
        Self {
            inner: std::sync::Arc::new(RawChunksPtr(ptr)),
        }
    }

    #[cfg(not(fox_stub))]
    pub(crate) fn as_raw(&self) -> *mut std::ffi::c_void {
        self.inner.0
    }

    /// Total KV positions this multimodal prompt will occupy (text + image
    /// tokens combined; for M-RoPE architectures this can differ from the raw
    /// token count) — used for scheduler block accounting exactly like
    /// `prompt_tokens.len()` is for a plain text prompt.
    pub fn n_positions(&self) -> usize {
        #[cfg(not(fox_stub))]
        {
            unsafe { crate::engine::ffi::mtmd_helper_get_n_pos(self.inner.0.cast()) as usize }
        }
        #[cfg(fox_stub)]
        {
            0
        }
    }
}

/// Sampling parameters recommended by the model's GGUF metadata.
/// Fields are `None` when the model doesn't specify a recommendation for that parameter.
#[derive(Debug, Clone)]
pub struct RecommendedSampling {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
}

/// A model's fill-in-the-middle special tokens.
///
/// FIM models are trained on a specific token order — llama.cpp's own infill path and
/// every mainstream code model use *suffix before prefix* (`[SUF] suffix [PRE] prefix
/// [MID]`), which lets the model see what it must join up to before it starts writing.
/// Emitting them prefix-first still produces fluent text, just text that ignores the
/// suffix, so the ordering is load-bearing rather than cosmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FimTokens {
    pub prefix: i32,
    pub suffix: i32,
    pub middle: i32,
}

/// Model architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    /// Embedding dimension, read from `llama_model_n_embd`. This is NOT
    /// `num_heads * head_dim` — for Gemma/MLA-class models the two differ, which
    /// is why it is stored explicitly rather than reconstructed.
    pub n_embd: usize,
    pub vocab_size: usize,
}

/// Logits from a single decode step (vocab_size floats).
#[derive(Debug, Clone)]
pub struct Logits {
    pub values: Vec<f32>,
    pub sampled_token: i32,
}

impl Logits {
    pub fn new(values: Vec<f32>, sampled_token: i32) -> Self {
        Self {
            values,
            sampled_token,
        }
    }
}

/// Result of one prefill step for a request. When the prompt is chunked
/// (`max_prefill_chunk`), prefill spans several steps, so a step reports how far it
/// advanced and carries `logits` only on the **final** chunk (once the last prompt
/// token has been submitted and sampled).
#[derive(Debug, Clone)]
pub struct PrefillStep {
    pub req_id: u64,
    /// Absolute prompt position now in the KV cache — the next chunk starts here.
    pub prefill_pos: usize,
    /// Sampled logits. `Some` only on the final chunk (prompt fully prefilled);
    /// `None` for intermediate chunks (no token is sampled yet).
    pub logits: Option<Logits>,
    /// Total prompt tokens submitted to llama.cpp for this request. Non-zero only on
    /// the final chunk; the engine records it as the request's `prefilled_tokens`.
    pub tokens_in_kv: usize,
}

/// `llama_decode` failed with ret==1 ("no KV slot for batch") even at the
/// minimum possible batch size — `batch.rs`'s bisection retry already
/// narrowed the batch down to this one request and it still doesn't fit.
/// Distinct from a generic decode failure so the engine layer (which owns
/// the `Scheduler` and `--context-shift` config that `LlamaCppModel`/
/// `batch.rs` have no access to) can attempt one targeted context roll
/// before giving up — see `docs/design/reactive-context-rolling.md`.
#[derive(Debug)]
pub(crate) struct KvCacheFullAtMinimum {
    pub req_id: u64,
}

impl std::fmt::Display for KvCacheFullAtMinimum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "llama_decode: no KV slot for request {} even at the minimum batch size",
            self.req_id
        )
    }
}

impl std::error::Error for KvCacheFullAtMinimum {}

/// Inference request (minimal view for model).
#[derive(Debug, Clone)]
pub struct InferenceRequestForModel {
    pub id: u64,
    pub prompt_tokens: Vec<i32>,
    pub last_token: Option<i32>,
    pub generated_tokens: usize,
    pub max_new_tokens: usize,
    pub context_len: usize,
    /// Stable llama.cpp sequence ID assigned at admission — never changes for the lifetime of
    /// a request. Using the batch index here would cause seq_id collisions across decode steps.
    pub kv_seq_id: SeqId,
    /// Sampling temperature (0 = greedy, 1 = unscaled).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Top-K filter (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
    /// OpenAI-style frequency penalty (additive; 0 = disabled).
    pub frequency_penalty: f32,
    /// OpenAI-style presence penalty (additive; 0 = disabled).
    pub presence_penalty: f32,
    /// Trailing window the penalties look at, in generated tokens (llama.cpp
    /// `repeat_last_n`): `-1` = whole history, `0` = disabled, `n` = last `n`.
    pub repeat_last_n: i32,
    /// Top-nσ logit cutoff in standard deviations (0 = disabled).
    pub top_n_sigma: f32,
    /// Floor on candidates left by any truncation step (0 = just the top one).
    pub min_keep: usize,
    /// RNG seed for reproducible sampling (None = random).
    pub seed: Option<u64>,
    /// Previously generated token IDs (for repetition penalty).
    pub generated_token_ids: Vec<i32>,
    /// Number of prompt tokens already in the KV cache from a prefix hit.
    /// `do_prefill` submits only `prompt_tokens[skip_prefix_tokens..]` starting at
    /// position `skip_prefix_tokens`.
    pub skip_prefix_tokens: usize,
    /// Sequence ID that holds the cached prefix KV data. When set, `do_prefill` calls
    /// `llama_memory_seq_cp` to transfer positions 0..skip_prefix_tokens before adding
    /// the remaining tokens to the batch.
    pub prefix_seq_id: Option<SeqId>,
    /// Absolute prompt position where this prefill call should start submitting tokens.
    /// Advances by up to `max_prefill_chunk` each call until it reaches
    /// `prompt_tokens.len()`. Starts at the effective skip (0, or the prefix-hit
    /// boundary). Lets a long prompt be prefilled in chunks across scheduler steps.
    pub prefill_pos: usize,
    /// GBNF grammar constraining generation (None = unconstrained). The model keeps a
    /// stateful grammar sampler per request id and, when this is set, masks forbidden
    /// tokens before sampling and advances the grammar with the chosen token.
    pub grammar: Option<std::sync::Arc<str>>,
    /// Min-P sampling threshold (0.0 = disabled).
    pub min_p: f32,
    /// Suppress end-of-generation tokens until this many tokens are generated (0 = off).
    pub min_tokens: usize,
    /// Additive per-token logit bias (OpenAI `logit_bias`).
    pub logit_bias: Option<std::sync::Arc<std::collections::HashMap<i32, f32>>>,
    /// Tokenized multimodal (text+image) prompt, when this request carries images.
    /// `prompt_tokens` is empty for such requests — `do_prefill` dispatches them to
    /// an atomic `mtmd_helper_eval_chunks` call instead of the normal token batch
    /// path (see `docs/design/vision-support.md`).
    pub multimodal: Option<MultimodalChunks>,
    /// LoRA adapter to apply while decoding this request, if any — see
    /// `docs/design/lora-support.md`. `do_prefill`/`do_decode` group requests by
    /// this value and call `llama_set_adapters_lora` once per group, since the
    /// adapter set is a property of the whole `llama_context`, not a sequence.
    pub lora: Option<crate::scheduler::LoraSelection>,
    /// Whether the caller asked for `logprobs` on this request. When `false`, the
    /// model layer skips copying the full vocab-sized logits vector into `Logits`
    /// (a real per-token cost — see `docs/design/rocm-benchmarking-2026-08.md`) since
    /// nothing downstream will read it.
    pub needs_logits: bool,
}

// ---------------------------------------------------------------------------
// Model trait
// ---------------------------------------------------------------------------

/// Backend model trait.
pub trait Model: Send + Sync {
    /// Sync prefill (called by engine from spawn_blocking).
    ///
    /// Submits at most `max_prefill_chunk` prompt tokens per request per call (0 =
    /// unbounded, single-shot), starting at each request's `prefill_pos`. Returns one
    /// [`PrefillStep`] per request reporting the new `prefill_pos` and, once the prompt
    /// is fully prefilled, the sampled `logits` and total `tokens_in_kv`. Chunking lets
    /// a long prompt interleave with other requests' decode steps instead of blocking
    /// the engine loop for the whole prefill.
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
        max_prefill_chunk: usize,
    ) -> Result<Vec<PrefillStep>>;

    /// Sync decode step (called by engine from spawn_blocking).
    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    /// Speculative decode for a single request: verify the given `drafts` (already
    /// proposed by an `engine::speculative::Proposer` — n-gram lookup or a draft
    /// model) in one pass, and return the committed tokens' logits (always ≥ 1).
    /// Exactness holds regardless of where `drafts` came from: every committed token
    /// is still sampled from THIS model's own logits; a wrong draft is simply
    /// rejected. The default performs an ordinary single-token decode (ignoring
    /// `drafts`), so the stub and non-speculative backends still work.
    fn speculative_decode_sync(
        &self,
        req_id: u64,
        request: &InferenceRequestForModel,
        drafts: Vec<i32>,
    ) -> Result<Vec<Logits>> {
        let _ = drafts;
        let out = self.decode_sync(&[req_id], std::slice::from_ref(request))?;
        Ok(out.into_iter().map(|(_, l)| l).collect())
    }

    /// Whether this model carries a trained multi-token-prediction head, attached at
    /// load time from a paired `mtp-*.gguf`. Default: no.
    fn has_mtp(&self) -> bool {
        false
    }

    /// Tell the MTP head a new generation is starting on `seq_id` with `prompt`.
    /// Default: nothing to tell.
    fn mtp_begin(&self, seq_id: SeqId, prompt: &[i32]) {
        let _ = (seq_id, prompt);
    }

    /// Draft up to `draft_len` tokens from the MTP head, continuing after `id_last` at
    /// position `n_past`. `seq` is the request's full logical sequence so far.
    ///
    /// Only ever called on a model whose `has_mtp` is true. Default: empty, which the
    /// caller treats as "no draft this step" and falls back to an ordinary decode.
    fn mtp_propose(
        &self,
        seq_id: SeqId,
        n_past: i32,
        id_last: i32,
        seq: &[i32],
        draft_len: usize,
    ) -> Vec<i32> {
        let _ = (seq_id, n_past, id_last, seq, draft_len);
        Vec::new()
    }

    /// Report how many drafted tokens the target accepted, so the head can keep its
    /// per-sequence state aligned with what was actually committed.
    fn mtp_accept(&self, seq_id: SeqId, n_accepted: u16) {
        let _ = (seq_id, n_accepted);
    }

    /// Feed `new_tokens` into this model's own KV at `seq_id` (starting at `base_pos`),
    /// then greedily (no penalty context — a draft proposer only needs to be
    /// plausible, not calibrated) decode up to `draft_len` further tokens, extending
    /// the same KV sequence for the next call. Stops early on an end-of-generation
    /// token. Used only by `engine::speculative::DraftModelProposer` (0.16
    /// draft-model speculation) — a model not loaded as a draft never calls this.
    /// Default: empty (stubs and models not acting as a draft don't implement it).
    fn draft_propose(
        &self,
        seq_id: SeqId,
        new_tokens: &[i32],
        base_pos: i32,
        draft_len: usize,
    ) -> Vec<i32> {
        let _ = (seq_id, new_tokens, base_pos, draft_len);
        Vec::new()
    }

    /// A hash identifying this model's tokenizer (vocab size, BOS/EOS, every token's
    /// piece text). Two models with the same fingerprint share a tokenizer — the
    /// precondition draft-model speculation requires (a draft token id is meaningless
    /// input to the target's verify batch if the tokenizers differ). Default: `0`,
    /// meaning "unknown/unchecked" (stubs never load two real tokenizers to compare).
    fn vocab_fingerprint(&self) -> u64 {
        0
    }

    /// Lifetime count of batch-size-bisection retries triggered by a recoverable
    /// `llama_decode` failure ("no KV slot for batch") during prefill/decode.
    /// Default 0 — only `LlamaCppModel` (real) tracks this.
    fn bisection_retry_count(&self) -> u64 {
        0
    }

    fn model_config(&self) -> ModelConfig;

    fn eos_token_id(&self) -> i32;

    /// Returns true if `token_id` is ANY end-of-generation token for this model
    /// (e.g. `<|im_end|>`, `<|endoftext|>`, etc.).  More reliable than comparing
    /// with `eos_token_id()` alone because models like Qwen3.5 have multiple EOG tokens.
    fn is_eog_token(&self, token_id: i32) -> bool;

    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    /// The model's fill-in-the-middle special tokens, if it has them:
    /// `(prefix, suffix, middle)`. `None` for any model not trained for FIM — most
    /// chat models — which is what `/infill` checks before accepting a request.
    ///
    /// These are a property of the vocabulary, not something a prompt can fake: a
    /// model without them has no notion of "generate between these two spans", so
    /// synthesising a prompt would produce plausible-looking nonsense. Default `None`
    /// so the stub and any non-llama.cpp model report honestly.
    fn fim_tokens(&self) -> Option<FimTokens> {
        None
    }

    fn token_to_piece(&self, token: i32) -> Result<String>;

    /// Returns the raw bytes produced by `llama_token_to_piece` without UTF-8
    /// validation or lossy replacement.  Used by the engine to accumulate
    /// per-request byte buffers so that multi-token UTF-8 sequences (e.g. emoji
    /// split across BPE tokens) are decoded correctly.
    ///
    /// The default implementation encodes the `token_to_piece` String back to
    /// bytes, which is safe for stub/mock models that already return valid UTF-8.
    /// `LlamaCppModel` overrides this to return the actual raw C bytes.
    fn token_to_piece_bytes(&self, token: i32) -> Vec<u8> {
        self.token_to_piece(token).unwrap_or_default().into_bytes()
    }

    /// Apply chat template to messages. Returns formatted prompt for tokenization.
    /// Fallback: simple "role: content\n" concatenation if template unavailable.
    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String>;

    /// Build the final prompt token ids for a chat request. Backends with a real
    /// Jinja template (`LlamaCppModel`) execute it — threading `enable_thinking`,
    /// emitting the model's real control tokens, and tokenizing them AS special
    /// tokens (not literal text). `tools` (OpenAI-shaped tool definitions, as JSON)
    /// is threaded into the template context so a model whose real template has
    /// native tool-formatting macros (e.g. Hermes/Qwen tool-use templates) renders
    /// its own tool listing; the default implementation ignores it. The default
    /// applies `apply_chat_template`, adds a `<think>` prefill when
    /// `enable_thinking`, and tokenizes generically.
    fn build_prompt_tokens(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
    ) -> Result<Vec<i32>> {
        let _ = tools;
        let mut prompt = self.apply_chat_template(messages)?;
        if enable_thinking {
            prompt.push_str("<think>\n");
        }
        self.tokenize(&prompt)
    }

    /// Which tool-call wire format the model's own chat template natively speaks (see
    /// [`NativeToolFormat`]), as opposed to relying on fox's generic prompt-injected
    /// tool listing. Default `None` — only `LlamaCppModel` can inspect a real template.
    fn native_tool_call_format(&self) -> Option<NativeToolFormat> {
        None
    }

    /// Effective per-sequence context length (tokens) this model was configured with.
    /// For `LlamaCppModel` this is the value used in `llama_init_from_model`.
    fn context_len(&self) -> u32 {
        4096
    }

    /// Total KV-cache capacity in tokens, as ACTUALLY allocated by the backend
    /// (llama.cpp `llama_n_ctx`). fox's paged block pool must be sized from this
    /// — never from a hand-rolled `n_head_kv * head_dim * n_layer` formula, which
    /// is wrong for Gemma (shared/SWA KV), MLA (latent KV) and recurrent models.
    /// Sizing from the real capacity guarantees the pool never claims room the
    /// backend doesn't have (the "fox thinks there's room → llama_decode fails"
    /// class of crashes). Default: the per-sequence context length (stubs/mocks).
    fn kv_cache_capacity(&self) -> usize {
        self.context_len() as usize
    }

    /// Returns `true` when the model has native thinking support — i.e. `<think>` is a
    /// single special token in the vocabulary (Qwen3, DeepSeek-R1, etc.).
    /// Models without native thinking always return `false`.
    fn supports_thinking(&self) -> bool {
        false
    }

    /// Returns `true` when a vision projector (mmproj) is loaded alongside this
    /// model, i.e. image content in a request can actually be encoded. Models
    /// loaded without a paired mmproj — the overwhelming majority — return `false`.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Names of the LoRA adapters loaded alongside this model (`--lora-modules`),
    /// available for a client to select via the `model` field. Empty for the
    /// overwhelming majority of models (no adapters configured).
    fn lora_adapter_names(&self) -> Vec<String> {
        Vec::new()
    }

    /// Render the chat prompt (same inputs as `build_prompt_tokens`) and tokenize
    /// it together with `images` via mtmd, splitting on `MEDIA_MARKER` occurrences
    /// in the rendered text. Callers should check `supports_vision()` first — the
    /// default errors, since only a model loaded with a paired mmproj can do this.
    fn tokenize_multimodal(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
        images: &[Vec<u8>],
    ) -> Result<MultimodalChunks> {
        let _ = (messages, enable_thinking, tools, images);
        Err(anyhow!("model has no vision support (no mmproj loaded)"))
    }

    /// Return sampling parameters recommended by the model's GGUF metadata, if any.
    /// Returns `None` when the model file contains no sampling hints.
    fn recommended_sampling(&self) -> Option<RecommendedSampling> {
        None
    }

    /// Remove all KV cache / recurrent state for the given sequence ID.
    /// Must be called before a seq_id is reused for a new request; otherwise the new request
    /// will inherit stale positions from the previous occupant and llama_decode will fail.
    fn clear_sequence(&self, seq_id: SeqId);

    /// Remove the KV cells of `seq_id` from position `from_pos` (inclusive) onward,
    /// keeping `[0, from_pos)`. Used when a finished request donates its prompt prefix
    /// to the prefix cache: the donated sequence must hold EXACTLY the cached prefix —
    /// leaving the old tail (rest of prompt + generated tokens) in place makes the next
    /// occupant re-submit tokens at positions that already have cells, which
    /// llama_decode rejects. Default: no-op (stubs have no real KV).
    /// Returns whether the trim actually happened. It can legitimately fail: on
    /// recurrent and hybrid caches a partial rollback is only possible while the
    /// distance is within the retained snapshot window (`n_rs_seq`,
    /// `llama-memory-recurrent.cpp:181`).
    ///
    /// **`false` is reliable; `true` is not.** llama.cpp only performs that range check
    /// while the sequence's tail cell is live. A preceding full clear sets `tail = -1`
    /// ("invalidate tails which will be cleared"), and from then on a partial rollback
    /// skips the check entirely, falls through, and returns `true` having rewound
    /// nothing. Measured on Qwen3.8-27B on 2026-08-17: a 60-token rollback with
    /// `n_rs_seq = 4` returned `true`, and every request after it answered with a bare
    /// EOS. So the `false` branch is worth handling (callers re-prefill), but it is a
    /// backstop, not the guard — bound the distance up front via
    /// [`Self::rollback_budget`] instead.
    #[must_use = "`true` from trim_sequence is not proof the rollback happened — see `rollback_budget`; `false` means the caller must re-prefill. `f5214df` records what discarding this cost."]
    fn trim_sequence(&self, _seq_id: SeqId, _from_pos: usize) -> bool {
        true
    }

    /// Copy `token_count` tokens worth of KV cache from `src_seq_id` to `dst_seq_id`
    /// (positions 0..token_count). Used by prefix caching: before prefilling a request whose
    /// prompt matches a completed one, we copy the KV data so only the non-cached suffix
    /// needs to be computed.
    fn copy_sequence_range(&self, src_seq_id: SeqId, dst_seq_id: SeqId, token_count: i32);

    /// Roll a sequence's KV window when it fills the context: discard the `n_discard`
    /// oldest tokens *after* the preserved head of `n_keep` tokens (e.g. BOS + system
    /// prompt) and shift the survivors down by `n_discard`, so decode can continue past
    /// `n_ctx` instead of `llama_decode` failing. Requires a shiftable KV cache
    /// (`supports_seq_copy()` / `llama_memory_can_shift`); the caller only invokes this
    /// for models where that holds. Default: no-op success (stub models never reach the
    /// context limit in tests).
    fn roll_context(&self, _seq_id: SeqId, _n_keep: usize, _n_discard: usize) -> Result<()> {
        Ok(())
    }

    /// Free the per-request grammar sampler (guided decoding), if any. Called by the
    /// engine on every terminal path a request can take (completion, length, stop,
    /// disconnect) so grammar samplers never leak. A no-op for requests without a
    /// grammar and for backends that don't do constrained decoding (the default).
    fn free_grammar(&self, _req_id: u64) {}

    /// Returns true if the loaded model's memory backend supports sequence copying
    /// (`llama_memory_seq_cp`).  Standard transformer (attention-only) models return true;
    /// recurrent / hybrid models (Mamba, Qwen3.5, etc.) return false.
    /// Prefix caching must be disabled when this returns false.
    fn supports_seq_copy(&self) -> bool;

    /// Whether a request may inherit the KV a sequence *already holds* and skip
    /// prefilling those tokens.
    ///
    /// Deliberately separate from [`Self::supports_seq_copy`], which asks a stricter
    /// question — may KV be copied *between* sequences. Conflating the two cost fox
    /// prompt reuse on every hybrid model: `supports_seq_copy()` is false for them
    /// (`seq_cp` on recurrent state is not a partial operation), and gating slot reuse
    /// on the same flag disabled the kind that needs no copy at all. `llama-server`
    /// does exactly that kind on the same architectures and the same llama.cpp —
    /// measured reusing 14680 tokens on Qwen3.5-9B where fox reused none.
    ///
    /// Default true, because the question this asks really is "can KV be inherited at
    /// all", and almost every model can.
    ///
    /// It is NOT the whole guard. This method says nothing about how far back a given
    /// offer would have to rewind, and an earlier version of this comment claimed
    /// [`Self::trim_sequence`]'s return value covered that at the point of use. It does
    /// not — see [`Self::rollback_budget`], which is the distance check and must run
    /// *before* the offer is accepted.
    fn supports_slot_reuse(&self) -> bool {
        true
    }

    /// How far back this model's memory can actually be rewound, in tokens.
    ///
    /// `None` means unbounded — an attention KV cache drops cells at any position, so
    /// any divergence point is reachable. A recurrent or hybrid cache returns
    /// `Some(n_rs_seq)`: it only keeps that many per-token state snapshots, and a
    /// rollback further than that is impossible.
    ///
    /// This exists because [`Self::trim_sequence`]'s return value is **not** a
    /// sufficient guard, contrary to what its own docs assumed.
    /// `llama_memory_recurrent::seq_rm` only range-checks the rollback while the
    /// sequence's tail cell is live; a preceding full clear sets `tail = -1`
    /// (`llama-memory-recurrent.cpp`, "invalidate tails which will be cleared"), after
    /// which a partial rollback skips the check entirely and reports success without
    /// rewinding anything. Measured on Qwen3.8-27B (`qwen35`, hybrid) on 2026-08-17: a
    /// 60-position rollback with `n_rs_seq = 4` returned `true`, and every request from
    /// the next one on came back empty. Callers must therefore bound the distance
    /// *before* deciding to reuse, not detect the failure afterwards.
    fn rollback_budget(&self) -> Option<usize> {
        None
    }

    /// Return the embedding dimension (n_embd) for the model.
    fn embedding_dim(&self) -> usize;

    /// Run a forward pass in embedding mode and return the sequence embedding vector.
    /// Uses sequence slot 0; caller must not have an active inference request on slot 0.
    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>>;

    /// Serialise one sequence's KV state to host memory.
    ///
    /// The blob is opaque and only valid for *this* model in *this* process — it
    /// encodes cell layout, not just tokens. Restoring it into a different model, or
    /// after the context was recreated, is undefined; the caller must key it by model
    /// and drop it on unload.
    ///
    /// `Err` when the backend cannot serialise (the stub) or the sequence is empty.
    fn state_seq_save(&self, _seq_id: SeqId) -> Result<Vec<u8>> {
        anyhow::bail!("this model backend cannot serialise sequence state")
    }

    /// Restore a blob produced by [`Self::state_seq_save`] into `seq_id`, returning the
    /// number of bytes consumed. The destination sequence must be empty: llama.cpp
    /// writes the cells at their original positions and does not clear first.
    fn state_seq_load(&self, _seq_id: SeqId, _data: &[u8]) -> Result<usize> {
        anyhow::bail!("this model backend cannot restore sequence state")
    }

    /// Score one `(query, document)` pair for reranking: a single relevance number
    /// read from the model's classification head.
    ///
    /// Only a reranker model can answer this. fox never sets `pooling_type`, so it
    /// stays `UNSPECIFIED` and llama.cpp resolves it from the model's own metadata
    /// (`llama-context.cpp:182-188`) — `RANK` for a reranker, `NONE` for everything
    /// else. Under `NONE`, `llama_get_embeddings_seq` returns NULL, which is exactly
    /// the signal used to reject a non-reranker model with a clear error instead of
    /// inventing a score from a mean-pooled vector.
    ///
    /// `Err` when the model is not a reranker or the forward pass fails.
    fn rerank_score(&self, _tokens: &[i32]) -> Result<f32> {
        anyhow::bail!("this model backend does not support reranking")
    }

    /// The vocabulary's separator token, used to join query and document in the
    /// rerank prompt. `None` when the model has no SEP.
    fn sep_token_id(&self) -> Option<i32> {
        None
    }

    /// Return the text forms of the model's EOS and EOT tokens.
    /// Used as base stop sequences so generation halts on model-native terminators
    /// even when the token ID is not caught by `is_eog_token`.
    fn stop_tokens(&self) -> Vec<String>;

    /// The (open, close) delimiters that wrap this model's reasoning in its
    /// generated output, when it uses non-default markers. `None` means the
    /// standard `<think>…</think>` (the output filter's default). `LlamaCppModel`
    /// detects e.g. Gemma's channel format (`<|channel>…<channel|>`) from the
    /// chat template, so the filter separates reasoning from the answer correctly.
    fn reasoning_delimiters(&self) -> Option<(String, String)> {
        None
    }

    /// Human description of the compute backend this model runs on (e.g.
    /// `Vulkan0 — AMD Radeon 890M`, or `CPU`). Reported at startup so users can see
    /// whether inference is GPU-accelerated. Default: `"cpu"` (stubs/mocks).
    fn active_backend(&self) -> String {
        "cpu".to_string()
    }

    /// Build a `ModelInfo` snapshot of this model's facts (used by `fox probe`).
    ///
    /// The default implementation assembles what it can from the generic trait
    /// methods; backends with direct GGUF access (`LlamaCppModel`) override it to
    /// report metadata-derived truth (real arch name, `n_embd`, trained context,
    /// embedded-template presence) instead of the reconstructed values.
    fn model_info(&self) -> ModelInfo {
        let c = self.model_config();
        let arch_name = "unknown".to_string();
        let supports_seq_copy = self.supports_seq_copy();
        ModelInfo {
            kv_memory_class: model_info::classify_kv_memory(&arch_name, supports_seq_copy),
            arch_name,
            backend: self.active_backend(),
            n_embd: self.embedding_dim(),
            n_head: c.num_heads,
            n_head_kv: c.num_heads_kv,
            head_dim: c.head_dim,
            n_layer: c.num_layers,
            n_ctx_train: self.context_len(),
            effective_ctx: self.context_len(),
            vocab_size: c.vocab_size,
            n_params: 0, // backends that cannot report it say so, rather than guessing
            eos_token_id: self.eos_token_id(),
            has_chat_template: false,
            supports_thinking: self.supports_thinking(),
            supports_seq_copy,
            stop_token_count: self.stop_tokens().len(),
            recommended_sampling: self.recommended_sampling(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheFullAtMinimum;

    #[test]
    fn kv_cache_full_at_minimum_downcasts_from_anyhow_error() {
        let err: anyhow::Error = anyhow::Error::new(KvCacheFullAtMinimum { req_id: 42 });
        let downcast = err
            .downcast_ref::<KvCacheFullAtMinimum>()
            .expect("must downcast back to KvCacheFullAtMinimum");
        assert_eq!(downcast.req_id, 42);
    }

    #[test]
    fn kv_cache_full_at_minimum_display_mentions_req_id() {
        let err = KvCacheFullAtMinimum { req_id: 7 };
        assert!(err.to_string().contains('7'));
    }

    #[test]
    fn unrelated_error_does_not_downcast() {
        let err = anyhow::anyhow!("some other decode failure");
        assert!(err.downcast_ref::<KvCacheFullAtMinimum>().is_none());
    }
}
