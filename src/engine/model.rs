// Model trait and LlamaCppModel implementation.
// Uses llama.cpp FFI for GGUF loading and inference.

#[cfg(not(fox_stub))]
use std::cmp::Ordering;
#[cfg(not(fox_stub))]
use std::ffi::CString;
#[cfg(not(fox_stub))]
use std::os::raw::c_char;
#[cfg(not(fox_stub))]
use std::ptr::NonNull;
#[cfg(not(fox_stub))]
use std::sync::Arc;

#[cfg(not(fox_stub))]
use anyhow::anyhow;
use anyhow::Result;
#[cfg(not(fox_stub))]
use rand::rngs::StdRng;
#[cfg(not(fox_stub))]
use rand::{Rng, SeedableRng};

#[cfg(not(fox_stub))]
use super::ffi;

/// Model architecture configuration.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
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
    pub kv_seq_id: i32,
    /// Sampling temperature (0 = greedy, 1 = unscaled).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (1.0 = disabled).
    pub top_p: f32,
    /// Top-K filter (0 = disabled).
    pub top_k: u32,
    /// Repetition penalty (1.0 = disabled).
    pub repetition_penalty: f32,
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
    pub prefix_seq_id: Option<i32>,
}

/// Backend model trait.
pub trait Model: Send + Sync {
    /// Sync prefill (called by engine from spawn_blocking).
    /// Returns `(req_id, logits, tokens_submitted)` — `tokens_submitted` is how many tokens
    /// were actually placed in the KV cache for each request (may differ from
    /// `prompt_tokens.len()` when effective_skip > 0).
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>>;

    /// Sync decode step (called by engine from spawn_blocking).
    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    fn model_config(&self) -> ModelConfig;

    fn eos_token_id(&self) -> i32;

    /// Returns true if `token_id` is ANY end-of-generation token for this model
    /// (e.g. `<|im_end|>`, `<|endoftext|>`, etc.).  More reliable than comparing
    /// with `eos_token_id()` alone because models like Qwen3.5 have multiple EOG tokens.
    fn is_eog_token(&self, token_id: i32) -> bool;

    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    fn token_to_piece(&self, token: i32) -> Result<String>;

    /// Apply chat template to messages. Returns formatted prompt for tokenization.
    /// Fallback: simple "role: content\n" concatenation if template unavailable.
    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String>;

    /// Remove all KV cache / recurrent state for the given sequence ID.
    /// Must be called before a seq_id is reused for a new request; otherwise the new request
    /// will inherit stale positions from the previous occupant and llama_decode will fail.
    fn clear_sequence(&self, seq_id: i32);

    /// Copy `token_count` tokens worth of KV cache from `src_seq_id` to `dst_seq_id`
    /// (positions 0..token_count). Used by prefix caching: before prefilling a request whose
    /// prompt matches a completed one, we copy the KV data so only the non-cached suffix
    /// needs to be computed.
    fn copy_sequence_range(&self, src_seq_id: i32, dst_seq_id: i32, token_count: i32);

    /// Returns true if the loaded model's memory backend supports sequence copying
    /// (`llama_memory_seq_cp`).  Standard transformer (attention-only) models return true;
    /// recurrent / hybrid models (Mamba, Qwen3.5, etc.) return false.
    /// Prefix caching must be disabled when this returns false.
    fn supports_seq_copy(&self) -> bool;

    /// Return the embedding dimension (n_embd) for the model.
    fn embedding_dim(&self) -> usize;

    /// Run a forward pass in embedding mode and return the sequence embedding vector.
    /// Uses sequence slot 0; caller must not have an active inference request on slot 0.
    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>>;
}

/// Sample the highest-probability token (deterministic).
#[cfg(not(fox_stub))]
fn sample_greedy(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Apply repetition penalty in-place: divide positive logits and multiply negative ones.
#[cfg(not(fox_stub))]
fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[i32], penalty: f32) {
    for &tid in token_ids {
        if tid >= 0 && (tid as usize) < logits.len() {
            let l = logits[tid as usize];
            logits[tid as usize] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
}

/// Parameters for the full stochastic sampler.
#[cfg(not(fox_stub))]
struct SamplerParams<'a> {
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repetition_penalty: f32,
    generated_ids: &'a [i32],
    seed: Option<u64>,
    token_count: usize,
}

/// Full stochastic sampler: repetition penalty → temperature → top-K → top-P → weighted draw.
///
/// When `temperature` ≤ 0 the function falls back to greedy regardless of other parameters.
/// The RNG is seeded per-request for reproducibility when `seed` is provided.
#[cfg(not(fox_stub))]
fn sample_token(logits: &[f32], p: SamplerParams<'_>) -> i32 {
    let SamplerParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty,
        generated_ids,
        seed,
        token_count,
    } = p;
    let mut logits = logits.to_vec();

    // 1. Repetition penalty
    if repetition_penalty != 1.0 && !generated_ids.is_empty() {
        apply_repetition_penalty(&mut logits, generated_ids, repetition_penalty);
    }

    // 2. Greedy shortcut
    if temperature <= 0.0 {
        return sample_greedy(&logits);
    }

    // 3. Temperature scaling
    for l in &mut logits {
        *l /= temperature;
    }

    // 4. Top-K masking
    let k = top_k as usize;
    if k > 0 && k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = indexed[k - 1].1;
        for l in &mut logits {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // 5. Softmax + sort by descending probability
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_l).exp() / exp_sum))
        .collect();
    probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // 6. Top-P nucleus truncation
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut end = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                end = idx + 1;
                break;
            }
        }
        probs.truncate(end);
    }

    // 7. Weighted random draw
    let mut rng: Box<dyn rand::RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s ^ (token_count as u64))),
        None => Box::new(rand::thread_rng()),
    };

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let r: f32 = rng.gen::<f32>() * total;
    let mut cum = 0.0f32;
    for (idx, p) in &probs {
        cum += p;
        if cum >= r {
            return *idx as i32;
        }
    }
    probs.last().map(|(idx, _)| *idx as i32).unwrap_or(0)
}

#[cfg(not(fox_stub))]
/// Llama.cpp model via FFI.
pub struct LlamaCppModel {
    _model: NonNull<ffi::llama_model>,
    _ctx: Arc<std::sync::Mutex<NonNull<ffi::llama_context>>>,
    vocab: *const ffi::llama_vocab,
    config: ModelConfig,
    eos_token: i32,
}

#[cfg(not(fox_stub))]
impl LlamaCppModel {
    /// Load a GGUF model from path.
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
        max_context_len: u32,
    ) -> Result<Self> {
        unsafe {
            ffi::llama_backend_init();
        }

        let path_cstr = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path not valid UTF-8"))?;
        let path_c = CString::new(path_cstr)?;

        let model_params = unsafe { ffi::llama_model_default_params() };
        let model = unsafe { ffi::llama_model_load_from_file(path_c.as_ptr(), model_params) };
        let model =
            NonNull::new(model).ok_or_else(|| anyhow!("llama_model_load_from_file failed"))?;

        let vocab = unsafe { ffi::llama_model_get_vocab(model.as_ptr()) };
        if vocab.is_null() {
            unsafe { ffi::llama_model_free(model.as_ptr()) };
            anyhow::bail!("llama_model_get_vocab returned null");
        }

        let eos_token = unsafe { ffi::llama_vocab_eos(vocab) };
        let n_vocab = unsafe { ffi::llama_vocab_n_tokens(vocab) };
        let n_layer = unsafe { ffi::llama_model_n_layer(model.as_ptr()) } as usize;
        let n_head = unsafe { ffi::llama_model_n_head(model.as_ptr()) } as usize;
        let n_head_kv = unsafe { ffi::llama_model_n_head_kv(model.as_ptr()) } as usize;
        let n_embd = unsafe { ffi::llama_model_n_embd(model.as_ptr()) } as usize;
        let head_dim = if n_head > 0 { n_embd / n_head } else { 128 };

        let config = ModelConfig {
            num_layers: n_layer,
            num_heads: n_head,
            num_heads_kv: n_head_kv,
            head_dim,
            vocab_size: n_vocab as usize,
        };

        let mut ctx_params = unsafe { ffi::llama_context_default_params() };
        // n_seq_max controls how many concurrent sequences the KV cache tracks.
        // Must be >= max_batch_size for serving, and large enough for prefix-cache
        // seq_cp operations (needs at least 2 slots). Using max(max_batch_size, 4)
        // ensures n_ctx_seq = n_ctx / n_seq_max gives reasonable per-seq context.
        let n_seq = (max_batch_size as u32).max(4);
        ctx_params.n_ctx = max_context_len * n_seq;
        // n_batch must be at least as large as n_ctx to handle full prompts in one pass
        ctx_params.n_batch = max_context_len.max(max_batch_size as u32);
        ctx_params.n_seq_max = n_seq;

        let ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), ctx_params) };
        let ctx = NonNull::new(ctx).ok_or_else(|| {
            unsafe { ffi::llama_model_free(model.as_ptr()) };
            anyhow!("llama_init_from_model failed")
        })?;

        // SAFETY: We manually implement Send + Sync for LlamaCppModel below.
        // The Arc<Mutex<NonNull<...>>> is intentionally used here for shared ownership
        // across clone (e.g. future multi-backend); the unsafe impls guarantee thread safety.
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx_arc = Arc::new(std::sync::Mutex::new(ctx));
        Ok(Self {
            _model: model,
            _ctx: ctx_arc,
            vocab,
            config,
            eos_token,
        })
    }

    fn tokenize_impl(&self, text: &str) -> Result<Vec<i32>> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return Err(anyhow!("vocab is null"));
        }
        // First call with null buffer to get required size
        let n_max = text.len() + 4;
        let mut tokens: Vec<ffi::llama_token> = vec![0; n_max];
        let n = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                tokens.as_mut_ptr(),
                n_max as i32,
                true,  // add_special
                false, // parse_special
            )
        };
        if n < 0 {
            let need = (-n) as usize;
            let mut tokens: Vec<ffi::llama_token> = vec![0; need];
            let n = unsafe {
                ffi::llama_tokenize(
                    vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    need as i32,
                    true,
                    false,
                )
            };
            if n < 0 {
                return Err(anyhow!("llama_tokenize failed: {}", n));
            }
            tokens.truncate(n as usize);
            Ok(tokens)
        } else {
            tokens.truncate(n as usize);
            Ok(tokens)
        }
    }

    fn token_to_piece_impl(&self, token: i32) -> Result<String> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return Err(anyhow!("vocab is null"));
        }
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0, // lstrip: keep leading spaces so words don't concatenate (e.g. "¡Hola!¿Enquépuedo...")
                    true, // special
                )
            };
            if n < 0 {
                let need = (-n) as usize;
                buf.resize(need, 0);
                let n = unsafe {
                    ffi::llama_token_to_piece(
                        vocab,
                        token,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len() as i32,
                        0, // lstrip: keep leading spaces
                        true,
                    )
                };
                if n < 0 {
                    return Err(anyhow!("llama_token_to_piece failed: {}", n));
                }
                let s = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                return Ok(s);
            }
            if n as usize <= buf.len() {
                let s = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                return Ok(s);
            }
            buf.resize(n as usize, 0);
        }
    }

    fn apply_chat_template_impl(&self, messages: &[(String, String)]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let role_cstrings: Vec<CString> = messages
            .iter()
            .map(|(r, _)| {
                // Replace interior null bytes that would truncate the C string
                let sanitized = r.replace('\0', "");
                CString::new(sanitized).unwrap_or_else(|_| CString::new("user").unwrap())
            })
            .collect();
        let content_cstrings: Vec<CString> = messages
            .iter()
            .map(|(_, c)| {
                let sanitized = c.replace('\0', "");
                CString::new(sanitized).unwrap_or_else(|_| CString::new("").unwrap())
            })
            .collect();
        let chat: Vec<ffi::llama_chat_message> = (0..messages.len())
            .map(|i| ffi::llama_chat_message {
                role: role_cstrings[i].as_ptr(),
                content: content_cstrings[i].as_ptr(),
            })
            .collect();

        let estimated_len = messages
            .iter()
            .map(|(r, c)| r.len() + c.len() + 128)
            .sum::<usize>()
            .max(512);

        let model = self._model.as_ptr();

        // Try model's template first, then fallback names
        let templates_to_try: Vec<Option<&str>> = {
            let from_model = unsafe {
                let p = ffi::llama_model_chat_template(model, std::ptr::null());
                if p.is_null() {
                    None
                } else {
                    std::ffi::CStr::from_ptr(p).to_str().ok().map(|s| s as &str)
                }
            };
            if let Some(s) = from_model {
                vec![Some(s)]
            } else {
                vec![None]
            }
        };

        // Built-in template names - llama_chat_apply_template looks them up by name
        let fallback_names = [
            "chatml",     // Qwen, Phi, OpenHermes, many models
            "llama3",     // Meta Llama 3
            "phi3",       // Microsoft Phi-3
            "llama2",     // Meta Llama 2, Mistral
            "mistral-v1", // Mistral
            "gemma",      // Google Gemma
        ];

        let mut template_strings: Vec<String> = templates_to_try
            .into_iter()
            .filter_map(|o| o.map(String::from))
            .collect();
        for name in fallback_names {
            if !template_strings.iter().any(|s| s == name) {
                template_strings.push(name.to_string());
            }
        }

        for tmpl_str in &template_strings {
            let mut buf = vec![0u8; estimated_len];
            let tmpl_c = match CString::new(tmpl_str.as_str()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let n = unsafe {
                ffi::llama_chat_apply_template(
                    tmpl_c.as_ptr(),
                    chat.as_ptr(),
                    messages.len(),
                    true, // add_ass
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n >= 0 {
                let len = (n as usize).min(buf.len());
                let result = String::from_utf8_lossy(&buf[..len]).into_owned();
                if !result.is_empty() {
                    tracing::debug!(template = tmpl_str, "applied chat template");
                    return Ok(result);
                }
            }
            if n > 0 {
                let need = n as usize;
                buf.resize(need, 0);
                let n2 = unsafe {
                    ffi::llama_chat_apply_template(
                        tmpl_c.as_ptr(),
                        chat.as_ptr(),
                        messages.len(),
                        true,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len() as i32,
                    )
                };
                if n2 >= 0 {
                    let len = (n2 as usize).min(buf.len());
                    let result = String::from_utf8_lossy(&buf[..len]).into_owned();
                    if !result.is_empty() {
                        return Ok(result);
                    }
                }
            }
        }

        Err(anyhow!("no chat template could be applied"))
    }

    fn do_prefill(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        // Copy cached prefix KV data into each request's sequence BEFORE building the batch.
        //
        // We copy positions 0..skip_prefix_tokens-1 (exclusive of the last "cached" position) so
        // that the last prefix token is always re-submitted in the batch below.  This guarantees:
        //   (a) the logits for the boundary position are freshly computed (not stale from seq_cp),
        //   (b) total_tokens is always ≥ 1, avoiding an invalid pos=context_len decode call.
        {
            let ctx_guard = self
                ._ctx
                .lock()
                .map_err(|e| anyhow!("lock poisoned: {}", e))?;
            let ctx = ctx_guard.as_ptr();
            unsafe {
                let mem = ffi::llama_get_memory(ctx as *const _);
                if !mem.is_null() {
                    // Only attempt seq_cp if the memory backend supports it.
                    let can_copy = ffi::llama_memory_can_shift(mem);
                    for req in requests.iter() {
                        if let Some(src) = req.prefix_seq_id {
                            if !can_copy {
                                // Recurrent/hybrid model — seq_cp not supported; skip.
                                continue;
                            }
                            // copy 0..skip-1; skip-1 position will be re-submitted in the batch
                            let copy_end = req.skip_prefix_tokens.saturating_sub(1) as i32;
                            if copy_end > 0 {
                                ffi::llama_memory_seq_cp(mem, src, req.kv_seq_id, 0, copy_end);
                            }
                        }
                    }
                }
            }
        }

        // Effective start of submission: one token before skip_prefix_tokens so the boundary
        // position is always freshly computed (see seq_cp comment above).
        let effective_skip = |r: &InferenceRequestForModel| {
            r.skip_prefix_tokens
                .saturating_sub(1)
                .min(r.prompt_tokens.len())
        };

        let total_tokens: usize = requests
            .iter()
            .map(|r| r.prompt_tokens.len() - effective_skip(r))
            .sum();

        // total_tokens ≥ 1 because effective_skip < prompt_tokens.len() for any non-empty prompt.
        let n_seq_max = requests.len().max(1) as i32;

        let mut batch = unsafe { ffi::llama_batch_init(total_tokens as i32, 0, n_seq_max) };

        let mut batch_logits_indices: Vec<i32> = Vec::with_capacity(requests.len());
        for req in requests.iter() {
            let seq_id = req.kv_seq_id;
            let start = effective_skip(req);
            let tokens_to_submit = &req.prompt_tokens[start..];
            if tokens_to_submit.is_empty() {
                // Should be handled by the total_tokens == 0 branch above, but be defensive.
                batch_logits_indices.push(-1);
                continue;
            }
            for (local_pos, &token) in tokens_to_submit.iter().enumerate() {
                let abs_pos = start + local_pos;
                let idx = batch.n_tokens as usize;
                let has_logits = abs_pos == req.prompt_tokens.len() - 1;
                unsafe {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = abs_pos as i32;
                    *batch.n_seq_id.add(idx) = 1;
                    let arr = *batch.seq_id.add(idx);
                    *arr.add(0) = seq_id;
                    *batch.logits.add(idx) = if has_logits { 1i8 } else { 0i8 };
                }
                batch.n_tokens += 1;
                if has_logits {
                    batch_logits_indices.push(idx as i32);
                }
            }
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (i, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(i);
            // tokens_in_kv = tokens actually placed in the KV during this prefill call.
            // Used by the engine to set prefilled_tokens on the request so decode positions
            // are always consecutive (fixes the position gap for recurrent/hybrid models).
            let tokens_in_kv = req
                .map(|r| r.prompt_tokens.len() - effective_skip(r))
                .unwrap_or(0);
            let batch_idx = batch_logits_indices.get(i).copied().unwrap_or(-1);
            let logits_ptr = if batch_idx >= 0 {
                unsafe { ffi::llama_get_logits_ith(ctx, batch_idx) }
            } else {
                std::ptr::null_mut()
            };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token), tokens_in_kv));
                continue;
            }
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = if let Some(r) = req {
                sample_token(
                    logits_slice,
                    SamplerParams {
                        temperature: r.temperature,
                        top_p: r.top_p,
                        top_k: r.top_k,
                        repetition_penalty: r.repetition_penalty,
                        generated_ids: &r.generated_token_ids,
                        seed: r.seed,
                        token_count: r.generated_tokens,
                    },
                )
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled), tokens_in_kv));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    fn do_decode(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let n_tokens = requests.len() as i32;
        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, n_tokens) };

        for (batch_slot, req) in requests.iter().enumerate() {
            let input_token = req
                .last_token
                .or_else(|| req.prompt_tokens.last().copied())
                .unwrap_or(self.eos_token);
            // context_len = prompt_len + generated_tokens (already incremented after prefill),
            // so the correct KV position for this token is context_len - 1.
            // Example: 47-token prompt → prefill covers pos 0..46; first decode token goes
            // at pos 47 (= context_len 48 - 1), not 48.
            let pos = req.context_len as i32 - 1;
            let seq_id = req.kv_seq_id; // stable ID — never the batch slot index

            unsafe {
                *batch.token.add(batch_slot) = input_token;
                *batch.pos.add(batch_slot) = pos;
                *batch.n_seq_id.add(batch_slot) = 1;
                let arr = *batch.seq_id.add(batch_slot);
                *arr.add(0) = seq_id;
                *batch.logits.add(batch_slot) = 1i8;
            }
            batch.n_tokens += 1;
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (out_idx, &req_id) in req_ids.iter().enumerate() {
            let req = requests.get(out_idx);
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, out_idx as i32) };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token)));
                continue;
            }
            let logits_slice: &[f32] =
                unsafe { std::slice::from_raw_parts(logits_ptr, n_vocab as usize) };
            let sampled = if let Some(r) = req {
                sample_token(
                    logits_slice,
                    SamplerParams {
                        temperature: r.temperature,
                        top_p: r.top_p,
                        top_k: r.top_k,
                        repetition_penalty: r.repetition_penalty,
                        generated_ids: &r.generated_token_ids,
                        seed: r.seed,
                        token_count: r.generated_tokens,
                    },
                )
            } else {
                sample_greedy(logits_slice)
            };
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled)));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }

    fn do_get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let n_embd = self.config.num_heads * self.config.head_dim;
        let n_tokens = tokens.len() as i32;

        let mut batch = unsafe { ffi::llama_batch_init(n_tokens, 0, 1) };
        for (i, &token) in tokens.iter().enumerate() {
            unsafe {
                *batch.token.add(i) = token;
                *batch.pos.add(i) = i as i32;
                *batch.n_seq_id.add(i) = 1;
                let arr = *batch.seq_id.add(i);
                *arr.add(0) = 0; // dedicated seq slot for embeddings
                *batch.logits.add(i) = 0i8; // no logits needed for embeddings
            }
            batch.n_tokens += 1;
        }

        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        unsafe { ffi::llama_set_embeddings(ctx, true) };

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe {
                ffi::llama_set_embeddings(ctx, false);
                ffi::llama_batch_free(batch);
            }
            return Err(anyhow!("llama_decode (embeddings) failed: {}", ret));
        }

        let emb_ptr = unsafe { ffi::llama_get_embeddings_seq(ctx, 0) };
        let embeddings = if emb_ptr.is_null() {
            vec![0.0f32; n_embd]
        } else {
            unsafe { std::slice::from_raw_parts(emb_ptr, n_embd) }.to_vec()
        };

        unsafe {
            let mem = ffi::llama_get_memory(ctx as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_rm(mem, 0, 0, -1);
            }
            ffi::llama_set_embeddings(ctx, false);
            ffi::llama_batch_free(batch);
        }

        Ok(embeddings)
    }
}

#[cfg(not(fox_stub))]
unsafe impl Send for LlamaCppModel {}
#[cfg(not(fox_stub))]
unsafe impl Sync for LlamaCppModel {}

#[cfg(not(fox_stub))]
impl Model for LlamaCppModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        self.do_prefill(req_ids, requests)
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        self.do_decode(req_ids, requests)
    }

    fn model_config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn eos_token_id(&self) -> i32 {
        self.eos_token
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        unsafe { ffi::llama_vocab_is_eog(self.vocab, token_id) }
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenize_impl(text)
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        self.token_to_piece_impl(token)
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        self.apply_chat_template_impl(messages)
    }

    fn clear_sequence(&self, seq_id: i32) {
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        unsafe {
            let mem = ffi::llama_get_memory(ctx_guard.as_ptr() as *const _);
            if !mem.is_null() {
                // p0=0, p1=-1 means "remove all positions for this sequence"
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
        }
    }

    fn copy_sequence_range(&self, src_seq_id: i32, dst_seq_id: i32, token_count: i32) {
        if token_count <= 0 {
            return;
        }
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        unsafe {
            let mem = ffi::llama_get_memory(ctx_guard.as_ptr() as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_cp(mem, src_seq_id, dst_seq_id, 0, token_count);
            }
        }
    }

    fn supports_seq_copy(&self) -> bool {
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        unsafe {
            let mem = ffi::llama_get_memory(ctx_guard.as_ptr() as *const _);
            if mem.is_null() {
                return false;
            }
            // llama_memory_can_shift returns true for standard attention KV caches
            // (which also support seq_cp).  Recurrent/hybrid models return false.
            ffi::llama_memory_can_shift(mem)
        }
    }

    fn embedding_dim(&self) -> usize {
        self.config.num_heads * self.config.head_dim
    }

    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        self.do_get_embeddings(tokens)
    }
}

// ==================== Stub implementation (when FOX_SKIP_LLAMA or no llama.cpp) ====================

#[cfg(fox_stub)]
/// Stub LlamaCppModel when llama.cpp is not built.
pub struct LlamaCppModel {
    config: ModelConfig,
}

#[cfg(fox_stub)]
impl LlamaCppModel {
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
        max_context_len: u32,
    ) -> Result<Self> {
        let _ = (model_path, max_batch_size, max_context_len);
        let config = ModelConfig {
            num_layers: 32,
            num_heads: 32,
            num_heads_kv: 32,
            head_dim: 128,
            vocab_size: 32000,
        };
        Ok(Self { config })
    }
}

#[cfg(fox_stub)]
unsafe impl Send for LlamaCppModel {}
#[cfg(fox_stub)]
unsafe impl Sync for LlamaCppModel {}

#[cfg(fox_stub)]
impl Model for LlamaCppModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits, usize)>> {
        let results: Vec<_> = req_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                let tokens_in_kv = requests.get(i).map(|r| r.prompt_tokens.len()).unwrap_or(0);
                (id, Logits::new(vec![], 2), tokens_in_kv)
            })
            .collect();
        Ok(results)
    }

    fn decode_sync(
        &self,
        req_ids: &[u64],
        _requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
        let results: Vec<_> = req_ids
            .iter()
            .map(|&id| (id, Logits::new(vec![], 2)))
            .collect();
        Ok(results)
    }

    fn model_config(&self) -> ModelConfig {
        self.config.clone()
    }

    fn eos_token_id(&self) -> i32 {
        2
    }

    fn is_eog_token(&self, token_id: i32) -> bool {
        token_id == 2
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        // Stub: return byte-level tokens
        let tokens: Vec<i32> = text.bytes().map(|b| b as i32).collect();
        Ok(if tokens.is_empty() { vec![0] } else { tokens })
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        let _ = token;
        Ok(String::new())
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        Ok(messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n"))
    }

    fn clear_sequence(&self, _seq_id: i32) {}

    fn copy_sequence_range(&self, _src_seq_id: i32, _dst_seq_id: i32, _token_count: i32) {}

    fn supports_seq_copy(&self) -> bool {
        false
    }

    fn embedding_dim(&self) -> usize {
        self.config.num_heads * self.config.head_dim
    }

    fn get_embeddings(&self, tokens: &[i32]) -> Result<Vec<f32>> {
        let _ = tokens;
        Ok(vec![0.0f32; self.embedding_dim()])
    }
}

#[cfg(all(test, not(fox_stub)))]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // sample_greedy
    // -----------------------------------------------------------------------

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn greedy_last_element_wins_on_tie() {
        // Rust's Iterator::max_by returns the *last* equal maximum element.
        let logits = vec![1.0f32, 1.0, 0.5];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn greedy_handles_single_token() {
        assert_eq!(sample_greedy(&[42.0f32]), 0);
    }

    // -----------------------------------------------------------------------
    // apply_repetition_penalty
    // -----------------------------------------------------------------------

    #[test]
    fn rep_penalty_divides_positive_logits() {
        let mut logits = vec![2.0f32, 1.0, -1.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "positive logit should be halved"
        );
        assert!((logits[1] - 1.0).abs() < 1e-6, "untouched");
        assert!(
            (logits[2] - (-1.0)).abs() < 1e-6,
            "untouched negative not in token_ids"
        );
    }

    #[test]
    fn rep_penalty_multiplies_negative_logits() {
        let mut logits = vec![-1.0f32, 0.5];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(
            (logits[0] - (-2.0)).abs() < 1e-6,
            "negative logit multiplied by penalty"
        );
        assert!((logits[1] - 0.5).abs() < 1e-6, "untouched");
    }

    #[test]
    fn rep_penalty_noop_when_no_generated_tokens() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn rep_penalty_ignores_out_of_range_token_ids() {
        let original = vec![1.0f32, 2.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[99, -1], 2.0);
        assert_eq!(logits, original);
    }

    // -----------------------------------------------------------------------
    // sample_token — greedy path (temperature ≤ 0)
    // -----------------------------------------------------------------------

    #[test]
    fn sample_token_greedy_at_temperature_zero() {
        let logits = vec![0.1f32, 5.0, 0.3];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                generated_ids: &[],
                seed: None,
                token_count: 0,
            },
        );
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_token_negative_temperature_is_greedy() {
        let logits = vec![0.1f32, 0.2, 9.9];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: -1.0,
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 1.0,
                generated_ids: &[],
                seed: None,
                token_count: 0,
            },
        );
        assert_eq!(token, 2);
    }

    // -----------------------------------------------------------------------
    // sample_token — stochastic path with seeded RNG (reproducible)
    // -----------------------------------------------------------------------

    #[test]
    fn sample_token_seeded_is_reproducible() {
        let logits = vec![1.0f32, 2.0, 0.5, 1.5];
        let params = || SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            generated_ids: &[],
            seed: Some(42),
            token_count: 0,
        };
        assert_eq!(
            sample_token(&logits, params()),
            sample_token(&logits, params())
        );
    }

    #[test]
    fn sample_token_top_k_restricts_candidates() {
        // With logits heavily favouring token 3 but top_k=2, only tokens 1 and 3 are eligible
        // (they have the two highest logits). Token 0 and 2 must never be sampled.
        let logits = vec![0.0f32, 5.0, 0.0, 10.0];
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for seed in 0u64..50 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 2,
                    repetition_penalty: 1.0,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
                },
            );
            seen.insert(t);
        }
        assert!(
            !seen.contains(&0) && !seen.contains(&2),
            "tokens outside top-K window should never be sampled; got {:?}",
            seen
        );
    }

    #[test]
    fn sample_token_top_p_restricts_candidates() {
        // Token 3 has logit 10 (very dominant). With top_p = 0.5, only tokens
        // with cumulative mass ≥ 50 % survive; that should include token 3 at minimum.
        let logits = vec![0.0f32, 0.0, 0.0, 10.0];
        for seed in 0u64..20 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    temperature: 1.0,
                    top_p: 0.5,
                    top_k: 0,
                    repetition_penalty: 1.0,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
                },
            );
            assert_eq!(
                t, 3,
                "dominant token must always be sampled under top_p=0.5"
            );
        }
    }

    #[test]
    fn sample_token_repetition_penalty_reduces_repeated_token() {
        // Token 0 has the highest raw logit but we penalise it heavily.
        // After the penalty token 1 should win in greedy mode.
        let logits = vec![5.0f32, 3.0];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0, // greedy so result is deterministic
                top_p: 1.0,
                top_k: 0,
                repetition_penalty: 10.0,
                generated_ids: &[0], // token 0 was already generated
                seed: None,
                token_count: 1,
            },
        );
        assert_eq!(token, 1, "penalised token 0 should lose to token 1");
    }
}
