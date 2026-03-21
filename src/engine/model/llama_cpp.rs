// LlamaCppModel — llama.cpp FFI backend.
//
// This file contains two conditional implementations:
//   - Real build (#[cfg(not(fox_stub))]): full FFI, loading, batching, sampling.
//   - Stub build (#[cfg(fox_stub)]):       no-op placeholder for CI / stub builds.

// ---------------------------------------------------------------------------
// Real implementation
// ---------------------------------------------------------------------------

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
use super::super::ffi;
#[cfg(not(fox_stub))]
use super::sampling::{sample_greedy, sample_token, SamplerParams};
#[cfg(not(fox_stub))]
use super::{InferenceRequestForModel, Logits, Model, ModelConfig};

/// SentencePiece uses U+2581 (▁) for word boundaries.
#[cfg(not(fox_stub))]
const SPM_SPACE: char = '\u{2581}';

/// Query current free GPU memory in bytes via nvidia-smi.
/// Returns None on CPU-only systems or when nvidia-smi is unavailable.
#[cfg(not(fox_stub))]
fn query_gpu_free_bytes() -> Option<usize> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mib: usize = std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()?;
    Some(mib * 1024 * 1024)
}

/// Choose the effective per-sequence context length.
///
/// Returns `user_limit` when the user specified one explicitly, otherwise
/// falls back to `model_train_ctx` (the context the model was trained with).
pub(crate) fn resolve_context_len(user_limit: Option<u32>, model_train_ctx: u32) -> u32 {
    user_limit.unwrap_or(model_train_ctx)
}

#[cfg(not(fox_stub))]
/// Llama.cpp model via FFI.
pub struct LlamaCppModel {
    _model: NonNull<ffi::llama_model>,
    _ctx: Arc<std::sync::Mutex<NonNull<ffi::llama_context>>>,
    vocab: *const ffi::llama_vocab,
    config: ModelConfig,
    eos_token: i32,
    /// Effective per-sequence context length (tokens) used when creating the llama.cpp context.
    effective_ctx: u32,
}

#[cfg(not(fox_stub))]
impl LlamaCppModel {
    /// Load a GGUF model from path.
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
        max_context_len: Option<u32>,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        type_kv: u32,
    ) -> Result<Self> {
        // Suppress llama.cpp's verbose loading output (tensor info, repack, etc.).
        // Fox shows its own clean progress spinner instead.
        unsafe extern "C" fn noop_log(
            _level: ffi::ggml_log_level,
            _text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
        }
        unsafe { ffi::llama_log_set(Some(noop_log), std::ptr::null_mut()) };

        // Load GPU/CPU backends compiled as dynamic libraries (GGML_BACKEND_DL).
        // Passing null searches the executable's directory and cwd — fox ships
        // libggml-cuda.so and libggml-cpu.so next to the binary.
        // On non-DL builds this is a no-op (backends are statically linked).
        unsafe { ffi::ggml_backend_load_all_from_path(std::ptr::null()) };

        unsafe {
            ffi::llama_backend_init();
        }

        let path_cstr = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path not valid UTF-8"))?;
        let path_c = CString::new(path_cstr)?;

        let mut model_params = unsafe { ffi::llama_model_default_params() };
        // Offload all layers to GPU (-1 = all). On CPU-only builds llama.cpp ignores this.
        model_params.n_gpu_layers = -1;
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
        let n_seq = (max_batch_size as u32).max(4);

        // Resolve effective per-sequence context: use the user's explicit limit, or
        // auto-detect from the model's trained context length (llama_model_n_ctx_train).
        let model_train_ctx =
            unsafe { ffi::llama_model_n_ctx_train(model.as_ptr()) } as u32;
        let effective_max_ctx = resolve_context_len(max_context_len, model_train_ctx);
        if max_context_len.is_none() {
            tracing::info!(
                model_train_ctx,
                effective_ctx = effective_max_ctx,
                "auto context: using model's trained context length"
            );
        }

        // Cap total KV context to fit in available GPU (or RAM) memory.
        // Query FREE memory now (after model weights are loaded) so we don't OOM.
        // Falls back to gpu_memory_bytes * fraction if nvidia-smi is unavailable.
        let free_bytes = query_gpu_free_bytes()
            .unwrap_or((gpu_memory_bytes as f64 * gpu_memory_fraction as f64) as usize);
        let budget_bytes = (free_bytes as f64 * gpu_memory_fraction as f64) as usize;
        // bytes_per_token = 2 (K+V) * n_head_kv * head_dim * 2 (fp16) * n_layer
        let bytes_per_token = 2 * n_head_kv * head_dim * 2 * n_layer;
        let max_tokens_by_mem = if bytes_per_token > 0 && budget_bytes > 0 {
            (budget_bytes / bytes_per_token) as u32
        } else {
            effective_max_ctx * n_seq
        };
        // Honour the effective_max_ctx per sequence, but don't exceed memory budget.
        let n_ctx = (effective_max_ctx * n_seq)
            .min(max_tokens_by_mem)
            .max(effective_max_ctx);
        ctx_params.n_ctx = n_ctx;
        // n_batch must be at least as large as n_ctx to handle full prompts in one pass
        ctx_params.n_batch = effective_max_ctx.max(max_batch_size as u32);
        ctx_params.n_seq_max = n_seq;
        ctx_params.flash_attn_type = 1; // LLAMA_FLASH_ATTN_TYPE_ENABLED
        ctx_params.offload_kqv = true;
        ctx_params.type_k = type_kv as _;
        ctx_params.type_v = type_kv as _;

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
            effective_ctx: effective_max_ctx,
        })
    }

    /// Read a single GGUF metadata value by key name.
    /// Returns `None` when the key is absent or cannot be decoded as UTF-8.
    fn read_meta_str(&self, key: &str) -> Option<String> {
        let key_c = CString::new(key).ok()?;
        let mut buf = vec![0u8; 512];
        let n = unsafe {
            ffi::llama_model_meta_val_str(
                self._model.as_ptr(),
                key_c.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if n < 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }

    /// Read a GGUF metadata value as `f32`. Returns `None` when missing or not parseable.
    fn read_meta_f32(&self, key: &str) -> Option<f32> {
        self.read_meta_str(key)?.trim().parse::<f32>().ok()
    }

    /// Read a GGUF metadata value as `u32`. Returns `None` when missing or not parseable.
    fn read_meta_u32(&self, key: &str) -> Option<u32> {
        self.read_meta_str(key)?.trim().parse::<u32>().ok()
    }

    /// Iterate all GGUF metadata keys/values and look for sampling-related hints.
    /// Logs all keys at TRACE level. Returns a partial `RecommendedSampling`.
    fn read_sampling_from_meta(&self) -> super::RecommendedSampling {
        let count = unsafe { ffi::llama_model_meta_count(self._model.as_ptr()) };
        let mut temperature: Option<f32> = None;
        let mut top_p: Option<f32> = None;
        let mut top_k: Option<u32> = None;

        let mut key_buf = vec![0u8; 256];
        let mut val_buf = vec![0u8; 512];

        for i in 0..count {
            let kn = unsafe {
                ffi::llama_model_meta_key_by_index(
                    self._model.as_ptr(),
                    i,
                    key_buf.as_mut_ptr() as *mut c_char,
                    key_buf.len(),
                )
            };
            let vn = unsafe {
                ffi::llama_model_meta_val_str_by_index(
                    self._model.as_ptr(),
                    i,
                    val_buf.as_mut_ptr() as *mut c_char,
                    val_buf.len(),
                )
            };
            if kn < 0 || vn < 0 {
                continue;
            }
            let key = String::from_utf8_lossy(&key_buf[..kn as usize]).into_owned();
            let val = String::from_utf8_lossy(&val_buf[..vn as usize]).into_owned();
            tracing::trace!(key = %key, value = %val, "GGUF metadata");

            let key_lc = key.to_lowercase();
            if temperature.is_none() && key_lc.contains("temperature") {
                temperature = val.trim().parse::<f32>().ok();
            }
            if top_p.is_none() && key_lc.contains("top_p") {
                top_p = val.trim().parse::<f32>().ok();
            }
            if top_k.is_none() && key_lc.contains("top_k") {
                top_k = val.trim().parse::<u32>().ok();
            }
        }

        super::RecommendedSampling { temperature, top_p, top_k }
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

    /// Raw-bytes variant: same as `token_to_piece_impl` but returns `Vec<u8>`
    /// without any UTF-8 validation so the engine can accumulate partial
    /// multi-byte sequences (e.g. emoji split across BPE tokens) before decoding.
    fn token_to_piece_bytes_impl(&self, token: i32) -> Vec<u8> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return vec![];
        }
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n < 0 {
                let need = (-n) as usize;
                buf.resize(need, 0);
                let n2 = unsafe {
                    ffi::llama_token_to_piece(
                        vocab,
                        token,
                        buf.as_mut_ptr() as *mut std::ffi::c_char,
                        buf.len() as i32,
                        0,
                        true,
                    )
                };
                if n2 < 0 {
                    return vec![];
                }
                return buf[..n2 as usize].to_vec();
            }
            if (n as usize) <= buf.len() {
                return buf[..n as usize].to_vec();
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
            if n >= 0 && (n as usize) <= buf.len() {
                // Buffer was large enough — return the complete result.
                let result = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                if !result.is_empty() {
                    tracing::debug!(template = tmpl_str, "applied chat template");
                    return Ok(result);
                }
            } else if n > 0 {
                // Buffer was too small — resize to the exact size needed and retry.
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
                if n2 >= 0 && (n2 as usize) <= buf.len() {
                    let result = String::from_utf8_lossy(&buf[..n2 as usize]).into_owned();
                    if !result.is_empty() {
                        tracing::debug!(template = tmpl_str, "applied chat template (resized)");
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

    fn token_to_piece_bytes(&self, token: i32) -> Vec<u8> {
        self.token_to_piece_bytes_impl(token)
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        self.apply_chat_template_impl(messages)
    }

    fn context_len(&self) -> u32 {
        self.effective_ctx
    }

    fn supports_thinking(&self) -> bool {
        // Reasoning models (Qwen3, DeepSeek-R1, …) have `<think>` as a single
        // special token.  Tokenising it with add_special=true produces at most
        // [BOS, <think>] (2 tokens).  Non-reasoning models split it into many
        // character/subword pieces, so the count will be higher.
        self.tokenize_impl("<think>")
            .map(|t| t.len() <= 2)
            .unwrap_or(false)
    }

    fn recommended_sampling(&self) -> Option<super::RecommendedSampling> {
        let rec = self.read_sampling_from_meta();
        // Return Some only if at least one parameter was found in the metadata.
        if rec.temperature.is_some() || rec.top_p.is_some() || rec.top_k.is_some() {
            Some(rec)
        } else {
            None
        }
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

    fn stop_tokens(&self) -> Vec<String> {
        let mut result: Vec<String> = Vec::new();
        // Collect the text form of every control OR EOG token in the vocabulary.
        //
        // This covers:
        //   - Control tokens: role separators (`<|user|>`, `<|system|>`, …)
        //   - EOG tokens: EOS/EOT variants (`<|endoftext|>`, `<|im_end|>`, model-specific
        //     stop markers like `,<!__EOF teleport>`, etc.)
        //
        // `is_eog_token()` suppresses these by token ID when llama.cpp recognises them
        // correctly.  Adding their text forms here ensures the text-based filter also
        // catches them when the model spells out the stop sequence as regular tokens
        // (which happens on some quants where the EOG flag is missing from metadata).
        let n_tokens = unsafe { ffi::llama_vocab_n_tokens(self.vocab) };
        for token_id in 0..n_tokens {
            let is_control = unsafe { ffi::llama_vocab_is_control(self.vocab, token_id) };
            let is_eog = unsafe { ffi::llama_vocab_is_eog(self.vocab, token_id) };
            if !is_control && !is_eog {
                continue;
            }
            if let Ok(s) = self.token_to_piece_impl(token_id) {
                let s = s.replace(SPM_SPACE, " ");
                let s = s.trim().to_string();
                if !s.is_empty() && !result.contains(&s) {
                    result.push(s);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Stub implementation (when fox_stub cfg is set)
// ---------------------------------------------------------------------------

#[cfg(fox_stub)]
use super::{InferenceRequestForModel, Logits, Model, ModelConfig};

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
        max_context_len: Option<u32>,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        type_kv: u32,
    ) -> Result<Self> {
        let _ = (
            model_path,
            max_batch_size,
            max_context_len,
            gpu_memory_bytes,
            gpu_memory_fraction,
            type_kv,
        );
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

    fn stop_tokens(&self) -> Vec<String> {
        vec![]
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::resolve_context_len;

    // --- resolve_context_len ---

    #[test]
    fn auto_uses_model_trained_ctx() {
        // When user passes None, the model's trained context is used as-is.
        assert_eq!(resolve_context_len(None, 8192), 8192);
    }

    #[test]
    fn auto_uses_model_trained_ctx_large() {
        // Works for models with very large trained context (e.g. Qwen2.5 128k).
        assert_eq!(resolve_context_len(None, 131_072), 131_072);
    }

    #[test]
    fn explicit_limit_overrides_model_ctx() {
        // When the user specifies a value, it takes priority over the model's context.
        assert_eq!(resolve_context_len(Some(4096), 131_072), 4096);
    }

    #[test]
    fn explicit_limit_equal_to_model_ctx() {
        // Setting the limit to exactly the model's trained context is valid.
        assert_eq!(resolve_context_len(Some(8192), 8192), 8192);
    }

    #[test]
    fn explicit_limit_larger_than_model_ctx() {
        // User can request more than trained context; model/memory budget caps it later.
        assert_eq!(resolve_context_len(Some(16_384), 8192), 16_384);
    }

    #[test]
    fn explicit_limit_of_one_is_respected() {
        // Pathological but valid: user forces a minimal context.
        assert_eq!(resolve_context_len(Some(1), 32_768), 1);
    }
}
