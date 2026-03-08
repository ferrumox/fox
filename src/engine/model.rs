// Model trait and LlamaCppModel implementation.
// Uses llama.cpp FFI for GGUF loading and inference.

use std::ffi::CString;
use std::os::raw::c_char;
use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::{anyhow, Result};

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
        Self { values, sampled_token }
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
}

/// Backend model trait.
pub trait Model: Send + Sync {
    /// Sync prefill (called by engine from spawn_blocking).
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    /// Sync decode step (called by engine from spawn_blocking).
    fn decode_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>>;

    fn model_config(&self) -> ModelConfig;

    fn eos_token_id(&self) -> i32;

    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;

    fn token_to_piece(&self, token: i32) -> Result<String>;

    /// Apply chat template to messages. Returns formatted prompt for tokenization.
    /// Fallback: simple "role: content\n" concatenation if template unavailable.
    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String>;
}

/// Sample token from logits (greedy).
fn sample_greedy(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Sample token using top-p (nucleus) sampling.
fn sample_top_p(logits: &[f32], top_p: f32) -> i32 {
    if top_p >= 1.0 {
        return sample_greedy(logits);
    }
    // Softmax and cumulative sum
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_logit).exp() / exp_sum))
        .collect();
    probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    for (i, p) in &probs {
        cum += p;
        if cum >= top_p {
            return *i as i32;
        }
    }
    probs.last().map(|(i, _)| *i as i32).unwrap_or(0)
}

#[cfg(not(ferrum_stub))]
/// Llama.cpp model via FFI.
pub struct LlamaCppModel {
    _model: NonNull<ffi::llama_model>,
    _ctx: Arc<std::sync::Mutex<NonNull<ffi::llama_context>>>,
    vocab: *const ffi::llama_vocab,
    config: ModelConfig,
    eos_token: i32,
}

#[cfg(not(ferrum_stub))]
impl LlamaCppModel {
    /// Load a GGUF model from path.
    pub fn load(model_path: &std::path::Path, max_batch_size: usize) -> Result<Self> {
        unsafe {
            ffi::llama_backend_init();
        }

        let path_cstr = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path not valid UTF-8"))?;
        let path_c = CString::new(path_cstr)?;

        let model_params = unsafe { ffi::llama_model_default_params() };
        let model = unsafe {
            ffi::llama_model_load_from_file(path_c.as_ptr(), model_params)
        };
        let model = NonNull::new(model).ok_or_else(|| anyhow!("llama_model_load_from_file failed"))?;

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
        // n_batch = max tokens per decode call (must fit full prompt; llama default 2048)
        ctx_params.n_batch = 2048u32.max(max_batch_size as u32);
        ctx_params.n_ctx = 2048;
        ctx_params.n_seq_max = 64;

        let ctx = unsafe {
            ffi::llama_init_from_model(model.as_ptr(), ctx_params)
        };
        let ctx = NonNull::new(ctx).ok_or_else(|| {
            unsafe { ffi::llama_model_free(model.as_ptr()) };
            anyhow!("llama_init_from_model failed")
        })?;

        Ok(Self {
            _model: model,
            _ctx: Arc::new(std::sync::Mutex::new(ctx)),
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
                    0,   // lstrip: keep leading spaces so words don't concatenate (e.g. "¡Hola!¿Enquépuedo...")
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
                        0,   // lstrip: keep leading spaces
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
            .map(|(r, _)| CString::new(r.as_str()).unwrap())
            .collect();
        let content_cstrings: Vec<CString> = messages
            .iter()
            .map(|(_, c)| CString::new(c.as_str()).unwrap())
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
            if let Some(ref s) = from_model {
                vec![Some(s)]
            } else {
                vec![None]
            }
        };

        // Built-in template names - llama_chat_apply_template looks them up by name
        let fallback_names = [
            "chatml",      // Qwen, Phi, OpenHermes, many models
            "llama3",      // Meta Llama 3
            "phi3",        // Microsoft Phi-3
            "llama2",      // Meta Llama 2, Mistral
            "mistral-v1",  // Mistral
            "gemma",       // Google Gemma
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
    ) -> Result<Vec<(u64, Logits)>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }

        let total_tokens: usize = requests.iter().map(|r| r.prompt_tokens.len()).sum();
        let n_seq_max = requests.len().max(1) as i32;

        let mut batch = unsafe {
            ffi::llama_batch_init(total_tokens as i32, 0, n_seq_max)
        };

        let mut batch_logits_indices: Vec<i32> = Vec::with_capacity(requests.len());
        for (seq_idx, req) in requests.iter().enumerate() {
            let seq_id = seq_idx as i32;
            for (pos, &token) in req.prompt_tokens.iter().enumerate() {
                let idx = batch.n_tokens as usize;
                let has_logits = pos == req.prompt_tokens.len() - 1;
                unsafe {
                    *batch.token.add(idx) = token;
                    *batch.pos.add(idx) = pos as i32;
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

        let ctx_guard = self._ctx.lock().map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (i, &req_id) in req_ids.iter().enumerate() {
            let batch_idx = batch_logits_indices.get(i).copied().unwrap_or(-1);
            let logits_ptr = if batch_idx >= 0 {
                unsafe { ffi::llama_get_logits_ith(ctx, batch_idx) }
            } else {
                std::ptr::null_mut()
            };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token)));
                continue;
            }
            let logits_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(logits_ptr, n_vocab as usize)
            };
            let sampled = sample_greedy(logits_slice);
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled)));
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

        for (idx, req) in requests.iter().enumerate() {
            let input_token = req.last_token.or_else(|| req.prompt_tokens.last().copied())
                .unwrap_or(self.eos_token);
            let pos = req.context_len as i32;
            let seq_id = idx as i32;
            let idx = idx as usize;

            unsafe {
                *batch.token.add(idx) = input_token;
                *batch.pos.add(idx) = pos;
                *batch.n_seq_id.add(idx) = 1;
                let arr = *batch.seq_id.add(idx);
                *arr.add(0) = seq_id;
                *batch.logits.add(idx) = 1i8;
            }
            batch.n_tokens += 1;
        }

        let ctx_guard = self._ctx.lock().map_err(|e| anyhow!("lock poisoned: {}", e))?;
        let ctx = ctx_guard.as_ptr();

        let ret = unsafe { ffi::llama_decode(ctx, batch) };
        if ret != 0 {
            unsafe { ffi::llama_batch_free(batch) };
            return Err(anyhow!("llama_decode failed: {}", ret));
        }

        let n_vocab = self.config.vocab_size as i32;
        let mut results = Vec::with_capacity(requests.len());

        for (out_idx, &req_id) in req_ids.iter().enumerate() {
            let logits_ptr = unsafe { ffi::llama_get_logits_ith(ctx, out_idx as i32) };
            if logits_ptr.is_null() {
                results.push((req_id, Logits::new(vec![], self.eos_token)));
                continue;
            }
            let logits_slice: &[f32] = unsafe {
                std::slice::from_raw_parts(logits_ptr, n_vocab as usize)
            };
            let sampled = sample_greedy(logits_slice);
            let values: Vec<f32> = logits_slice.to_vec();
            results.push((req_id, Logits::new(values, sampled)));
        }

        unsafe { ffi::llama_batch_free(batch) };
        Ok(results)
    }
}

#[cfg(not(ferrum_stub))]
unsafe impl Send for LlamaCppModel {}
#[cfg(not(ferrum_stub))]
unsafe impl Sync for LlamaCppModel {}

#[cfg(not(ferrum_stub))]
impl Model for LlamaCppModel {
    fn prefill_sync(
        &self,
        req_ids: &[u64],
        requests: &[InferenceRequestForModel],
    ) -> Result<Vec<(u64, Logits)>> {
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

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenize_impl(text)
    }

    fn token_to_piece(&self, token: i32) -> Result<String> {
        self.token_to_piece_impl(token)
    }

    fn apply_chat_template(&self, messages: &[(String, String)]) -> Result<String> {
        self.apply_chat_template_impl(messages)
    }
}

// ==================== Stub implementation (when FERRUM_SKIP_LLAMA or no llama.cpp) ====================

#[cfg(ferrum_stub)]
/// Stub LlamaCppModel when llama.cpp is not built.
pub struct LlamaCppModel {
    config: ModelConfig,
}

#[cfg(ferrum_stub)]
impl LlamaCppModel {
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
    ) -> Result<Self> {
        let _ = (model_path, max_batch_size);
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

#[cfg(ferrum_stub)]
unsafe impl Send for LlamaCppModel {}
#[cfg(ferrum_stub)]
unsafe impl Sync for LlamaCppModel {}

#[cfg(ferrum_stub)]
impl Model for LlamaCppModel {
    fn prefill_sync(
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
}
