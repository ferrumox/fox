// LlamaCppModel — llama.cpp FFI backend.
//
// This module contains two conditional implementations:
//   - Real build (#[cfg(not(fox_stub))]): full FFI, loading, batching, sampling.
//   - Stub build (#[cfg(fox_stub)]):       no-op placeholder for CI / stub builds.

#[cfg(not(fox_stub))]
mod batch;
#[cfg(not(fox_stub))]
mod metadata;
mod stub;
#[cfg(not(fox_stub))]
mod vocab;

#[cfg(fox_stub)]
pub use stub::LlamaCppModel;

#[cfg(not(fox_stub))]
use anyhow::Result;

// ---------------------------------------------------------------------------
// Real implementation
// ---------------------------------------------------------------------------

#[cfg(not(fox_stub))]
use std::ptr::NonNull;
#[cfg(not(fox_stub))]
use std::sync::Arc;

#[cfg(not(fox_stub))]
use anyhow::anyhow;

#[cfg(not(fox_stub))]
use crate::engine::ffi;
#[cfg(not(fox_stub))]
use crate::engine::model::{InferenceRequestForModel, Logits, Model, ModelConfig};

/// SentencePiece uses U+2581 (▁) for word boundaries.
#[cfg(not(fox_stub))]
pub(super) const SPM_SPACE: char = '\u{2581}';

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

/// Read available system RAM in bytes from /proc/meminfo (Linux).
/// Returns None on non-Linux systems or parse errors.
#[cfg(not(fox_stub))]
fn available_ram_bytes() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if line.starts_with("MemAvailable:") {
            let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// Diagnose why `llama_model_load_from_file` returned null and return a
/// human-readable error with actionable suggestions.
#[cfg(not(fox_stub))]
fn diagnose_load_failure(model_path: &std::path::Path) -> anyhow::Error {
    // 1. Check GGUF magic bytes (0x47 0x47 0x55 0x46 == "GGUF").
    let magic_ok = std::fs::File::open(model_path)
        .ok()
        .and_then(|mut f| {
            use std::io::Read;
            let mut buf = [0u8; 4];
            f.read_exact(&mut buf).ok().map(|_| buf)
        })
        .map(|b| b == [0x47, 0x47, 0x55, 0x46])
        .unwrap_or(false);

    if !magic_ok {
        return anyhow!(
            "failed to load '{}': the file is not a valid GGUF model.\n\
             It may be corrupt or from an incomplete download.\n\
             → Delete the file and run `fox pull` again.",
            model_path.display()
        );
    }

    // 2. Compare file size to available memory.
    let file_size = std::fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    let file_gb = file_size as f64 / 1_073_741_824.0;

    let gpu_free = query_gpu_free_bytes();
    let ram_free = available_ram_bytes();

    let file_size_usize = file_size as usize;
    let memory_likely_cause = match (gpu_free, ram_free) {
        (Some(vram), _) if file_size > 0 && vram < file_size_usize => true,
        (None, Some(ram)) if file_size > 0 && ram < file_size_usize => true,
        _ => false,
    };

    if memory_likely_cause || file_size > 0 {
        let mut msg = format!(
            "failed to load '{}' ({:.1} GB): not enough memory to fit the model.\n",
            model_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("?"),
            file_gb
        );
        if let Some(vram) = gpu_free {
            msg.push_str(&format!(
                "  GPU free:  {:.1} GB\n",
                vram as f64 / 1_073_741_824.0
            ));
        }
        if let Some(ram) = ram_free {
            msg.push_str(&format!(
                "  RAM free:  {:.1} GB\n",
                ram as f64 / 1_073_741_824.0
            ));
        }
        msg.push_str("\nSuggestions:\n");
        msg.push_str("  • Use a smaller quantization — pull a Q4_K_M or Q3_K_M variant instead of Q8_0/F16.\n");
        msg.push_str(
            "  • Reduce context length with --max-context-len (e.g. 2048 instead of 8192).\n",
        );
        msg.push_str("  • Close other GPU processes (other models, browsers with WebGL, etc.).\n");
        msg.push_str("  • Unload other loaded models first (fox models or POST /api/delete).\n");
        if gpu_free.is_some() {
            msg.push_str("  • If you have multiple GPUs, ensure CUDA_VISIBLE_DEVICES targets the right one.\n");
        }
        anyhow!("{}", msg.trim_end())
    } else {
        anyhow!(
            "failed to load '{}': llama.cpp could not open the model.\n\
             The file may use a GGUF version not supported by this build of Fox.\n\
             → Check for a newer Fox release or try a different model variant.",
            model_path.display()
        )
    }
}

/// Choose the effective per-sequence context length.
///
/// Returns `user_limit` when the user specified one explicitly, otherwise
/// falls back to `model_train_ctx` (the context the model was trained with).
#[cfg(not(fox_stub))]
pub(crate) fn resolve_context_len(user_limit: Option<u32>, model_train_ctx: u32) -> u32 {
    user_limit.unwrap_or(model_train_ctx)
}

/// Llama.cpp model via FFI.
#[cfg(not(fox_stub))]
pub struct LlamaCppModel {
    pub(super) _model: NonNull<ffi::llama_model>,
    pub(super) _ctx: Arc<std::sync::Mutex<NonNull<ffi::llama_context>>>,
    pub(super) vocab: *const ffi::llama_vocab,
    pub(super) config: ModelConfig,
    pub(super) eos_token: i32,
    /// Effective per-sequence context length (tokens) used when creating the llama.cpp context.
    pub(super) effective_ctx: u32,
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

        use std::ffi::CString;
        let path_cstr = model_path
            .to_str()
            .ok_or_else(|| anyhow!("model path not valid UTF-8"))?;
        let path_c = CString::new(path_cstr)?;

        let mut model_params = unsafe { ffi::llama_model_default_params() };
        // Offload all layers to GPU (-1 = all). On CPU-only builds llama.cpp ignores this.
        model_params.n_gpu_layers = -1;
        let model = unsafe { ffi::llama_model_load_from_file(path_c.as_ptr(), model_params) };
        let model = NonNull::new(model).ok_or_else(|| diagnose_load_failure(model_path))?;

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
        let model_train_ctx = unsafe { ffi::llama_model_n_ctx_train(model.as_ptr()) } as u32;
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

    fn recommended_sampling(&self) -> Option<crate::engine::model::RecommendedSampling> {
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
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(all(test, not(fox_stub)))]
mod tests {
    use super::resolve_context_len;

    #[test]
    fn auto_uses_model_trained_ctx() {
        assert_eq!(resolve_context_len(None, 8192), 8192);
    }

    #[test]
    fn auto_uses_model_trained_ctx_large() {
        assert_eq!(resolve_context_len(None, 131_072), 131_072);
    }

    #[test]
    fn explicit_limit_overrides_model_ctx() {
        assert_eq!(resolve_context_len(Some(4096), 131_072), 4096);
    }

    #[test]
    fn explicit_limit_equal_to_model_ctx() {
        assert_eq!(resolve_context_len(Some(8192), 8192), 8192);
    }

    #[test]
    fn explicit_limit_larger_than_model_ctx() {
        assert_eq!(resolve_context_len(Some(16_384), 8192), 16_384);
    }

    #[test]
    fn explicit_limit_of_one_is_respected() {
        assert_eq!(resolve_context_len(Some(1), 32_768), 1);
    }
}
