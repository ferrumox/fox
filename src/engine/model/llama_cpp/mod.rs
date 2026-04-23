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
#[cfg(not(fox_stub))]
use crate::engine::mtmd_ffi;

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

    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("?");

    if memory_likely_cause {
        let mut msg = format!(
            "failed to load '{model_name}' ({file_gb:.1} GB): not enough memory to fit the model.\n",
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
        let mut msg = format!(
            "failed to load '{model_name}': llama.cpp could not load the model.\n",
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
        msg.push_str("\nPossible causes:\n");
        msg.push_str("  • GPU backend libraries (libggml-cuda.so / libggml-vulkan.so) not found.\n");
        msg.push_str("  • KV cache context allocation too large — try --max-context-len 2048.\n");
        msg.push_str("  • GGUF version not supported by this build of Fox.\n");
        msg.push_str("  • File corrupt or from an incomplete download.\n");
        anyhow!("{}", msg.trim_end())
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

/// Pool of mtmd contexts for parallel CLIP encoding.
#[cfg(not(fox_stub))]
pub(super) struct MtmdPool {
    contexts: std::sync::Mutex<Vec<NonNull<mtmd_ffi::mtmd_context>>>,
    available: std::sync::Condvar,
    /// Input embedding dimension from the model (needed for embedding buffer sizing).
    pub(super) n_embd_inp: usize,
}

#[cfg(not(fox_stub))]
impl MtmdPool {
    fn new(contexts: Vec<NonNull<mtmd_ffi::mtmd_context>>, n_embd_inp: usize) -> Self {
        Self {
            contexts: std::sync::Mutex::new(contexts),
            available: std::sync::Condvar::new(),
            n_embd_inp,
        }
    }

    pub(super) fn acquire(&self) -> NonNull<mtmd_ffi::mtmd_context> {
        let mut pool = self.contexts.lock().expect("mtmd pool lock poisoned");
        loop {
            if let Some(ctx) = pool.pop() {
                return ctx;
            }
            pool = self.available.wait(pool).expect("mtmd pool lock poisoned");
        }
    }

    pub(super) fn release(&self, ctx: NonNull<mtmd_ffi::mtmd_context>) {
        if let Ok(mut pool) = self.contexts.lock() {
            pool.push(ctx);
        }
        self.available.notify_one();
    }

    fn free_all(&self) {
        if let Ok(mut pool) = self.contexts.lock() {
            for ctx in pool.drain(..) {
                unsafe { mtmd_ffi::mtmd_free(ctx.as_ptr()) };
            }
        }
    }
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
    /// Whether this instance owns the model pointer and should free it on drop.
    /// `false` when sharing weights with another `LlamaCppModel` (e.g. bench-kv).
    owns_model: bool,
    /// Pool of multimodal (vision) contexts for parallel CLIP encoding.
    /// None when no multimodal projector is loaded.
    pub(super) mtmd_pool: Option<MtmdPool>,
}

#[cfg(not(fox_stub))]
impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        // Free mtmd pool contexts first (they reference the model internally).
        if let Some(ref pool) = self.mtmd_pool {
            pool.free_all();
        }
        // Free the llama context (must happen before model is freed).
        if let Ok(ctx) = self._ctx.lock() {
            unsafe { ffi::llama_free(ctx.as_ptr()) };
        }
        if self.owns_model {
            unsafe { ffi::llama_model_free(self._model.as_ptr()) };
        }
    }
}

#[cfg(not(fox_stub))]
impl LlamaCppModel {
    /// Load a GGUF model from path.
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        model_path: &std::path::Path,
        max_batch_size: usize,
        max_context_len: Option<u32>,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        type_k: u32,
        type_v: u32,
        main_gpu: i32,
        split_mode: u32,
        tensor_split: &[f32],
        moe_offload_cpu: bool,
        mmproj_path: Option<&std::path::Path>,
        flash_attn: bool,
        vision_contexts: usize,
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
        model_params.main_gpu = main_gpu;
        model_params.split_mode = split_mode as ffi::llama_split_mode;

        // tensor_split: raw pointer must stay valid until llama_model_load_from_file returns.
        // ts_buf is kept alive on the stack for the duration of the call.
        let ts_buf: Vec<f32>;
        if !tensor_split.is_empty() {
            let max_devs = unsafe { ffi::llama_max_devices() };
            let mut buf = vec![0.0f32; max_devs];
            for (i, &v) in tensor_split.iter().enumerate().take(max_devs) {
                buf[i] = v;
            }
            ts_buf = buf;
            model_params.tensor_split = ts_buf.as_ptr();
        } else {
            ts_buf = vec![]; // kept to satisfy the borrow checker
        }

        // MoE expert tensor CPU offload.
        // When enabled, all MoE expert weight tensors are pinned to CPU RAM so they are not
        // loaded into VRAM. This lets models like DeepSeek or Mixtral run on GPUs with limited
        // VRAM — the attention layers stay on GPU while expert weights are read from RAM on demand.
        //
        // Pattern covers: blk.<N>.ffn_up_exps, ffn_down_exps, ffn_gate_exps and the
        // chunked variants (ffn_up_chexps, …) used by some architectures.
        //
        // SAFETY: `moe_pattern_cstr` and `buft_overrides` must remain alive until
        // `llama_model_load_from_file` returns — both are declared before the call and
        // dropped explicitly afterwards.
        let moe_pattern_cstr: std::ffi::CString;
        let buft_overrides: Vec<ffi::llama_model_tensor_buft_override>;
        if moe_offload_cpu {
            let cpu_buft = unsafe { ffi::ggml_backend_cpu_buffer_type() };
            moe_pattern_cstr = std::ffi::CString::new("blk\\.\\d+\\.ffn_(up|down|gate)_(ch|)exps")
                .expect("MoE pattern is valid C string");
            // NULL-terminated: one real entry + one sentinel with null pattern.
            buft_overrides = vec![
                ffi::llama_model_tensor_buft_override {
                    pattern: moe_pattern_cstr.as_ptr(),
                    buft: cpu_buft,
                },
                ffi::llama_model_tensor_buft_override {
                    pattern: std::ptr::null(),
                    buft: std::ptr::null_mut(),
                },
            ];
            model_params.tensor_buft_overrides = buft_overrides.as_ptr();
            tracing::info!("MoE CPU offload enabled — expert tensors will be allocated in RAM");
        } else {
            moe_pattern_cstr = std::ffi::CString::new("").expect("empty string is valid");
            buft_overrides = vec![];
        }

        let model = unsafe { ffi::llama_model_load_from_file(path_c.as_ptr(), model_params) };
        drop(ts_buf); // explicit: ts_buf outlives model_params usage above
        drop(buft_overrides); // keep overrides alive until after the load call
        drop(moe_pattern_cstr);
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
        ctx_params.flash_attn_type = if flash_attn { 1 } else { 0 };
        ctx_params.offload_kqv = true;
        ctx_params.type_k = type_k as _;
        ctx_params.type_v = type_v as _;

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

        // Load multimodal projector (vision/CLIP encoder) pool if provided.
        let mtmd_pool = if let Some(mmproj) = mmproj_path {
            let mmproj_cstr = std::ffi::CString::new(
                mmproj
                    .to_str()
                    .ok_or_else(|| anyhow!("mmproj path not valid UTF-8"))?,
            )?;
            let mut mtmd_params = unsafe { mtmd_ffi::mtmd_context_params_default() };
            mtmd_params.use_gpu = true;
            unsafe {
                mtmd_ffi::mtmd_log_set(Some(noop_log), std::ptr::null_mut());
            }

            let n_contexts = vision_contexts.max(1);
            let mut pool_ctxs: Vec<NonNull<mtmd_ffi::mtmd_context>> = Vec::new();
            for i in 0..n_contexts {
                let raw = unsafe {
                    mtmd_ffi::mtmd_init_from_file(
                        mmproj_cstr.as_ptr(),
                        model.as_ptr() as *const _,
                        mtmd_params,
                    )
                };
                match NonNull::new(raw) {
                    Some(ptr) => pool_ctxs.push(ptr),
                    None => {
                        if i == 0 {
                            tracing::warn!(mmproj = ?mmproj, "failed to load multimodal projector");
                        } else {
                            tracing::warn!(
                                context_idx = i,
                                "failed to create additional mtmd context, using {} contexts",
                                pool_ctxs.len()
                            );
                        }
                        break;
                    }
                }
            }
            if pool_ctxs.is_empty() {
                None
            } else {
                let n_embd_inp =
                    unsafe { ffi::llama_model_n_embd_inp(model.as_ptr()) } as usize;
                if n_contexts > 1 {
                    tracing::info!(
                        count = pool_ctxs.len(),
                        "created mtmd context pool for parallel CLIP encoding"
                    );
                }
                Some(MtmdPool::new(pool_ctxs, n_embd_inp))
            }
        } else {
            None
        };

        Ok(Self {
            _model: model,
            _ctx: ctx_arc,
            vocab,
            config,
            eos_token,
            effective_ctx: effective_max_ctx,
            owns_model: true,
            mtmd_pool,
        })
    }

    /// Create a new context from this model's weights with different KV cache types.
    ///
    /// The returned instance shares the underlying model pointer but owns a fresh
    /// llama.cpp context. Use this to compare KV types without reloading weights.
    ///
    /// # Safety
    /// The original model must outlive all instances returned by this method.
    pub fn new_context(
        &self,
        max_batch_size: usize,
        max_context_len: Option<u32>,
        gpu_memory_bytes: usize,
        gpu_memory_fraction: f32,
        type_k: u32,
        type_v: u32,
        flash_attn: bool,
    ) -> Result<Self> {
        let model = self._model;

        let mut ctx_params = unsafe { ffi::llama_context_default_params() };
        let n_seq = (max_batch_size as u32).max(4);

        let model_train_ctx = unsafe { ffi::llama_model_n_ctx_train(model.as_ptr()) } as u32;
        let effective_max_ctx = resolve_context_len(max_context_len, model_train_ctx);

        let free_bytes = query_gpu_free_bytes()
            .unwrap_or((gpu_memory_bytes as f64 * gpu_memory_fraction as f64) as usize);
        let budget_bytes = (free_bytes as f64 * gpu_memory_fraction as f64) as usize;
        let n_head_kv = self.config.num_heads_kv;
        let head_dim = self.config.head_dim;
        let n_layer = self.config.num_layers;
        // Use the actual KV type byte ratios rather than assuming F16.
        let (k_num, k_den) = crate::kv_cache::kv_type_bytes(type_k);
        let (v_num, v_den) = crate::kv_cache::kv_type_bytes(type_v);
        let elems_per_token = (n_head_kv * head_dim * n_layer) as u64;
        let bytes_per_token_u64 =
            (elems_per_token * k_num).div_ceil(k_den) + (elems_per_token * v_num).div_ceil(v_den);
        let max_tokens_by_mem = if bytes_per_token_u64 > 0 && budget_bytes > 0 {
            (budget_bytes as u64 / bytes_per_token_u64) as u32
        } else {
            effective_max_ctx * n_seq
        };
        let n_ctx = (effective_max_ctx * n_seq)
            .min(max_tokens_by_mem)
            .max(effective_max_ctx);

        ctx_params.n_ctx = n_ctx;
        ctx_params.n_batch = effective_max_ctx.max(max_batch_size as u32);
        ctx_params.n_seq_max = n_seq;
        ctx_params.flash_attn_type = if flash_attn { 1 } else { 0 };
        ctx_params.offload_kqv = true;
        ctx_params.type_k = type_k as _;
        ctx_params.type_v = type_v as _;

        let ctx = unsafe { ffi::llama_init_from_model(model.as_ptr(), ctx_params) };
        let ctx = NonNull::new(ctx)
            .ok_or_else(|| anyhow!("llama_init_from_model failed for new_context"))?;

        #[allow(clippy::arc_with_non_send_sync)]
        let ctx_arc = Arc::new(std::sync::Mutex::new(ctx));
        Ok(Self {
            _model: model,
            _ctx: ctx_arc,
            vocab: self.vocab,
            config: self.config.clone(),
            eos_token: self.eos_token,
            effective_ctx: effective_max_ctx,
            owns_model: false, // weights are owned by the original LlamaCppModel
            mtmd_pool: None,   // new_context is for bench-kv; vision not needed
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

    fn supports_vision(&self) -> bool {
        self.mtmd_pool.is_some()
    }

    fn vision_prefill_sync(&self, params: &super::VisionPrefillParams) -> Result<(usize, Logits)> {
        self.do_vision_prefill(
            params.seq_id,
            &params.text_prompt,
            &params.image_bytes,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
            params.seed,
        )
    }

    fn vision_preprocess_sync(
        &self,
        params: &super::VisionPreprocessParams,
    ) -> Result<super::PreprocessedVision> {
        self.do_vision_preprocess(&params.text_prompt, &params.image_bytes)
    }

    fn vision_decode_prefill_sync(
        &self,
        params: &super::VisionDecodeParams,
    ) -> Result<(usize, Logits)> {
        self.do_vision_decode_prefill(
            params.seq_id,
            &params.preprocessed,
            params.temperature,
            params.top_p,
            params.top_k,
            params.repetition_penalty,
            params.seed,
        )
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
