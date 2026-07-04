// LlamaCppModel — llama.cpp FFI backend.
//
// This module contains two conditional implementations:
//   - Real build (#[cfg(not(fox_stub))]): full FFI, loading, batching, sampling.
//   - Stub build (#[cfg(fox_stub)]):       no-op placeholder for CI / stub builds.

#[cfg(not(fox_stub))]
mod batch;
#[cfg(all(test, not(fox_stub)))]
mod golden;
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
use crate::engine::model::{
    InferenceRequestForModel, Logits, Model, ModelConfig, ModelInfo, PrefillStep,
};

/// SentencePiece uses U+2581 (▁) for word boundaries.
#[cfg(not(fox_stub))]
pub(super) const SPM_SPACE: char = '\u{2581}';

/// Known non-default reasoning-delimiter formats: `(open, close)` marker pairs,
/// matched against the model's OWN chat template (never its name). The default
/// `<think>`/`</think>` covers most reasoning models (Qwen3, DeepSeek-R1), so they
/// need no entry here. Adding support for a new format = one line + a golden test.
#[cfg(not(fox_stub))]
const REASONING_FORMATS: &[(&str, &str)] = &[
    // Gemma / GPT-OSS "channel" (harmony) format — note the mirrored brackets.
    ("<|channel>", "<channel|>"),
];

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

/// Read a GGUF metadata string by key directly from a model pointer.
///
/// Standalone variant of `read_meta_str` for use during `load()`, before a
/// `LlamaCppModel` (and thus `&self`) exists. Returns `None` when the key is
/// absent or the value cannot be decoded as UTF-8.
#[cfg(not(fox_stub))]
fn meta_str(model: *const ffi::llama_model, key: &str) -> Option<String> {
    use std::ffi::CString;
    let key_c = CString::new(key).ok()?;
    let mut buf = vec![0u8; 256];
    let n = unsafe {
        ffi::llama_model_meta_val_str(
            model,
            key_c.as_ptr(),
            buf.as_mut_ptr() as *mut std::os::raw::c_char,
            buf.len(),
        )
    };
    if n < 0 {
        return None;
    }
    Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
}

/// Resolve the per-head dimension for KV cache sizing.
///
/// `n_embd / n_head` is WRONG for architectures that pin an explicit head
/// dimension — notably Gemma-2/3 (head_dim = 256, independent of n_embd/n_head)
/// and DeepSeek-V2/V3 (MLA). Such models publish the real value in the GGUF
/// `<arch>.attention.key_length` key; using the derived value instead mis-sizes
/// the KV cache and produces corrupt output. Prefer the metadata key, falling
/// back to `n_embd / n_head` for architectures that omit it.
#[cfg(not(fox_stub))]
fn resolve_head_dim(model: *const ffi::llama_model, n_embd: usize, n_head: usize) -> usize {
    let from_meta = meta_str(model, "general.architecture")
        .and_then(|arch| meta_str(model, &format!("{arch}.attention.key_length")))
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&d| d > 0);
    from_meta.unwrap_or(if n_head > 0 { n_embd / n_head } else { 128 })
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

    if memory_likely_cause {
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
        // Load failed but memory is NOT the obvious cause — don't assert OOM. The
        // common case here is a missing compute backend (no GPU driver AND the CPU
        // backend .so not found next to the binary).
        let mut msg = format!(
            "failed to load '{}' ({:.1} GB): llama.cpp returned no model.\n",
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
        msg.push_str("\nPossible causes:\n");
        msg.push_str(
            "  • No compute backend — GPU driver missing AND the CPU backend library\n    \
             (libggml-cpu.so) is not next to the fox binary.\n",
        );
        msg.push_str("  • GGUF version/architecture not supported by this llama.cpp build.\n");
        msg.push_str("  • The model is larger than free memory (see figures above).\n");
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

/// Human description of the active compute backend, read from the ggml devices
/// registered by `ggml_backend_load_all`. Prefers a GPU/iGPU device (that is where
/// inference runs when one is present); otherwise reports CPU. Shown at startup so
/// users can tell whether they are running on the GPU.
#[cfg(not(fox_stub))]
pub(crate) fn active_backend_description() -> String {
    use std::ffi::CStr;
    let read = |p: *const std::os::raw::c_char| -> String {
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let count = unsafe { ffi::ggml_backend_dev_count() };
    let mut accelerator: Option<String> = None;
    let mut has_cpu = false;
    for i in 0..count {
        let dev = unsafe { ffi::ggml_backend_dev_get(i) };
        if dev.is_null() {
            continue;
        }
        // 0 = CPU, 1 = GPU, 2 = iGPU.
        match unsafe { ffi::ggml_backend_dev_type(dev) } {
            0 => has_cpu = true,
            _ if accelerator.is_none() => {
                let name = read(unsafe { ffi::ggml_backend_dev_name(dev) });
                let desc = read(unsafe { ffi::ggml_backend_dev_description(dev) });
                accelerator = Some(if desc.is_empty() {
                    name
                } else {
                    format!("{name} — {desc}")
                });
            }
            _ => {}
        }
    }
    accelerator.unwrap_or_else(|| {
        if has_cpu {
            "CPU".to_string()
        } else {
            "unknown".to_string()
        }
    })
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
    /// Lazily-built, cached minijinja environment holding the model's compiled chat
    /// template (pycompat callback + the GGUF template added once). The inner `None`
    /// means the model has no usable embedded template. Cached so the template is
    /// parsed once, not on every request (see `render_chat_jinja`).
    pub(super) chat_env: std::sync::OnceLock<Option<minijinja::Environment<'static>>>,
}

#[cfg(not(fox_stub))]
impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        // Free the context first (must happen before model is freed).
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
        let head_dim = resolve_head_dim(model.as_ptr(), n_embd, n_head);

        let config = ModelConfig {
            num_layers: n_layer,
            num_heads: n_head,
            num_heads_kv: n_head_kv,
            head_dim,
            n_embd,
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
        // AUTO (-1): let llama.cpp enable flash attention only when the active
        // backend supports it for this model/KV type. Forcing ENABLED (1) caused
        // decode failures and garbage output on Vulkan / some ROCm setups and with
        // quantized KV caches — matching upstream/Ollama, which default to AUTO.
        ctx_params.flash_attn_type = -1; // LLAMA_FLASH_ATTN_TYPE_AUTO
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
        Ok(Self {
            _model: model,
            _ctx: ctx_arc,
            vocab,
            config,
            eos_token,
            effective_ctx: effective_max_ctx,
            owns_model: true,
            chat_env: std::sync::OnceLock::new(),
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
        ctx_params.flash_attn_type = -1; // LLAMA_FLASH_ATTN_TYPE_AUTO (see load())
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
            chat_env: std::sync::OnceLock::new(),
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
        max_prefill_chunk: usize,
    ) -> Result<Vec<PrefillStep>> {
        self.do_prefill(req_ids, requests, max_prefill_chunk)
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

    fn build_prompt_tokens(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
    ) -> Result<Vec<i32>> {
        self.build_prompt_tokens_impl(messages, enable_thinking)
    }

    fn reasoning_delimiters(&self) -> Option<(String, String)> {
        // Detect the reasoning format from the model's OWN chat template — never
        // from its name. `REASONING_FORMATS` is a small, documented, extensible
        // registry of known non-default (open, close) marker pairs; a model matches
        // a format when its template references BOTH markers. No match → the caller
        // uses the default `<think>`/`</think>`. Adding a format is one line + a
        // golden test (see docs/design/model-architecture-rework.md §4.3).
        let t = self.raw_chat_template()?;
        REASONING_FORMATS
            .iter()
            .find(|(open, close)| t.contains(open) && t.contains(close))
            .map(|(open, close)| (open.to_string(), close.to_string()))
    }

    fn context_len(&self) -> u32 {
        self.effective_ctx
    }

    fn active_backend(&self) -> String {
        active_backend_description()
    }

    fn kv_cache_capacity(&self) -> usize {
        // The real total KV capacity llama.cpp allocated for this context — read
        // back rather than recomputed, so fox's block pool matches it exactly.
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return self.effective_ctx as usize,
        };
        unsafe { ffi::llama_n_ctx(ctx_guard.as_ptr() as *const _) as usize }
    }

    fn supports_thinking(&self) -> bool {
        // Primary signal: the model's chat template exposes an `enable_thinking`
        // toggle (Gemma-4, Qwen3, …). This is robust regardless of what the model
        // names its reasoning tokens.
        if self
            .raw_chat_template()
            .is_some_and(|t| t.contains("enable_thinking"))
        {
            return true;
        }
        // Fallback signal: `<think>` is a single special token (DeepSeek-R1, some
        // Qwen). Tokenising it with add_special=true yields at most [BOS, <think>].
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
        self.config.n_embd
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

    fn model_info(&self) -> ModelInfo {
        // Read metadata-derived truth directly from the model, rather than the
        // reconstructed values the generic default would produce.
        let model = self._model.as_ptr();
        let n_ctx_train = unsafe { ffi::llama_model_n_ctx_train(model) } as u32;
        let arch_name = self
            .read_meta_str("general.architecture")
            .unwrap_or_else(|| "unknown".to_string());
        let has_chat_template =
            unsafe { !ffi::llama_model_chat_template(model, std::ptr::null()).is_null() };

        ModelInfo {
            arch_name,
            backend: self.active_backend(),
            n_embd: self.config.n_embd,
            n_head: self.config.num_heads,
            n_head_kv: self.config.num_heads_kv,
            head_dim: self.config.head_dim,
            n_layer: self.config.num_layers,
            n_ctx_train,
            effective_ctx: self.effective_ctx,
            vocab_size: self.config.vocab_size,
            eos_token_id: self.eos_token,
            has_chat_template,
            supports_thinking: self.supports_thinking(),
            supports_seq_copy: self.supports_seq_copy(),
            stop_token_count: self.stop_tokens().len(),
            recommended_sampling: self.recommended_sampling(),
        }
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
