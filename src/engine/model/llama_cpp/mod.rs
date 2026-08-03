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
    InferenceRequestForModel, Logits, Model, ModelConfig, ModelInfo, NativeToolFormat, PrefillStep,
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

/// Number of threads llama.cpp should use for compute.
///
/// llama.cpp's own default is `GGML_DEFAULT_N_THREADS` = 4 (marked
/// `// TODO: better default` in `ggml.h`) and applies regardless of machine —
/// leaving it alone means fox uses 4 threads on a 24-core box, roughly halving
/// CPU-backend throughput. `llama-server` doesn't inherit that default either;
/// it resolves `n_threads` from `common_cpu_get_num_math()`.
///
/// Mirrors `common_cpu_get_num_physical_cores()` (`common/common.cpp`): on
/// Linux, count distinct `thread_siblings` masks, which yields *physical*
/// cores. Physical rather than logical is deliberate — the two SMT siblings of
/// one core share execution units, so counting them doubles thread count without
/// doubling math throughput and typically costs performance for this workload.
/// Falls back to half the logical CPUs (a reasonable SMT-aware guess) and then
/// to llama.cpp's own 4.
#[cfg(not(fox_stub))]
fn resolve_n_threads() -> i32 {
    if let Some(n) = std::env::var("FOX_N_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    #[cfg(target_os = "linux")]
    {
        let mut siblings = std::collections::HashSet::new();
        for cpu in 0..u32::MAX {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings");
            match std::fs::read_to_string(&path) {
                Ok(line) => {
                    siblings.insert(line.trim().to_string());
                }
                Err(_) => break, // no more CPUs
            }
        }
        if !siblings.is_empty() {
            return siblings.len() as i32;
        }
    }
    std::thread::available_parallelism()
        .map(|n| (n.get() / 2).max(1) as i32)
        .unwrap_or(4)
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
    from_meta.unwrap_or(n_embd.checked_div(n_head).unwrap_or(128))
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

/// Halve `current` toward `floor` for a context-creation OOM retry. `None`
/// once no further shrinking is possible (already at or below `floor`) — the
/// caller should treat that as a genuine, unrecoverable failure rather than
/// retry again. Mirrors `batch.rs`'s `bisection_split` shape: fox doesn't
/// predict whether `n_ctx` tokens fit in memory (no formula is correct across
/// architectures — see `docs/design/mla-recurrent-kv-sizing.md`), it asks
/// llama.cpp by trying, and shrinks only on a real, observed failure.
#[cfg(not(fox_stub))]
pub(crate) fn shrink_n_ctx(current: u32, floor: u32) -> Option<u32> {
    if current <= floor {
        None
    } else {
        Some((current / 2).max(floor))
    }
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

/// Owning handle to a llama.cpp GBNF grammar sampler for one in-flight request.
///
/// The raw `*mut llama_sampler` is not `Send`/`Sync` on its own, but every access
/// happens under the model's `_ctx` mutex (held during `do_prefill`/`do_decode`
/// sampling), so concurrent use is already serialized. `Drop` frees the sampler, so
/// removing the entry from the `grammars` map — or dropping the model — releases it.
#[cfg(not(fox_stub))]
pub(super) struct GrammarSampler {
    ptr: *mut ffi::llama_sampler,
}

#[cfg(not(fox_stub))]
unsafe impl Send for GrammarSampler {}
#[cfg(not(fox_stub))]
unsafe impl Sync for GrammarSampler {}

#[cfg(not(fox_stub))]
impl Drop for GrammarSampler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::llama_sampler_free(self.ptr) };
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
    /// All end-of-generation token ids (`llama_vocab_is_eog`), precomputed once so
    /// `min_tokens` can mask them without a per-token vocab scan.
    pub(super) eog_tokens: Vec<i32>,
    /// Sequence id reserved for embeddings (the extra `n_seq - 1` slot, OUTSIDE the
    /// scheduler's 0..max_batch pool). Embeddings write + wipe this sequence per call;
    /// using a pool id here would clobber a live generation's KV under load.
    pub(super) embed_seq_id: i32,
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
    /// Per-request GBNF grammar samplers for guided decoding, keyed by request id.
    /// Created lazily on the first constrained sample and freed via `free_grammar` on
    /// every terminal path (so they never leak). Empty unless a request set a grammar.
    pub(super) grammars: dashmap::DashMap<u64, GrammarSampler>,
    /// Lifetime count of batch-size-bisection retries triggered by `llama_decode`
    /// returning `1` ("no KV slot for batch") in `do_prefill`/`do_decode`. Surfaced
    /// via `Model::bisection_retry_count()` and diffed into a Prometheus counter in
    /// `run_loop`, same pattern as `spec_proposed`/`spec_accepted`.
    pub(super) decode_bisection_retries: std::sync::atomic::AtomicU64,
    /// Vision/multimodal context (`mtmd`), present only when this model was loaded
    /// with a paired mmproj GGUF. `None` for the overwhelming majority of models —
    /// every multimodal code path is gated on this being `Some`.
    pub(super) mtmd_ctx: Option<NonNull<ffi::mtmd_context>>,
    /// Named LoRA adapters loaded alongside this model (`--lora-modules`),
    /// keyed by the operator-chosen name a client selects via the `model`
    /// field. Value is the loaded adapter handle plus its configured default
    /// scale. Empty for the overwhelming majority of models.
    pub(super) lora_adapters:
        std::collections::HashMap<String, (NonNull<ffi::llama_adapter_lora>, f32)>,
}

#[cfg(not(fox_stub))]
impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        // mtmd holds no ownership over the llama model/context (it only borrows a
        // `llama_model*` at init and a `llama_context*` per eval call), but free it
        // first regardless, before either of the resources it borrowed goes away.
        if let Some(mtmd_ctx) = self.mtmd_ctx {
            unsafe { ffi::mtmd_free(mtmd_ctx.as_ptr()) };
        }
        // LoRA adapters borrow the model's weights (loaded via `llama_adapter_lora_init(model, ..)`)
        // but are otherwise independent objects — free them before the model itself.
        for (adapter, _scale) in self.lora_adapters.values() {
            unsafe { ffi::llama_adapter_lora_free(adapter.as_ptr()) };
        }
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
        mmproj_path: Option<&std::path::Path>,
        lora_modules: &[(String, std::path::PathBuf, f32)],
        reranking: bool,
    ) -> Result<Self> {
        // Suppress llama.cpp's verbose loading output (tensor info, repack, etc.).
        // Fox shows its own clean progress spinner instead.
        unsafe extern "C" fn noop_log(
            _level: ffi::ggml_log_level,
            _text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
        }
        // `FOX_LLAMA_LOG=1` forwards llama.cpp's own log to stderr instead of
        // dropping it. Without this escape hatch, llama.cpp's internal
        // diagnostics are unreachable from a fox build — notably its
        // `LLAMA_BATCH_DEBUG=1` ubatch tracing, which is the only way to observe
        // how a submitted `llama_batch` actually gets split into ubatches (the
        // batching behaviour investigated in
        // docs/design/rocm-benchmarking-2026-08.md). Previously the only way to
        // see any of it was to patch the vendored source and rebuild.
        unsafe extern "C" fn passthrough_log(
            _level: ffi::ggml_log_level,
            text: *const std::os::raw::c_char,
            _user_data: *mut std::os::raw::c_void,
        ) {
            if text.is_null() {
                return;
            }
            let text = unsafe { std::ffi::CStr::from_ptr(text) };
            eprint!("{}", text.to_string_lossy());
        }
        // Process-global llama.cpp/ggml initialisation — must run exactly once,
        // NOT once per model load.
        //
        // `ggml_backend_load_all_from_path` appends to a global
        // `std::vector<ggml_backend_reg_entry>` inside libggml with no internal
        // locking. Two threads loading two models at the same time can therefore
        // reallocate that vector concurrently and corrupt it: observed as an
        // intermittent SIGSEGV (~1 run in 15) in the golden suite, caught under
        // gdb in `_M_realloc_insert<ggml_backend_reg_entry>` while other threads
        // were inside `llama_model_load_from_file`. `llama_backend_init` is
        // likewise a once-per-process call per llama.cpp's own API contract.
        //
        // `ModelRegistry::get_or_load` happens to serialise its loads behind
        // `load_lock`, which is why this never surfaced in the server — but that
        // is the registry's single-flight policy, not a guarantee this layer can
        // rely on: any other caller loading two models concurrently (tests,
        // a draft model, future callers) would hit it.
        static LLAMA_GLOBAL_INIT: std::sync::Once = std::sync::Once::new();
        LLAMA_GLOBAL_INIT.call_once(|| {
            let forward_llama_log = std::env::var_os("FOX_LLAMA_LOG").is_some_and(|v| v != "0");
            unsafe {
                if forward_llama_log {
                    ffi::llama_log_set(Some(passthrough_log), std::ptr::null_mut())
                } else {
                    ffi::llama_log_set(Some(noop_log), std::ptr::null_mut())
                }
            };

            // Load GPU/CPU backends compiled as dynamic libraries (GGML_BACKEND_DL).
            // Passing null searches the executable's directory and cwd — fox ships
            // libggml-cuda.so and libggml-cpu.so next to the binary.
            // On non-DL builds this is a no-op (backends are statically linked).
            unsafe { ffi::ggml_backend_load_all_from_path(std::ptr::null()) };

            unsafe {
                ffi::llama_backend_init();
            }
        });

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
        let n_seq = (max_batch_size as u32).max(4) + 1; // +1: dedicated embeddings slot (last id)

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

        // fox does not predict KV/state memory usage — no formula is correct across
        // architectures (MLA's latent KV is far smaller than the positional formula
        // below assumes; recurrent/hybrid models have no per-token KV at all, so the
        // formula's inputs aren't even meaningful — see
        // docs/design/mla-recurrent-kv-sizing.md). Instead: ask llama.cpp by trying.
        //
        // The positional formula below still computes a soft *ceiling* — it keeps
        // `--gpu-memory-fraction`'s documented meaning (don't be needlessly greedy on
        // constrained hardware) as a first-guess upper bound, not a precise
        // prediction. Real correctness comes from the retry loop: attempt the full
        // desired n_ctx, and only shrink in response to an actual `llama_init_from_model`
        // failure (mirrors the decode-time OOM bisection retry in `batch.rs` — same
        // "observe real failure, retry smaller" philosophy, one layer earlier).
        let free_bytes = query_gpu_free_bytes()
            .unwrap_or((gpu_memory_bytes as f64 * gpu_memory_fraction as f64) as usize);
        let budget_bytes = (free_bytes as f64 * gpu_memory_fraction as f64) as usize;
        // bytes_per_token = 2 (K+V) * n_head_kv * head_dim * 2 (fp16) * n_layer
        let bytes_per_token = 2 * n_head_kv * head_dim * 2 * n_layer;
        let ceiling_tokens = if bytes_per_token > 0 && budget_bytes > 0 {
            (budget_bytes / bytes_per_token) as u32
        } else {
            effective_max_ctx * n_seq
        };
        // Honour the effective_max_ctx per sequence as a floor — never shrink below
        // what a single sequence needs, since that would silently truncate the
        // user's requested context length rather than fail loudly.
        let desired_n_ctx = (effective_max_ctx * n_seq)
            .min(ceiling_tokens)
            .max(effective_max_ctx);

        // n_batch must be at least as large as n_ctx to handle full prompts in one pass
        ctx_params.n_batch = effective_max_ctx.max(max_batch_size as u32);
        ctx_params.n_seq_max = n_seq;
        // Unified KV cache (one shared buffer, `n_stream = 1`) instead of one
        // stream per sequence. This is what makes `llama_kv_cache::init_batch`
        // select `split_simple` over `split_equal` — and `split_simple` has no
        // "seq_ids must be consecutive and increasing" requirement, so a decode
        // batch folds into ONE full-width ubatch regardless of which IDs the
        // scheduler happens to hold. Without it, a prefix-cache hit inheriting a
        // donated, non-dense seq_id silently fragments the batch (measured: 1.74
        // of a possible 4 under sustained load) — see
        // docs/design/rocm-benchmarking-2026-08.md's "Known limitation".
        ctx_params.kv_unified = true;
        // Never inherit llama.cpp's 4-thread default — see resolve_n_threads().
        let n_threads = resolve_n_threads();
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
        tracing::debug!(n_threads, "llama.cpp compute threads");
        // AUTO (-1): let llama.cpp enable flash attention only when the active
        // backend supports it for this model/KV type. Forcing ENABLED (1) caused
        // decode failures and garbage output on Vulkan / some ROCm setups and with
        // quantized KV caches — matching upstream/Ollama, which default to AUTO.
        ctx_params.flash_attn_type = -1; // LLAMA_FLASH_ATTN_TYPE_AUTO
        ctx_params.offload_kqv = true;
        ctx_params.type_k = type_k as _;
        ctx_params.type_v = type_v as _;
        if reranking {
            // Must be set explicitly. A reranker GGUF does NOT necessarily carry a
            // `<arch>.pooling_type` key — jina-reranker-v1-tiny-en, for one, has none —
            // so llama.cpp's UNSPECIFIED fallback resolves to NONE and there is no
            // sequence score to read. llama-server sets it the same way, from
            // `--reranking` (arg.cpp:3067-3070), rather than trusting metadata.
            ctx_params.pooling_type = ffi::llama_pooling_type_LLAMA_POOLING_TYPE_RANK;
        }

        let mut n_ctx_candidate = desired_n_ctx;
        let ctx = loop {
            ctx_params.n_ctx = n_ctx_candidate;
            let raw = unsafe { ffi::llama_init_from_model(model.as_ptr(), ctx_params) };
            if let Some(ctx) = NonNull::new(raw) {
                if n_ctx_candidate != desired_n_ctx {
                    tracing::info!(
                        requested = desired_n_ctx,
                        allocated = n_ctx_candidate,
                        "context created at a smaller n_ctx after retrying — the initial size didn't fit"
                    );
                }
                break ctx;
            }
            match shrink_n_ctx(n_ctx_candidate, effective_max_ctx) {
                Some(next) => {
                    tracing::warn!(
                        attempted = n_ctx_candidate,
                        retrying_at = next,
                        "llama_init_from_model failed (likely OOM) — retrying with a smaller n_ctx"
                    );
                    n_ctx_candidate = next;
                }
                None => {
                    unsafe { ffi::llama_model_free(model.as_ptr()) };
                    return Err(anyhow!(
                        "llama_init_from_model failed even at the minimum viable context \
                         ({effective_max_ctx} tokens) — not enough memory for this model at \
                         this context length"
                    ));
                }
            }
        };

        // Vision/multimodal: load the paired mmproj GGUF via mtmd, if given. A bad
        // pairing (wrong architecture, corrupt file) fails loudly here rather than
        // producing garbage output at inference time.
        let mtmd_ctx = match mmproj_path {
            Some(p) => {
                let mmproj_cstr = CString::new(
                    p.to_str()
                        .ok_or_else(|| anyhow!("mmproj path not valid UTF-8"))?,
                )?;
                let mut mtmd_params = unsafe { ffi::mtmd_context_params_default() };
                let marker_cstr = CString::new(crate::engine::model::MEDIA_MARKER)
                    .expect("MEDIA_MARKER is a valid C string");
                mtmd_params.media_marker = marker_cstr.as_ptr();
                let raw = unsafe {
                    ffi::mtmd_init_from_file(mmproj_cstr.as_ptr(), model.as_ptr(), mtmd_params)
                };
                // marker_cstr/mmproj_cstr only need to outlive the call above — mtmd
                // copies both into its own storage during init.
                Some(NonNull::new(raw).ok_or_else(|| {
                    unsafe { ffi::llama_free(ctx.as_ptr()) };
                    unsafe { ffi::llama_model_free(model.as_ptr()) };
                    anyhow!(
                        "mtmd_init_from_file failed for {p:?} — check the mmproj matches this model's architecture"
                    )
                })?)
            }
            None => None,
        };

        // Named LoRA adapters (--lora-modules): loaded once alongside the model,
        // attached/detached per decode step by do_prefill/do_decode via
        // llama_set_adapters_lora — see docs/design/lora-support.md. A bad
        // adapter file fails loudly here, same posture as mmproj above.
        let mut lora_adapters = std::collections::HashMap::new();
        for (name, path, scale) in lora_modules {
            let path_cstr = CString::new(
                path.to_str()
                    .ok_or_else(|| anyhow!("lora adapter path not valid UTF-8: {path:?}"))?,
            )?;
            let raw = unsafe { ffi::llama_adapter_lora_init(model.as_ptr(), path_cstr.as_ptr()) };
            let adapter = NonNull::new(raw).ok_or_else(|| {
                for (adapter, _) in lora_adapters.values() {
                    let adapter: &NonNull<ffi::llama_adapter_lora> = adapter;
                    unsafe { ffi::llama_adapter_lora_free(adapter.as_ptr()) };
                }
                if let Some(mtmd_ctx) = mtmd_ctx {
                    unsafe { ffi::mtmd_free(mtmd_ctx.as_ptr()) };
                }
                unsafe { ffi::llama_free(ctx.as_ptr()) };
                unsafe { ffi::llama_model_free(model.as_ptr()) };
                anyhow!(
                    "llama_adapter_lora_init failed for '{name}' ({path:?}) — check the \
                     adapter matches this model's architecture"
                )
            })?;
            lora_adapters.insert(name.clone(), (adapter, *scale));
        }

        // SAFETY: We manually implement Send + Sync for LlamaCppModel below.
        // The Arc<Mutex<NonNull<...>>> is intentionally used here for shared ownership
        // across clone (e.g. future multi-backend); the unsafe impls guarantee thread safety.
        #[allow(clippy::arc_with_non_send_sync)]
        let ctx_arc = Arc::new(std::sync::Mutex::new(ctx));
        let eog_tokens: Vec<i32> = (0..config.vocab_size as i32)
            .filter(|&id| unsafe { ffi::llama_vocab_is_eog(vocab, id) })
            .collect();
        Ok(Self {
            _model: model,
            _ctx: ctx_arc,
            vocab,
            config,
            eos_token,
            eog_tokens,
            embed_seq_id: (n_seq - 1) as i32,
            effective_ctx: effective_max_ctx,
            owns_model: true,
            chat_env: std::sync::OnceLock::new(),
            grammars: dashmap::DashMap::new(),
            decode_bisection_retries: std::sync::atomic::AtomicU64::new(0),
            mtmd_ctx,
            lora_adapters,
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
        let n_seq = (max_batch_size as u32).max(4) + 1; // +1: dedicated embeddings slot (last id)

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
        ctx_params.kv_unified = true; // see load() for why
        let n_threads = resolve_n_threads(); // see load() for why
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
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
            eog_tokens: self.eog_tokens.clone(),
            embed_seq_id: (n_seq - 1) as i32,
            effective_ctx: effective_max_ctx,
            owns_model: false, // weights are owned by the original LlamaCppModel
            chat_env: std::sync::OnceLock::new(),
            grammars: dashmap::DashMap::new(),
            decode_bisection_retries: std::sync::atomic::AtomicU64::new(0),
            // bench-kv compares KV cache types, not vision/LoRA; the original
            // instance (if any) keeps ownership of its mtmd context and adapters.
            mtmd_ctx: None,
            lora_adapters: std::collections::HashMap::new(),
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

    fn fim_tokens(&self) -> Option<crate::engine::model::FimTokens> {
        // llama.cpp returns LLAMA_TOKEN_NULL (-1) for a vocabulary that has no FIM
        // tokens. All three are required: a model with only some of them cannot be
        // driven through the infill format, and guessing the missing one would
        // silently produce a prompt the model was never trained on.
        let (prefix, suffix, middle) = unsafe {
            (
                ffi::llama_vocab_fim_pre(self.vocab),
                ffi::llama_vocab_fim_suf(self.vocab),
                ffi::llama_vocab_fim_mid(self.vocab),
            )
        };
        (prefix >= 0 && suffix >= 0 && middle >= 0).then_some(crate::engine::model::FimTokens {
            prefix,
            suffix,
            middle,
        })
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
        tools: Option<&serde_json::Value>,
    ) -> Result<Vec<i32>> {
        self.build_prompt_tokens_impl(messages, enable_thinking, tools)
    }

    fn native_tool_call_format(&self) -> Option<NativeToolFormat> {
        // Detected from the model's OWN chat template, never its name — same
        // principle as `reasoning_delimiters`/`supports_thinking` above. Hermes and
        // Qwen tool-use templates instruct the model to wrap tool calls in
        // `<tool_call>...</tool_call>`; Mistral's instructs the `[TOOL_CALLS]`
        // marker. A model without either marker in its template gets fox's generic
        // prompt-injected tool listing instead. Llama3 is deliberately not detected
        // here — see `NativeToolFormat`'s doc comment.
        let t = self.raw_chat_template()?;
        if t.contains("<tool_call>") {
            Some(NativeToolFormat::Hermes)
        } else if t.contains("[TOOL_CALLS]") {
            Some(NativeToolFormat::Mistral)
        } else {
            None
        }
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

    fn supports_vision(&self) -> bool {
        self.mtmd_ctx.is_some()
    }

    fn lora_adapter_names(&self) -> Vec<String> {
        self.lora_adapters.keys().cloned().collect()
    }

    fn tokenize_multimodal(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
        tools: Option<&serde_json::Value>,
        images: &[Vec<u8>],
    ) -> Result<crate::engine::model::MultimodalChunks> {
        self.tokenize_multimodal_impl(messages, enable_thinking, tools, images)
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

    fn trim_sequence(&self, seq_id: i32, from_pos: usize) {
        let ctx_guard = match self._ctx.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        unsafe {
            let mem = ffi::llama_get_memory(ctx_guard.as_ptr() as *const _);
            if !mem.is_null() {
                // p1 = -1 → remove [from_pos, ∞) for this sequence.
                ffi::llama_memory_seq_rm(mem, seq_id, from_pos as i32, -1);
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
        // NOT `llama_memory_can_shift` — verified against a real Mamba model
        // (2026-08-01) that it returns `true` for recurrent memory too
        // ("shifting the pos is trivial for recurrent models",
        // `llama-memory-recurrent.cpp`), the opposite of what this method
        // needs. `llama_model_is_recurrent`/`llama_model_is_hybrid` are the
        // model-level, architecture-authoritative answer to what this method
        // actually asks: does fox's block-level KV copy-on-write (the prefix
        // cache's mechanism) apply to this model at all. See
        // docs/design/mla-recurrent-kv-sizing.md.
        unsafe {
            let model = self._model.as_ptr();
            !ffi::llama_model_is_recurrent(model) && !ffi::llama_model_is_hybrid(model)
        }
    }

    fn roll_context(&self, seq_id: i32, n_keep: usize, n_discard: usize) -> Result<()> {
        if n_discard == 0 {
            return Ok(());
        }
        // Recurrent/hybrid models have no per-token positional KV to drop/shift at
        // all (a fixed-size state, not growing blocks) — `llama_memory_can_shift`
        // alone is not a reliable guard here (it reports `true` for recurrent
        // memory, since repositioning is a cheap no-op for it, not because fox's
        // seq_rm/seq_add-based rolling is meaningful for it). Checked in addition
        // to, not instead of, the existing can_shift check below.
        if unsafe { ffi::llama_model_is_recurrent(self._model.as_ptr()) }
            || unsafe { ffi::llama_model_is_hybrid(self._model.as_ptr()) }
        {
            return Err(anyhow!(
                "context rolling is not supported for recurrent/hybrid models"
            ));
        }
        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {}", e))?;
        unsafe {
            let mem = ffi::llama_get_memory(ctx_guard.as_ptr() as *const _);
            if mem.is_null() {
                return Err(anyhow!("no memory backend for context roll"));
            }
            if !ffi::llama_memory_can_shift(mem) {
                return Err(anyhow!("KV cache is not shiftable"));
            }
            let keep = n_keep as i32;
            let discard = n_discard as i32;
            // Drop [n_keep, n_keep + n_discard) …
            if !ffi::llama_memory_seq_rm(mem, seq_id, keep, keep + discard) {
                return Err(anyhow!("llama_memory_seq_rm failed during context roll"));
            }
            // … then shift every surviving token after the hole down by n_discard so
            // positions stay contiguous (p1 = -1 → [keep + discard, ∞)).
            ffi::llama_memory_seq_add(mem, seq_id, keep + discard, -1, -discard);
        }
        Ok(())
    }

    fn free_grammar(&self, req_id: u64) {
        // Removing the entry drops its GrammarSampler, which frees the llama.cpp sampler.
        self.grammars.remove(&req_id);
    }

    fn speculative_decode_sync(
        &self,
        _req_id: u64,
        request: &InferenceRequestForModel,
        drafts: Vec<i32>,
    ) -> Result<Vec<Logits>> {
        self.do_speculative_decode(request, drafts)
    }

    fn draft_propose(
        &self,
        seq_id: i32,
        new_tokens: &[i32],
        base_pos: i32,
        draft_len: usize,
    ) -> Vec<i32> {
        self.do_draft_propose(seq_id, new_tokens, base_pos, draft_len)
    }

    fn rerank_score(&self, tokens: &[i32]) -> Result<f32> {
        self.do_rerank_score(tokens)
    }

    fn state_seq_save(&self, seq_id: i32) -> Result<Vec<u8>> {
        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        let ctx = ctx_guard.as_ptr();
        // FLAGS_NONE, never ON_DEVICE: llama.h:883-885 warns that the on-device
        // variant keeps the data in device buffers AND invalidates every prior state
        // for that seq_id — the opposite of what a host-RAM cache needs.
        let size = unsafe {
            ffi::llama_state_seq_get_size_ext(ctx, seq_id, ffi::LLAMA_STATE_SEQ_FLAGS_NONE)
        };
        if size == 0 {
            return Err(anyhow!("sequence {seq_id} has no state to save"));
        }
        let mut buf = vec![0u8; size];
        let written = unsafe {
            ffi::llama_state_seq_get_data_ext(
                ctx,
                buf.as_mut_ptr(),
                size,
                seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
            )
        };
        if written == 0 {
            return Err(anyhow!("llama_state_seq_get_data_ext wrote nothing"));
        }
        buf.truncate(written);
        Ok(buf)
    }

    fn state_seq_load(&self, seq_id: i32, data: &[u8]) -> Result<usize> {
        if data.is_empty() {
            return Err(anyhow!("cannot restore an empty state blob"));
        }
        let ctx_guard = self
            ._ctx
            .lock()
            .map_err(|e| anyhow!("lock poisoned: {e}"))?;
        let ctx = ctx_guard.as_ptr();
        // The destination must be empty first: set_data_ext writes cells at their
        // recorded positions and does not clear, so leftovers would survive
        // underneath the restored state and corrupt the sequence.
        unsafe {
            let mem = ffi::llama_get_memory(ctx as *const _);
            if !mem.is_null() {
                ffi::llama_memory_seq_rm(mem, seq_id, 0, -1);
            }
        }
        let read = unsafe {
            ffi::llama_state_seq_set_data_ext(
                ctx,
                data.as_ptr(),
                data.len(),
                seq_id,
                ffi::LLAMA_STATE_SEQ_FLAGS_NONE,
            )
        };
        if read == 0 {
            return Err(anyhow!("llama_state_seq_set_data_ext rejected the blob"));
        }
        Ok(read)
    }

    fn sep_token_id(&self) -> Option<i32> {
        let sep = unsafe { ffi::llama_vocab_sep(self.vocab) };
        (sep >= 0).then_some(sep)
    }

    fn vocab_fingerprint(&self) -> u64 {
        self.compute_vocab_fingerprint()
    }

    fn bisection_retry_count(&self) -> u64 {
        self.decode_bisection_retries
            .load(std::sync::atomic::Ordering::Relaxed)
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
        let supports_seq_copy = self.supports_seq_copy();

        ModelInfo {
            kv_memory_class: crate::engine::model::model_info::classify_kv_memory(
                &arch_name,
                supports_seq_copy,
            ),
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
            n_params: unsafe { ffi::llama_model_n_params(self._model.as_ptr()) },
            eos_token_id: self.eos_token,
            has_chat_template,
            supports_thinking: self.supports_thinking(),
            supports_seq_copy,
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
    use super::{resolve_context_len, shrink_n_ctx};

    #[test]
    fn shrink_n_ctx_halves_toward_floor() {
        assert_eq!(shrink_n_ctx(8192, 2048), Some(4096));
        assert_eq!(shrink_n_ctx(4096, 2048), Some(2048));
    }

    #[test]
    fn shrink_n_ctx_clamps_at_floor_not_below() {
        // 2048/2=1024, which is below the 2048 floor — clamp up to the floor.
        assert_eq!(shrink_n_ctx(3000, 2048), Some(2048));
    }

    #[test]
    fn shrink_n_ctx_none_once_at_floor() {
        assert_eq!(shrink_n_ctx(2048, 2048), None);
    }

    #[test]
    fn shrink_n_ctx_none_below_floor() {
        assert_eq!(shrink_n_ctx(1000, 2048), None);
    }

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
