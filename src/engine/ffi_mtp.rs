//! Rust side of the NextN/MTP staging API, isolated from the rest of the FFI.
//!
//! Everything else MTP speculative decoding needs is public C in `llama.h` and already
//! comes out of bindgen: `LLAMA_CONTEXT_TYPE_MTP`, `llama_context_params::ctx_other`,
//! `llama_model_n_layer_nextn`, `llama_model_n_embd_out`, `llama_set_sampler`. Only the
//! four functions below are missing, because they live in llama.cpp's `src/llama-ext.h`
//! — a header that declares itself staging, with "breaking changes and C++ are allowed".
//!
//! They are reached through `csrc/mtp_shim.cpp` rather than by naming their mangled
//! symbols here, so an upstream signature change fails while compiling that one file,
//! with the real signature in the error, instead of surfacing as an unresolved
//! `_Z26llama_set_embeddings_nextn...` at link time.
//!
//! Build with `FOX_NO_MTP=1` to drop the shim entirely (sets `cfg(fox_no_mtp)`).

#![cfg(fox_mtp)]
// The MTP proposer that consumes these lands on top of this bridge; until then nothing
// calls them and a real (non-stub) build warns on all four.
#![allow(dead_code)]

use super::ffi::{llama_batch, llama_context};

extern "C" {
    fn fox_mtp_set_embeddings_nextn(ctx: *mut llama_context, value: bool, masked: bool);
    fn fox_mtp_get_embeddings_nextn_ith(ctx: *mut llama_context, i: i32) -> *mut f32;
    fn fox_mtp_set_nextn_layer_offset(ctx: *mut llama_context, offset: i32);
    fn fox_mtp_get_ctx_other(ctx: *mut llama_context) -> *mut llama_context;
}

/// Make `ctx` emit NextN hidden states alongside its normal outputs.
///
/// The driver sets `masked = false` on the target context and `true` on the MTP one.
///
/// # Safety
/// `ctx` must be a live context.
pub unsafe fn set_embeddings_nextn(ctx: *mut llama_context, value: bool, masked: bool) {
    fox_mtp_set_embeddings_nextn(ctx, value, masked);
}

/// Hidden row `i` of the most recent decode: row 0 is the sampled token, row N the Nth
/// accepted draft token.
///
/// Returns `None` when the context produced no such row. The slice borrows the
/// context's own output buffer and is invalidated by the next decode, so callers must
/// copy out of it before decoding again — hence the returned lifetime is tied to
/// nothing, and this stays `unsafe`.
///
/// # Safety
/// `ctx` must be a live context that was configured with [`set_embeddings_nextn`], and
/// `n_embd` must be its NextN row width (`llama_model_n_embd_out`). The returned pointer
/// must not outlive the next `llama_decode` on `ctx`.
pub unsafe fn get_embeddings_nextn_ith<'a>(
    ctx: *mut llama_context,
    i: i32,
    n_embd: usize,
) -> Option<&'a [f32]> {
    let ptr = fox_mtp_get_embeddings_nextn_ith(ctx, i);
    (!ptr.is_null()).then(|| std::slice::from_raw_parts(ptr, n_embd))
}

/// Select which appended NextN block the MTP graph runs.
///
/// Qwen3.5/3.8 ship a single trained head (`mtp_num_hidden_layers: 1`), so fox leaves
/// this at the default 0; it exists for chained-head architectures.
///
/// # Safety
/// `ctx` must be a live MTP context.
pub unsafe fn set_nextn_layer_offset(ctx: *mut llama_context, offset: i32) {
    fox_mtp_set_nextn_layer_offset(ctx, offset);
}

/// The context `ctx` was linked to via `llama_context_params::ctx_other`, or null.
///
/// Equal to the target context when the MTP head shares the target's KV cache instead
/// of owning one — a per-architecture distinction the driver branches on.
///
/// # Safety
/// `ctx` must be a live context.
pub unsafe fn get_ctx_other(ctx: *mut llama_context) -> *mut llama_context {
    fox_mtp_get_ctx_other(ctx)
}

// ── MTP speculative driver ───────────────────────────────────────────────────
// fox wraps llama.cpp's own driver (common/speculative.cpp) rather than porting its
// 444 lines of hidden-state carryover to Rust. `csrc/mtp_shim.cpp` flattens its C++
// API (std::vector, references) to the C entry points below.

/// Opaque driver handle owned by the shim.
#[repr(C)]
pub struct FoxSpec {
    _private: [u8; 0],
}

extern "C" {
    fn fox_spec_set_log_verbosity(verbosity: i32);
    fn fox_spec_mtp_init(
        ctx_tgt: *mut llama_context,
        ctx_dft: *mut llama_context,
        n_max: i32,
        n_seq: u32,
    ) -> *mut FoxSpec;
    fn fox_spec_free(spec: *mut FoxSpec);
    fn fox_spec_begin(spec: *mut FoxSpec, seq: i32, prompt: *const i32, n: usize);
    fn fox_spec_process(spec: *mut FoxSpec, batch: *const llama_batch) -> bool;
    fn fox_spec_need_embd(spec: *mut FoxSpec) -> bool;
    fn fox_spec_need_embd_nextn(spec: *mut FoxSpec) -> bool;
    #[allow(clippy::too_many_arguments)]
    fn fox_spec_draft(
        spec: *mut FoxSpec,
        seq: i32,
        n_past: i32,
        id_last: i32,
        prompt: *const i32,
        n_prompt: usize,
        out: *mut i32,
        out_cap: i32,
    ) -> i32;
    fn fox_spec_accept(spec: *mut FoxSpec, seq: i32, n_accepted: u16);
    fn fox_spec_save_state(spec: *mut FoxSpec, seq: i32) -> bool;
    fn fox_spec_restore_state(spec: *mut FoxSpec, seq: i32) -> bool;
    fn fox_spec_saved_bytes(spec: *mut FoxSpec, seq: i32) -> usize;
}

/// Raise common/'s log threshold so the driver's own traces are visible.
///
/// fox never calls `common_init`, so the threshold sits at its default and the driver's
/// SPC_* lines — including `begin`'s warning about missing prefill hidden states — are
/// dropped. Set `FOX_MTP_VERBOSE=<n>` to see them.
pub fn set_log_verbosity(verbosity: i32) {
    unsafe { fox_spec_set_log_verbosity(verbosity) }
}

/// Owning handle to the MTP speculative driver.
///
/// Deliberately neither `Send` nor `Sync`: the driver keeps per-sequence hidden-state
/// carryover tied to the two contexts it was built over, and llama.cpp drives it from
/// one decode loop. Making it cross threads would need a claim about llama.cpp's
/// internals that nothing here establishes.
pub struct MtpDriver {
    ptr: *mut FoxSpec,
}

impl MtpDriver {
    /// Build a driver over an existing target context and MTP context.
    ///
    /// The caller creates both through public `llama.h`: the MTP context with
    /// `ctx_type = LLAMA_CONTEXT_TYPE_MTP` and `ctx_other` pointing at the target.
    /// Returns `None` if llama.cpp declined to build the driver.
    ///
    /// # Safety
    /// Both contexts must be live and must outlive the returned driver.
    pub unsafe fn new(
        ctx_tgt: *mut llama_context,
        ctx_mtp: *mut llama_context,
        n_max: i32,
        n_seq: u32,
    ) -> Option<Self> {
        let ptr = fox_spec_mtp_init(ctx_tgt, ctx_mtp, n_max, n_seq);
        (!ptr.is_null()).then_some(Self { ptr })
    }

    /// Start a new generation on `seq`.
    pub fn begin(&mut self, seq: i32, prompt: &[i32]) {
        unsafe { fox_spec_begin(self.ptr, seq, prompt.as_ptr(), prompt.len()) }
    }

    /// Feed the target's decode batch to the driver so it can capture hidden rows.
    ///
    /// # Safety
    /// `batch` must be the batch just submitted to the target context.
    pub unsafe fn process(&mut self, batch: *const llama_batch) -> bool {
        fox_spec_process(self.ptr, batch)
    }

    /// Whether the target context has to emit post-norm embeddings.
    pub fn needs_embd(&mut self) -> bool {
        unsafe { fox_spec_need_embd(self.ptr) }
    }

    /// Whether the target context has to emit NextN embeddings.
    pub fn needs_embd_nextn(&mut self) -> bool {
        unsafe { fox_spec_need_embd_nextn(self.ptr) }
    }

    /// Draft up to `out.len()` tokens continuing after `id_last` at position `n_past`.
    /// Returns the drafted slice, empty when the driver proposed nothing.
    pub fn draft<'a>(
        &mut self,
        seq: i32,
        n_past: i32,
        id_last: i32,
        prompt: &[i32],
        out: &'a mut [i32],
    ) -> &'a [i32] {
        let n = unsafe {
            fox_spec_draft(
                self.ptr,
                seq,
                n_past,
                id_last,
                prompt.as_ptr(),
                prompt.len(),
                out.as_mut_ptr(),
                out.len() as i32,
            )
        };
        &out[..n.max(0) as usize]
    }

    /// Snapshot the driver's state for `seq` so a draft can be undone.
    ///
    /// Drafting advances the head's KV past where the target is, and on a hybrid context
    /// a partial trim will not undo it — so the state is saved before drafting and put
    /// back afterwards, the way llama-server does it. `false` means the driver had no
    /// state to give yet.
    pub fn save_state(&mut self, seq: i32) -> bool {
        unsafe { fox_spec_save_state(self.ptr, seq) }
    }

    /// Restore the state saved by [`save_state`]. `false` when there was none.
    pub fn restore_state(&mut self, seq: i32) -> bool {
        unsafe { fox_spec_restore_state(self.ptr, seq) }
    }

    /// Size of the last snapshot for `seq`, so the cost of doing this per step is
    /// observable rather than assumed.
    pub fn saved_bytes(&mut self, seq: i32) -> usize {
        unsafe { fox_spec_saved_bytes(self.ptr, seq) }
    }

    /// Report how many drafted tokens the target accepted.
    pub fn accept(&mut self, seq: i32, n_accepted: u16) {
        unsafe { fox_spec_accept(self.ptr, seq, n_accepted) }
    }
}

impl Drop for MtpDriver {
    fn drop(&mut self) {
        unsafe { fox_spec_free(self.ptr) }
    }
}
