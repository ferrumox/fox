// C shim over llama.cpp's staging NextN/MTP API.
//
// Everything else fox needs for MTP speculative decoding is already public C in
// `llama.h` (LLAMA_CONTEXT_TYPE_MTP, llama_context_params::ctx_other,
// llama_model_n_layer_nextn, llama_model_n_embd_out, llama_set_sampler). These four
// functions are not: they live in `src/llama-ext.h`, which declares itself staging
// ("breaking changes and C++ are allowed") and, being outside any `extern "C"`, exports
// C++-mangled symbols:
//
//   _Z26llama_set_embeddings_nextnP13llama_contextbb
//   _Z30llama_get_embeddings_nextn_ithP13llama_contexti
//   _Z28llama_set_nextn_layer_offsetP13llama_contexti
//   _Z19llama_get_ctx_otherP13llama_context
//
// Rust could name those mangled symbols directly via #[link_name], but then a change to
// any of these signatures upstream would surface as an unresolved symbol at link time,
// naming a mangled string and pointing at no source line. Letting the C++ compiler
// resolve them here means the same change fails *compiling this file*, with the real
// signature in the error. That is the whole point of the shim: one file to fix, and it
// says what broke.
//
// This file is only compiled when llama.cpp is actually built (never under
// FOX_SKIP_LLAMA=1 / cfg(fox_stub)).

#include "llama.h"
#include "llama-ext.h"

#include "common.h"
#include "log.h"
#include "speculative.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <vector>

extern "C" {

// Make the context emit NextN hidden states alongside its normal outputs.
// `masked` is set on the MTP context and clear on the target — see the driver in
// common/speculative.cpp (common_speculative_impl_draft_mtp).
void fox_mtp_set_embeddings_nextn(struct llama_context * ctx, bool value, bool masked) {
    llama_set_embeddings_nextn(ctx, value, masked);
}

// Hidden row `i` of the last decode: row 0 is the sampled token, row N the Nth
// accepted draft token. Returns a pointer into the context's own buffer, valid until
// the next decode — the caller must copy before decoding again.
float * fox_mtp_get_embeddings_nextn_ith(struct llama_context * ctx, int32_t i) {
    return llama_get_embeddings_nextn_ith(ctx, i);
}

// Select which appended NextN block the MTP graph runs. Qwen3.5/3.8 ship a single
// trained head, so fox leaves this at the default 0; it exists for the chained-head
// architectures (step35).
void fox_mtp_set_nextn_layer_offset(struct llama_context * ctx, int32_t offset) {
    llama_set_nextn_layer_offset(ctx, offset);
}

// The context this one was linked to via llama_context_params::ctx_other. The driver
// uses it to detect the shared-memory variant (gemma4), where the MTP head shares the
// target's KV cache instead of owning one.
struct llama_context * fox_mtp_get_ctx_other(struct llama_context * ctx) {
    return llama_get_ctx_other(ctx);
}

} // extern "C"

// ─────────────────────────────────────────────────────────────────────────────
// The MTP speculative driver itself.
//
// fox wraps common/speculative.cpp instead of reimplementing it. That driver is 444
// lines with three architecture modes and documented traps (cross-batch carryover of
// hidden rows, a deferred pairing boundary); porting it to Rust would mean owning
// exactly the kind of logic that is easy to get subtly, silently wrong, and then
// keeping it in step with upstream. Wrapping costs one CMake flag: measured at 15s of
// extra compile time from scratch, with no network fetches (common's third-party deps
// are vendored under vendor/llama.cpp/vendor/).
//
// The API is C++ (std::vector, references, default-heavy structs), so the translation
// to plain C lives here. Everything below is a thin adapter: no MTP logic of its own.

struct fox_spec {
    common_params_speculative params;
    common_speculative *      spec = nullptr;

    // draft_params holds raw pointers into these, so they must outlive each draft()
    // call and keep a stable address — hence members, not locals.
    llama_tokens prompt_buf;
    llama_tokens result_buf;

    // Per-sequence snapshots of the driver's state, taken around a draft. Kept here
    // rather than handed to Rust because nothing on that side needs to read them: they
    // are written and restored in the same pair of calls, and copying a recurrent state
    // across the FFI boundary twice per speculative step would be pure cost.
    std::map<llama_seq_id, std::vector<uint8_t>> saved;
};

extern "C" {

// Raise common/'s log threshold so the driver's own SPC_TRC/SPC_WRN traces appear.
//
// They are the only view into what the driver thinks it received — `begin()` warns when
// the prefill hidden states never reached the MTP context, and the trace lines show what
// it paired up per step. fox never calls `common_init`, so without this the threshold
// sits at its default and those lines are dropped. Driven by FOX_MTP_VERBOSE.
void fox_spec_set_log_verbosity(int verbosity) {
    common_log_set_verbosity_thold(verbosity);
}

// Build an MTP driver over an already-created target context and MTP context.
// fox creates both contexts itself through public llama.h (ctx_type =
// LLAMA_CONTEXT_TYPE_MTP, ctx_other = target); this only wires up the driver.
// Returns null if the driver could not be created.
struct fox_spec * fox_spec_mtp_init(struct llama_context * ctx_tgt,
                                    struct llama_context * ctx_dft,
                                    int32_t                n_max,
                                    uint32_t               n_seq) {
    if (!ctx_tgt || !ctx_dft) {
        return nullptr;
    }

    auto * s = new fox_spec();
    s->params.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    s->params.draft.ctx_tgt = ctx_tgt;
    s->params.draft.ctx_dft = ctx_dft;
    if (n_max > 0) {
        s->params.draft.n_max = n_max;
    }

    s->spec = common_speculative_init(s->params, n_seq);
    if (!s->spec) {
        delete s;
        return nullptr;
    }
    return s;
}

void fox_spec_free(struct fox_spec * s) {
    if (!s) {
        return;
    }
    if (s->spec) {
        common_speculative_free(s->spec);
    }
    delete s;
}

// Start a new generation on `seq` with the given prompt.
void fox_spec_begin(struct fox_spec * s, llama_seq_id seq, const llama_token * prompt, size_t n) {
    if (!s) {
        return;
    }
    s->prompt_buf.assign(prompt, prompt + n);
    common_speculative_begin(s->spec, seq, s->prompt_buf);
}

// Feed the target's decode batch to the driver so it can capture the hidden rows it
// needs. Must be called for every target decode while drafting is active.
bool fox_spec_process(struct fox_spec * s, const struct llama_batch * batch) {
    return (s && batch) ? common_speculative_process(s->spec, *batch) : false;
}

// Whether the target context must be configured to emit post-norm / NextN embeddings.
bool fox_spec_need_embd(struct fox_spec * s) {
    return s ? common_speculative_need_embd(s->spec) : false;
}

bool fox_spec_need_embd_nextn(struct fox_spec * s) {
    return s ? common_speculative_need_embd_nextn(s->spec) : false;
}

// Draft up to `out_cap` tokens for `seq`, continuing after `id_last` at `n_past`.
// `prompt`/`n_prompt` is the sequence so far (prompt ++ generated), which the driver
// still needs as an explicit argument upstream. Returns how many tokens were written
// to `out`, or -1 on error.
int32_t fox_spec_draft(struct fox_spec *   s,
                       llama_seq_id        seq,
                       llama_pos           n_past,
                       llama_token         id_last,
                       const llama_token * prompt,
                       size_t              n_prompt,
                       llama_token *       out,
                       int32_t             out_cap) {
    if (!s || !out || out_cap <= 0) {
        return -1;
    }

    s->prompt_buf.assign(prompt, prompt + n_prompt);
    s->result_buf.clear();

    auto & dp = common_speculative_get_draft_params(s->spec, seq);
    dp.drafting = true;
    dp.n_max    = out_cap;
    dp.n_past   = n_past;
    dp.id_last  = id_last;
    dp.prompt   = &s->prompt_buf;
    dp.result   = &s->result_buf;

    common_speculative_draft(s->spec);

    const int32_t n = (int32_t) std::min<size_t>(s->result_buf.size(), (size_t) out_cap);
    if (n > 0) {
        std::memcpy(out, s->result_buf.data(), n * sizeof(llama_token));
    }
    return n;
}

// Snapshot the driver's state for `seq`, so a draft can be rolled back.
//
// Drafting advances the head's own KV by the tokens it proposes, which leaves it ahead
// of the target and makes llama.cpp reject the verification batch that follows
// ("inconsistent sequence positions"). A partial `llama_memory_seq_rm` does not undo it
// on a hybrid context — llama-server checkpoints and restores instead, and so does fox.
// Returns false when the driver has no state to give for this sequence yet.
bool fox_spec_save_state(struct fox_spec * s, llama_seq_id seq) {
    if (!s) {
        return false;
    }
    std::vector<uint8_t> data;
    if (!common_speculative_get_state(s->spec, seq, data)) {
        return false;
    }
    s->saved[seq] = std::move(data);
    return true;
}

// Put back the state saved by the matching `fox_spec_save_state`. No-op (false) when
// nothing was saved for this sequence.
bool fox_spec_restore_state(struct fox_spec * s, llama_seq_id seq) {
    if (!s) {
        return false;
    }
    auto it = s->saved.find(seq);
    if (it == s->saved.end()) {
        return false;
    }
    common_speculative_set_state(s->spec, seq, it->second);
    return true;
}

// How many bytes the last snapshot for `seq` took, for cost reporting. 0 when none.
size_t fox_spec_saved_bytes(struct fox_spec * s, llama_seq_id seq) {
    if (!s) {
        return 0;
    }
    auto it = s->saved.find(seq);
    return it == s->saved.end() ? 0 : it->second.size();
}

// Tell the driver how many of its drafted tokens the target actually accepted.
void fox_spec_accept(struct fox_spec * s, llama_seq_id seq, uint16_t n_accepted) {
    if (s) {
        common_speculative_accept(s->spec, seq, n_accepted);
    }
}

} // extern "C"
