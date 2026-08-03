// GET /props and GET /slots — server and per-sequence introspection
// (llama-server's server-context.cpp:4551-4600 and :4475-4516).
//
// `/props` answers "what is this server actually serving, and how is it configured",
// which today can only be guessed at from flags and filenames. Every field comes from
// the *loaded model*, never from its filename — `/api/show` derives its answers from
// the file stem and consequently reports `parameters: ""`, `template: ""` and
// `parameter_size: "unknown"`.
//
// `/slots` reports what each llama.cpp sequence holds. It deliberately does NOT report
// the resident tokens themselves: a parked sequence is another user's conversation, and
// this endpoint has no per-user authorisation of its own. llama-server redacts the same
// fields unless LLAMA_SERVER_SLOTS_DEBUG is set.

use axum::{extract::State, Json};
use serde::Serialize;

use crate::api::router::AppState;
use crate::api::shared::sampling_defaults as defaults;
use crate::scheduler::SlotSnapshot;

#[derive(Debug, Serialize)]
pub struct PropsResponse {
    /// Which model these properties describe, or `null` when none is resident —
    /// fox loads lazily, so "no model yet" is a normal state, not an error.
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_info: Option<PropsModelInfo>,
    pub total_slots: usize,
    pub build_info: String,
    /// Which fox-level features this server was started with.
    pub features: PropsFeatures,
    /// The sampling defaults a request gets when it sets nothing — **per API
    /// surface**, because fox's differ on purpose and a caller cannot otherwise find
    /// out which set applies to it.
    pub default_generation_settings: DefaultGenerationSettings,
}

/// fox serves two API families whose upstreams ship different sampling defaults, so
/// fox's differ too (see `api/shared/sampling_defaults.rs`, where a test locks the
/// divergence). llama-server has a single set and publishes it under the same
/// `/props` key; fox publishes both, since which one applies depends on the endpoint
/// a client happens to use.
///
/// This exists because the divergence has a measurable cost a caller cannot see:
/// `/v1/*` mirrors OpenAI, which has no `top_k`, so `top_k = 0` means the sampler
/// softmaxes the whole vocabulary instead of 40 candidates — worth **8.7%** of decode
/// throughput on a Radeon 890M (`docs/design/rocm-benchmarking-2026-08.md`). The
/// default stays as it is; publishing it makes an informed `"top_k": 40` possible.
#[derive(Debug, Serialize)]
pub struct DefaultGenerationSettings {
    pub openai: SurfaceDefaults,
    pub ollama: SurfaceDefaults,
    /// Server-wide, applies to both (`--repeat-last-n`).
    pub repeat_last_n: i32,
}

#[derive(Debug, Serialize)]
pub struct SurfaceDefaults {
    pub temperature: f32,
    pub top_p: f32,
    /// `0` = disabled, i.e. no truncation before the softmax.
    pub top_k: u32,
    /// `1.0` = disabled.
    pub repeat_penalty: f32,
    pub max_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct PropsModelInfo {
    pub architecture: String,
    pub backend: String,
    /// Context actually available per sequence — what fox allocated, which is not
    /// necessarily the model's trained maximum.
    pub n_ctx: u32,
    pub n_ctx_train: u32,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub vocab_size: usize,
    pub eos_token_id: i32,
    pub has_chat_template: bool,
    pub supports_thinking: bool,
    pub supports_vision: bool,
    /// Whether KV can be reused across requests at all — false for recurrent/hybrid
    /// memory, where fox disables prefix caching entirely.
    pub supports_kv_reuse: bool,
    /// Fill-in-the-middle capable, i.e. whether `/infill` will accept this model.
    pub supports_infill: bool,
}

#[derive(Debug, Serialize)]
pub struct PropsFeatures {
    pub kv_reuse: bool,
    pub reranking: bool,
    pub speculative: bool,
}

pub async fn props(State(state): State<AppState>) -> Json<PropsResponse> {
    let cfg = state.registry.config();
    // Report on whichever model is resident. fox is lazy-loading, so asking for
    // properties must never *cause* a load — that would evict the running model
    // under the default --max-models 1.
    let resident = state
        .registry
        .loaded()
        .into_iter()
        .find(|(name, _)| *name == state.primary_model)
        .or_else(|| state.registry.loaded().into_iter().next());

    let (model, model_info) = match resident {
        Some((name, entry)) => {
            let info = entry.engine.model_info();
            let mi = PropsModelInfo {
                architecture: info.arch_name.clone(),
                backend: info.backend.clone(),
                n_ctx: info.effective_ctx,
                n_ctx_train: info.n_ctx_train,
                n_embd: info.n_embd,
                n_layer: info.n_layer,
                n_head: info.n_head,
                n_head_kv: info.n_head_kv,
                vocab_size: info.vocab_size,
                eos_token_id: info.eos_token_id,
                has_chat_template: info.has_chat_template,
                supports_thinking: info.supports_thinking,
                supports_vision: entry.engine.supports_vision(),
                supports_kv_reuse: info.supports_seq_copy,
                supports_infill: entry.engine.fim_tokens().is_some(),
            };
            (Some(name), Some(mi))
        }
        None => (None, None),
    };

    Json(PropsResponse {
        model,
        model_info,
        total_slots: cfg.max_batch_size,
        build_info: format!("fox {}", env!("CARGO_PKG_VERSION")),
        default_generation_settings: DefaultGenerationSettings {
            openai: SurfaceDefaults {
                temperature: defaults::TEMPERATURE,
                top_p: defaults::TOP_P,
                top_k: defaults::openai::TOP_K,
                repeat_penalty: defaults::openai::REPETITION_PENALTY,
                max_tokens: defaults::openai::MAX_TOKENS,
            },
            ollama: SurfaceDefaults {
                temperature: defaults::TEMPERATURE,
                top_p: defaults::TOP_P,
                top_k: defaults::ollama::TOP_K,
                repeat_penalty: defaults::ollama::REPEAT_PENALTY,
                max_tokens: defaults::ollama::MAX_TOKENS,
            },
            repeat_last_n: state.repeat_last_n,
        },
        features: PropsFeatures {
            kv_reuse: cfg.kv_reuse,
            reranking: cfg.reranking,
            speculative: cfg.speculative,
        },
    })
}

#[derive(Debug, Serialize)]
pub struct SlotsResponse {
    pub model: Option<String>,
    pub slots: Vec<SlotSnapshot>,
    /// Blocks held in the KV pool, and its capacity. Summing `slots[].blocks` does
    /// NOT give this: a shared prefix block is counted by each slot that references
    /// it, so only this figure falls when sharing takes effect.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_blocks_used: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_blocks_total: Option<usize>,
}

pub async fn slots(State(state): State<AppState>) -> Json<SlotsResponse> {
    let resident = state
        .registry
        .loaded()
        .into_iter()
        .find(|(name, _)| *name == state.primary_model)
        .or_else(|| state.registry.loaded().into_iter().next());

    match resident {
        Some((name, entry)) => {
            let (used, total) = entry.engine.kv_blocks();
            Json(SlotsResponse {
                model: Some(name),
                slots: entry.engine.slots_snapshot(),
                kv_blocks_used: Some(used),
                kv_blocks_total: Some(total),
            })
        }
        // No model resident yet: an empty list, not a 404. Nothing is wrong.
        None => Json(SlotsResponse {
            model: None,
            slots: Vec::new(),
            kv_blocks_used: None,
            kv_blocks_total: None,
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::api::test_helpers::*;

    #[tokio::test]
    async fn props_reports_the_loaded_model_not_the_filename() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/props").await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

        assert_eq!(v["model"], "stub");
        assert!(v["total_slots"].is_number(), "{v}");
        assert!(v["build_info"].as_str().unwrap().starts_with("fox "), "{v}");
        // Facts that /api/show reports as empty strings must be real here.
        let mi = &v["model_info"];
        assert!(
            mi["n_ctx"].is_number(),
            "n_ctx must come from the model: {v}"
        );
        assert!(mi["vocab_size"].as_u64().unwrap() > 0, "{v}");
        assert!(mi["supports_infill"].is_boolean(), "{v}");
        assert!(v["features"]["kv_reuse"].is_boolean(), "{v}");

        // The two surfaces' sampling defaults must both be visible: they differ on
        // purpose, and `/v1/*`'s top_k = 0 costs ~8.7% of decode throughput, which a
        // caller has no other way to discover.
        let d = &v["default_generation_settings"];
        assert_eq!(d["openai"]["top_k"], 0, "{v}");
        assert_eq!(d["ollama"]["top_k"], 40, "{v}");
        assert!(d["repeat_last_n"].is_number(), "{v}");
    }

    #[tokio::test]
    async fn slots_lists_one_entry_per_sequence() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let resp = get_req(app, "/slots").await;
        assert_eq!(resp.status(), 200);
        let v: serde_json::Value = serde_json::from_slice(&body_bytes(resp).await).unwrap();

        let slots = v["slots"].as_array().expect("slots array");
        assert!(!slots.is_empty(), "{v}");
        assert_eq!(slots[0]["state"], "free", "nothing has run yet: {v}");
        assert!(slots[0]["id"].is_number(), "{v}");
        // The resident tokens themselves are another user's conversation and must
        // never be exposed here.
        assert!(
            slots[0].get("tokens").is_none() && slots[0].get("prompt").is_none(),
            "slot contents must not be exposed: {v}"
        );
    }
}
