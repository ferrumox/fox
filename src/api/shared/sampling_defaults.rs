//! Default sampling parameters, deliberately keyed by API surface.
//!
//! fox serves two API families whose upstreams ship different sampling defaults, so
//! fox's defaults diverge **on purpose** — this is a decision, not an accident:
//!
//! * **OpenAI (`/v1/*`)** mirrors OpenAI: no `top_k`, and no repeat penalty (OpenAI
//!   uses `frequency_penalty`/`presence_penalty`, both `0.0` = off by default).
//! * **Ollama (`/api/*`)** mirrors upstream Ollama: `top_k = 40`,
//!   `repeat_penalty = 1.1`.
//!
//! Shared knobs (`temperature`, `top_p`) are the same on both surfaces. A value the
//! caller sets explicitly always wins; these apply only when a field is absent from
//! the request. Keep this the single source of truth — handlers must reference these
//! constants instead of inlining literals, so the divergence stays visible here.

/// Sampling temperature — shared by both surfaces.
pub const TEMPERATURE: f32 = 0.8;
/// Nucleus sampling cutoff — shared by both surfaces.
pub const TOP_P: f32 = 0.9;
/// Cap on characters consumed by a `<think>` block before it is force-closed,
/// so a runaway reasoning trace can't eat the whole generation budget.
pub const MAX_THINKING_CHARS: usize = 8192;

/// OpenAI (`/v1/*`) surface defaults.
pub mod openai {
    /// Disabled — OpenAI exposes no `top_k`. `0` means "off" in the sampler.
    pub const TOP_K: u32 = 0;
    /// Off — OpenAI has no repeat penalty; it uses frequency/presence penalties.
    pub const REPETITION_PENALTY: f32 = 1.0;
    /// Additive frequency penalty, off by default (OpenAI semantics).
    pub const FREQUENCY_PENALTY: f32 = 0.0;
    /// Additive presence penalty, off by default (OpenAI semantics).
    pub const PRESENCE_PENALTY: f32 = 0.0;
    /// Generated-token cap when the caller sends neither `max_tokens` nor
    /// `max_completion_tokens`.
    pub const MAX_TOKENS: usize = 256;
}

/// Ollama (`/api/*`) surface defaults.
pub mod ollama {
    /// Upstream Ollama default.
    pub const TOP_K: u32 = 40;
    /// Upstream Ollama default (`1.1`; `1.0` would be off).
    pub const REPEAT_PENALTY: f32 = 1.1;
    /// `num_predict` default when the caller omits it.
    pub const MAX_TOKENS: usize = 512;
    /// Larger `num_predict` default when thinking is on, so the reasoning block
    /// doesn't consume the entire visible-answer budget.
    pub const MAX_TOKENS_THINKING: usize = 2048;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Lock the deliberate cross-surface divergence: if someone "unifies" these by
    // accident, this test fails and forces the decision back into the open.
    #[test]
    fn surfaces_share_temperature_and_top_p() {
        assert_eq!(TEMPERATURE, 0.8);
        assert_eq!(TOP_P, 0.9);
    }

    #[test]
    fn openai_has_no_top_k_or_repeat_penalty() {
        assert_eq!(openai::TOP_K, 0, "OpenAI exposes no top_k");
        assert_eq!(
            openai::REPETITION_PENALTY,
            1.0,
            "OpenAI has no repeat penalty (uses frequency/presence)"
        );
    }

    #[test]
    fn ollama_mirrors_upstream_ollama() {
        assert_eq!(ollama::TOP_K, 40);
        assert_eq!(ollama::REPEAT_PENALTY, 1.1);
    }
}
