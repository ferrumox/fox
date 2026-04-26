//! Mirostat v2 — perplexity-targeted sampling.
//!
//! Reference: Basu et al. 2020, "Mirostat: A neural text decoding algorithm
//! that directly controls perplexity". The algorithm keeps an internal
//! `μ` that adapts the truncation cutoff so the *average surprise* of
//! emitted tokens converges to a user-supplied target `τ`. Useful as a
//! drop-in replacement for top-p / top-k that doesn't need per-prompt
//! tuning to avoid both repetition (when τ is too low) and incoherence
//! (when τ is too high).
//!
//! Surprise here is `-log2(p)` for the chosen token; τ ≈ 5 corresponds
//! to roughly the entropy of natural English at the token level.

use std::cmp::Ordering;
use std::collections::HashMap;

use rand::Rng;

use super::sampling::apply_logit_bias;

/// Mirostat v2 state carried between decode steps for the same sequence.
#[derive(Debug, Clone)]
pub struct MirostatV2 {
    /// Current truncation surprise. Initialised to `2 * tau` (paper default).
    pub mu: f32,
    /// Target surprise per token.
    pub tau: f32,
    /// Learning rate for `mu` updates. Typical value ≈ 0.1.
    pub eta: f32,
}

impl MirostatV2 {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self {
            mu: 2.0 * tau,
            tau,
            eta,
        }
    }

    /// Re-arm the state to its initial μ. Call this at the start of a new
    /// sequence (after `clear_sequence` on the model).
    pub fn reset(&mut self) {
        self.mu = 2.0 * self.tau;
    }
}

/// Sample a token using Mirostat v2.
///
/// 1. Softmax `logits` → probabilities.
/// 2. Sort descending; truncate to the prefix whose tokens have surprise ≤ μ.
/// 3. Renormalise and draw multinomially.
/// 4. Update μ ← μ − η · (observed_surprise − τ).
///
/// `seed` and `token_count` are combined the same way as in the regular
/// stochastic sampler so two requests with the same seed produce the same
/// trajectory regardless of decoding mode.
pub fn sample(
    logits: &[f32],
    state: &mut MirostatV2,
    seed: Option<u64>,
    token_count: usize,
    logit_bias: Option<&HashMap<i32, f32>>,
) -> i32 {
    if logits.is_empty() {
        return 0;
    }

    // 0. Apply logit bias on a private copy. The mirostat path operates on
    //    raw logits (no temperature/top-k/top-p), so the bias step is the
    //    only pre-softmax adjustment available.
    let mut owned: Vec<f32>;
    let logits: &[f32] = match logit_bias {
        Some(bias) => {
            owned = logits.to_vec();
            apply_logit_bias(&mut owned, bias);
            &owned
        }
        None => logits,
    };

    // 1. Softmax → probabilities, with the usual numerical-stability shift.
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let total: f32 = exps.iter().sum();

    // 2. Sort descending while keeping original indices.
    let mut sorted: Vec<(usize, f32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / total))
        .collect();
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    // 3. Truncate at the first index whose surprise exceeds μ.
    let mu = state.mu.max(0.0);
    let mut keep = sorted.len();
    for (i, &(_, p)) in sorted.iter().enumerate() {
        if p <= 0.0 {
            keep = i.max(1);
            break;
        }
        let surprise = -p.log2();
        if surprise > mu {
            keep = i.max(1);
            break;
        }
    }
    sorted.truncate(keep);

    // 4. Multinomial draw over the kept tail.
    let total_kept: f32 = sorted.iter().map(|&(_, p)| p).sum();
    let mut rng = build_rng(seed, token_count);
    let r: f32 = rng.gen::<f32>() * total_kept;
    let mut cum = 0.0f32;
    let mut chosen = sorted.last().copied().unwrap_or((0, 0.0));
    for &(idx, p) in &sorted {
        cum += p;
        if cum >= r {
            chosen = (idx, p);
            break;
        }
    }

    // 5. Update μ from the observed surprise. Clamp at zero to prevent the
    //    cutoff from collapsing to nothing if the model emits a very high-
    //    probability token (which would otherwise drive μ negative).
    let observed_surprise = if chosen.1 > 0.0 {
        -chosen.1.log2()
    } else {
        state.mu
    };
    state.mu = (state.mu - state.eta * (observed_surprise - state.tau)).max(0.0);

    chosen.0 as i32
}

/// Construct a deterministic RNG when `seed` is set, otherwise a thread-local
/// non-deterministic one. Mirrors the helper used by the regular sampler so
/// both sampling paths produce identical trajectories for identical seeds.
fn build_rng(seed: Option<u64>, token_count: usize) -> Box<dyn rand::RngCore> {
    use rand::SeedableRng;
    match seed {
        Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s ^ (token_count as u64))),
        None => Box::new(rand::thread_rng()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_initialises_mu_to_two_tau() {
        let m = MirostatV2::new(5.0, 0.1);
        assert!((m.mu - 10.0).abs() < 1e-6);
        assert!((m.tau - 5.0).abs() < 1e-6);
        assert!((m.eta - 0.1).abs() < 1e-6);
    }

    #[test]
    fn reset_restores_initial_mu() {
        let mut m = MirostatV2::new(5.0, 0.1);
        m.mu = 0.3;
        m.reset();
        assert!((m.mu - 10.0).abs() < 1e-6);
    }

    #[test]
    fn deterministic_when_seeded() {
        let logits: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut s1 = MirostatV2::new(3.0, 0.1);
        let mut s2 = MirostatV2::new(3.0, 0.1);
        for step in 0..5 {
            let a = sample(&logits, &mut s1, Some(42), step, None);
            let b = sample(&logits, &mut s2, Some(42), step, None);
            assert_eq!(a, b, "seeded mirostat must be reproducible");
        }
    }

    #[test]
    fn mu_stays_non_negative_under_high_probability_token() {
        // A heavily peaked distribution: one logit dominates.
        let mut logits = vec![0.0f32; 32];
        logits[0] = 100.0;
        let mut state = MirostatV2::new(5.0, 0.5);
        for step in 0..20 {
            let _ = sample(&logits, &mut state, Some(7), step, None);
            assert!(
                state.mu >= 0.0,
                "mu went negative at step {step}: {}",
                state.mu
            );
        }
    }

    #[test]
    fn empty_logits_return_zero_without_panicking() {
        let mut state = MirostatV2::new(5.0, 0.1);
        assert_eq!(sample(&[], &mut state, Some(1), 0, None), 0);
    }

    #[test]
    fn always_picks_a_valid_index() {
        let logits: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
        let mut state = MirostatV2::new(5.0, 0.1);
        for step in 0..100 {
            let token = sample(&logits, &mut state, Some(99), step, None);
            assert!(token >= 0);
            assert!((token as usize) < logits.len(), "token id out of range");
        }
    }

    #[test]
    fn mu_converges_toward_tau_on_long_run() {
        // Roughly uniform logits — the average surprise of a uniform
        // distribution over N tokens is log2(N). For N=64 that's 6.0, so
        // with τ=4.0 μ should drift well below the initial 2*τ=8.
        let logits = vec![0.0f32; 64];
        let mut state = MirostatV2::new(4.0, 0.2);
        for step in 0..200 {
            let _ = sample(&logits, &mut state, Some(13), step, None);
        }
        assert!(
            state.mu < 8.0,
            "mu did not move from initial 2*tau=8 (got {})",
            state.mu
        );
        assert!(state.mu >= 0.0);
    }
}
