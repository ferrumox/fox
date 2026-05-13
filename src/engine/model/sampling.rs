// Token sampling: greedy, temperature, top-K, top-P, min-p, repetition/presence/frequency penalty.
// This module is excluded entirely when fox_stub is set (no llama.cpp builds).

use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sample the highest-probability token (deterministic).
pub(crate) fn sample_greedy(logits: &[f32]) -> i32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

/// Apply repetition penalty in-place: divide positive logits and multiply negative ones.
pub(crate) fn apply_repetition_penalty(logits: &mut [f32], token_ids: &[i32], penalty: f32) {
    for &tid in token_ids {
        if tid >= 0 && (tid as usize) < logits.len() {
            let l = logits[tid as usize];
            logits[tid as usize] = if l > 0.0 { l / penalty } else { l * penalty };
        }
    }
}

/// Add caller-supplied biases to the corresponding logits in-place. Mirrors
/// the OpenAI `logit_bias` knob — a positive value boosts a token, negative
/// suppresses it (`-100` ≈ ban). Out-of-range token ids are silently
/// ignored so a request that targets a token from a different vocabulary
/// can't crash the sampler.
pub(crate) fn apply_logit_bias(logits: &mut [f32], bias: &std::collections::HashMap<i32, f32>) {
    for (&tid, &b) in bias.iter() {
        if tid >= 0 && (tid as usize) < logits.len() {
            logits[tid as usize] += b;
        }
    }
}

/// Compute a temperature in `[low, high]` based on the normalised entropy
/// of `logits`. When the model is confident (low entropy) the result is
/// closer to `low`; when it's hesitant (high entropy) the result drifts
/// toward `high`. Useful as a single knob that produces near-greedy text in
/// code-like contexts and looser sampling in prose-like contexts.
///
/// Returns `low` immediately when `low >= high` so the caller can use
/// `(0, 0)` as a "disabled" sentinel without changing behaviour.
pub(crate) fn dynamic_temperature(logits: &[f32], low: f32, high: f32) -> f32 {
    if !(low >= 0.0 && high > low) {
        return low.max(0.0);
    }
    if logits.is_empty() {
        return low;
    }

    // Softmax with the standard numerical-stability shift.
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return low;
    }
    let mut entropy = 0.0f32;
    for e in &exps {
        let p = e / sum;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    let max_entropy = (logits.len() as f32).log2().max(1.0);
    let normalised = (entropy / max_entropy).clamp(0.0, 1.0);
    low + (high - low) * normalised
}

/// Parameters for the full stochastic sampler.
pub(crate) struct SamplerParams<'a> {
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    /// Min-p filter: remove tokens with prob < min_p × max_prob (0.0 = disabled).
    pub(crate) min_p: f32,
    pub(crate) repetition_penalty: f32,
    /// Per-token count map for presence/frequency penalty (None = skip penalty).
    pub(crate) token_counts: Option<&'a std::collections::HashMap<i32, usize>>,
    /// Presence penalty: subtract from logit for any token seen at least once (0.0 = disabled).
    pub(crate) presence_penalty: f32,
    /// Frequency penalty: subtract penalty × count from logit for each token (0.0 = disabled).
    pub(crate) frequency_penalty: f32,
    pub(crate) generated_ids: &'a [i32],
    pub(crate) seed: Option<u64>,
    pub(crate) token_count: usize,
    /// Optional per-token logit adjustment (OpenAI-compatible `logit_bias`).
    /// `None` skips the step entirely.
    pub(crate) logit_bias: Option<&'a std::collections::HashMap<i32, f32>>,
    /// Dynamic temperature range. When `Some((low, high))` and `high > low`,
    /// the sampler computes the post-softmax entropy of the logits and
    /// blends a temperature in `[low, high]` — replacing the static
    /// `temperature` for that step. `None` keeps the static behaviour.
    pub(crate) dynamic_temp: Option<(f32, f32)>,
}

/// Full stochastic sampler: repetition/presence/frequency penalty → temperature → top-K → top-P → min-p → weighted draw.
///
/// When `temperature` ≤ 0 the function falls back to greedy regardless of other parameters.
/// The RNG is seeded per-request for reproducibility when `seed` is provided.
pub(crate) fn sample_token(logits: &[f32], p: SamplerParams<'_>) -> i32 {
    let SamplerParams {
        temperature: static_temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        token_counts,
        presence_penalty,
        frequency_penalty,
        generated_ids,
        seed,
        token_count,
        logit_bias,
        dynamic_temp,
    } = p;
    // Resolve the effective temperature once. If the caller supplied a
    // dynamic range we recompute it from the *raw* logits so the entropy
    // measurement isn't biased by penalties applied later.
    let temperature = match dynamic_temp {
        Some((lo, hi)) if hi > lo => dynamic_temperature(logits, lo, hi),
        _ => static_temperature,
    };
    let mut logits = logits.to_vec();

    // 0. Logit bias — applied first so positive boosts compete cleanly with
    //    repetition / presence / frequency penalties further down.
    if let Some(bias) = logit_bias {
        apply_logit_bias(&mut logits, bias);
    }

    // 1. Repetition penalty
    if repetition_penalty != 1.0 && !generated_ids.is_empty() {
        apply_repetition_penalty(&mut logits, generated_ids, repetition_penalty);
    }

    // 2. Presence / frequency penalty
    if presence_penalty != 0.0 || frequency_penalty != 0.0 {
        if let Some(counts) = token_counts {
            for (&tid, &count) in counts.iter() {
                if tid >= 0 && (tid as usize) < logits.len() {
                    if presence_penalty != 0.0 {
                        logits[tid as usize] -= presence_penalty;
                    }
                    if frequency_penalty != 0.0 {
                        logits[tid as usize] -= frequency_penalty * count as f32;
                    }
                }
            }
        }
    }

    // 3. Greedy shortcut
    if temperature <= 0.0 {
        return sample_greedy(&logits);
    }

    // 4. Temperature scaling
    for l in &mut logits {
        *l /= temperature;
    }

    // 5. Top-K masking
    let k = top_k as usize;
    if k > 0 && k < logits.len() {
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold = indexed[k - 1].1;
        for l in &mut logits {
            if *l < threshold {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    // 6. Softmax + sort by descending probability
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_l).exp() / exp_sum))
        .collect();
    probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // 7. Min-p filter: remove tokens with prob < min_p × max_prob
    if min_p > 0.0 {
        let max_prob = probs.first().map(|(_, p)| *p).unwrap_or(0.0);
        let threshold = min_p * max_prob;
        probs.retain(|(_, p)| *p >= threshold);
        if probs.is_empty() {
            return logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
                .map(|(i, _)| i as i32)
                .unwrap_or(0);
        }
    }

    // 8. Top-P nucleus truncation
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut end = probs.len();
        for (idx, (_, p)) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
                end = idx + 1;
                break;
            }
        }
        probs.truncate(end);
    }

    // 9. Weighted random draw
    let mut rng: Box<dyn rand::RngCore> = match seed {
        Some(s) => Box::new(StdRng::seed_from_u64(s ^ (token_count as u64))),
        None => Box::new(rand::thread_rng()),
    };

    let total: f32 = probs.iter().map(|(_, p)| p).sum();
    let r: f32 = rng.gen::<f32>() * total;
    let mut cum = 0.0f32;
    for (idx, p) in &probs {
        cum += p;
        if cum >= r {
            return *idx as i32;
        }
    }
    probs.last().map(|(idx, _)| *idx as i32).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // sample_greedy
    // -----------------------------------------------------------------------

    #[test]
    fn greedy_picks_argmax() {
        let logits = vec![0.1f32, 0.9, 0.3, 0.7];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn greedy_last_element_wins_on_tie() {
        // Rust's Iterator::max_by returns the *last* equal maximum element.
        let logits = vec![1.0f32, 1.0, 0.5];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn greedy_handles_single_token() {
        assert_eq!(sample_greedy(&[42.0f32]), 0);
    }

    // -----------------------------------------------------------------------
    // apply_repetition_penalty
    // -----------------------------------------------------------------------

    #[test]
    fn rep_penalty_divides_positive_logits() {
        let mut logits = vec![2.0f32, 1.0, -1.0];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "positive logit should be halved"
        );
        assert!((logits[1] - 1.0).abs() < 1e-6, "untouched");
        assert!(
            (logits[2] - (-1.0)).abs() < 1e-6,
            "untouched negative not in token_ids"
        );
    }

    #[test]
    fn rep_penalty_multiplies_negative_logits() {
        let mut logits = vec![-1.0f32, 0.5];
        apply_repetition_penalty(&mut logits, &[0], 2.0);
        assert!(
            (logits[0] - (-2.0)).abs() < 1e-6,
            "negative logit multiplied by penalty"
        );
        assert!((logits[1] - 0.5).abs() < 1e-6, "untouched");
    }

    #[test]
    fn rep_penalty_noop_when_no_generated_tokens() {
        let original = vec![1.0f32, 2.0, 3.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[], 2.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn rep_penalty_ignores_out_of_range_token_ids() {
        let original = vec![1.0f32, 2.0];
        let mut logits = original.clone();
        apply_repetition_penalty(&mut logits, &[99, -1], 2.0);
        assert_eq!(logits, original);
    }

    // -----------------------------------------------------------------------
    // sample_token — greedy path (temperature ≤ 0)
    // -----------------------------------------------------------------------

    #[test]
    fn dynamic_temperature_returns_low_when_distribution_is_peaked() {
        // One logit dominates → entropy ≈ 0 → temperature ≈ low.
        let mut logits = vec![0.0f32; 64];
        logits[0] = 100.0;
        let t = dynamic_temperature(&logits, 0.2, 1.5);
        assert!(
            (t - 0.2).abs() < 0.01,
            "peaked distribution should map close to low; got {t}"
        );
    }

    #[test]
    fn dynamic_temperature_returns_high_when_distribution_is_uniform() {
        // All logits equal → entropy = log2(N) → normalised = 1 → temp = high.
        let logits = vec![0.0f32; 64];
        let t = dynamic_temperature(&logits, 0.2, 1.5);
        assert!(
            (t - 1.5).abs() < 0.01,
            "uniform distribution should map close to high; got {t}"
        );
    }

    #[test]
    fn dynamic_temperature_clamps_invalid_range() {
        // high < low → return clamped low (no panic, no negative output).
        assert_eq!(dynamic_temperature(&[0.0, 1.0, 2.0], 0.5, 0.2), 0.5);
        assert_eq!(dynamic_temperature(&[0.0, 1.0, 2.0], -1.0, 0.5), 0.0);
    }

    #[test]
    fn apply_logit_bias_adds_to_target_tokens_only() {
        let mut logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bias = std::collections::HashMap::new();
        bias.insert(1, 10.0);
        bias.insert(3, -5.0);
        bias.insert(99, 1.0); // out of range — silently ignored
        bias.insert(-1, 1.0); // negative id — silently ignored
        apply_logit_bias(&mut logits, &bias);
        assert_eq!(logits, vec![1.0, 12.0, 3.0, -1.0]);
    }

    #[test]
    fn logit_bias_can_force_a_token_under_greedy() {
        // Without bias the greedy pick is index 1 (highest logit).
        let logits = vec![0.1f32, 5.0, 0.3];
        // With a +10 bias on index 2 the new max is 10.3 → index 2.
        let mut bias = std::collections::HashMap::new();
        bias.insert(2, 10.0);
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                token_counts: None,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                generated_ids: &[],
                seed: None,
                token_count: 0,
                logit_bias: Some(&bias),
                dynamic_temp: None,
            },
        );
        assert_eq!(token, 2);
    }

    #[test]
    fn sample_token_greedy_at_temperature_zero() {
        let logits = vec![0.1f32, 5.0, 0.3];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                token_counts: None,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                generated_ids: &[],
                seed: None,
                token_count: 0,
                logit_bias: None,
                dynamic_temp: None,
            },
        );
        assert_eq!(token, 1);
    }

    #[test]
    fn sample_token_negative_temperature_is_greedy() {
        let logits = vec![0.1f32, 0.2, 9.9];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: -1.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 1.0,
                token_counts: None,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                generated_ids: &[],
                seed: None,
                token_count: 0,
                logit_bias: None,
                dynamic_temp: None,
            },
        );
        assert_eq!(token, 2);
    }

    // -----------------------------------------------------------------------
    // sample_token — stochastic path with seeded RNG (reproducible)
    // -----------------------------------------------------------------------

    #[test]
    fn sample_token_seeded_is_reproducible() {
        let logits = vec![1.0f32, 2.0, 0.5, 1.5];
        let params = || SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            token_counts: None,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            generated_ids: &[],
            seed: Some(42),
            token_count: 0,
            logit_bias: None,
            dynamic_temp: None,
        };
        assert_eq!(
            sample_token(&logits, params()),
            sample_token(&logits, params())
        );
    }

    #[test]
    fn sample_token_top_k_restricts_candidates() {
        // With logits heavily favouring token 3 but top_k=2, only tokens 1 and 3 are eligible
        // (they have the two highest logits). Token 0 and 2 must never be sampled.
        let logits = vec![0.0f32, 5.0, 0.0, 10.0];
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        for seed in 0u64..50 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 2,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                    token_counts: None,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
                    logit_bias: None,
                    dynamic_temp: None,
                },
            );
            seen.insert(t);
        }
        assert!(
            !seen.contains(&0) && !seen.contains(&2),
            "tokens outside top-K window should never be sampled; got {:?}",
            seen
        );
    }

    #[test]
    fn sample_token_top_p_restricts_candidates() {
        // Token 3 has logit 10 (very dominant). With top_p = 0.5, only tokens
        // with cumulative mass ≥ 50 % survive; that should include token 3 at minimum.
        let logits = vec![0.0f32, 0.0, 0.0, 10.0];
        for seed in 0u64..20 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    temperature: 1.0,
                    top_p: 0.5,
                    top_k: 0,
                    min_p: 0.0,
                    repetition_penalty: 1.0,
                    token_counts: None,
                    presence_penalty: 0.0,
                    frequency_penalty: 0.0,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
                    logit_bias: None,
                    dynamic_temp: None,
                },
            );
            assert_eq!(
                t, 3,
                "dominant token must always be sampled under top_p=0.5"
            );
        }
    }

    #[test]
    fn sample_token_repetition_penalty_reduces_repeated_token() {
        // Token 0 has the highest raw logit but we penalise it heavily.
        // After the penalty token 1 should win in greedy mode.
        let logits = vec![5.0f32, 3.0];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0, // greedy so result is deterministic
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 10.0,
                token_counts: None,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
                generated_ids: &[0], // token 0 was already generated
                seed: None,
                token_count: 1,
                logit_bias: None,
                dynamic_temp: None,
            },
        );
        assert_eq!(token, 1, "penalised token 0 should lose to token 1");
    }
}
