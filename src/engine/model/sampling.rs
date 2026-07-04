// Token sampling: greedy, temperature, top-K, top-P, repetition penalty.
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

/// Apply OpenAI-style frequency and presence penalties in-place.
/// `logit -= presence * (token appeared) + frequency * (times it appeared)`.
/// Both default to 0.0 (disabled). Unlike `repetition_penalty` (multiplicative),
/// these are additive and match the OpenAI `frequency_penalty`/`presence_penalty`
/// semantics, so those request fields are honoured instead of silently ignored.
pub(crate) fn apply_frequency_presence_penalty(
    logits: &mut [f32],
    token_ids: &[i32],
    frequency: f32,
    presence: f32,
) {
    if frequency == 0.0 && presence == 0.0 {
        return;
    }
    let mut counts: std::collections::HashMap<i32, u32> = std::collections::HashMap::new();
    for &tid in token_ids {
        if tid >= 0 {
            *counts.entry(tid).or_insert(0) += 1;
        }
    }
    for (&tid, &count) in &counts {
        let idx = tid as usize;
        if idx < logits.len() {
            logits[idx] -= frequency * count as f32 + presence;
        }
    }
}

/// Parameters for the full stochastic sampler.
pub(crate) struct SamplerParams<'a> {
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: u32,
    /// Minimum probability relative to the top token (0 = disabled). Tokens whose
    /// probability is below `min_p × max_prob` are dropped before the draw.
    pub(crate) min_p: f32,
    pub(crate) repetition_penalty: f32,
    pub(crate) frequency_penalty: f32,
    pub(crate) presence_penalty: f32,
    /// Additive per-token bias applied to the raw logits (OpenAI `logit_bias`).
    pub(crate) logit_bias: Option<&'a std::collections::HashMap<i32, f32>>,
    pub(crate) generated_ids: &'a [i32],
    pub(crate) seed: Option<u64>,
    pub(crate) token_count: usize,
}

/// Full stochastic sampler: repetition penalty → temperature → top-K → top-P → weighted draw.
///
/// When `temperature` ≤ 0 the function falls back to greedy regardless of other parameters.
/// The RNG is seeded per-request for reproducibility when `seed` is provided.
pub(crate) fn sample_token(logits: &[f32], p: SamplerParams<'_>) -> i32 {
    let SamplerParams {
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        frequency_penalty,
        presence_penalty,
        logit_bias,
        generated_ids,
        seed,
        token_count,
    } = p;
    let mut logits = logits.to_vec();

    // 0. logit_bias: additive per-token bias on the raw logits (OpenAI semantics,
    //    where ±100 effectively forces or bans a token).
    if let Some(bias) = logit_bias {
        for (&id, &b) in bias {
            if id >= 0 && (id as usize) < logits.len() {
                logits[id as usize] += b;
            }
        }
    }

    // 1. Repetition + frequency/presence penalties
    if repetition_penalty != 1.0 && !generated_ids.is_empty() {
        apply_repetition_penalty(&mut logits, generated_ids, repetition_penalty);
    }
    if !generated_ids.is_empty() {
        apply_frequency_presence_penalty(
            &mut logits,
            generated_ids,
            frequency_penalty,
            presence_penalty,
        );
    }

    // 2. Greedy shortcut
    if temperature <= 0.0 {
        return sample_greedy(&logits);
    }

    // 3. Temperature scaling
    for l in &mut logits {
        *l /= temperature;
    }

    // 4. Top-K masking
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

    // 5. Softmax + sort by descending probability
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &l)| (i, (l - max_l).exp() / exp_sum))
        .collect();
    probs.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(Ordering::Equal));

    // 5b. Min-P: drop tokens whose probability is below `min_p × max_prob`. Probs are
    // sorted descending, so keep the leading run above the threshold (at least the top).
    if min_p > 0.0 && !probs.is_empty() {
        let threshold = min_p * probs[0].1;
        let keep = probs
            .iter()
            .take_while(|(_, p)| *p >= threshold)
            .count()
            .max(1);
        probs.truncate(keep);
    }

    // 6. Top-P nucleus truncation
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

    // 7. Weighted random draw
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
    // apply_frequency_presence_penalty
    // -----------------------------------------------------------------------

    #[test]
    fn freq_presence_penalty_additive() {
        let mut logits = vec![1.0f32, 1.0, 1.0];
        // token 0 appears twice, token 1 once, token 2 not at all.
        apply_frequency_presence_penalty(&mut logits, &[0, 0, 1], 0.5, 0.2);
        assert!((logits[0] - (-0.2)).abs() < 1e-6, "1 - (0.5*2 + 0.2)");
        assert!((logits[1] - 0.3).abs() < 1e-6, "1 - (0.5*1 + 0.2)");
        assert!((logits[2] - 1.0).abs() < 1e-6, "unseen token untouched");
    }

    #[test]
    fn freq_presence_penalty_noop_when_zero() {
        let orig = vec![1.0f32, 2.0];
        let mut logits = orig.clone();
        apply_frequency_presence_penalty(&mut logits, &[0, 1], 0.0, 0.0);
        assert_eq!(logits, orig);
    }

    // -----------------------------------------------------------------------
    // sample_token — greedy path (temperature ≤ 0)
    // -----------------------------------------------------------------------

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
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: None,
                generated_ids: &[],
                seed: None,
                token_count: 0,
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
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: None,
                generated_ids: &[],
                seed: None,
                token_count: 0,
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
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: None,
            generated_ids: &[],
            seed: Some(42),
            token_count: 0,
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
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    logit_bias: None,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
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
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    logit_bias: None,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
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
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: None,
                generated_ids: &[0], // token 0 was already generated
                seed: None,
                token_count: 1,
            },
        );
        assert_eq!(token, 1, "penalised token 0 should lose to token 1");
    }

    fn greedy_params<'a>(
        logit_bias: Option<&'a std::collections::HashMap<i32, f32>>,
    ) -> SamplerParams<'a> {
        SamplerParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias,
            generated_ids: &[],
            seed: None,
            token_count: 0,
        }
    }

    #[test]
    fn logit_bias_forces_a_token() {
        // Token 2 has the lowest logit but a huge positive bias makes it win.
        let logits = vec![5.0f32, 4.0, 0.0];
        let mut bias = std::collections::HashMap::new();
        bias.insert(2, 100.0);
        assert_eq!(sample_token(&logits, greedy_params(Some(&bias))), 2);
    }

    #[test]
    fn logit_bias_bans_a_token() {
        // Token 0 is the top logit but a large negative bias eliminates it.
        let logits = vec![5.0f32, 4.0, 0.0];
        let mut bias = std::collections::HashMap::new();
        bias.insert(0, -100.0);
        assert_eq!(sample_token(&logits, greedy_params(Some(&bias))), 1);
    }

    #[test]
    fn min_p_keeps_only_dominant_token() {
        // Token 3 dominates; min_p = 0.5 drops every token below half its probability.
        let logits = vec![0.0f32, 0.0, 0.0, 10.0];
        for seed in 0u64..20 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    temperature: 1.0,
                    top_p: 1.0,
                    top_k: 0,
                    min_p: 0.5,
                    repetition_penalty: 1.0,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    logit_bias: None,
                    generated_ids: &[],
                    seed: Some(seed),
                    token_count: 0,
                },
            );
            assert_eq!(t, 3, "min_p=0.5 must keep only the dominant token");
        }
    }
}
