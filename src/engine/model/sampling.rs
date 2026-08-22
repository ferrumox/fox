// Token sampling: greedy, temperature, top-K, top-P, repetition penalty.
// This module is excluded entirely when fox_stub is set (no llama.cpp builds).

use std::cmp::Ordering;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Sample the highest-probability token (deterministic).
///
/// Ties go to the **highest** token id, matching `Iterator::max_by`'s
/// last-maximum rule, which this function used to be written in terms of and which
/// callers at `temperature = 0` observe directly.
///
/// NaN never wins. That is not a refinement, it is a bug fix: `max_by` replaces its
/// accumulator whenever the comparison is not `Greater`, and
/// `partial_cmp(..).unwrap_or(Equal)` reports `Equal` for an incomparable NaN — so
/// a NaN always displaced the running maximum, and every element after it competed
/// from scratch. Measured against the old implementation:
///
/// ```text
/// [5.0, NaN]       -> 1   the NaN itself
/// [5.0, NaN, 1.0]  -> 2   neither the maximum nor the NaN
/// ```
///
/// One NaN anywhere in a 128K-wide logit vector was enough to emit an essentially
/// arbitrary token. NaN logits are reachable in practice — fp16 overflow, a bad
/// quantisation, corrupted KV — so this was a live path, not a theoretical one.
/// A vector that is *entirely* NaN has no meaningful answer; it returns 0, as the
/// empty case does.
pub(crate) fn sample_greedy(logits: &[f32]) -> i32 {
    let mut best = 0usize;
    let mut best_logit = f32::NEG_INFINITY;
    let mut have_candidate = false;
    let mut nan_count = 0usize;

    for (i, &l) in logits.iter().enumerate() {
        if l.is_nan() {
            nan_count += 1;
            continue;
        }
        // `>=`, not `>`: keep the last maximum, so `[1.0, 1.0, 0.5]` picks 1.
        if !have_candidate || l >= best_logit {
            best = i;
            best_logit = l;
            have_candidate = true;
        }
    }

    if nan_count > 0 {
        // Worth a warning every time rather than once: the model is producing
        // garbage, and which requests it happened on is the diagnostic.
        tracing::warn!(
            nan_count,
            vocab = logits.len(),
            picked = best,
            "NaN logits in greedy sampling — ignored, but the forward pass is suspect"
        );
    }

    best as i32
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

/// Slice the trailing window of generated tokens the penalties may look at.
///
/// Mirrors llama.cpp's `repeat_last_n` semantics (`common/sampling.h`):
/// `-1` = the whole history, `0` = penalties disabled, `n > 0` = the last `n`
/// tokens. Without a window, both penalty passes scan every token generated so
/// far on *every* step, which is `O(generated²)` per request and — with the
/// Ollama surface's `repeat_penalty = 1.1` default — keeps penalising tokens
/// from thousands of positions back, degrading long outputs.
///
/// Deliberate divergence from llama.cpp: fox's window covers only *generated*
/// tokens, while llama.cpp's spans prompt+generated. fox has never penalised
/// prompt tokens and changing that would silently alter output for every
/// caller, so the window narrows an existing behaviour rather than redefining it.
pub(crate) fn penalty_window(generated_ids: &[i32], repeat_last_n: i32) -> &[i32] {
    match repeat_last_n {
        0 => &[],
        n if n < 0 => generated_ids,
        n => &generated_ids[generated_ids.len().saturating_sub(n as usize)..],
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
    /// How far back the three penalties above may look, in generated tokens.
    /// `-1` = whole history, `0` = disabled, `n` = last `n`. See [`penalty_window`].
    pub(crate) repeat_last_n: i32,
    /// Top-nσ: keep only tokens whose logit is within `n` standard deviations of the
    /// highest logit (`<= 0` disables). Unlike top-p, the cutoff is computed on the
    /// raw logit *scale*, so it does not shift as temperature changes.
    pub(crate) top_n_sigma: f32,
    /// Floor on how few candidates any truncation step (min-p, top-p, top-nσ) may
    /// leave. `0`/`1` behave identically — one candidate is always kept regardless.
    pub(crate) min_keep: usize,
    /// Additive per-token bias applied to the raw logits (OpenAI `logit_bias`).
    pub(crate) logit_bias: Option<&'a std::collections::HashMap<i32, f32>>,
    pub(crate) generated_ids: &'a [i32],
    pub(crate) seed: Option<u64>,
    pub(crate) token_count: usize,
}

/// Indices of the `n` largest logits, descending, without materialising the vocabulary.
///
/// Replaces `(0..logits.len()).collect()` + `select_nth_unstable_by`, which profiling
/// showed cost fox **6.6% of wall time** against `llama-server`'s 1.4% in
/// `llama_token_data_array_partial_sort_inplace` — the whole of the decode-throughput
/// deficit at concurrency ≥ 4. Two costs were being paid per token *per sequence*:
///
///   - a 1 MB allocation (128256 × `usize`) that was immediately truncated to `n`;
///   - a comparator that dereferenced into a separate 512 KB logits array on every
///     comparison, while the index array it was permuting moved underneath it.
///
/// This keeps a sorted-descending buffer of at most `n` entries and streams the logits
/// once. The common case per element is a single `f32` compare against the running
/// threshold — a sequential read of the logits array, no indirection, no allocation
/// proportional to the vocabulary. Insertions cost an O(n) memmove but happen rarely
/// after the buffer fills, and `n` is small (top-k is typically ≤ 100; the adaptive
/// pool starts at 64).
///
/// Ties: `partition_point` keeps the earlier-scanned index first among equal logits,
/// which for a descending scan means the lower token id — the same order the caller's
/// subsequent `sort_by` would produce, and stable across runs.
fn select_top_n(logits: &[f32], n: usize) -> Vec<usize> {
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(n + 1);
    // NEG_INFINITY until the buffer fills, so every early element is taken.
    let mut threshold = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        // NaN compares false here and is therefore treated as smaller than everything.
        // The negation is load-bearing and must not be rewritten as `l <= threshold`:
        // that is false for NaN, which would *admit* a NaN into the candidate pool.
        // Clippy cannot know which of the two a partial order was meant to pick.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let below_threshold = !(l > threshold);
        if top.len() == n && below_threshold {
            continue;
        }
        let pos = top.partition_point(|&(_, v)| v > l);
        top.insert(pos, (i, l));
        if top.len() > n {
            top.pop();
        }
        if top.len() == n {
            threshold = top[n - 1].1;
        }
    }
    top.into_iter().map(|(i, _)| i).collect()
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
        repeat_last_n,
        top_n_sigma,
        min_keep,
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

    // 1. Repetition + frequency/presence penalties, over the trailing window only.
    let penalised = penalty_window(generated_ids, repeat_last_n);
    if repetition_penalty != 1.0 && !penalised.is_empty() {
        apply_repetition_penalty(&mut logits, penalised, repetition_penalty);
    }
    if !penalised.is_empty() {
        apply_frequency_presence_penalty(
            &mut logits,
            penalised,
            frequency_penalty,
            presence_penalty,
        );
    }

    // 2. Greedy shortcut
    if temperature <= 0.0 {
        return sample_greedy(&logits);
    }

    // (see `select_top_n` for why candidate selection is a linear scan rather than a
    // partition over a materialised index vector)

    // 3. Temperature scaling
    for l in &mut logits {
        *l /= temperature;
    }

    // 4/5. Candidate selection + softmax, avoiding an O(n log n) sort of the
    // entire vocab (128K+ entries on real models) whenever possible:
    //
    // - `top_k > 0`: `select_nth_unstable_by` partitions in O(n) average to find
    //   the top-k set directly — softmax normalizes over just this set, which
    //   matches the pre-refactor semantics of masking non-top-k logits to
    //   `-inf` then softmaxing over everything (masked entries contribute
    //   `exp(-inf) == 0` to the sum either way, so restricting the sum to the
    //   survivors is numerically identical).
    // - `top_k` disabled (0, OpenAI's default — see `sampling_defaults.rs`):
    //   softmax must still normalize over the *whole* vocab (that's the actual
    //   probability distribution being sampled from), so `max_l`/`exp_sum`
    //   are computed with one full linear pass each — unavoidable, but only
    //   `O(n)` and comparisons/`exp()`, not a sort. What *is* avoidable: fully
    //   sorting/materializing probabilities for all 128K entries just to find
    //   min-p/top-p's cutoff, when in practice a real model's softmax output
    //   concentrates almost all its mass in a small head. So: adaptively grow
    //   a by-logit candidate pool (64 → 256 → 1024 → …) via the same
    //   `select_nth_unstable_by`, and stop as soon as it provably contains
    //   enough of the distribution to make min-p/top-p truncation below give
    //   the exact same result as if the whole vocab had been sorted — falling
    //   back to the whole vocab only if a request's parameters genuinely
    //   require it (e.g. `top_p` at/near `1.0`).
    let k = top_k as usize;
    let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let (mut candidates, exp_sum): (Vec<usize>, f32) = if k > 0 && k < logits.len() {
        let idx = select_top_n(&logits, k);
        let sum: f32 = idx.iter().map(|&i| (logits[i] - max_l).exp()).sum();
        (idx, sum)
    } else {
        let sum: f32 = logits.iter().map(|&l| (l - max_l).exp()).sum();
        let top_p_needed = top_p.clamp(0.0, 1.0);
        // `min_p × max_prob`; `max_prob == 1/sum` since the single highest logit
        // (always inside any non-empty candidate pool) has probability
        // `exp(max_l - max_l)/sum == 1/sum`.
        let min_p_threshold = if min_p > 0.0 { min_p / sum } else { 0.0 };
        // `top_p >= 1.0` asks for the whole distribution and `min_p <= 0` removes the
        // other reason the pool could stop early, so the answer is the entire
        // vocabulary — known up front, not something to discover. The adaptive loop
        // below would reach the same set by growing 64 → 256 → … → n, paying a
        // `select_top_n` pass and a fresh allocation at every step and never once
        // satisfying `covered >= 1.0` (a float sum of probabilities does not reach
        // exactly 1.0). On Qwen3.8-27B's 248,320-entry vocabulary that is seven
        // discarded passes per token, and `top_p: 1.0` is what `fox bench` and the
        // OpenAI defaults both ask for.
        if top_p_needed >= 1.0 && min_p <= 0.0 {
            ((0..logits.len()).collect::<Vec<usize>>(), sum)
        } else {
            let mut bound = 64usize.min(logits.len());
            let idx = loop {
                let cand: Vec<usize> = if bound < logits.len() {
                    select_top_n(&logits, bound)
                } else {
                    (0..logits.len()).collect()
                };
                if bound >= logits.len() {
                    break cand;
                }
                let mut covered = 0.0f32;
                let mut min_prob_in_pool = f32::INFINITY;
                for &i in &cand {
                    let prob = (logits[i] - max_l).exp() / sum;
                    covered += prob;
                    min_prob_in_pool = min_prob_in_pool.min(prob);
                }
                // `top_p == 1.0` truncates nothing, so it imposes no requirement on the
                // pool and must not hold the loop open — `covered` is a float sum of
                // probabilities and never reaches exactly 1.0, so testing it would grow
                // the pool to the whole vocabulary no matter what the other filters
                // wanted. Reaching here at all means some filter *does* truncate (the
                // no-truncation case took the whole-vocabulary branch above), so this
                // lets that filter alone decide how big the pool has to be: `min_p`
                // with `top_p: 1.0` went from 5.2 to 25 tok/s on Qwen3.8-27B.
                let top_p_satisfied = top_p_needed >= 1.0 || covered >= top_p_needed;
                let min_p_satisfied = min_p <= 0.0 || min_prob_in_pool < min_p_threshold;
                if top_p_satisfied && min_p_satisfied {
                    break cand;
                }
                bound = (bound * 4).min(logits.len());
            };
            (idx, sum)
        }
    };
    // Descending order exists solely so the truncation steps below can keep a *prefix*.
    // When none of them will run, the pool goes to the weighted draw whole, and a draw
    // over a fixed set is order-independent: the same tokens carry the same
    // probabilities, so the distribution is identical either way. Sorting the entire
    // vocabulary to then truncate nothing is the other half of the `top_p: 1.0` cost.
    //
    // Behaviour note: with a fixed `seed` this changes *which* token a given draw lands
    // on (the linear walk over the pool now visits it in vocabulary order). Seeded
    // reproducibility is intact — the order is deterministic — but a seed does not map
    // to the same token it did before this change.
    let truncates = min_p > 0.0 || top_n_sigma > 0.0 || top_p < 1.0;
    if truncates {
        candidates.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Equal));
    }
    let mut probs: Vec<(usize, f32)> = candidates
        .iter()
        .map(|&i| (i, (logits[i] - max_l).exp() / exp_sum))
        .collect();

    // Every truncation below keeps at least one candidate, and at least `min_keep`
    // when the caller asked for a floor.
    let floor = min_keep.max(1);

    // 5b. Min-P: drop tokens whose probability is below `min_p × max_prob`. Probs are
    // sorted descending, so keep the leading run above the threshold (at least the top).
    if min_p > 0.0 && !probs.is_empty() {
        let threshold = min_p * probs[0].1;
        let keep = probs
            .iter()
            .take_while(|(_, p)| *p >= threshold)
            .count()
            .max(floor);
        probs.truncate(keep);
    }

    // 5c. Top-nσ: keep tokens within `n` standard deviations of the top logit.
    //
    // The statistics are taken over the WHOLE vocabulary, not the candidate pool —
    // a pool-local σ would shrink as the pool shrinks and the cutoff would drift.
    //
    // Note this is invariant under temperature: scaling every logit by `1/temperature`
    // scales `max_l` and `σ` identically, so `logit >= max_l - n·σ` selects the same
    // set before and after step 3. That is why it can safely be applied here, after
    // scaling, rather than needing its own pass over the raw logits.
    if top_n_sigma > 0.0 && !probs.is_empty() {
        let n = logits.len() as f32;
        let mean = logits.iter().sum::<f32>() / n;
        let variance = logits.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / n;
        let sigma = variance.sqrt();
        if sigma.is_finite() && sigma > 0.0 {
            let threshold = max_l - top_n_sigma * sigma;
            let keep = probs
                .iter()
                .take_while(|(i, _)| logits[*i] >= threshold)
                .count()
                .max(floor);
            probs.truncate(keep);
        }
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
        probs.truncate(end.max(floor).min(probs.len()));
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

    #[test]
    fn greedy_handles_empty_logits() {
        assert_eq!(sample_greedy(&[]), 0);
    }

    #[test]
    fn greedy_all_negative_infinity_keeps_the_last() {
        // `max_by` returned the last element here; the rewrite must not change it.
        assert_eq!(sample_greedy(&[f32::NEG_INFINITY; 3]), 2);
    }

    #[test]
    fn greedy_ignores_a_nan_instead_of_picking_it() {
        assert_eq!(sample_greedy(&[5.0, f32::NAN]), 0);
        assert_eq!(sample_greedy(&[f32::NAN, 5.0]), 1);
    }

    #[test]
    fn greedy_nan_does_not_destroy_the_running_maximum() {
        // The regression that motivated the fix: the old implementation returned 2
        // here — neither the maximum nor the NaN, just whatever followed it.
        assert_eq!(sample_greedy(&[5.0, f32::NAN, 1.0]), 0);
        assert_eq!(sample_greedy(&[1.0, f32::NAN, 5.0, 2.0]), 2);
    }

    #[test]
    fn greedy_all_nan_returns_zero() {
        assert_eq!(sample_greedy(&[f32::NAN; 4]), 0);
    }

    #[test]
    fn greedy_nan_does_not_disturb_tie_breaking() {
        assert_eq!(sample_greedy(&[1.0, f32::NAN, 1.0, 0.5]), 2);
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
                repeat_last_n: -1,
                top_n_sigma: 0.0,
                min_keep: 0,
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
                repeat_last_n: -1,
                top_n_sigma: 0.0,
                min_keep: 0,
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
            repeat_last_n: -1,
            top_n_sigma: 0.0,
            min_keep: 0,
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
                    repeat_last_n: -1,
                    top_n_sigma: 0.0,
                    min_keep: 0,
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

    /// Base params for the pool-size regression tests below: every filter disabled, so
    /// the sampler must draw from the entire vocabulary.
    fn unfiltered(seed: u64) -> SamplerParams<'static> {
        SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: -1,
            top_n_sigma: 0.0,
            min_keep: 0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: None,
            generated_ids: &[],
            seed: Some(seed),
            token_count: 0,
        }
    }

    /// `top_p = 1.0` with no other filter truncates NOTHING, so the whole vocabulary
    /// stays reachable — including the tail.
    ///
    /// This guards the shortcut that makes that case fast. The adaptive candidate pool
    /// starts at 64 and grows only while a filter still demands more; with no filter
    /// demanding anything, a plausible-looking "optimisation" is to stop at the first
    /// bound, which would silently make every token past the 64th unsamplable. The
    /// vocabulary here is uniform, so a correct sampler reaches deep into it.
    #[test]
    fn unfiltered_sampling_can_reach_beyond_the_initial_pool() {
        let logits = vec![0.0f32; 1000]; // uniform: every token equally likely
        let deepest = (0u64..200)
            .map(|seed| sample_token(&logits, unfiltered(seed)))
            .max()
            .expect("200 draws");
        assert!(
            deepest >= 64,
            "the pool must span the vocabulary when nothing truncates; deepest={deepest}"
        );
    }

    /// The same shortcut must not cost the tail its weight: on a uniform vocabulary the
    /// draws should spread across it, not cluster in whatever prefix the pool started at.
    #[test]
    fn unfiltered_sampling_spreads_across_the_vocabulary() {
        let logits = vec![0.0f32; 1000];
        let seen: std::collections::HashSet<i32> = (0u64..200)
            .map(|seed| sample_token(&logits, unfiltered(seed)))
            .collect();
        let beyond_pool = seen.iter().filter(|&&t| t >= 64).count();
        assert!(
            beyond_pool > 100,
            "a uniform vocabulary should be sampled broadly; only {beyond_pool} of \
             {} distinct draws landed past the initial pool",
            seen.len()
        );
    }

    /// `min_p` with `top_p = 1.0` still truncates exactly.
    ///
    /// This pair used to be the sampler's worst case: `top_p`'s pool requirement was
    /// tested as `covered >= 1.0`, which a float sum of probabilities never reaches, so
    /// the pool grew to the whole vocabulary and was then fully sorted — 5.2 tok/s
    /// against 25 on Qwen3.8-27B. `top_p = 1.0` now imposes no requirement, leaving
    /// `min_p` to size the pool; this asserts the truncation it performs is unchanged.
    #[test]
    fn min_p_truncates_with_top_p_disabled() {
        // Token 3 dominates; min_p = 0.9 keeps only tokens within 90% of its probability.
        let logits = vec![0.0f32, 0.0, 0.0, 20.0];
        for seed in 0u64..40 {
            let t = sample_token(
                &logits,
                SamplerParams {
                    min_p: 0.9,
                    ..unfiltered(seed)
                },
            );
            assert_eq!(t, 3, "min_p must still drop the tail when top_p is 1.0");
        }
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
                    repeat_last_n: -1,
                    top_n_sigma: 0.0,
                    min_keep: 0,
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
                repeat_last_n: -1,
                top_n_sigma: 0.0,
                min_keep: 0,
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

    // top_n_sigma / min_keep

    fn sigma_params<'a>(top_n_sigma: f32, min_keep: usize) -> SamplerParams<'a> {
        SamplerParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repeat_last_n: -1,
            top_n_sigma,
            min_keep,
            logit_bias: None,
            generated_ids: &[],
            seed: Some(7),
            token_count: 0,
        }
    }

    #[test]
    fn top_n_sigma_excludes_far_below_the_top() {
        // One dominant logit and a long flat tail: a tight sigma cutoff must leave
        // only the dominant token, so every draw returns it.
        let mut logits = vec![0.0f32; 64];
        logits[5] = 30.0;
        for i in 0..200 {
            let mut p = sigma_params(0.5, 0);
            p.token_count = i;
            assert_eq!(sample_token(&logits, p), 5);
        }
    }

    #[test]
    fn top_n_sigma_zero_is_disabled() {
        // With the cutoff off, the flat tail is reachable — proving the previous
        // test measured the cutoff and not just the logit gap.
        let mut logits = vec![0.0f32; 64];
        logits[5] = 3.0;
        let mut seen_other = false;
        for i in 0..200 {
            let mut p = sigma_params(0.0, 0);
            p.token_count = i;
            if sample_token(&logits, p) != 5 {
                seen_other = true;
                break;
            }
        }
        assert!(
            seen_other,
            "with top_n_sigma disabled the tail must be reachable"
        );
    }

    #[test]
    fn top_n_sigma_is_temperature_invariant() {
        // Scaling every logit by 1/temperature scales max and sigma identically, so
        // the surviving set must not depend on temperature. Greedy would hide this,
        // so compare the full candidate outcome across many seeds.
        let mut logits = vec![0.0f32; 64];
        logits[5] = 30.0;
        logits[9] = 29.0;
        for i in 0..100 {
            let mut hot = sigma_params(1.0, 0);
            hot.temperature = 2.0;
            hot.token_count = i;
            let mut cold = sigma_params(1.0, 0);
            cold.temperature = 0.5;
            cold.token_count = i;
            // Both must stay within the surviving {5, 9} set.
            assert!(matches!(sample_token(&logits, hot), 5 | 9));
            assert!(matches!(sample_token(&logits, cold), 5 | 9));
        }
    }

    #[test]
    fn min_keep_floors_an_aggressive_truncation() {
        // min_p = 1.0 would normally leave exactly the top token; min_keep = 3 must
        // keep three candidates alive, so a run of draws hits more than one.
        let logits = vec![10.0f32, 9.5, 9.0, 1.0];
        let mut distinct = std::collections::HashSet::new();
        for i in 0..300 {
            let mut p = sigma_params(0.0, 3);
            p.min_p = 1.0;
            p.token_count = i;
            distinct.insert(sample_token(&logits, p));
        }
        assert!(
            distinct.len() > 1,
            "min_keep must prevent the truncation collapsing to one token, saw {distinct:?}"
        );
        assert!(
            !distinct.contains(&3),
            "min_keep is a floor, not a bypass — the 4th token must stay excluded"
        );
    }

    // penalty_window

    #[test]
    fn penalty_window_negative_is_whole_history() {
        let ids = [1, 2, 3, 4, 5];
        assert_eq!(penalty_window(&ids, -1), &ids[..]);
        // Any negative value means "all", matching llama.cpp.
        assert_eq!(penalty_window(&ids, -64), &ids[..]);
    }

    #[test]
    fn penalty_window_zero_disables_penalties() {
        let ids = [1, 2, 3];
        assert!(penalty_window(&ids, 0).is_empty());
    }

    #[test]
    fn penalty_window_takes_the_trailing_n() {
        let ids = [1, 2, 3, 4, 5];
        assert_eq!(penalty_window(&ids, 2), &[4, 5]);
        assert_eq!(penalty_window(&ids, 5), &ids[..]);
    }

    #[test]
    fn penalty_window_larger_than_history_does_not_panic() {
        // saturating_sub keeps this in bounds instead of underflowing.
        let ids = [1, 2];
        assert_eq!(penalty_window(&ids, 1000), &ids[..]);
        assert!(penalty_window(&[], 64).is_empty());
    }

    #[test]
    fn repeat_last_n_zero_matches_no_history() {
        // A window of 0 must be indistinguishable from having generated nothing:
        // this is what makes `--repeat-last-n 0` a true kill switch.
        let logits = vec![5.0f32, 3.0];
        let with_window_off = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 10.0,
                repeat_last_n: 0,
                top_n_sigma: 0.0,
                min_keep: 0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: None,
                generated_ids: &[0],
                seed: None,
                token_count: 1,
            },
        );
        assert_eq!(
            with_window_off, 0,
            "with the window disabled the penalty must not fire"
        );
    }

    #[test]
    fn repeat_last_n_excludes_tokens_outside_the_window() {
        // Token 0 was generated long ago; a window of 1 only sees token 1, so the
        // heavy penalty must land on token 1 and leave token 0 the winner.
        let logits = vec![5.0f32, 6.0];
        let token = sample_token(
            &logits,
            SamplerParams {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 0,
                min_p: 0.0,
                repetition_penalty: 10.0,
                repeat_last_n: 1,
                top_n_sigma: 0.0,
                min_keep: 0,
                frequency_penalty: 0.0,
                presence_penalty: 0.0,
                logit_bias: None,
                generated_ids: &[0, 1],
                seed: None,
                token_count: 2,
            },
        );
        assert_eq!(token, 0, "only the last token should have been penalised");
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
            repeat_last_n: -1,
            top_n_sigma: 0.0,
            min_keep: 0,
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
                    repeat_last_n: -1,
                    top_n_sigma: 0.0,
                    min_keep: 0,
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

    /// `select_top_n` must pick the same *set* as a full sort, and in the same order.
    ///
    /// The reference is deliberately the naive thing it replaced — sort every index by
    /// logit descending and take the first n — because the point of the change was
    /// performance, and a performance change that quietly alters which tokens are
    /// candidates is a correctness regression wearing a speedup's clothes.
    #[test]
    fn select_top_n_matches_a_full_sort() {
        fn reference(logits: &[f32], n: usize) -> Vec<usize> {
            let mut idx: Vec<usize> = (0..logits.len()).collect();
            // Ties broken by index, matching the scan order of `select_top_n`.
            idx.sort_by(|&a, &b| {
                logits[b]
                    .partial_cmp(&logits[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });
            idx.truncate(n);
            idx
        }

        // A deterministic pseudo-random spread, plus deliberate ties and negatives.
        let mut logits: Vec<f32> = (0..5000)
            .map(|i| (((i * 2654435761u64 as usize) % 10007) as f32) / 100.0 - 50.0)
            .collect();
        logits[10] = 9.5;
        logits[20] = 9.5; // exact tie, must resolve to the lower index first
        logits[30] = 9.5;

        for n in [1usize, 2, 3, 40, 64, 999] {
            assert_eq!(
                select_top_n(&logits, n),
                reference(&logits, n),
                "top-{n} disagrees with a full sort"
            );
        }
    }

    /// n larger than the vocabulary must not panic or lose entries.
    #[test]
    fn select_top_n_handles_n_beyond_len() {
        let logits = vec![0.5f32, -1.0, 3.0];
        assert_eq!(select_top_n(&logits, 10), vec![2, 0, 1]);
    }
}
