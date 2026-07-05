// Speculative decoding — n-gram / prompt-lookup proposer (0.15, S1).
//
// The draft step of speculative decoding: guess the next few tokens by finding where the
// recent output has occurred before in the same sequence. Pure logic (no llama.cpp), so
// it lives here and is unit-tested; the verify/accept/cleanup half is the real-build
// `LlamaCppModel::do_speculative_decode`. See `docs/design/speculative-decoding.md`.

/// Propose up to `draft_len` draft tokens for the sequence `seq`.
///
/// Matches the last `ngram` tokens (the suffix) against the most recent *earlier*
/// occurrence of that same n-gram in `seq`, and proposes the tokens that followed it.
/// Returns an empty vec when there is no history to match (too short) or no earlier
/// occurrence — the caller then falls back to an ordinary one-token decode.
// The only non-test caller is `do_speculative_decode`, which is compiled out in stub
// builds; keep the proposer (and its unit tests) available without a dead-code warning.
#[cfg_attr(fox_stub, allow(dead_code))]
pub(crate) fn propose_ngram(seq: &[i32], ngram: usize, draft_len: usize) -> Vec<i32> {
    if ngram == 0 || draft_len == 0 || seq.len() <= ngram {
        return Vec::new();
    }
    let suffix = &seq[seq.len() - ngram..];
    // Candidate match start positions are 0..(len-ngram): everything strictly before the
    // suffix itself. Scan from the most recent backwards so we extend the freshest match.
    for start in (0..seq.len() - ngram).rev() {
        if &seq[start..start + ngram] == suffix {
            let from = start + ngram;
            let take = draft_len.min(seq.len() - from);
            return seq[from..from + take].to_vec();
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::propose_ngram;

    #[test]
    fn too_short_or_disabled_returns_empty() {
        assert!(propose_ngram(&[1, 2], 2, 4).is_empty()); // len == ngram, no history
        assert!(propose_ngram(&[1, 2, 3], 0, 4).is_empty()); // ngram disabled
        assert!(propose_ngram(&[1, 2, 3], 2, 0).is_empty()); // draft_len disabled
    }

    #[test]
    fn no_earlier_occurrence_returns_empty() {
        // suffix [4,5] never appears earlier.
        assert!(propose_ngram(&[1, 2, 3, 4, 5], 2, 4).is_empty());
    }

    #[test]
    fn simple_repetition_proposes_the_pattern() {
        // seq = A B C A B C A B ; suffix [A B] (7,? -> last two) matched earlier → propose C…
        let seq = [10, 20, 30, 10, 20, 30, 10, 20];
        // suffix = [10, 20]; most recent earlier occurrence starts at index 3;
        // followers = seq[5..] = [30, 10, 20] → capped to draft_len.
        assert_eq!(propose_ngram(&seq, 2, 4), vec![30, 10, 20]);
    }

    #[test]
    fn draft_len_caps_the_proposal() {
        let seq = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2];
        // suffix [1,2] earlier at index 4 → followers seq[6..] = [3,4,1,2] → cap to 2.
        assert_eq!(propose_ngram(&seq, 2, 2), vec![3, 4]);
    }

    #[test]
    fn matches_the_most_recent_occurrence() {
        // suffix [1,2] appears earlier at index 0 and index 3. The proposer picks the
        // most recent (index 3), so the first draft is its follower `8`, not `9`.
        let seq = [1, 2, 9, 1, 2, 8, 1, 2];
        assert_eq!(propose_ngram(&seq, 2, 3), vec![8, 1, 2]);
    }

    #[test]
    fn longer_ngram_needs_a_longer_match() {
        // suffix [2,3,4] (ngram=3) matches the earlier [2,3,4] at index 1 → followers 5,9.
        let seq = [1, 2, 3, 4, 5, 9, 2, 3, 4];
        assert_eq!(propose_ngram(&seq, 3, 2), vec![5, 9]);
    }
}
