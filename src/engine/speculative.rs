// Speculative decoding proposers.
//
// The draft step of speculative decoding: guess the next few tokens before the target
// model verifies them. Two implementations:
//   - `NgramProposer` (0.15, S1): guess by finding where the recent output has occurred
//     before in the same sequence. Pure logic (no llama.cpp).
//   - `DraftModelProposer` (0.16): a second, smaller resident model predicts ahead.
// The verify/accept/cleanup half (`LlamaCppModel::do_speculative_decode`) is proposer-
// agnostic — it verifies whatever `Vec<i32>` of draft tokens it's given, so exactness
// (byte-identical output on/off) holds regardless of which proposer produced them; a
// wrong draft is simply rejected. See `docs/design/speculative-decoding.md` and
// `docs/design/speculative-roadmap.md` (Level 2).

use std::sync::{Arc, Mutex};

use super::model::Model;
use crate::seq::SeqId;

/// Proposes candidate tokens for speculative decoding, to be verified against the
/// target model's real logits. Implementations must be cheap to call once per decode
/// step (a `DraftModelProposer` call runs the draft model's own decode — callers must
/// invoke it from a blocking context, same as any other llama.cpp call).
pub trait Proposer: Send + Sync {
    /// Propose up to `draft_len` candidate tokens for `seq` (the request's full
    /// logical sequence so far: prompt ++ generated). `req_id` lets a stateful
    /// proposer (the draft model) detect when a different request has taken over
    /// the single speculative "slot" and reset its own state accordingly.
    fn propose(&self, req_id: u64, seq: &[i32], draft_len: usize) -> Vec<i32>;

    /// The KV sequence the request occupies, for proposers whose state is keyed by it.
    ///
    /// Separate from `req_id`: fox's request ids are monotonic and unique for the life
    /// of the process, while `kv_seq_id` is a slot index that gets reused. The MTP
    /// driver keys its per-sequence hidden-state carryover by the *llama.cpp* sequence,
    /// so it has to be told which one, not which request.
    fn set_kv_seq_id(&self, kv_seq_id: SeqId) {
        let _ = kv_seq_id;
    }

    /// Report how many of the last `propose` call's tokens the target committed.
    ///
    /// Only the MTP proposer needs this: its drafts come from a stateful driver that
    /// has to know where the sequence actually ended up. n-gram is stateless, and the
    /// draft model resyncs from `seq` on the next call, so both ignore it.
    fn accept(&self, n_accepted: usize) {
        let _ = n_accepted;
    }
}

/// Thin wrapper around `propose_ngram` implementing `Proposer` (0.15's proposer,
/// unchanged — this refactor only moves where it's called from).
pub struct NgramProposer {
    pub ngram: usize,
}

impl Proposer for NgramProposer {
    fn propose(&self, _req_id: u64, seq: &[i32], draft_len: usize) -> Vec<i32> {
        propose_ngram(seq, self.ngram, draft_len)
    }
}

/// Proposer backed by the model's own trained MTP head.
///
/// Unlike [`DraftModelProposer`] this holds no second model and no KV state of its own:
/// the head reads the target's hidden states, and llama.cpp's driver — fed by
/// `mtp_process` after every generation decode — owns whatever carryover it needs. So
/// there is nothing here to resync when a different request takes over the speculative
/// slot; the driver tracks that per sequence itself.
pub struct MtpProposer {
    model: Arc<dyn Model>,
    /// The llama.cpp sequence the current request occupies. The driver keys its
    /// hidden-state carryover by this, and `mtp_process` files those rows under the
    /// sequence the batch actually used — so asking for a draft under any other id
    /// returns nothing, or something that belongs to a different conversation.
    kv_seq_id: Mutex<Option<SeqId>>,
    /// The request currently holding the speculative slot, so a new one gets a `begin`.
    last_req_id: Mutex<Option<u64>>,
}

impl MtpProposer {
    fn seq_id(&self) -> Option<SeqId> {
        *self.kv_seq_id.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn new(model: Arc<dyn Model>) -> Self {
        Self {
            model,
            kv_seq_id: Mutex::new(None),
            last_req_id: Mutex::new(None),
        }
    }
}

impl Proposer for MtpProposer {
    fn propose(&self, req_id: u64, seq: &[i32], draft_len: usize) -> Vec<i32> {
        // The driver wants the sequence WITHOUT its last token, plus that token as
        // `id_last` — llama-server keeps them apart the same way (`prompt.tokens` gets
        // `{ids.begin(), ids.end() - 1}` while `slot.sampled = ids.back()`), because the
        // last token has not been decoded yet: it goes into the next batch. Passing the
        // full sequence here instead puts one extra token in front of every hidden row
        // the driver pairs up, which reads as drafts that are almost always wrong.
        let Some((&id_last, head)) = seq.split_last() else {
            return Vec::new(); // nothing to continue from
        };
        // A new request on the speculative slot starts a new generation as far as the
        // driver is concerned. `begin` is also where upstream checks that the prefill's
        // hidden states actually reached the MTP context, and warns when they did not.
        {
            let mut last = self.last_req_id.lock().unwrap_or_else(|e| e.into_inner());
            if *last != Some(req_id) {
                *last = Some(req_id);
                if let Some(seq_id) = self.seq_id() {
                    self.model.mtp_begin(seq_id, head);
                }
            }
        }
        // No sequence yet means the engine has not told us which slot the request
        // occupies; drafting under a guessed one is how `a4171eb` happened.
        let Some(seq_id) = self.seq_id() else {
            return Vec::new();
        };
        self.model
            .mtp_propose(seq_id, head.len() as i32, id_last, head, draft_len)
    }

    fn set_kv_seq_id(&self, kv_seq_id: SeqId) {
        *self.kv_seq_id.lock().unwrap_or_else(|e| e.into_inner()) = Some(kv_seq_id);
    }

    fn accept(&self, n_accepted: usize) {
        if let Some(seq_id) = self.seq_id() {
            self.model.mtp_accept(seq_id, n_accepted as u16);
        }
    }
}

struct DraftState {
    /// Count of tokens from `seq` that are currently, correctly present in the draft
    /// model's own KV at `seq_id` — i.e. real, target-verified tokens only, never the
    /// draft's own unconfirmed speculative tail.
    synced_len: usize,
    /// The request this state belongs to; a change means the shared speculative slot
    /// switched to an unrelated request, and the draft's KV must be reset rather than
    /// incrementally reused (it holds a different, unrelated sequence's content).
    last_req_id: Option<u64>,
}

/// Draft-model speculation (0.16): a second, smaller resident model proposes tokens
/// by running its own decode loop, reusing the exact same verify/accept machinery as
/// n-gram speculation. Only one speculative sequence is ever live at a time (fox only
/// speculates on a single decoding request per step), so a single dedicated `seq_id`
/// in the draft's own KV is used — no pool needed.
pub struct DraftModelProposer {
    draft_model: Arc<dyn Model>,
    seq_id: SeqId,
    state: Mutex<DraftState>,
}

impl DraftModelProposer {
    pub fn new(draft_model: Arc<dyn Model>) -> Self {
        Self {
            draft_model,
            // The draft model has its own llama.cpp context, so this id shares no
            // address space with the scheduler's slots — see `SeqId::dedicated`.
            seq_id: SeqId::dedicated(0),
            state: Mutex::new(DraftState {
                synced_len: 0,
                last_req_id: None,
            }),
        }
    }
}

impl Proposer for DraftModelProposer {
    fn propose(&self, req_id: u64, seq: &[i32], draft_len: usize) -> Vec<i32> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if state.last_req_id != Some(req_id) || state.synced_len > seq.len() {
            // A different request took over the speculative slot (or, defensively,
            // synced_len is somehow ahead of seq — can't happen for the same
            // monotonically-growing request) — the draft's KV holds an unrelated
            // sequence's content, so start fresh rather than reuse it.
            self.draft_model.clear_sequence(self.seq_id);
            state.synced_len = 0;
            state.last_req_id = Some(req_id);
        } else if !self
            .draft_model
            .trim_sequence(self.seq_id, state.synced_len)
        {
            // Discard last round's unconfirmed speculative tail (the target may not
            // have accepted all of it) — keep only the target-verified prefix.
            //
            // A refused trim means the tail is still there, so feeding the next tokens
            // would write at positions that already hold cells. Same escape hatch the
            // engine uses for a refused trim: wipe and re-sync from scratch. Costs one
            // round of drafting, never produces drafts from a state the model did not
            // reach. (Found by `#[must_use]` — this return value used to be discarded,
            // which is `f5214df` item 3 in a second place.)
            self.draft_model.clear_sequence(self.seq_id);
            state.synced_len = 0;
        }

        let new_tokens = &seq[state.synced_len..];
        let base_pos = state.synced_len as i32;
        let produced = self
            .draft_model
            .draft_propose(self.seq_id, new_tokens, base_pos, draft_len);
        state.synced_len += new_tokens.len();
        produced
    }
}

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
    use super::{propose_ngram, DraftModelProposer, NgramProposer, Proposer};
    use crate::engine::model::{InferenceRequestForModel, Logits, Model, ModelConfig, PrefillStep};
    use crate::seq::SeqId;
    use std::sync::{Arc, Mutex as StdMutex};

    #[test]
    fn ngram_proposer_delegates_to_propose_ngram() {
        // Thin-wrapper regression check: NgramProposer must produce exactly what the
        // bare function does for the same inputs.
        let seq = [10, 20, 30, 10, 20, 30, 10, 20];
        let p = NgramProposer { ngram: 2 };
        assert_eq!(p.propose(1, &seq, 4), propose_ngram(&seq, 2, 4));
    }

    /// Minimal `Model` mock recording every call to the methods
    /// `DraftModelProposer` calls, so its bookkeeping can be asserted directly.
    /// Methods irrelevant to that bookkeeping panic if ever called.
    #[derive(Default)]
    struct RecordingModel {
        calls: StdMutex<Vec<String>>,
    }

    impl Model for RecordingModel {
        fn prefill_sync(
            &self,
            _: &[u64],
            _: &[InferenceRequestForModel],
            _: usize,
        ) -> anyhow::Result<Vec<PrefillStep>> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn decode_sync(
            &self,
            _: &[u64],
            _: &[InferenceRequestForModel],
        ) -> anyhow::Result<Vec<(u64, Logits)>> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn model_config(&self) -> ModelConfig {
            unimplemented!("not used by DraftModelProposer")
        }
        fn eos_token_id(&self) -> i32 {
            -1
        }
        fn is_eog_token(&self, _token_id: i32) -> bool {
            false
        }
        fn tokenize(&self, _text: &str) -> anyhow::Result<Vec<i32>> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn token_to_piece(&self, _token: i32) -> anyhow::Result<String> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn apply_chat_template(&self, _messages: &[(String, String)]) -> anyhow::Result<String> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn clear_sequence(&self, seq_id: SeqId) {
            self.calls.lock().unwrap().push(format!("clear({seq_id})"));
        }
        fn trim_sequence(&self, seq_id: SeqId, from_pos: usize) -> bool {
            self.calls
                .lock()
                .unwrap()
                .push(format!("trim({seq_id},{from_pos})"));
            true
        }
        fn copy_sequence_range(&self, _src_seq_id: SeqId, _dst_seq_id: SeqId, _token_count: i32) {}
        fn supports_seq_copy(&self) -> bool {
            false
        }
        fn embedding_dim(&self) -> usize {
            0
        }
        fn get_embeddings(&self, _tokens: &[i32]) -> anyhow::Result<Vec<f32>> {
            unimplemented!("not used by DraftModelProposer")
        }
        fn stop_tokens(&self) -> Vec<String> {
            vec![]
        }
        fn draft_propose(
            &self,
            seq_id: SeqId,
            new_tokens: &[i32],
            base_pos: i32,
            draft_len: usize,
        ) -> Vec<i32> {
            self.calls.lock().unwrap().push(format!(
                "draft_propose({seq_id},{new_tokens:?},{base_pos},{draft_len})"
            ));
            // Fixed fake proposal — just enough to prove `synced_len` only ever
            // advances by real (fed) tokens, never by this speculative output.
            vec![900, 901]
        }
    }

    #[test]
    fn draft_proposer_first_call_clears_and_feeds_from_zero() {
        let model = Arc::new(RecordingModel::default());
        let p = DraftModelProposer::new(model.clone());

        let produced = p.propose(1, &[10, 11, 12], 2);
        assert_eq!(produced, vec![900, 901]);

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec!["clear(0)", "draft_propose(0,[10, 11, 12],0,2)"],
            "a fresh proposer must clear before its first feed and start at position 0"
        );
    }

    #[test]
    fn draft_proposer_same_request_trims_the_speculative_tail_not_the_full_seq() {
        let model = Arc::new(RecordingModel::default());
        let p = DraftModelProposer::new(model.clone());

        let _ = p.propose(1, &[10, 11, 12], 2);
        model.calls.lock().unwrap().clear();

        // Same request, seq grew by 2 real tokens (as if the target committed 2 more).
        let _ = p.propose(1, &[10, 11, 12, 100, 101], 2);

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec!["trim(0,3)", "draft_propose(0,[100, 101],3,2)"],
            "continuing the same request must trim back to the trusted prefix (3), \
             not to seq.len(), and feed only the newly-real tokens"
        );
    }

    #[test]
    fn draft_proposer_different_request_resets_instead_of_reusing_state() {
        let model = Arc::new(RecordingModel::default());
        let p = DraftModelProposer::new(model.clone());

        let _ = p.propose(1, &[10, 11, 12], 2);
        model.calls.lock().unwrap().clear();

        // A different request takes over the shared speculative slot — the draft's
        // KV holds request 1's unrelated content and must be reset, not incrementally
        // reused (even though this seq is shorter, which would also trigger the
        // defensive synced_len > seq.len() branch).
        let _ = p.propose(2, &[50, 51], 2);

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec!["clear(0)", "draft_propose(0,[50, 51],0,2)"],
            "a different request must reset (clear), never trim/reuse another \
             request's synced state"
        );
    }

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
