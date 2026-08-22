//! Per-sequence resident-prompt tracking and longest-common-prefix slot affinity.
//!
//! This is fox's port of llama-server's slot model (`server-context.cpp:1586-1694`,
//! `:3166-3243`) and it replaces the previous block-hash prefix cache.
//!
//! ## What changed and why
//!
//! The old design kept an `lru::LruCache` of *donated whole-block prompt prefixes* —
//! 8 entries at default settings, keyed by the chain hash of the last complete
//! 16-token block, popped (moved, not shared) on a hit. Three consequences, all
//! fixed here:
//!
//! 1. **The generated response was always discarded.** Only the prompt's complete
//!    blocks were donated. In a multi-turn chat the next turn's prompt *contains*
//!    the previous assistant reply, so it could never match past the point where the
//!    previous turn's prompt ended — exactly the case the cache exists to serve.
//! 2. **Reuse was block-aligned.** A prompt matching 31 of 32 tokens reused 16.
//! 3. **Capacity was tiny and not configurable** (`max_batch_size / 4`).
//!
//! Here, every sequence permanently remembers the tokens resident in its KV, prompt
//! *and* generation. A finished request "parks" its slot as [`SlotState::Idle`]
//! rather than freeing it, so the slot table *is* the cache: `max_batch_size`
//! conversations instead of 8, each holding its full tail, matched token-exactly.
//!
//! ## Blocks here are budget, not addresses
//!
//! fox's `BlockId`s never reach llama.cpp — see
//! `docs/design/llama-server-gap-analysis.md` §0.2. llama.cpp allocates its own cells
//! by `(seq_id, pos)`. So a slot's `blocks` is only *how much pool budget this
//! sequence is charged*, and block identity is irrelevant: handing a claiming request
//! `min(have, needed)` of them and topping up the difference is sound. That is why
//! token-exact reuse composes with a 16-token block allocator without any
//! reconciliation — they are different layers.

use std::time::Instant;

use crate::kv_cache::BlockId;
use crate::seq::SeqId;

/// Ownership state of one llama.cpp sequence slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlotState {
    /// Holds no KV and no blocks. Free to claim.
    Free,
    /// Owned by a live request. **Never touched by reclamation** — that is what
    /// keeps the never-preempt-on-admission invariant intact.
    Busy(u64),
    /// The request that owned this slot finished, but its KV is intact and is a
    /// valid prefix for a future prompt. Reclaimable under block pressure.
    Idle,
}

/// One llama.cpp sequence and everything fox knows about what it holds.
#[derive(Debug)]
pub(crate) struct Slot {
    /// Stable llama.cpp `seq_id`. Equal to the slot's index, like `slot.id = i` in
    /// llama-server (`server-context.cpp:1355-1374`). This table is the only place
    /// in fox that mints one — see [`SeqId`].
    pub(super) seq_id: SeqId,
    /// Tokens currently resident in llama.cpp's KV for this sequence, at positions
    /// `0..tokens.len()`. Only meaningful while [`SlotState::Idle`]: while `Busy`
    /// the owning request is the source of truth, and while `Free` this is empty.
    pub(super) tokens: Vec<i32>,
    /// KV pool blocks charged to this slot.
    pub(super) blocks: Vec<BlockId>,
    pub(super) state: SlotState,
    /// When this slot last stopped being `Busy`. Drives LRU selection and reclaim.
    pub(super) t_last_used: Instant,
}

/// One slot's externally visible state, for `GET /slots`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SlotSnapshot {
    /// llama.cpp `seq_id`, stable for the process lifetime.
    pub id: i32,
    /// `free` | `processing` | `idle`. `idle` means the slot holds reusable KV from
    /// a finished request — a cache entry, not work in progress.
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<u64>,
    /// How many tokens of KV this slot currently holds.
    pub resident_tokens: usize,
    /// KV pool blocks charged to it.
    pub blocks: usize,
    /// Seconds since it stopped being busy — only meaningful while idle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub idle_secs: Option<u64>,
}

/// The slot chosen for an incoming request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SlotChoice {
    pub(super) index: usize,
    /// How many leading tokens of the new prompt are already resident in this
    /// slot's KV. `0` for a fresh or LRU-recycled slot.
    pub(super) lcp: usize,
}

/// Length of the longest common prefix of two token sequences.
pub(super) fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Fixed-size table of sequence slots, one per concurrent request the engine allows.
#[derive(Debug)]
pub(crate) struct SlotTable {
    slots: Vec<Slot>,
}

impl SlotTable {
    pub(super) fn new(n_slots: usize) -> Self {
        let now = Instant::now();
        Self {
            slots: (0..n_slots)
                .map(|index| Slot {
                    seq_id: SeqId::slot(index),
                    tokens: Vec::new(),
                    blocks: Vec::new(),
                    state: SlotState::Free,
                    t_last_used: now,
                })
                .collect(),
        }
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Pick the slot an incoming request should occupy, mirroring llama-server's
    /// three-stage `get_available_slot()` (`server-context.cpp:1586-1694`):
    ///
    /// 1. **Affinity** — among *idle* slots, the one whose resident tokens share the
    ///    longest prefix with this prompt, provided the match covers at least
    ///    `min_similarity` of the prompt.
    /// 2. **Free** — an untouched slot, so an unrelated prompt doesn't evict a
    ///    warm conversation.
    /// 3. **LRU** — the idle slot unused for longest.
    ///
    /// Returns `None` only when every slot is `Busy`, which the caller must treat as
    /// "wait", never as "evict something".
    ///
    /// `allow_reuse` is false for requests that must not inherit anyone's KV (LoRA:
    /// KV computed under one adapter is invalid input for another — see
    /// `docs/design/lora-support.md`), and for a globally disabled `--kv-reuse`.
    pub(super) fn select(
        &self,
        prompt: &[i32],
        min_similarity: f32,
        allow_reuse: bool,
    ) -> Option<SlotChoice> {
        // Stage 1: longest-common-prefix affinity.
        //
        // `prompt` is empty for multimodal requests (their positions come from image
        // chunks, not `prompt_tokens`), so this loop is naturally skipped for them and
        // they fall through to a Free/LRU slot with lcp = 0. That also keeps the
        // similarity division below safe from a zero denominator.
        if allow_reuse && !prompt.is_empty() {
            let mut best: Option<SlotChoice> = None;
            for (index, slot) in self.slots.iter().enumerate() {
                if slot.state != SlotState::Idle || slot.tokens.is_empty() {
                    continue;
                }
                let lcp = common_prefix_len(&slot.tokens, prompt);
                if lcp == 0 {
                    continue;
                }
                if (lcp as f32 / prompt.len() as f32) < min_similarity {
                    continue;
                }
                if best.is_none_or(|b| lcp > b.lcp) {
                    best = Some(SlotChoice { index, lcp });
                }
            }
            if best.is_some() {
                return best;
            }
        }

        // Stage 2: a Free slot.
        if let Some(index) = self.slots.iter().position(|s| s.state == SlotState::Free) {
            return Some(SlotChoice { index, lcp: 0 });
        }

        // Stage 3: least-recently-used idle slot.
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.state == SlotState::Idle)
            .min_by_key(|(_, s)| s.t_last_used)
            .map(|(index, _)| SlotChoice { index, lcp: 0 })
    }

    /// The `seq_id` of the slot at `index`.
    pub(super) fn seq_id_at(&self, index: usize) -> SeqId {
        self.slots[index].seq_id
    }

    /// Overwrite what a slot is considered to hold, after its KV was replaced from the
    /// host-RAM cache. The slot's previous contents are gone — `state_seq_load` clears
    /// the destination — so the bookkeeping has to follow, or a later LCP match would
    /// be computed against tokens that are no longer resident.
    pub(super) fn set_resident(&mut self, index: usize, tokens: Vec<i32>) {
        self.slots[index].tokens = tokens;
    }

    /// Forget what a sequence was believed to hold, by `seq_id`. Used when a restore
    /// failed: the slot's recorded tokens describe a state that never landed.
    pub(super) fn forget_resident(&mut self, seq_id: SeqId) {
        if let Some(slot) = self.slots.iter_mut().find(|s| s.seq_id == seq_id) {
            slot.tokens.clear();
        }
    }

    /// How many blocks the slot at `index` currently holds.
    pub(super) fn blocks_at(&self, index: usize) -> usize {
        self.slots[index].blocks.len()
    }

    /// How many tokens this slot's KV currently holds — prompt *and* generated reply.
    /// Taking up a slot's LCP offer means rolling the sequence back from here to the
    /// divergence point, so this is the distance a bounded cache has to be checked
    /// against before the offer is accepted.
    pub(super) fn resident_len(&self, index: usize) -> usize {
        self.slots[index].tokens.len()
    }

    /// Hand the slot to `req_id`, yielding its `seq_id` and the blocks it already
    /// holds (which the request inherits rather than re-allocating). The resident
    /// token list is dropped: while `Busy`, the owning request is the source of
    /// truth, and [`Self::park`] restores it on completion.
    pub(super) fn claim(&mut self, index: usize, req_id: u64) -> (SeqId, Vec<BlockId>) {
        let slot = &mut self.slots[index];
        slot.state = SlotState::Busy(req_id);
        slot.tokens.clear();
        (slot.seq_id, std::mem::take(&mut slot.blocks))
    }

    /// Park a finished request's sequence as reusable: KV stays resident, blocks stay
    /// charged, and `tokens` records exactly what llama.cpp holds at positions
    /// `0..tokens.len()`.
    ///
    /// Returns `false` if `seq_id` isn't a slot this table owns.
    pub(super) fn park(&mut self, seq_id: SeqId, tokens: Vec<i32>, blocks: Vec<BlockId>) -> bool {
        let Some(slot) = self.slots.iter_mut().find(|s| s.seq_id == seq_id) else {
            return false;
        };
        slot.tokens = tokens;
        slot.blocks = blocks;
        slot.state = SlotState::Idle;
        slot.t_last_used = Instant::now();
        true
    }

    /// Drop a slot's KV entirely and return the blocks it held so the caller can free
    /// them. Used when a finished request's KV must not be reused (context-rolled,
    /// LoRA, engine error) and when reclaiming under pressure.
    pub(super) fn release(&mut self, seq_id: SeqId) -> Vec<BlockId> {
        let Some(slot) = self.slots.iter_mut().find(|s| s.seq_id == seq_id) else {
            return Vec::new();
        };
        slot.tokens.clear();
        slot.state = SlotState::Free;
        slot.t_last_used = Instant::now();
        std::mem::take(&mut slot.blocks)
    }

    /// Free the least-recently-used *idle* slot to make room, skipping `except`
    /// (the slot the incoming request was matched to).
    ///
    /// **This is not preemption.** An idle slot belongs to a request that already
    /// *finished* and whose output the client already received; its KV is a cache
    /// entry, not work in progress. `Busy` slots are never considered, so the
    /// never-preempt-on-admission invariant (see
    /// `admission_never_preempts_running_requests`) is untouched.
    /// Returns `(seq_id, resident_tokens, blocks)`. The token list comes back rather
    /// than being dropped so the caller can serialise the sequence to the host-RAM
    /// prompt cache before its KV is wiped — the victim is only known here, so
    /// reading it beforehand is not possible.
    pub(super) fn reclaim_lru(&mut self, except: usize) -> Option<(SeqId, Vec<i32>, Vec<BlockId>)> {
        let index = self
            .slots
            .iter()
            .enumerate()
            .filter(|(i, s)| *i != except && s.state == SlotState::Idle)
            .min_by_key(|(_, s)| s.t_last_used)
            .map(|(i, _)| i)?;
        let slot = &mut self.slots[index];
        slot.state = SlotState::Free;
        Some((
            slot.seq_id,
            std::mem::take(&mut slot.tokens),
            std::mem::take(&mut slot.blocks),
        ))
    }

    /// Total blocks charged across every slot — the accounting counterpart to
    /// `KVCacheManager::allocated_blocks()`. Used by the conservation assertions in
    /// `stress_slot_reuse_no_leak`.
    #[cfg(test)]
    pub(super) fn charged_blocks(&self) -> usize {
        self.slots.iter().map(|s| s.blocks.len()).sum()
    }

    pub(super) fn count(&self, f: impl Fn(&Slot) -> bool) -> usize {
        self.slots.iter().filter(|s| f(s)).count()
    }

    #[cfg(test)]
    pub(super) fn iter(&self) -> impl Iterator<Item = &Slot> {
        self.slots.iter()
    }

    /// Per-slot state for `GET /slots`.
    ///
    /// Deliberately reports only what a slot *holds*, never the tokens themselves:
    /// a resident sequence is another user's conversation, and an unauthenticated
    /// introspection endpoint must not hand it out. llama-server redacts the same
    /// fields unless `LLAMA_SERVER_SLOTS_DEBUG` is set.
    pub(crate) fn snapshot(&self) -> Vec<SlotSnapshot> {
        self.slots
            .iter()
            .map(|s| SlotSnapshot {
                // The wire format is llama-server's, so the snapshot carries the raw id.
                id: s.seq_id.raw(),
                state: match s.state {
                    SlotState::Free => "free",
                    SlotState::Busy(_) => "processing",
                    SlotState::Idle => "idle",
                },
                request_id: match s.state {
                    SlotState::Busy(id) => Some(id),
                    _ => None,
                },
                resident_tokens: s.tokens.len(),
                blocks: s.blocks.len(),
                idle_secs: matches!(s.state, SlotState::Idle)
                    .then(|| s.t_last_used.elapsed().as_secs()),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_with(entries: &[(SlotState, &[i32], usize)]) -> SlotTable {
        let mut t = SlotTable::new(entries.len());
        for (i, (state, tokens, nblocks)) in entries.iter().enumerate() {
            t.slots[i].state = *state;
            t.slots[i].tokens = tokens.to_vec();
            t.slots[i].blocks = (0..*nblocks).collect();
        }
        t
    }

    #[test]
    fn common_prefix_len_basics() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2, 3]), 2);
        assert_eq!(common_prefix_len(&[], &[1]), 0);
        assert_eq!(common_prefix_len(&[9], &[1]), 0);
    }

    #[test]
    fn select_prefers_longest_common_prefix() {
        let t = table_with(&[
            (SlotState::Idle, &[1, 2, 3, 4, 5], 1),
            (SlotState::Idle, &[1, 2], 1),
        ]);
        let c = t.select(&[1, 2, 3, 4, 9], 0.1, true).unwrap();
        assert_eq!(c.index, 0);
        assert_eq!(c.lcp, 4);
    }

    #[test]
    fn select_ignores_matches_below_the_similarity_floor() {
        // 1 of 10 tokens shared = 0.1 similarity; a 0.5 floor must reject it and
        // fall through to the Free slot instead of hijacking a warm one.
        let t = table_with(&[(SlotState::Idle, &[1, 99], 1), (SlotState::Free, &[], 0)]);
        let c = t
            .select(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5, true)
            .unwrap();
        assert_eq!(c.index, 1);
        assert_eq!(c.lcp, 0);
    }

    #[test]
    fn select_skips_affinity_when_reuse_is_disallowed() {
        let t = table_with(&[(SlotState::Idle, &[1, 2, 3], 1), (SlotState::Free, &[], 0)]);
        // A LoRA request must never inherit another request's KV.
        let c = t.select(&[1, 2, 3], 0.1, false).unwrap();
        assert_eq!(c.index, 1);
        assert_eq!(c.lcp, 0);
    }

    #[test]
    fn select_prefers_free_over_recycling_an_idle_slot() {
        let t = table_with(&[(SlotState::Idle, &[7, 7, 7], 1), (SlotState::Free, &[], 0)]);
        // No shared prefix at all → don't evict the warm slot, take the free one.
        let c = t.select(&[1, 2, 3], 0.1, true).unwrap();
        assert_eq!(c.index, 1);
    }

    #[test]
    fn select_falls_back_to_lru_idle() {
        let mut t = table_with(&[(SlotState::Idle, &[7], 1), (SlotState::Idle, &[8], 1)]);
        // Make slot 1 the older one.
        t.slots[1].t_last_used = Instant::now() - std::time::Duration::from_secs(60);
        let c = t.select(&[1, 2, 3], 0.1, true).unwrap();
        assert_eq!(c.index, 1);
        assert_eq!(c.lcp, 0);
    }

    #[test]
    fn select_returns_none_when_every_slot_is_busy() {
        let t = table_with(&[(SlotState::Busy(1), &[], 1), (SlotState::Busy(2), &[], 1)]);
        assert!(t.select(&[1, 2, 3], 0.1, true).is_none());
    }

    #[test]
    fn select_with_empty_prompt_does_not_divide_by_zero() {
        // Multimodal requests carry no `prompt_tokens`.
        let t = table_with(&[(SlotState::Idle, &[1, 2], 1), (SlotState::Free, &[], 0)]);
        let c = t.select(&[], 0.1, true).unwrap();
        assert_eq!(c.lcp, 0);
    }

    #[test]
    fn claim_takes_blocks_and_clears_resident_tokens() {
        let mut t = table_with(&[(SlotState::Idle, &[1, 2, 3], 3)]);
        let (seq_id, blocks) = t.claim(0, 42);
        assert_eq!(seq_id, SeqId::slot(0));
        assert_eq!(blocks.len(), 3);
        assert_eq!(t.slots[0].state, SlotState::Busy(42));
        assert!(t.slots[0].tokens.is_empty());
        assert!(t.slots[0].blocks.is_empty());
        assert_eq!(t.charged_blocks(), 0, "blocks moved to the request");
    }

    #[test]
    fn park_makes_the_whole_sequence_reusable() {
        let mut t = table_with(&[(SlotState::Busy(42), &[], 0)]);
        assert!(t.park(SeqId::slot(0), vec![1, 2, 3, 4], vec![0, 1]));
        assert_eq!(t.slots[0].state, SlotState::Idle);
        // The parked tokens include the generated tail, which is the whole point:
        // the next turn of a conversation matches past the previous prompt's end.
        assert_eq!(t.slots[0].tokens, vec![1, 2, 3, 4]);
        assert_eq!(t.charged_blocks(), 2);
    }

    #[test]
    fn release_frees_the_slot_and_hands_back_its_blocks() {
        let mut t = table_with(&[(SlotState::Idle, &[1, 2], 2)]);
        let blocks = t.release(SeqId::slot(0));
        assert_eq!(blocks.len(), 2);
        assert_eq!(t.slots[0].state, SlotState::Free);
        assert!(t.slots[0].tokens.is_empty());
        assert_eq!(t.charged_blocks(), 0);
    }

    #[test]
    fn reclaim_lru_never_touches_a_busy_slot() {
        let mut t = table_with(&[(SlotState::Busy(1), &[1], 5), (SlotState::Idle, &[2], 2)]);
        t.slots[0].t_last_used = Instant::now() - std::time::Duration::from_secs(600);
        // Slot 0 is by far the oldest, but it is Busy — reclaiming it would be
        // preemption, which admission must never do.
        let (seq_id, tokens, blocks) = t.reclaim_lru(usize::MAX).unwrap();
        assert_eq!(seq_id, SeqId::slot(1));
        assert_eq!(
            tokens,
            vec![2],
            "the victim's tokens come back for the RAM cache"
        );
        assert_eq!(blocks.len(), 2);
        assert_eq!(t.slots[0].state, SlotState::Busy(1));
    }

    #[test]
    fn reclaim_lru_skips_the_excepted_slot_and_reports_exhaustion() {
        let mut t = table_with(&[(SlotState::Idle, &[1], 2), (SlotState::Busy(9), &[], 1)]);
        // The only idle slot is the one we're about to claim — nothing to reclaim.
        assert!(t.reclaim_lru(0).is_none());
    }
}
