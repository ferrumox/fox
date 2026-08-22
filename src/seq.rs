//! `SeqId` — a llama.cpp sequence id that cannot be written down as a literal.
//!
//! # Why this is a newtype
//!
//! llama.cpp addresses its KV cells by `(seq_id, pos)`, and until now fox passed
//! that id around as a bare `i32`. Two of fox's KV lifecycle bugs came directly
//! from that:
//!
//! * `a4171eb` — `do_get_embeddings` built its batch with `*arr.add(0) = 0` and then
//!   `seq_rm`'d sequence 0. The scheduler hands out `0..max_batch`, and issues `0`
//!   *last*, which is why nothing below full concurrency ever hit it. Under load an
//!   embedding request corrupted, then erased, a live generation's KV.
//! * `a4171eb` again, one layer up: nothing distinguished "an id the slot table
//!   issued" from "an integer someone typed".
//!
//! The guarantee this type provides is deliberately narrow, and worth stating
//! exactly so nobody relies on more than it gives:
//!
//! * A bare integer is a **type error** in every Rust signature that expects a
//!   sequence — 60 of them, and no raw `seq_id: i32` survives outside the FFI layer.
//! * Minting a `SeqId` is still possible — it has to be, since the ids ultimately
//!   come from a range fox chooses — but only through the two named constructors
//!   below, both `pub(crate)`, both greppable. It is an act, not an accident.
//!
//! What it does **not** do, stated here because an earlier version of this comment
//! claimed it did: it does not make `a4171eb` unwriteable. `llama_batch` is a bag of
//! bindgen pointers and no newtype reaches through one, so a raw store into
//! `batch.seq_id` remains expressible — see `set_batch_row` in
//! `engine/model/llama_cpp/batch.rs`, which says the same from the other side. What
//! stops the bug is that `set_batch_row` is the only sanctioned write path and it
//! takes a `SeqId`, so re-writing that literal means hand-rolling pointer arithmetic
//! next to a helper that already exists. The type turns an invisible literal into a
//! visible act; it does not turn a runtime bug into a compile error. Claiming
//! otherwise would be the exact overselling this module was written to stop.
//!
//! It is *not* an ownership token: `SeqId` is `Copy`, and holding one says nothing
//! about whether the sequence is still yours. That would be
//! `docs/design/kv-typed-classification.md`'s phase 2, which the classification
//! recommends against.

use std::fmt;

/// A llama.cpp sequence id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(i32);

impl SeqId {
    /// The sequence belonging to slot `index` of the scheduler's slot table.
    ///
    /// Mirrors llama-server's `slot.id = i` (`server-context.cpp:1355-1374`): the
    /// slot table owns the range `0..n_slots` and this is the only way into it.
    pub(crate) fn slot(index: usize) -> Self {
        Self(index as i32)
    }

    /// A sequence that is *not* one of the scheduler's slots.
    ///
    /// Two legitimate callers, and the constructor is named so that a third has to
    /// argue for itself:
    ///
    /// * the embeddings sequence, allocated at `n_seq - 1` **beyond** the slot range
    ///   precisely so it cannot collide with a generation (this is `a4171eb`);
    /// * the draft model's own sequence, which lives in a different llama.cpp context
    ///   with its own address space and so cannot collide with anything.
    ///
    /// Anything else reaching for this is very likely `a4171eb` being written again
    /// under a different name.
    pub(crate) fn dedicated(raw: i32) -> Self {
        Self(raw)
    }

    /// The `i32` llama.cpp's C API wants. Call this at the FFI boundary and nowhere
    /// else — every use inside fox should keep the `SeqId`.
    pub fn raw(self) -> i32 {
        self.0
    }
}

impl fmt::Display for SeqId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_ids_are_the_slot_index() {
        assert_eq!(SeqId::slot(0).raw(), 0);
        assert_eq!(SeqId::slot(7).raw(), 7);
    }

    #[test]
    fn the_embeddings_sequence_is_outside_the_slot_range() {
        let n_slots = 4;
        let embed = SeqId::dedicated(n_slots as i32);
        assert!(
            (0..n_slots).all(|i| SeqId::slot(i) != embed),
            "the a4171eb collision must be impossible by construction"
        );
    }

    /// `Option<SeqId>` replaced the `-1` sentinel that `kv_seq_id` used to carry, so
    /// "no sequence" is a variant rather than a magic number a comparison can miss.
    #[test]
    fn unassigned_is_a_variant_not_a_sentinel() {
        let unassigned: Option<SeqId> = None;
        assert!(unassigned.is_none());
        assert_eq!(std::mem::size_of::<Option<SeqId>>(), 8);
    }
}
