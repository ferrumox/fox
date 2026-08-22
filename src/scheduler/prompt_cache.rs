//! Host-RAM prompt cache — llama-server's `server_prompt_cache` / `--cache-ram`
//! (`server-task.cpp:1600-1793`).
//!
//! The slot table (see `slots.rs`) keeps a finished conversation warm by *holding its
//! GPU blocks*. That caps the number of warm conversations at `--max-batch-size` and
//! makes each one cost VRAM even while nobody is using it. This cache is the other
//! half: a reclaimed sequence's KV is serialised to host memory first, so it can be
//! restored later without having occupied a block in the meantime.
//!
//! That trade — host RAM and a memcpy, instead of VRAM and a re-prefill — is what makes
//! it worth having on a single-model laptop, where VRAM is the scarce resource and
//! re-prefilling a long conversation is the expensive operation.
//!
//! Entries are opaque, model-specific blobs (see `Model::state_seq_save`), so the cache
//! belongs to one loaded model and must be dropped when that model unloads.

use std::time::Instant;

/// One cached sequence: the tokens it holds and its serialised KV.
#[derive(Debug)]
struct Entry {
    tokens: Vec<i32>,
    data: Vec<u8>,
    last_used: Instant,
}

/// What a lookup found.
#[derive(Debug, Clone)]
pub(crate) struct PromptCacheHit {
    /// The serialised state to restore.
    pub(crate) data: Vec<u8>,
    /// How many leading tokens of the query prompt this state covers.
    pub(crate) matched: usize,
    /// Total tokens the restored sequence will hold, which is what the caller must
    /// treat as resident afterwards — it can exceed `matched` when the cached
    /// conversation continued past the point the new prompt diverges.
    pub(crate) resident: usize,
}

/// Fixed-byte-budget store of serialised sequence states.
#[derive(Debug)]
pub(crate) struct PromptCache {
    entries: Vec<Entry>,
    /// Byte budget. `0` disables the cache entirely (the default).
    limit_bytes: usize,
    hits: u64,
    misses: u64,
}

fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

impl PromptCache {
    pub(crate) fn new(limit_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            limit_bytes,
            hits: 0,
            misses: 0,
        }
    }

    pub(crate) fn enabled(&self) -> bool {
        self.limit_bytes > 0
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    pub(crate) fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Store a serialised sequence.
    ///
    /// Skips a state whose tokens are a prefix of one already cached: the longer entry
    /// serves every prompt the shorter one would, so keeping both spends bytes for
    /// nothing. Conversely, erases any cached entry that the newcomer supersedes.
    /// Both rules come from llama-server's `alloc()` (`server-task.cpp:1623-1676`).
    pub(crate) fn store(&mut self, tokens: Vec<i32>, data: Vec<u8>) {
        if !self.enabled() || tokens.is_empty() || data.is_empty() {
            return;
        }
        // A single state larger than the whole budget can never be stored without
        // immediately evicting itself.
        if data.len() > self.limit_bytes {
            return;
        }
        if self
            .entries
            .iter()
            .any(|e| common_prefix_len(&e.tokens, &tokens) == tokens.len())
        {
            return; // already covered by a longer (or equal) entry
        }
        self.entries
            .retain(|e| common_prefix_len(&e.tokens, &tokens) != e.tokens.len());

        self.entries.push(Entry {
            tokens,
            data,
            last_used: Instant::now(),
        });
        self.evict_to_budget();
    }

    /// Find the cached state that covers the longest prefix of `prompt`, if any is
    /// worth restoring.
    ///
    /// `min_matched` is the caller's floor — normally what the live slot table can
    /// already serve, so the cache is only consulted when it can do strictly better.
    /// Restoring a state that covers less than the slots already hold would be a
    /// memcpy that loses information.
    ///
    /// The entry is removed on a hit: its state is about to become resident in a
    /// sequence, and keeping a second copy in RAM would double-count the budget.
    pub(crate) fn take_best(
        &mut self,
        prompt: &[i32],
        min_matched: usize,
    ) -> Option<PromptCacheHit> {
        if !self.enabled() || prompt.is_empty() {
            return None;
        }
        // Ties are the common case, not the exception: N clients arriving behind one
        // shared system prompt leave N checkpoints that all cover exactly that prefix and
        // differ only in the question tacked on after it. Any of them "matches" equally.
        //
        // They are not equally useful. Whatever the chosen entry holds *beyond* the match
        // has to be rolled back after the restore, and on a hybrid/recurrent cache that
        // rollback is refused past `--rs-rollback` snapshots — which turns the restore
        // into a full re-prefill. So among equal matches, take the entry carrying the
        // least dead tail.
        //
        // Picking arbitrarily (`max_by_key` keeps the first maximum, i.e. insertion
        // order) is what made the burst non-deterministic: identical runs landed on 2 or
        // 4 refused trims depending on which client's checkpoint each request happened to
        // draw, and the wall time swung with it.
        let best = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (i, common_prefix_len(&e.tokens, prompt), e.tokens.len()))
            .filter(|&(_, matched, _)| matched > min_matched)
            .max_by(|a, b| {
                // Longest match first; then the shortest resident tail; then the lowest
                // index, so a full tie resolves the same way every run rather than by
                // whichever entry was stored first.
                a.1.cmp(&b.1)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(i, matched, _)| (i, matched));

        match best {
            Some((index, matched)) => {
                let entry = self.entries.remove(index);
                self.hits += 1;
                Some(PromptCacheHit {
                    data: entry.data,
                    matched,
                    resident: entry.tokens.len(),
                })
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    /// Drop everything — called when the owning model unloads, since the blobs encode
    /// that model's cell layout and mean nothing to another.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }

    /// Evict least-recently-used entries until the byte budget is met.
    fn evict_to_budget(&mut self) {
        while self.bytes() > self.limit_bytes && !self.entries.is_empty() {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i);
            match oldest {
                Some(i) => {
                    self.entries.remove(i);
                }
                None => break,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(n: usize) -> Vec<u8> {
        vec![7u8; n]
    }

    #[test]
    fn disabled_cache_stores_nothing() {
        let mut c = PromptCache::new(0);
        c.store(vec![1, 2, 3], blob(10));
        assert_eq!(c.len(), 0);
        assert!(c.take_best(&[1, 2, 3], 0).is_none());
    }

    #[test]
    fn take_best_picks_the_longest_covering_prefix() {
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2], blob(10));
        c.store(vec![1, 2, 3, 4], blob(10));
        c.store(vec![9, 9], blob(10));
        // [1,2] was superseded by [1,2,3,4], so only two entries survive the stores.
        assert_eq!(c.len(), 2);

        let hit = c.take_best(&[1, 2, 3, 4, 5], 0).expect("hit");
        assert_eq!(hit.matched, 4);
        assert_eq!(hit.resident, 4);
        // Removed on hit — its state is about to live in a sequence instead.
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn take_best_respects_the_callers_floor() {
        // The live slots can already serve 4 tokens; a 4-token cache entry is not an
        // improvement and restoring it would be a pointless memcpy.
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2, 3, 4], blob(10));
        assert!(c.take_best(&[1, 2, 3, 4, 5], 4).is_none());
        assert_eq!(c.len(), 1, "a rejected candidate stays cached");
        // One more token of coverage does clear the floor.
        assert!(c.take_best(&[1, 2, 3, 4, 5], 3).is_some());
    }

    #[test]
    fn take_best_breaks_ties_on_the_shortest_dead_tail() {
        // The shape a concurrent burst leaves behind: several checkpoints covering the
        // same shared prefix [1,2,3], each with a different question after it. All match
        // the incoming prompt equally, so the tie-break is the whole decision — and what
        // the winner carries past the match is what has to be rolled back afterwards.
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2, 3, 7, 7, 7, 7], blob(10)); // stored first, longest tail
        c.store(vec![1, 2, 3, 8], blob(10)); // shortest tail
        c.store(vec![1, 2, 3, 9, 9], blob(10));

        let hit = c.take_best(&[1, 2, 3, 4], 0).expect("hit");
        assert_eq!(hit.matched, 3, "all three cover the same prefix");
        assert_eq!(
            hit.resident, 4,
            "the entry with the least to roll back wins, not the one stored first"
        );
    }

    #[test]
    fn take_best_is_deterministic_across_equal_candidates() {
        // Two entries that tie on match *and* on tail length: the choice must not depend
        // on insertion order, or identical runs diverge — which is exactly how the burst
        // benchmark ended up with 2 refused trims one run and 4 the next.
        let pick = |order: [i32; 2]| {
            let mut c = PromptCache::new(1024);
            for q in order {
                c.store(vec![1, 2, 3, q], blob(10));
            }
            c.take_best(&[1, 2, 3, 5], 0).expect("hit").resident
        };
        assert_eq!(pick([8, 9]), pick([9, 8]));
    }

    #[test]
    fn store_skips_a_prompt_already_covered_by_a_longer_entry() {
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2, 3, 4], blob(10));
        c.store(vec![1, 2], blob(10));
        assert_eq!(c.len(), 1, "the shorter prefix adds no coverage");
    }

    #[test]
    fn store_supersedes_shorter_entries() {
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2], blob(10));
        c.store(vec![1, 2, 3, 4], blob(10));
        assert_eq!(c.len(), 1, "the longer entry replaces what it covers");
        let hit = c.take_best(&[1, 2, 3, 4], 0).expect("hit");
        assert_eq!(hit.matched, 4);
    }

    #[test]
    fn budget_is_enforced_by_evicting_the_least_recently_used() {
        let mut c = PromptCache::new(25);
        c.store(vec![1], blob(10));
        c.store(vec![2], blob(10));
        assert_eq!(c.len(), 2);
        // Third entry pushes past 25 bytes; the oldest goes.
        c.store(vec![3], blob(10));
        assert!(c.bytes() <= 25, "budget exceeded: {}", c.bytes());
        assert!(c.take_best(&[1], 0).is_none(), "oldest should be gone");
        assert!(c.take_best(&[3], 0).is_some(), "newest should be kept");
    }

    #[test]
    fn a_state_larger_than_the_whole_budget_is_refused() {
        // Storing it would evict everything including itself, so it is rejected up
        // front rather than thrashing the cache empty.
        let mut c = PromptCache::new(10);
        c.store(vec![1], blob(50));
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn clear_drops_everything() {
        let mut c = PromptCache::new(1024);
        c.store(vec![1, 2], blob(10));
        c.clear();
        assert_eq!(c.len(), 0);
        assert_eq!(c.bytes(), 0);
    }
}
