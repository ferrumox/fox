//! Device residency as a type — paged, ref-counted, copy-on-write.
//!
//! # Why the paper's model is not enough here
//!
//! The paper derives transfer direction from ownership: `&T` maps host→device,
//! `&mut T` maps both ways. That is per-launch. `Preload`/`PreloadMut` widen it to
//! two states — resident until dropped — which is right for HPC, where you upload a
//! mesh, run a stencil, and download the result.
//!
//! An inference server needs neither. Its KV cache is tens of gigabytes resident
//! across *millions* of launches, carved into fixed-size blocks, ref-counted,
//! shared between sequences that happen to have a common prefix, and copied only
//! when one of them writes. fox already implements exactly that, at runtime, in
//! `src/kv_cache/` — `PageTable`, `allocate`/`free_blocks`/`retain_block`,
//! `copy_on_write`, `is_shared`.
//!
//! This module is that discipline expressed as types, so the compiler tracks it
//! rather than a runtime data structure. It is the piece the paper does not have
//! and its future-work section does not claim (that lists async transfers,
//! multi-device, ABI validation). vLLM's PagedAttention is the same idea with no
//! static guarantees at all.
//!
//! # Status
//!
//! Allocation policy only — this owns no memory and calls no runtime. It is the
//! bookkeeping half, written and tested first because it is the half that does not
//! need a working `offload!`. Binding a block to a real device allocation waits on
//! the toolchain (see `docs/design/rust-gpu-offload.md`).

use alloc::vec::Vec;

/// A fixed-size unit of device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BlockId(pub u32);

/// Where a buffer's contents currently are.
///
/// `Preload`/`PreloadMut` in the paper are `Host`/`Device` with no third case. The
/// third case is the one that matters for serving: two sequences reading the same
/// prefix must not each pay for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// Not on the device.
    Host,
    /// On the device, referenced by exactly one owner. Writable in place.
    Exclusive,
    /// On the device, referenced by more than one owner. A write must copy first.
    Shared,
}

/// What went wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyError {
    /// Not enough free blocks. The caller's job is to evict, not to retry.
    OutOfBlocks { requested: usize, available: usize },
    /// A block id that this pool never handed out, or already reclaimed.
    UnknownBlock(BlockId),
}

/// A pool of equally-sized device blocks with reference counts.
///
/// Deliberately mirrors `KVCacheManager` in fox: same block-size-and-refcount
/// shape, same copy-on-write rule, so the two can be reconciled later instead of
/// being two different models of the same thing.
#[derive(Debug)]
pub struct BlockPool {
    block_size: usize,
    refcount: Vec<u32>,
    free: Vec<BlockId>,
}

impl BlockPool {
    /// A pool of `total_blocks` blocks of `block_size` elements each.
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            block_size,
            refcount: alloc::vec![0; total_blocks],
            // Hand out low ids first: deterministic, which makes tests meaningful.
            free: (0..total_blocks).rev().map(|i| BlockId(i as u32)).collect(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn total_blocks(&self) -> usize {
        self.refcount.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free.len()
    }

    /// Blocks with at least one reference.
    pub fn blocks_in_use(&self) -> usize {
        self.refcount.iter().filter(|&&r| r > 0).count()
    }

    /// How many blocks `elements` needs.
    pub fn blocks_for(&self, elements: usize) -> usize {
        elements.div_ceil(self.block_size)
    }

    /// Take `n` blocks, each at refcount 1. All-or-nothing: a partial allocation
    /// would leave the caller to unwind a failure it did not ask for.
    pub fn allocate(&mut self, n: usize) -> Result<Vec<BlockId>, ResidencyError> {
        if n > self.free.len() {
            return Err(ResidencyError::OutOfBlocks {
                requested: n,
                available: self.free.len(),
            });
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let id = self.free.pop().expect("checked above");
            self.refcount[id.0 as usize] = 1;
            out.push(id);
        }
        Ok(out)
    }

    /// Add a reference — a second owner now shares this block.
    pub fn retain(&mut self, id: BlockId) -> Result<u32, ResidencyError> {
        let r = self
            .refcount
            .get_mut(id.0 as usize)
            .ok_or(ResidencyError::UnknownBlock(id))?;
        if *r == 0 {
            return Err(ResidencyError::UnknownBlock(id));
        }
        *r += 1;
        Ok(*r)
    }

    /// Drop a reference; the block returns to the pool at zero.
    pub fn release(&mut self, id: BlockId) -> Result<u32, ResidencyError> {
        let r = self
            .refcount
            .get_mut(id.0 as usize)
            .ok_or(ResidencyError::UnknownBlock(id))?;
        if *r == 0 {
            return Err(ResidencyError::UnknownBlock(id));
        }
        *r -= 1;
        let now = *r;
        if now == 0 {
            self.free.push(id);
        }
        Ok(now)
    }

    pub fn residency(&self, id: BlockId) -> Residency {
        match self.refcount.get(id.0 as usize) {
            None | Some(0) => Residency::Host,
            Some(1) => Residency::Exclusive,
            Some(_) => Residency::Shared,
        }
    }

    pub fn is_shared(&self, id: BlockId) -> bool {
        matches!(self.residency(id), Residency::Shared)
    }

    /// Make `id` writable by its caller.
    ///
    /// `Exclusive` → `Ok(None)`: write in place, nothing to do. `Shared` → a fresh
    /// block and one reference dropped from the original. The caller must copy the
    /// contents; this only decides *that* a copy is owed and reserves somewhere to
    /// put it.
    pub fn copy_on_write(&mut self, id: BlockId) -> Result<Option<BlockId>, ResidencyError> {
        match self.residency(id) {
            Residency::Host => Err(ResidencyError::UnknownBlock(id)),
            Residency::Exclusive => Ok(None),
            Residency::Shared => {
                let fresh = self.allocate(1)?[0];
                self.release(id)?;
                Ok(Some(fresh))
            }
        }
    }
}

/// The blocks one owner holds, in order — fox's `PageTable` under another name.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockTable {
    blocks: Vec<BlockId>,
}

impl BlockTable {
    pub fn new(blocks: Vec<BlockId>) -> Self {
        Self { blocks }
    }

    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    pub fn blocks(&self) -> &[BlockId] {
        &self.blocks
    }

    pub fn push(&mut self, id: BlockId) {
        self.blocks.push(id);
    }

    /// Share the first `n` blocks with a second owner, taking a reference on each.
    ///
    /// This is prefix sharing: two sequences with a common prefix reference the
    /// same blocks instead of each holding a copy. The block straddling the
    /// divergence point is *not* shared — sharing it would mean a shared block
    /// eventually receives a write, which is what makes copy-on-write necessary
    /// rather than merely available. Same rule fox's scheduler follows.
    pub fn share_prefix(
        &self,
        n: usize,
        pool: &mut BlockPool,
    ) -> Result<BlockTable, ResidencyError> {
        let n = n.min(self.blocks.len());
        let mut taken = Vec::with_capacity(n);
        for &id in &self.blocks[..n] {
            match pool.retain(id) {
                Ok(_) => taken.push(id),
                Err(e) => {
                    // Never leave half a table referenced.
                    for &t in &taken {
                        let _ = pool.release(t);
                    }
                    return Err(e);
                }
            }
        }
        Ok(BlockTable::new(taken))
    }

    /// Drop every reference this table holds.
    pub fn release_all(&mut self, pool: &mut BlockPool) {
        for id in self.blocks.drain(..) {
            let _ = pool.release(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_is_all_or_nothing() {
        let mut pool = BlockPool::new(4, 16);
        assert_eq!(
            pool.allocate(5),
            Err(ResidencyError::OutOfBlocks {
                requested: 5,
                available: 4
            })
        );
        assert_eq!(
            pool.free_blocks(),
            4,
            "a failed allocation must not consume anything"
        );
        assert_eq!(pool.allocate(4).map(|b| b.len()), Ok(4));
        assert_eq!(pool.free_blocks(), 0);
    }

    #[test]
    fn blocks_return_to_the_pool_at_zero_references() {
        let mut pool = BlockPool::new(2, 8);
        let b = pool.allocate(1).unwrap()[0];
        assert_eq!(pool.residency(b), Residency::Exclusive);
        assert_eq!(pool.retain(b), Ok(2));
        assert_eq!(pool.residency(b), Residency::Shared);
        assert_eq!(pool.release(b), Ok(1));
        assert_eq!(pool.free_blocks(), 1, "still held by one owner");
        assert_eq!(pool.release(b), Ok(0));
        assert_eq!(pool.free_blocks(), 2);
        assert_eq!(pool.residency(b), Residency::Host);
    }

    #[test]
    fn writing_to_an_exclusive_block_copies_nothing() {
        let mut pool = BlockPool::new(4, 8);
        let b = pool.allocate(1).unwrap()[0];
        assert_eq!(pool.copy_on_write(b), Ok(None));
    }

    #[test]
    fn writing_to_a_shared_block_yields_a_fresh_one() {
        let mut pool = BlockPool::new(4, 8);
        let b = pool.allocate(1).unwrap()[0];
        pool.retain(b).unwrap();
        let fresh = pool
            .copy_on_write(b)
            .unwrap()
            .expect("shared block must copy");
        assert_ne!(fresh, b);
        assert_eq!(
            pool.residency(b),
            Residency::Exclusive,
            "the other owner keeps it"
        );
        assert_eq!(pool.residency(fresh), Residency::Exclusive);
    }

    #[test]
    fn a_shared_prefix_is_charged_once() {
        let mut pool = BlockPool::new(8, 16);
        let first = BlockTable::new(pool.allocate(5).unwrap());
        let used = pool.blocks_in_use();
        let second = first.share_prefix(3, &mut pool).unwrap();
        assert_eq!(second.len(), 3);
        assert_eq!(
            pool.blocks_in_use(),
            used,
            "sharing must not consume a new block"
        );
        assert_eq!(pool.free_blocks(), 3);
        for &id in second.blocks() {
            assert!(pool.is_shared(id));
        }
        assert!(
            !pool.is_shared(first.blocks()[3]),
            "past the divergence point stays private"
        );
    }

    #[test]
    fn releasing_a_table_returns_every_block() {
        let mut pool = BlockPool::new(8, 16);
        let mut t = BlockTable::new(pool.allocate(5).unwrap());
        let mut shared = t.share_prefix(2, &mut pool).unwrap();
        t.release_all(&mut pool);
        assert_eq!(
            pool.free_blocks(),
            6,
            "the two shared blocks are still held"
        );
        shared.release_all(&mut pool);
        assert_eq!(pool.free_blocks(), 8);
        assert_eq!(pool.blocks_in_use(), 0);
    }

    #[test]
    fn blocks_for_rounds_up() {
        let pool = BlockPool::new(4, 16);
        assert_eq!(pool.blocks_for(0), 0);
        assert_eq!(pool.blocks_for(1), 1);
        assert_eq!(pool.blocks_for(16), 1);
        assert_eq!(pool.blocks_for(17), 2);
    }

    #[test]
    fn unknown_blocks_are_rejected_rather_than_panicking() {
        let mut pool = BlockPool::new(2, 8);
        assert_eq!(
            pool.retain(BlockId(99)),
            Err(ResidencyError::UnknownBlock(BlockId(99)))
        );
        assert_eq!(
            pool.release(BlockId(0)),
            Err(ResidencyError::UnknownBlock(BlockId(0)))
        );
        assert_eq!(
            pool.copy_on_write(BlockId(0)),
            Err(ResidencyError::UnknownBlock(BlockId(0))),
            "a block nobody holds cannot be made writable"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The same discipline as types.
//
// Everything above is the *runtime* form: a caller who calls `release` twice, or
// writes to a block `is_shared` would have said yes about, gets no complaint from
// the compiler. What follows is the typed form, written to answer one question and
// no other — which of fox's real KV lifecycle bugs would have failed to compile.
// The verdict is in `docs/design/kv-typed-classification.md`; these types exist so
// that verdict cites evidence instead of assertion.
//
// Read the `compile_fail` doctests as the load-bearing part. A test that does not
// compile is the only direct evidence that a rule became unstateable rather than
// merely documented.
// ─────────────────────────────────────────────────────────────────────────────

use core::cell::RefCell;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;

/// A [`BlockPool`] plus one word of per-block contents, reachable by shared
/// reference so that handles can return themselves to it on drop.
///
/// The interior mutability is not incidental — it is the price of the design. A
/// linear handle frees itself in `Drop`, `Drop` gets only `&self`, so the pool must
/// be reachable through a shared reference and therefore must carry a `RefCell`
/// (a `Mutex` in fox, where the scheduler is concurrent). The runtime failure mode
/// the types remove from the call sites reappears, smaller, at the pool.
#[derive(Debug)]
pub struct Pool {
    inner: RefCell<BlockPool>,
    /// One `u32` per block, standing in for its contents. Enough to observe that a
    /// copy-on-write actually copied.
    cells: RefCell<Vec<u32>>,
}

impl Pool {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        Self {
            inner: RefCell::new(BlockPool::new(total_blocks, block_size)),
            cells: RefCell::new(alloc::vec![0; total_blocks]),
        }
    }

    /// Take one block, exclusively owned.
    pub fn alloc(&self) -> Result<Owned<'_>, ResidencyError> {
        let id = self.inner.borrow_mut().allocate(1)?[0];
        Ok(Owned { id, pool: self })
    }

    pub fn free_blocks(&self) -> usize {
        self.inner.borrow().free_blocks()
    }

    pub fn blocks_in_use(&self) -> usize {
        self.inner.borrow().blocks_in_use()
    }

    pub fn residency(&self, id: BlockId) -> Residency {
        self.inner.borrow().residency(id)
    }
}

/// A block this handle owns alone. The only writable form.
///
/// Not `Clone`, not `Copy`, and it releases in `Drop`: the reference count cannot be
/// forgotten and cannot be dropped twice.
///
/// ```
/// use fox_offload::resident::Pool;
/// let pool = Pool::new(4, 16);
/// let mut b = pool.alloc().unwrap();
/// b.write(7);
/// assert_eq!(b.read(), 7);
/// drop(b);
/// assert_eq!(pool.free_blocks(), 4, "Drop returned it");
/// ```
///
/// A handle cannot outlive its pool:
///
/// ```compile_fail
/// use fox_offload::resident::{Owned, Pool};
/// let escaped: Owned<'_> = {
///     let pool = Pool::new(4, 16);
///     pool.alloc().unwrap()
/// };
/// ```
///
/// And it cannot be released twice, because releasing is not a method:
///
/// ```compile_fail
/// use fox_offload::resident::Pool;
/// let pool = Pool::new(4, 16);
/// let b = pool.alloc().unwrap();
/// drop(b);
/// drop(b); // use of moved value
/// ```
#[derive(Debug)]
pub struct Owned<'p> {
    id: BlockId,
    pool: &'p Pool,
}

impl<'p> Owned<'p> {
    pub fn id(&self) -> BlockId {
        self.id
    }

    pub fn read(&self) -> u32 {
        self.pool.cells.borrow()[self.id.0 as usize]
    }

    /// Write. Exists on `Owned` and on nothing else.
    pub fn write(&mut self, stamp: u32) {
        self.pool.cells.borrow_mut()[self.id.0 as usize] = stamp;
    }

    /// Hand this block to two owners. Consumes the exclusive handle, so writing
    /// after sharing is not an error to detect — there is no handle left to write
    /// through.
    ///
    /// ```compile_fail
    /// use fox_offload::resident::Pool;
    /// let pool = Pool::new(4, 16);
    /// let mut b = pool.alloc().unwrap();
    /// let (_x, _y) = b.share();
    /// b.write(1); // use of moved value: `b`
    /// ```
    pub fn share(self) -> (Shared<'p>, Shared<'p>) {
        let me = ManuallyDrop::new(self);
        let (id, pool) = (me.id, me.pool);
        pool.inner
            .borrow_mut()
            .retain(id)
            .expect("an owned block is live by construction");
        (Shared { id, pool }, Shared { id, pool })
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        let _ = self.pool.inner.borrow_mut().release(self.id);
    }
}

/// A block someone else also references. **No `write`.**
///
/// ```compile_fail
/// use fox_offload::resident::Pool;
/// let pool = Pool::new(4, 16);
/// let (mut x, _y) = pool.alloc().unwrap().share();
/// x.write(1); // no method named `write` found for struct `Shared`
/// ```
///
/// The only route back to writable is [`Shared::make_mut`], which copies:
///
/// ```
/// use fox_offload::resident::Pool;
/// let pool = Pool::new(4, 16);
/// let mut b = pool.alloc().unwrap();
/// b.write(7);
/// let (x, y) = b.share();
/// let mut w = x.make_mut().unwrap();
/// w.write(9);
/// assert_ne!(w.id(), y.id(), "the writer got a fresh block");
/// assert_eq!(y.read(), 7, "the other sharer still sees the old contents");
/// ```
#[derive(Debug)]
pub struct Shared<'p> {
    id: BlockId,
    pool: &'p Pool,
}

impl<'p> Shared<'p> {
    pub fn id(&self) -> BlockId {
        self.id
    }

    pub fn read(&self) -> u32 {
        self.pool.cells.borrow()[self.id.0 as usize]
    }

    /// Become writable. Copies first if anyone else still holds the block.
    pub fn make_mut(self) -> Result<Owned<'p>, ResidencyError> {
        let pool = self.pool;
        let cow = pool.inner.borrow_mut().copy_on_write(self.id);
        match cow {
            // Nothing was consumed; let `self` drop as usual.
            Err(e) => Err(e),
            // Sole referent after all: keep the block, keep the reference.
            Ok(None) => {
                let me = ManuallyDrop::new(self);
                Ok(Owned {
                    id: me.id,
                    pool: me.pool,
                })
            }
            // `copy_on_write` already dropped this handle's reference.
            Ok(Some(fresh)) => {
                let me = ManuallyDrop::new(self);
                let stamp = pool.cells.borrow()[me.id.0 as usize];
                pool.cells.borrow_mut()[fresh.0 as usize] = stamp;
                Ok(Owned { id: fresh, pool })
            }
        }
    }
}

impl Drop for Shared<'_> {
    fn drop(&mut self) {
        let _ = self.pool.inner.borrow_mut().release(self.id);
    }
}

// ── Sequence leases ──────────────────────────────────────────────────────────
//
// The blocks above are what the phase-1 design proposed to type. The two fox bugs
// that a type system would actually have stopped are both here instead: fox's
// `BlockId`s are an admission budget that never reaches llama.cpp, while the
// resource that gets clobbered, poisoned and leaked is the llama.cpp *sequence*.

/// A llama.cpp sequence id that can only come from [`SeqPool::claim`].
///
/// The point is the missing constructor: there is no way to write down "sequence 0"
/// and hand it to the batch builder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SeqId(u32);

impl SeqId {
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// A pool of sequence ids that clears a sequence's KV as it is reclaimed.
#[derive(Debug)]
pub struct SeqPool {
    free: RefCell<Vec<SeqId>>,
    /// How many times each sequence has been cleared — the observable side effect
    /// that `Drop` guarantees.
    clears: RefCell<Vec<u32>>,
}

impl SeqPool {
    pub fn new(n: usize) -> Self {
        Self {
            free: RefCell::new((0..n).rev().map(|i| SeqId(i as u32)).collect()),
            clears: RefCell::new(alloc::vec![0; n]),
        }
    }

    /// Take a sequence. `None` when the pool is empty.
    pub fn claim(&self) -> Option<SeqLease<'_>> {
        let id = self.free.borrow_mut().pop()?;
        Some(SeqLease { id, pool: self })
    }

    pub fn available(&self) -> usize {
        self.free.borrow().len()
    }

    pub fn clears(&self, id: SeqId) -> u32 {
        self.clears.borrow()[id.0 as usize]
    }
}

/// A claimed sequence. Clearing happens in `Drop`, so an error path that returns
/// the id without clearing it is not a path anyone can write.
///
/// This is bug `3f9b6e8`'s second half: the prefill/decode error paths returned the
/// seq_id to the pool without `clear_sequence`, and every later occupant of that
/// sequence failed too, forever.
///
/// ```
/// use fox_offload::resident::SeqPool;
/// let pool = SeqPool::new(2);
/// let id = {
///     let lease = pool.claim().unwrap();
///     let id = lease.id();
///     assert_eq!(pool.clears(id), 0);
///     id // an early return on a decode failure looks exactly like this
/// };
/// assert_eq!(pool.clears(id), 1, "reclaimed sequences are cleared, always");
/// assert_eq!(pool.available(), 2);
/// ```
#[derive(Debug)]
pub struct SeqLease<'p> {
    id: SeqId,
    pool: &'p SeqPool,
}

impl SeqLease<'_> {
    pub fn id(&self) -> SeqId {
        self.id
    }
}

impl Drop for SeqLease<'_> {
    fn drop(&mut self) {
        self.pool.clears.borrow_mut()[self.id.0 as usize] += 1;
        self.pool.free.borrow_mut().push(self.id);
    }
}

/// Place a token in a batch. Takes a lease, not a number.
///
/// This is bug `a4171eb`'s second half in one line: `do_get_embeddings` wrote
/// `*arr.add(0) = 0`, a literal that happened to be a live generation's sequence.
/// Under this signature that line is a type error.
///
/// ```compile_fail
/// use fox_offload::resident::place_token;
/// let mut batch = Vec::new();
/// place_token(&mut batch, 0, 0); // expected `&SeqLease<'_>`, found integer
/// ```
///
/// ```
/// use fox_offload::resident::{place_token, SeqPool};
/// let pool = SeqPool::new(4);
/// let embed = pool.claim().unwrap();
/// let mut batch = Vec::new();
/// place_token(&mut batch, &embed, 0);
/// assert_eq!(batch, vec![(embed.id().raw(), 0)]);
/// ```
pub fn place_token(batch: &mut Vec<(u32, u32)>, seq: &SeqLease<'_>, pos: u32) {
    batch.push((seq.id().raw(), pos));
}

// ── Sequence states ──────────────────────────────────────────────────────────
//
// The typestate half of the proposal. It is cheap to write and it does hold, but
// note what it does *not* do: it enforces an invariant, it does not find one. Every
// rule below had to be discovered by a failing server first.

/// Prefill has finished and nothing has been generated: the KV holds exactly the
/// prompt.
#[derive(Debug)]
pub struct Prefilled;
/// Tokens have been appended past the prompt boundary.
#[derive(Debug)]
pub struct Generating;
/// The context window has slid: the cells at `[0, n)` are no longer the prompt.
#[derive(Debug)]
pub struct Rolled;

/// What a sequence holds, as a type parameter.
#[derive(Debug)]
pub struct Track<S> {
    resident: usize,
    _state: PhantomData<S>,
}

impl Track<Prefilled> {
    pub fn after_prefill(resident: usize) -> Self {
        Self {
            resident,
            _state: PhantomData,
        }
    }

    /// Serialize the prompt boundary. Bug `f9d051a`: fox checkpointed on eviction
    /// instead, by which time the blob held prompt *plus* response and reproduced
    /// the very rollback it was meant to avoid.
    pub fn checkpoint(&self) -> usize {
        self.resident
    }

    /// Donate the prompt prefix to the cache. Bug `a4171eb`: a request that had
    /// rolled its context donated anyway, and the cells at `[0, cached)` were
    /// mid-generation tokens, not the prefix the key promised.
    pub fn donate(&self) -> usize {
        self.resident
    }

    pub fn generate(self, n: usize) -> Track<Generating> {
        Track {
            resident: self.resident + n,
            _state: PhantomData,
        }
    }
}

impl Track<Generating> {
    /// ```compile_fail
    /// use fox_offload::resident::Track;
    /// let t = Track::after_prefill(100).generate(5);
    /// t.checkpoint(); // no method named `checkpoint` found for `Track<Generating>`
    /// ```
    pub fn roll(self, discarded: usize) -> Track<Rolled> {
        Track {
            resident: self.resident.saturating_sub(discarded),
            _state: PhantomData,
        }
    }

    pub fn resident(&self) -> usize {
        self.resident
    }
}

impl Track<Rolled> {
    /// ```compile_fail
    /// use fox_offload::resident::Track;
    /// let t = Track::after_prefill(100).generate(5).roll(20);
    /// t.donate(); // no method named `donate` found for `Track<Rolled>`
    /// ```
    pub fn resident(&self) -> usize {
        self.resident
    }
}

#[cfg(test)]
mod typed_tests {
    use super::*;

    #[test]
    fn sharing_charges_one_block_and_unwinds_completely() {
        let pool = Pool::new(4, 16);
        let b = pool.alloc().unwrap();
        assert_eq!(pool.free_blocks(), 3);
        let (x, y) = b.share();
        assert_eq!(pool.free_blocks(), 3, "sharing consumed nothing");
        assert_eq!(pool.residency(x.id()), Residency::Shared);
        drop(x);
        assert_eq!(pool.free_blocks(), 3, "one referent left");
        drop(y);
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn make_mut_on_a_sole_referent_copies_nothing() {
        let pool = Pool::new(4, 16);
        let (x, y) = pool.alloc().unwrap().share();
        let id = x.id();
        drop(y);
        let w = x.make_mut().unwrap();
        assert_eq!(w.id(), id, "nobody else held it — no copy owed");
        assert_eq!(pool.blocks_in_use(), 1);
    }

    #[test]
    fn a_lease_clears_before_it_returns_even_on_an_error_path() {
        let pool = SeqPool::new(2);
        // The shape of the bug: an error path that returns early.
        fn failing_decode(pool: &SeqPool) -> Result<(), ()> {
            let _lease = pool.claim().ok_or(())?;
            Err(())
        }
        assert!(failing_decode(&pool).is_err());
        assert_eq!(pool.available(), 2, "the id came back");
        // Ids are handed out lowest-first, so the failing call held sequence 0.
        assert_eq!(pool.clears(SeqId(0)), 1, "and it came back cleared");
    }

    #[test]
    fn embeddings_cannot_borrow_a_live_generations_sequence() {
        let pool = SeqPool::new(2);
        let live = pool.claim().unwrap();
        let embed = pool.claim().unwrap();
        assert_ne!(live.id(), embed.id());
        let mut batch = Vec::new();
        place_token(&mut batch, &embed, 0);
        assert_eq!(batch, alloc::vec![(embed.id().raw(), 0)]);
        // The pool is exhausted; there is no third id to accidentally reuse.
        assert!(pool.claim().is_none());
    }

    #[test]
    fn only_a_prompt_boundary_can_be_checkpointed_or_donated() {
        let t = Track::after_prefill(100);
        assert_eq!(t.checkpoint(), 100);
        assert_eq!(t.donate(), 100);
        let g = t.generate(5);
        assert_eq!(g.resident(), 105);
        let r = g.roll(20);
        assert_eq!(r.resident(), 85);
        // `checkpoint`/`donate` are gone from both later states — see the
        // `compile_fail` doctests on `Track<Generating>` and `Track<Rolled>`.
    }
}
