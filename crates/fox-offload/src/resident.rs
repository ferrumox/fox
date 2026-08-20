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
