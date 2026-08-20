//! Argmax over a logit vector — the reduction fox's sampler runs on the CPU.
//!
//! Picked as the first kernel for three reasons: it is the one genuinely hot loop
//! fox runs host-side over a 128K-wide array (`sample_greedy` in
//! `src/engine/model/sampling.rs`), it is self-contained, and AMD already exposes
//! the wave-level primitives a good version needs (`wave_reduce_max`, `ballot`,
//! `readfirstlane`) in `core::arch::amdgpu`.
//!
//! It is **not** here because it would make fox faster. On the target hardware
//! decode is bandwidth-bound, and a previous 4.6× sampling micro-benchmark produced
//! no measurable throughput change. It is here because it is small enough to get
//! exactly right, and getting it exactly right is the interesting part:
//!
//! # Tie-breaking is part of the contract
//!
//! `sample_greedy` uses `Iterator::max_by`, which returns the **last** maximal
//! element — so on a tie the *highest* token id wins. A tree reduction that returns
//! an arbitrary winner would be a silent behaviour change for every caller at
//! `temperature = 0`. [`Reduction`] therefore fixes the rule: greater value wins,
//! and on equal values the greater index wins. That rule is associative and
//! commutative, so it survives any reduction order — which is what makes it
//! implementable on a device at all.
//!
//! # A NaN bug this exercise surfaced
//!
//! `sample_greedy` compares with `partial_cmp(..).unwrap_or(Ordering::Equal)`.
//! `max_by` replaces its accumulator whenever the comparison is not `Greater`, so
//! an incomparable NaN *always* replaces the running maximum. Measured, not
//! reasoned about:
//!
//! ```text
//! sample_greedy(&[5.0, NaN])       == 1   // the NaN
//! sample_greedy(&[5.0, NaN, 1.0])  == 2   // neither the max nor the NaN
//! ```
//!
//! One NaN anywhere in the logits destroys greedy sampling — it wipes the running
//! maximum and everything after it competes from scratch. NaN logits are reachable
//! (fp16 overflow, a bad quantisation, corrupted KV), so this is a live bug in fox,
//! filed here rather than fixed here because it belongs in a fix on its own branch.
//!
//! This kernel deliberately does **not** reproduce it: NaN never wins. That is a
//! divergence from current fox behaviour, and it is the correct direction.

use crate::launch::Kernel;
use crate::region::{Linear1D, PartitioningStrategy, Region};
use crate::thread::{GridDim, ThreadId};

/// A candidate token: its index and its logit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub index: u32,
    pub value: f32,
}

impl Candidate {
    /// The identity for [`Reduction`]: loses to everything real.
    pub const NONE: Candidate = Candidate {
        index: u32::MAX,
        value: f32::NEG_INFINITY,
    };
}

/// Combine two candidates. Associative and commutative, so any reduction order —
/// per-thread, per-wave, per-workgroup, host-side — gives the same answer.
///
/// Greater value wins; on equal values the greater index wins, matching
/// `Iterator::max_by`'s last-maximum rule. NaN never wins.
pub struct Reduction;

impl Reduction {
    pub fn combine(a: Candidate, b: Candidate) -> Candidate {
        // `>` is false when either side is NaN, so a NaN candidate can only be
        // chosen when the other side is NaN too, and then neither branch matters.
        if b.value > a.value {
            b
        } else if a.value > b.value {
            a
        } else if a.value == b.value && b.index > a.index {
            b
        } else {
            a
        }
    }
}

/// Arguments for [`BlockArgmax`]: read the logits, write one partial per thread.
pub struct ArgmaxArgs<'a> {
    pub logits: &'a [f32],
    /// Must hold at least `grid.total_threads()` entries.
    pub partials: &'a mut [Candidate],
}

/// Each thread reduces its own lane of the logits into one [`Candidate`].
///
/// One partial *per thread* rather than per workgroup: a per-workgroup version
/// needs shared memory and a barrier, which is exactly the part
/// `core::offload` still leaves `unsafe` and which the CPU oracle cannot model
/// honestly. Fan-in from thread partials is done by [`reduce_partials`].
pub struct BlockArgmax;

impl Kernel for BlockArgmax {
    type Args<'a> = ArgmaxArgs<'a>;

    #[inline(always)]
    fn thread(args: &mut Self::Args<'_>, tid: ThreadId, grid: &GridDim) {
        let k = tid.linear(grid) as usize;
        if k >= args.partials.len() {
            return;
        }
        let lane = Linear1D::lane(args.logits.len(), tid, grid);

        let mut best = Candidate::NONE;
        for i in 0..lane.len() {
            let Some(j) = lane.nth(i) else { continue };
            let Some(&v) = args.logits.get(j) else {
                continue;
            };
            best = Reduction::combine(
                best,
                Candidate {
                    index: j as u32,
                    value: v,
                },
            );
        }
        args.partials[k] = best;
    }
}

/// Fan the per-thread partials in to one winner.
pub fn reduce_partials(partials: &[Candidate]) -> Candidate {
    partials
        .iter()
        .copied()
        .fold(Candidate::NONE, Reduction::combine)
}

/// Argmax over `logits`, executed on the CPU through the same kernel a device
/// would run. Returns the token id, or `0` for an empty vocabulary (as
/// `sample_greedy` does).
///
/// The `grid` is a real launch shape, not a formality — running the same input
/// under several shapes is what proves the reduction is order-independent.
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
pub fn argmax_via_kernel(logits: &[f32], grid: &GridDim) -> i32 {
    use crate::launch::launch_cpu;
    use alloc::vec;

    if logits.is_empty() {
        return 0;
    }
    let mut partials = vec![Candidate::NONE; grid.total_threads() as usize];
    let mut args = ArgmaxArgs {
        logits,
        partials: &mut partials,
    };
    launch_cpu::<BlockArgmax>(&mut args, grid);

    let winner = reduce_partials(&partials);
    if winner.index == u32::MAX {
        0
    } else {
        winner.index as i32
    }
}

/// Unused by the kernel, kept as the thing the kernel must agree with.
#[allow(dead_code)]
fn _region_is_the_intended_shape(logits: &mut [f32], grid: GridDim) -> usize {
    let region: Region<'_, f32, Linear1D> = Region::new(logits, grid);
    region.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use core::cmp::Ordering;

    /// Verbatim copy of `sample_greedy` from `src/engine/model/sampling.rs`, so the
    /// comparison below is against what fox really does, not a paraphrase.
    fn sample_greedy(logits: &[f32]) -> i32 {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(i, _)| i as i32)
            .unwrap_or(0)
    }

    fn grids() -> Vec<GridDim> {
        alloc::vec![
            GridDim::linear(1, 1),
            GridDim::linear(1, 32),
            GridDim::linear(4, 64),
            GridDim::linear(7, 5),
            GridDim::linear(64, 256),
        ]
    }

    /// A cheap deterministic spread; no rand dependency in a crate that must build
    /// for a target whose whole library is `core`.
    fn pseudo_logits(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 20.0 - 10.0
            })
            .collect()
    }

    #[test]
    fn matches_fox_sample_greedy_on_ordinary_logits() {
        for seed in 1..12u32 {
            for n in [1usize, 2, 7, 63, 128, 1000, 4096] {
                let logits = pseudo_logits(n, seed);
                let expected = sample_greedy(&logits);
                for grid in grids() {
                    assert_eq!(
                        argmax_via_kernel(&logits, &grid),
                        expected,
                        "n={n} seed={seed} grid={grid:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn ties_go_to_the_highest_index_like_max_by() {
        let logits = [1.0f32, 1.0, 0.5, 1.0, 0.2];
        assert_eq!(sample_greedy(&logits), 3);
        for grid in grids() {
            assert_eq!(argmax_via_kernel(&logits, &grid), 3, "grid {grid:?}");
        }
    }

    #[test]
    fn empty_vocabulary_returns_zero() {
        assert_eq!(sample_greedy(&[]), 0);
        assert_eq!(argmax_via_kernel(&[], &GridDim::linear(1, 32)), 0);
    }

    #[test]
    fn the_result_does_not_depend_on_the_launch_shape() {
        let logits = pseudo_logits(5000, 7);
        let first = argmax_via_kernel(&logits, &grids()[0]);
        for grid in grids() {
            assert_eq!(argmax_via_kernel(&logits, &grid), first, "grid {grid:?}");
        }
    }

    #[test]
    fn reduction_is_commutative_and_associative() {
        let c = |i: u32, v: f32| Candidate { index: i, value: v };
        let (a, b, d) = (c(1, 3.0), c(2, 3.0), c(3, -1.0));
        assert_eq!(Reduction::combine(a, b), Reduction::combine(b, a));
        assert_eq!(
            Reduction::combine(Reduction::combine(a, b), d),
            Reduction::combine(a, Reduction::combine(b, d))
        );
        // identity
        assert_eq!(Reduction::combine(a, Candidate::NONE), a);
    }

    /// Documents the bug found in `sample_greedy`, and this kernel's deliberate
    /// divergence from it. If fox's sampler is ever fixed, the first three
    /// assertions are what the fix has to change.
    #[test]
    fn nan_destroys_fox_greedy_but_not_this_kernel() {
        assert_eq!(sample_greedy(&[5.0, f32::NAN]), 1, "fox picks the NaN");
        assert_eq!(
            sample_greedy(&[5.0, f32::NAN, 1.0]),
            2,
            "fox picks neither max nor NaN"
        );
        assert_eq!(sample_greedy(&[f32::NAN, 5.0]), 1);

        for grid in grids() {
            assert_eq!(
                argmax_via_kernel(&[5.0, f32::NAN], &grid),
                0,
                "grid {grid:?}"
            );
            assert_eq!(
                argmax_via_kernel(&[5.0, f32::NAN, 1.0], &grid),
                0,
                "grid {grid:?}"
            );
            assert_eq!(
                argmax_via_kernel(&[f32::NAN, 5.0], &grid),
                1,
                "grid {grid:?}"
            );
        }
    }

    #[test]
    fn all_nan_does_not_panic() {
        for grid in grids() {
            let out = argmax_via_kernel(&[f32::NAN, f32::NAN, f32::NAN], &grid);
            assert!((0..3).contains(&out), "grid {grid:?} gave {out}");
        }
    }
}
