# Rust GPU offload — where this stands and what is left to decide

Written 2026-08-20 at the end of a long session, so the next one can pick this up
without re-deriving it. Read this first; the other three documents are the detail.

Everything below was measured on this machine or read in the primary source. Where
something is inferred it says so.

## How we got here, in four lines

A post claimed an inference engine on Rust's GPU offload reaching ~100 tok/s aggregated
on an H100. The question was whether that paper (arXiv:2608.13759) matters to fox. It
does not, directly — the paper works one layer below where fox lives. Answering it
properly turned into: verifying the toolchain, reading the paper closely, building a
crate, and finding two real bugs in fox along the way.

## What exists

All on branch `research/rust-gpu-offload`, **not pushed**, 12 commits, ~2,800 lines.

| Artifact | What it is |
|---|---|
| `docs/design/rust-gpu-offload.md` | Feasibility, measured. How far rustc's offload gets here and exactly where it stops |
| `docs/design/offload-paper-critique.md` | Close reading of the paper. Ten points, text and figures |
| `docs/design/offload-issue-draft.md` | **Finished** bug report for rust-lang/rust. Not filed |
| `scripts/probe_rust_offload.sh` | Reproduces the whole pipeline and says where it breaks. Re-run on nightly bumps |
| `crates/fox-offload/` | Workspace crate, 27 tests, zero dependencies, builds on stable |

Separately, on `feature/prompt-cache-determinism` (also **not pushed**): the NaN sampling
fix and the sampler's return to CI, 364 → 403 tests.

## The three findings that matter

**1. rustc's GPU offload compiles for both vendors and does not execute.** AMD `gfx1100`
and NVIDIA `sm_120` both build from the same source, and the release IR is clean. The
kernel is never bound to its device entry: rustc registers an all-null `__tgt_bin_desc`
and its own source calls that a placeholder. Reproduced with a fully matched toolchain
built from source at the same commit, so it is not a mixed-LLVM artifact. Not fileable
as "you broke it" — the feature's own docs say "not ready for usage" — but the specific
failure is not reported anywhere.

**2. The paper's opening survives a close reading, and narrows.** `Preload`/`PreloadMut`
is a two-state model; the refcount is on the immutable handle only and counts whole
mappings. No partial residency, no blocks, no shared prefix, no copy-on-write. Correct
for the HPC case they target, insufficient for serving. Upstream PR #158076 has the
`Region` trait in review with no concrete strategies and disjointness as an unverified
obligation. So the gap is real and smaller than it looked: concrete strategies plus a
way to check the obligation.

**3. The exercise found two live bugs in fox, which is the actual return of the day.**
A single NaN logit made `sample_greedy` emit an arbitrary token at `temperature = 0`,
and the sampler module was gated behind `#[cfg(not(fox_stub))]` so CI — which runs with
`FOX_SKIP_LLAMA=1` — had never compiled or tested it. Both fixed. Neither is released.

## Two things that were never projects, despite how they were discussed

Recorded because a future session will otherwise resurrect them.

- **"We were building an offload."** No. `fox-offload` is a library of type-level
  abstractions plus a CPU oracle. Nothing here is an offload engine.
- **"We were writing a paper."** No. Not one line was written. It was raised as an
  option repeatedly, which made it feel like an open workstream. It never started.

## Decisions left, with honest verdicts

### A. File the bug report

Ready, 244 lines, checked against existing issues (the same error appears inside #150391
but was filed as a diagnostics problem, so this is not a duplicate). Costs an hour of
reading and pasting. **This is not a project — it is pressing a button on finished work.**

What it actually buys is the reply: whether the host-side link automation is weeks or
quarters away. That single fact gates everything else. If nothing else is planned, it is
courtesy rather than strategy, and courtesy is a fine reason.

### B. A documentation PR to rust-lang/rust

`library/core/src/offload.md`'s example does not compile: it calls `thread_idx_x()`,
`block_idx_x()` and `block_dim_x()`, and none exist on either target (AMD has
`workitem_id_x`/`workgroup_id_x`, safe; NVIDIA has `_thread_idx_x`/`_block_idx_x`/
`_block_dim_x`, unsafe; AMD has no workgroup-size query at all). It rotted because the
block is marked `rust,ignore`, so doctests never compile it.

Small, verified by compiler error, uncontroversial, low risk. It is a real commit in
`rust-lang/rust` with your name in the history — and it is a documentation fix, which is
what it will look like to anyone who checks. Do not oversell it.

**Order matters: file the issue first, the PR second.** The issue carries a day of
analysis and establishes standing; the PR arriving behind it reads as "and here is the
thing that broke me, fixed". Alone, it is a typo fix from a stranger.

### C. Offer the strategies to PR #158076

`Linear1D`, `Strided1D` and `verify_disjoint` fill exactly what that PR lacks. The paper
even invites it: "By releasing our interface as a standalone crate, we hope to encourage
users to explore additional partitioning schemes."

Not a drop-in. Someone else's PR is mid-review; this needs coordinating, not arriving
with code. Worth doing only after A gets a reply.

### D. The typed KV residency work

Parked deliberately. See the `kv-tipado-decision-pendiente` memory for the design, the
decision rule agreed in advance, and the framing. Short version: classify fox's ~6
KV-lifecycle bugs against an `Owned`/`Shared` API and see how many the types would have
rejected. Four or five means the refactor justifies itself; one means drop it. Two days
either way.

It buys **no speed**. It changes the cost of touching that subsystem, which today only
the person who wrote the scheduler can do safely. And it is not free: rewriting a core
API can introduce bugs of its own.

### E. Ship the two fixes

Not part of this thread, but it is the only item with a user waiting at the other end.
Every released version of fox, v0.21.0 included, still has the NaN bug, and the
recurrent-model prompt-reuse corruption fix is also unreleased.

## What not to do

- **Do not rewrite fox's kernel layer.** No `std` on device, no multi-GPU, no quantised
  GEMM or tensor cores anywhere in the paper, shared memory explicitly left `unsafe`.
  And decode on the target hardware is bandwidth-bound, so a perfect kernel layer buys
  zero.
- **Do not write a performance paper.** Nothing executes; any number would be invented.
- **Do not treat the probe as maintenance.** It is one script. Run it when nightly moves;
  when the image survives into the binary, the runtime half is unblocked.
