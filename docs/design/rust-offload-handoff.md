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

### C'. Fix the bug rather than only reporting it

The escalation of A, and the largest contribution available here by a wide margin: a docs
fix corrects an example, this makes the feature execute. Prompt in
`rustc-offload-fix-prompt.md`.

Most of the cost is already paid — `~/src/rust-offload/rust` holds LLVM, clang and lld
built from source at the nightly's own commit, 9.1 GB and 27 minutes that do not need
repeating. What is left is the bootstrap step that builds the amdgcn device runtime with
gcc (PR #161118's known problem, fixable by pointing `build.cc`/`cxx` at the freshly built
clang), then the patch itself, then re-running the probe.

Two candidate fixes, both visible in `gpu_offload.rs`: emit the populated descriptor —
whose correct form sits in a comment beside the null one — or stop registering from rustc
entirely and leave it to the linker wrapper, which the file's own `FIXME` suggests is where
they are heading. Try the first: it is additive.

**It does not depend on A getting a reply**, which an earlier version of this document
claimed. The real check is two minutes of searching for an open PR touching
`register_offload`; on 2026-08-20 none of the five in-flight offload PRs did. File the
report anyway because it costs an hour and blocks nothing — and if the patch then works,
it belongs in that thread rather than behind it. A report carrying a working patch stops
being a report.

### D. Offer the strategies to PR #158076

`Linear1D`, `Strided1D` and `verify_disjoint` fill exactly what that PR lacks. The paper
even invites it: "By releasing our interface as a standalone crate, we hope to encourage
users to explore additional partitioning schemes."

Not a drop-in. Someone else's PR is mid-review; this needs coordinating, not arriving
with code. Worth doing only after A gets a reply.

### E. The thesis: ownership-typed paged device residency

This is the one idea in the whole thread with a life beyond fox, and it is parked in
three phases rather than dropped. Naming matters here, because in conversation it kept
getting called "the exercise" and disappearing.

**The idea.** The paper's contribution is deriving an invariant from a type you already
have: `&T` maps to the device, `&mut T` maps both ways, no pragma. It applies that to a
two-state model — on host, or on device until dropped. An inference server needs the
model underneath instead: fixed-size blocks, refcounted, shared between sequences with a
common prefix, copied only on write. `Owned<'p>` is writable; `Shared<'p>` has **no write
method**, so writing to a shared block is not a mistake you can make, it is a program you
cannot express. `Shared::make_mut` is the only path back to writable. Neither is `Copy`
or `Clone` and both `Drop` into the pool, so you cannot forget to release or release
twice.

It buys **no speed**. What it changes is the cost of touching that subsystem, which today
only the person who wrote the scheduler can do safely, because the rules live in
comments. In an engine, KV lifecycle bugs are silent — wrong tokens, truncated replies,
another user's conversation corrupted — so a class that keeps recurring is worth making
inexpressible. And it is not free: rewriting a core API can introduce bugs of its own.

**Phase 1 — the exercise. Two days. This is the gate.** Take fox's ~6 KV-lifecycle bugs
from the history — prefix-cache leak, `seq_cp` crash, admission preempting, stale cells
past the divergence point, `trim_sequence` reporting success without rewinding, duplicated
shared-prefix accounting — and classify each one: would the types have rejected it at
compile time? Decision rule, agreed before starting so it cannot be rationalised
afterwards: **four or five means go; one means drop it.** Prior estimate, to validate
rather than believe: the ordering constraint is caught outright, the recurrent rollback is
caught as a *capability* rather than as that bug (`trim` only on `Rewindable` sequences),
the accounting only partly.

**Phase 2 — the refactor.** Only if phase 1 says go.

**Phase 3 — the paper.** Optional, and only worth it for the credential — it will not win
fox users. But it costs almost nothing extra, because **phase 1 produces the evaluation
section as a side effect**: a type discipline plus real bugs it would have rejected is
exactly what a PL paper's evaluation is. There is no choosing between doing the
engineering and writing it up; it is the same work.

**The paper cannot start before phase 1.** Writing an introduction for evidence you have
not checked is how you end up committed to a claim that does not survive.

Design, decision rule and framing in the `kv-tipado-decision-pendiente` memory. The
runtime version already exists in `crates/fox-offload/src/resident.rs`, 27 tests, no GPU
and no working offload required — it is a type system.

### F. Ship the two fixes

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
