# Rust GPU offload — where this stands and what is left to decide

Written 2026-08-20 at the end of a long session, so the next one can pick this up
without re-deriving it. Read this first; the other documents are the detail.

**Revised 2026-08-21.** Two things happened after the first version was written, and
both invert conclusions it stated confidently. The kernel *does* execute — there is no
rustc bug, and finding 1 below is a correction rather than a finding. And the branch is
now merged, rebased and pushed rather than sitting on one disk. Every section that
changed says so inline; nothing was quietly deleted.

Everything below was measured on this machine or read in the primary source. Where
something is inferred it says so.

## How we got here, in four lines

A post claimed an inference engine on Rust's GPU offload reaching ~100 tok/s aggregated
on an H100. The question was whether that paper (arXiv:2608.13759) matters to fox. It
does not, directly — the paper works one layer below where fox lives. Answering it
properly turned into: verifying the toolchain, reading the paper closely, building a
crate, and finding two real bugs in fox along the way.

## What exists

All on branch `research/rust-gpu-offload`, rebased onto `develop` and **pushed**, 23
commits carrying only offload work — the sixteen product commits it had accumulated now
live in `develop` where they belong.

| Artifact | What it is |
|---|---|
| `docs/design/rust-gpu-offload.md` | Feasibility, measured. How far rustc's offload gets here and exactly where it stops |
| `docs/design/offload-paper-critique.md` | Close reading of the paper. Ten points, text and figures |
| `docs/design/offload-issue-draft.md` | Bug report for rust-lang/rust. **Retired** — the cause it names is refuted. Never filed |
| `docs/design/rustc-offload-fix-results.md` | The day the diagnosis collapsed. Three experiments, three wrong readings, and what it actually was |
| `docs/design/offload-oracle-differential.md` | The crate's kernel run on the real GPU against its CPU oracle. 256 partials, zero divergence |
| `docs/design/offload-pr-draft.md` | Documentation PR for `library/core/src/offload.md`. Ready, still valid |
| `scripts/probe_rust_offload.sh` | Reproduces the whole pipeline and says where it breaks. Re-run on nightly bumps |
| `crates/fox-offload/` | Workspace crate, 32 tests + 11 compile-fail doctests, zero dependencies, builds on stable **and for `amdgcn`** |

The NaN sampling fix and the sampler's return to CI are no longer separate: that branch
is merged into `develop` along with eighteen other commits. See F.

## The three findings that matter

**1. ~~rustc's GPU offload does not execute.~~ It does. The fault was one layer below
Rust.** This is the correction, and it is worth reading as one rather than as a fact,
because the wrong version survived a full day of competent-looking evidence.

The kernel runs on the stock rustup nightly, unpatched: `x[0]=0 x[42]=84 x[255]=510`,
repeatedly, and under `OMP_TARGET_OFFLOAD=MANDATORY` so it cannot be the host quietly
doing the work. What was broken is this machine's `libhsa-runtime64` — Ubuntu ships
1.11.0 from April 2024, it does not recognise gfx1150, and it enumerates zero GPUs. With
no devices there is no `TranslationTable`, and the launch fails with `does not have a
matching target pointer`, which reads exactly like a codegen bug. Replacing HSA with
1.18 from AMD's `.deb`s, no root required, fixes it; the same binary against the old
runtime fails again, which is the control that closes it.

The null `__tgt_bin_desc` is real and is still a placeholder in rustc's source. It is
simply not the cause: experiment C removed it entirely, corrected a constructor ordering
problem found on the way, and the failure did not change. Three candidate explanations
were eliminated by experiment rather than by reading, which is the one part of the day
that was done right.

The lesson is cheap to state and was expensive to learn: **the control — does *any*
offload work on this machine — was never run until the end.** A C++ OpenMP program would
have answered it in five minutes on the first morning.

**2. The paper's opening survives a close reading, and narrows.** `Preload`/`PreloadMut`
is a two-state model; the refcount is on the immutable handle only and counts whole
mappings. No partial residency, no blocks, no shared prefix, no copy-on-write. Correct
for the HPC case they target, insufficient for serving. Upstream PR #158076 has the
`Region` trait in review with no concrete strategies and disjointness as an unverified
obligation. So the gap is real and smaller than it looked: concrete strategies plus a
way to check the obligation.

**3'. The crate's kernel was checked against its own CPU oracle on real hardware.**
Unblocked by finding 1 and done the same evening. `BlockArgmax::thread`, one source
compiled for `x86_64` and for `amdgcn`/`gfx1100`, on input chosen to hurt: negatives, a
tie at 12.5, a NaN. All 256 partials agree one to one — not merely the winner — the
tie-break rule and the never-let-NaN-win rule both hold on device, and
`Candidate { index: u32, value: f32 }` crosses the device mapping intact. That last one
is a small real data point on the ABI gap the paper acknowledges without implementing.
This is the differential test the critique says nobody runs.

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

### A. ~~File the bug report~~ — **dead, and this is the good outcome**

There is no bug to report. The three drafts each carry a banner saying so and are kept as
a record, not as work in progress. An hour's delay in pressing the button is the reason
fox did not publish a false cause to a compiler team, which is a better result than the
report would have been.

What is worth keeping from it: the process notes cost real time to work out and transfer
to anything filed later — rust-lang forbids LLM-authored issue bodies, the LLM disclosure
goes inside `<!-- homu-ignore -->` so it stays out of git history, and duplicate checking
is expected before filing.

### B. A documentation PR to rust-lang/rust

`library/core/src/offload.md`'s example does not compile: it calls `thread_idx_x()`,
`block_idx_x()` and `block_dim_x()`, and none exist on either target (AMD has
`workitem_id_x`/`workgroup_id_x`, safe; NVIDIA has `_thread_idx_x`/`_block_idx_x`/
`_block_dim_x`, unsafe; AMD has no workgroup-size query at all). It rotted because the
block is marked `rust,ignore`, so doctests never compile it.

Small, verified by compiler error, uncontroversial, low risk. It is a real commit in
`rust-lang/rust` with your name in the history — and it is a documentation fix, which is
what it will look like to anyone who checks. Do not oversell it.

**The ordering advice that used to be here is void** — it said to file the issue first
so the PR would not arrive as a typo fix from a stranger, and there is no issue now. So
it does arrive alone, and that is fine: it is a correct fix to an example that does not
compile, offered on its own terms. This is the only rust-lang contribution left standing,
and the branch is ready at `ManuelSLemos/rust:offload-doc-example`. One thing is missing:
the LLM disclosure at the end, in your own words.

### C. ~~Fix the bug rather than only reporting it~~ — **dead, nothing to fix**

This was the largest contribution available and it evaporated with finding 1: the feature
already executes. `rustc-offload-fix-prompt.md` was run to completion and its answer is
`rustc-offload-fix-results.md` — a negative result reached properly. What follows is left
in place only because the toolchain notes at the end of that document are reusable, and
because it is the clearest illustration in this thread of a plausible plan aimed at the
wrong target.

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
with code. It used to be gated on A getting a reply — with A dead, nothing gates it, and
it is now the most substantial open item in the thread. It is also stronger than it was
yesterday: the strategies have been run on a real GPU (finding 3'), not only reasoned
about.

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

**Phase 1 — DONE, 2026-08-20, and the gate says no.** Twelve bugs classified in
`docs/design/kv-typed-classification.md`: two would have failed to compile, six are
catchable only as a class and only after a failing server had already revealed the
invariant, four are not type-shaped at all. Against the rule below that is 17% at its
strictest and 42% at its most generous, where the threshold was 60–70%.

The finding underneath the count is the one that matters, and it is why the answer is
"no" and not "maybe": **the `Owned`/`Shared` copy-on-write design proposed for phase 2
catches none of them.** It protects writes through block handles, and nothing in fox
writes through a block handle. Both bugs a type system would have stopped are in the
*sequence* lifecycle. The blocks are budget; the sequence is what corrupts.

So phase 2 is rejected, and the redirection was taken instead of proposed: `SeqId` ships
on `feature/seq-id-phase1`, a newtype with two named constructors that makes a bare
integer a type error where a sequence is expected. That is `a4171eb` made unwriteable.
Phase 3 loses its evaluation section and with it its reason to exist.

The original text of the gate follows, for the record.

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

### F. Ship the fixes — **half done**

Not part of this thread, and still the only item with a user waiting at the other end.
The nineteen commits are merged into `develop` and pushed, so they are no longer one
disk away from being lost: eleven fixes, two features (`--n-gpu-layers` and the
experimental `--mtp-model`), 446 tests green. What has *not* happened is a release, so
every version a user can install — v0.21.0 included — still has the NaN bug and the
recurrent-model prompt-reuse corruption. Two new CLI flags make it 0.22.0 rather than a
patch release, and it needs `make e2e` against a real server and model before the tag.

## What not to do

- **Do not rewrite fox's kernel layer.** No `std` on device, no multi-GPU, no quantised
  GEMM or tensor cores anywhere in the paper, shared memory explicitly left `unsafe`.
  And decode on the target hardware is bandwidth-bound, so a perfect kernel layer buys
  zero.
- **Do not write a performance paper.** This used to read "nothing executes; any number
  would be invented". Kernels execute now, so the reason changes rather than the advice:
  a numerically-correct argmax on a 890M says nothing about serving throughput, and the
  hardware here is bandwidth-bound at decode. A number you can measure is not the same
  as a number worth publishing.
- **Do not treat the probe as maintenance.** It is one script. Its original purpose —
  watch for the image surviving into the binary — is spent; what it is good for now is
  catching a regression when nightly moves.
- **Do not debug a toolchain before running the control.** Written down as a rule
  because a day was spent on it: before believing that a compiler is broken, check that
  *anything* of the same shape works on the machine. Here, a five-minute C++ OpenMP
  program would have redirected the entire investigation on the first morning.
