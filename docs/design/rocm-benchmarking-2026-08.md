# fox vs Ollama on the Radeon 890M — ROCm benchmarking (2026-08-01)

Status: **root cause found and fixed — fox now matches/beats Ollama on this
benchmark.** Two independent bugs in fox's own code (not llama.cpp, not the
GPU) combined to explain nearly the whole gap: an expensive default sampling
path, and — the bigger one — a `seq_id` allocation/ordering bug that
silently defeated llama.cpp's own multi-sequence batching. Fixed throughput:
**~46-52 t/s → ~122-146 t/s** on the standard benchmark, against Ollama's
~110-148 t/s (run-to-run) and vanilla `llama-server`'s 173 t/s. An earlier
pass through this investigation wrongly concluded the gap was a structural
llama.cpp/ggml-cuda kernel limit; that conclusion is **retracted** below,
with the evidence that overturned it, since it was briefly committed to
`vllm-gap-analysis.md`/`STATUS.md` before being corrected. Related:
[`engine-capabilities-checklist.md`](engine-capabilities-checklist.md)'s
target-machine section, [`vllm-gap-analysis.md`](vllm-gap-analysis.md) §1.
A follow-up caveat — that the fix held only for low-prefix-cache-reuse
traffic, and degraded back toward baseline under a shared system prompt —
was **also fixed since** (2026-08-01, third pass), by switching to a
unified KV cache: **median 158.2 t/s, range [155.0, 158.9]**, above
Ollama's 144-155. See "Known limitation" (now resolved), "Attempted fix"
(the ruled-out repair), and "The fix that actually closed it: `kv_unified`"
below. Since measured directly against vanilla `llama-server` on the same
hardware (178.1 t/s): fox is at **89% of it, an 11.1% remaining gap**, now
localised to fox's own request lifecycle rather than to batching — see
"Verified against vanilla `llama-server`". A separate `n_threads` fix found
along the way is worth **+54% on the CPU backend** (neutral on ROCm).

## Why this exists

The stated goal for this machine (AMD Ryzen + Radeon 890M) is for fox to beat
both Ollama and vLLM on it, not just be API-compatible with them. vLLM doesn't
run here at all (no NVIDIA GPU). Ollama does, so it's the real competitor —
this doc is a head-to-head throughput investigation against it, same
hardware, same GGUF weights, same underlying llama.cpp lineage.

## Setup

Both engines run in Docker (host stayed unmodified — no ROCm/Vulkan dev
packages installed outside containers):

- **fox:vulkan** — `Dockerfile.vulkan` (already existed), Ubuntu 24.04 +
  `glslc`/Vulkan dev headers at build time.
- **fox:rocm** — new `Dockerfile.rocm`, `rocm/dev-ubuntu-24.04` base so
  `hipcc`/clang are present at build time. Pinned to ROCm 7.2 (see Results).
- **ollama:rocm** — official `ollama/ollama:rocm` image.
- **llamacpp:rocm** — `Dockerfile.llama-server-rocm` (committed), compiling
  vanilla `llama.cpp`'s own `llama-server` from the **exact same vendored
  commit fox uses** and with the **same ggml/HIP flags `build.rs` sets**, so
  a comparison against it differs only in the serving layer. Used to isolate
  "is this fox's code or llama.cpp's" (see below). Earlier passes used a
  throwaway version of this image; it is committed now precisely so the
  reference number stops being folklore.

The 890M is `gfx1150` (RDNA 3.5), not in ROCm's supported-device list for
any of these engines. All three need `HSA_OVERRIDE_GFX_VERSION=11.0.0` to
make the driver present it as `gfx1100` (a supported RDNA3 desktop chip);
Ollama additionally needs `OLLAMA_IGPU_ENABLE=1` (it has its own iGPU
drop-guard). Fox's ROCm build targets `gfx1100` at compile time to match
(`AMDGPU_TARGETS=gfx1100`).

All engines serve the **exact same GGUF** (`llama-3.2-1b-instruct-q8_0`,
imported into Ollama via `docker cp` + a minimal `Modelfile` rather than
letting it pull its own copy) — isolates serving-engine differences from
model/quant differences.

`fox-bench` (`--concurrency 4 --requests 40 --max-tokens 256` for the
sustained-load numbers below, smaller for early exploratory runs) drove
every comparison.

## build.rs changes (kept)

1. **`AMDGPU_TARGETS` passthrough** — `build.rs` never forwarded a GPU
   architecture to CMake for the HIP backend. CMake's HIP language
   auto-detects the target by probing a visible GPU at *build* time, which
   doesn't exist in a build container. Added: if the `AMDGPU_TARGETS` env var
   is set, forward it as a CMake define. Without this, the ROCm build fails
   to configure at all inside Docker.
2. No other build.rs changes. `GGML_HIP_MMQ_MFMA` was considered
   (Ollama-parity flag) but **ruled out by reading the vendor source**: it's
   gated behind `defined(CDNA)` (`ggml-cuda/common.cuh`) — RDNA3/gfx11 never
   takes that branch, so it's a no-op for this GPU regardless of what fox
   sets.

`Dockerfile.rocm` (new) mirrors `Dockerfile.vulkan`'s two-stage structure;
its header comment has the exact build/run commands, including the
`--device /dev/kfd --device /dev/dri --group-add video --group-add <render-gid>`
passthrough (get the render GID via `getent group render` — passing the
group by *name* fails inside minimal containers that lack that `/etc/group`
entry). It also needed `hipblas-dev`/`rocblas-dev` explicitly (not just the
runtime `hipblas`/`rocblas` packages) — without them, `libggml-hip.so`
silently fails to dlopen and the server falls back to CPU with no error.

## Results — theories tested and ruled out

| Comparison | fox | Other | Notes |
|---|---|---|---|
| fox-ROCm vs fox-Vulkan (conc=4) | **50.7 t/s** | 43.0 t/s (Vulkan) | ROCm backend is a real, if modest, win over Vulkan for fox itself |
| fox-ROCm vs Ollama-ROCm (conc=4, `OLLAMA_NUM_PARALLEL=4`) | 48-52 t/s | **93.8-136.8 t/s** | the gap that mattered (run-to-run variance in this range; not yet root-caused at this point) |
| fox-ROCm vs Ollama-ROCm (conc=1, solo) | 44.7 t/s | 48.6 t/s | **parity** — same llama.cpp core, same per-token speed alone |
| fox-ROCm, `--max-context-len 4096` vs default (131072, model's trained ctx) | 44.3 t/s | — | **no improvement** — ruled out KV-cache-size/memory-bandwidth theory |
| fox-ROCm pinned to **ROCm 7.2** (matches Ollama's bundled `libamdhip64.so.7.2`) vs `rocm/dev-ubuntu-24.04:latest` (ROCm 7.14, bleeding-edge) | 48.8 t/s | — | **no improvement** — ruled out ROCm library version |
| Vanilla **llama-server** (same llama.cpp commit as fox, no fox/Ollama) | — | **173.0 t/s** | **the decisive test — see below** |
| fox-ROCm, after the sampling fix (this doc's actual finding) | **58-66 t/s** | 122-137 t/s (Ollama), 173 t/s (llama-server) | real +25-40% improvement, gap not fully closed |

Theories tested and ruled out, in the order investigated:

- **Backend choice (Vulkan vs ROCm)** — ROCm is ~15% faster than Vulkan for
  fox alone, but that's a fraction of the ~2x gap vs Ollama. Not the main
  cause.
- **KV-cache footprint** (fox defaults to the model's full trained context —
  131072 tokens — when `--max-context-len` isn't passed) — forcing
  `--max-context-len 4096` didn't move throughput. Not the cause.
- **ROCm library version** — pinning to Ollama's exact ROCm 7.2 gave a
  statistically identical result to the original ROCm 7.14 build. Not the
  cause. (`Dockerfile.rocm` stays pinned to 7.2 anyway, for a cleaner
  apples-to-apples comparison going forward.)
- **A newer llama.cpp version/patches than fox's vendored commit** — Ollama
  pins `b10091` (2026-07-22), ~3-4 weeks ahead of fox's vendored `6f4f53f2b`
  (2026-06-29), and applies no patches to the batching/KV-cache files.
  Diffed the relevant files directly between the two commits: no
  behavior-relevant difference. Not the cause.
- **Scheduler admission/pipelining** — does fox eagerly re-fill a freed
  decode slot, or does it lose a step? Measured directly via a temporary
  per-tick log (`prefill_ids.len()`/`decode_ids.len()`, reverted after use):
  realized decode batch width averages **~3.85 of a possible 4**, identical
  on CPU and ROCm. The scheduler hands the model a near-full batch on
  almost every step. Not the cause.
- **llama.cpp's ubatch-splitter selection** (`split_equal` vs
  `split_simple`) — confirmed via a temporary one-line `fprintf` in the
  vendored `llama-kv-cache.cpp` (reverted, never committed — this repo's
  submodule convention is to never commit vendor changes) that the correct,
  fully-batching `split_equal` path runs, not `split_simple`. An earlier
  `LLAMA_BATCH_DEBUG=1` capture that seemed to show `split_simple` running
  was a misread of a different, unrelated debug line (a `graph_reserve`
  warm-up ubatch, not the real decode ubatch). Not the cause.
- **GPU kernel dispatch width** (`rocprofv3` profiling) — a short benchmark
  (8 requests, 64 tokens) showed 94.5% of GPU time in single-token
  (`ncols_dst=1`) matmul kernels. A longer, more realistic capture (40
  requests, 256 tokens) found a *mix* instead — `ncols_dst=1: 47.5%,
  2: 37.5%, 3: 8.4%, 4: ~1.1%`, weighted average **~1.64 of 4** — which at
  the time was read as "not a wall of 1s, so probably just natural
  completion-timing variance, not a real bug." **That reading was itself
  wrong — this measurement was the earliest direct evidence of the actual
  seq_id-ordering bug** (see below), just not recognized as such until
  later. `ncols_dst<4` most of the time is exactly what a `seq_id`-ordering
  failure in llama.cpp's `split_equal` looks like from the outside: some
  ticks luck into an ascending-enough ID assignment to batch several
  sequences together (2-3), others don't and collapse to `1`, depending on
  the essentially-random order a LIFO pool happened to hand out IDs in that
  moment — not the "one prefill step of staggering" explanation this doc
  originally gave it.

## The wrong turn: "closed as structural" (retracted)

At this point in the investigation, with scheduling, splitting, and batch
width all ruled out or explained, the ~1.64-vs-~3.85 mismatch between what
the scheduler hands the model and what the GPU kernel actually executes was
concluded to be an llama.cpp/ggml-cuda kernel limitation — "the same 'fox
rides llama.cpp's kernels' ceiling as PagedAttention/FlashAttention" — and
was briefly committed as **closed, structural, not fox backlog** in
`vllm-gap-analysis.md` and `STATUS.md`.

**This was wrong, and a single test proved it**: compiling and running
vanilla `llama-server` — the same llama.cpp source, same vendored commit
fox uses, same GPU, no fox code and no Ollama Go layer at all — on the
identical benchmark. Result: **173.0 t/s**, not only far above fox's ~50
t/s but *also* above Ollama's own ~122 t/s. If the ggml-cuda kernel itself
couldn't sustain wide batching, `llama-server` built from the *exact same
kernel source* couldn't have hit 173 t/s either. The kernel is fine; the
bottleneck was always going to be found in code fox actually owns. The
"closed as structural" edits were reverted in both files.

## The real root cause: fox's own sampling defaults

Timing instrumentation added directly inside `do_decode_batch`
(`src/engine/model/llama_cpp/batch.rs`, all temporary and since removed)
isolated where the ~76-88ms average per-tick cost (4-wide batch) actually
went:

| Stage | Cost |
|---|---|
| `ffi::llama_decode` call itself | ~44-50ms |
| Per-request `sample_constrained`/`sample_token` | ~3.6-4.6ms **each**, ×4 |
| Everything else (post-processing, streaming) | ~0.4ms |

The per-request sampling cost was the first surprise — `sample_token`
(`src/engine/model/sampling.rs`) should be a cheap top-k/softmax operation,
not several milliseconds. Further breakdown pinpointed it: `k` (the
resolved `top_k`) was **`0`** on every single sample. From
`src/api/shared/sampling_defaults.rs`:

```rust
/// Disabled — OpenAI exposes no `top_k`. `0` means "off" in the sampler.
pub const TOP_K: u32 = 0;
```

This is a **deliberate** fox design choice (matching real OpenAI's API,
which has no `top_k` parameter at all) — not a bug in the sense of "wrong
value," but one with a severe, unanticipated performance cost: with `top_k`
disabled, `sample_token` computed `exp()` over the **entire ~128,256-token
vocabulary twice** (once for the normalizing sum, once building the
probability vector) and then **fully sorted** all 128,256 entries just to
apply `top_p`/`min_p` truncation — every single generated token, every
request. Ollama and `llama-server` never hit this path: they default
`top_k = 40` **regardless of API surface** (confirmed via `llama-server`'s
own `/props` endpoint), so real OpenAI-API-shaped requests against them
still get the cheap top-40 path fox's OpenAI surface deliberately opts out
of.

## The fix

Two changes in `src/engine/model/sampling.rs` and its call sites, both
behavior-preserving (same output distribution, no default changed):

1. **Adaptive candidate selection** (`sample_token`) — when `top_k` is
   disabled, instead of sorting/exponentiating the full vocab, adaptively
   grow a by-logit candidate pool (64 → 256 → 1024 → … via
   `select_nth_unstable_by`, an O(n)-average partition) until it provably
   contains enough of the distribution to make `top_p`/`min_p` truncation
   give the *exact same result* as sorting everything — falling back to the
   full vocab only if a request's parameters genuinely need it (e.g.
   `top_p` at/near `1.0`). `max_l`/`exp_sum` are still computed with one
   full linear pass each (unavoidable — that's the real normalizer), but the
   expensive full sort and the *second* full `exp()` pass are gone in the
   common case. When `top_k > 0`, the same partition primitive already
   finds the top-k set directly in O(n) average instead of a full sort.
2. **Skip the full-vocab logits copy when nothing reads it** — a new
   `needs_logits: bool` on `InferenceRequestForModel`
   (`src/engine/model/mod.rs`), set from `r.sampling.logprobs.is_some()`
   (`src/engine/run.rs`, both `run_prefill`/`run_decode`), gates the
   `logits_slice.to_vec()` copy in `do_decode_batch`/`do_prefill_batch` —
   only OpenAI `logprobs` ever reads `Logits.values`, so the ~513KB copy
   is now skipped whenever a request doesn't ask for it (the common case).

Correctness verified: all existing `sampling::` unit tests pass unchanged
(including `sample_token_top_p_restricts_candidates` and
`min_p_keeps_only_dominant_token`, which directly exercise the new adaptive
path), all 12 golden tests (real model) pass, the full stub integration
suite (40 tests, including `test_v1_chat_logprobs_present_when_requested`
and `test_v1_chat_logprobs_absent_by_default`) passes, and a live spot-check
with `logprobs: true` against the running ROCm build returned a coherent
completion with populated `logprobs`/`top_logprobs`.

## The second, bigger root cause: `seq_id` allocation order

The sampling fix alone (~58-66 t/s) still left fox at roughly half of
Ollama's throughput. Further instrumentation inside `do_decode_batch`
found `ffi::llama_decode` itself still costing ~44-50ms per 4-wide call
even after the sampling fix — and reading `llama-context.cpp` explained why
that number is misleading: `llama_decode` launches GPU compute
**asynchronously** (`ggml_backend_sched_graph_compute_async`) and returns
before the GPU finishes; the actual completion wait happens later, inside
`llama_get_logits_ith` (`ctx->synchronize()`). Timing both call sites
directly showed the *first* `llama_get_logits_ith` call per tick
consistently took ~17-18ms (the real GPU wait — plausibly close to what
`llama-server`'s 173 t/s implies is needed here), leaving **~40ms of
synchronous CPU-side cost inside `llama_decode` itself** unexplained.
Capping `n_batch`/`n_ubatch`/`n_seq_max` to match `llama-server`'s exact
launch config ruled out context/batch sizing as the cause.

**The real explanation was upstream of all of this**: this doc's own
earlier `rocprofv3` profiling (see "Results" above) had already found that
GPU kernel dispatch averaged only `ncols_dst≈1.64` of a possible 4 — and at
the time, that was (wrongly) chalked up to normal batch-width variance from
requests finishing at different times. It wasn't. The real cause: fox's
`seq_id_pool` (`src/scheduler/mod.rs`) was a plain `Vec`-backed LIFO stack,
handing out whatever ID was pushed back most recently — and
`do_decode_batch` (`src/engine/model/llama_cpp/batch.rs`) emitted the
`llama_batch` in **scheduler-admission order**, not seq_id order. But
llama.cpp's `split_equal` (the splitter used whenever the KV cache is
non-unified — i.e. always, here, since `n_stream = n_seq_max > 1`) is
called with `sequential=true`, and only keeps growing the *same* ubatch
while walking the batch as long as
`batch.seq_id[i][0] == last_seq_id + 1` (`llama-batch.cpp`) — **strictly
consecutive, strictly increasing, in emission order.** Four genuinely
concurrent requests with scattered or out-of-order seq_ids (entirely
possible with a LIFO pool and admission-order emission) silently fail this
check and get split into four separate 1-token ubatches — real GEMV-level
serialization, invisible from fox's own scheduler-level metrics (which only
see "4 requests decoding this step," never how llama.cpp's splitter grouped
them). This is the mechanism the whole `ncols_dst` investigation earlier in
this doc was circling without landing on.

Fixed in two places, both required together:

1. `src/scheduler/mod.rs`: `seq_id_pool` is now a `BinaryHeap<Reverse<i32>>`
   min-heap instead of a `Vec` stack — always hands out the *lowest* free
   ID, so N concurrent requests occupy IDs `0..N-1` densely (mirroring how
   `llama-server` assigns `slot.id = i`).
2. `src/engine/model/llama_cpp/batch.rs` (`do_decode_batch`): the
   `llama_batch` is now emitted in **ascending `kv_seq_id` order**, not
   scheduler-admission order — a dense seq_id pool alone isn't sufficient if
   the batch itself isn't walked in that order. Logits are read back via an
   inverse index (`slot_of`) so the returned per-request order is
   unaffected by the emission reordering.

## Measured result

| Metric (concurrency=4, 40 requests, 256 max tokens) | Baseline | +sampling fix | +seq_id fix |
|---|---|---|---|
| Throughput | ~46-52 t/s | ~58-66 t/s | **~122-146 t/s** |
| vs Ollama (~110-148 t/s, run-to-run) | ~half | ~half | **on par, sometimes ahead** |
| vs `llama-server` native (173 t/s) | ~30% | ~35% | **~70-85%** |

Both fixes were needed — the sampling fix alone left fox at roughly half of
Ollama; the seq_id fix on top of it took fox from "roughly half of Ollama"
to "matching or beating Ollama" on this exact benchmark, run twice to guard
against a fluke (122.0 t/s and 146.3 t/s, both far above every pre-fix
number recorded in this doc, both 40/40 requests with zero errors on fox's
side). The remaining gap to `llama-server`'s 173 t/s is unexplored but far
smaller than what this investigation started with, and no longer suggests
an upstream/structural ceiling — fox's own request-lifecycle overhead
(scheduling, HTTP/streaming layers) is the more likely remaining source,
not a fundamental limitation.

## Known limitation (RESOLVED — see "The fix that actually closed it" below): the seq_id fix degrades under heavy prefix-cache reuse

The ad-hoc single-shot numbers above (fresh container, one comparison, then
torn down) turned out to be an optimistic case. Built `scripts/repeat_bench.sh`
(committed — see below) to properly benchmark with warmup, multiple
repetitions, alternating engine order, and automatic discarding of runs with
request errors, specifically because single ad-hoc runs on this hardware
showed too much run-to-run variance to trust. Running it for 5 sustained
repetitions against the **same long-lived fox container** (rather than a
fresh one per comparison) surfaced a real, reproducible degradation the
one-shot numbers missed entirely: throughput settled at a rock-stable
**~52.7 t/s** (range 52.6-52.9 across all 5 repetitions) — back down to the
pre-seq_id-fix baseline, while Ollama stayed at 144-155 t/s the whole time.

Root cause, confirmed via `docker logs | grep seq_id` and `/metrics`: **not**
a resource leak (`ferrumox_kv_cache_usage_ratio` sits at ~0.4% at idle —
blocks are freed correctly). The seq_id min-heap and ascending-emission-order
fix above guarantees dense, consecutive IDs only for requests that get a
**fresh pool pop** (a genuine prefix-cache miss). But `try_insert_prefix`'s
block-level prefix cache donates a finished request's **existing** seq_id to
the cache entry, and a future cache **hit** (`schedule.rs`'s admission path,
`req.kv_seq_id = hit.seq_id`) inherits that donated ID as-is — not a fresh
pop from the ascending pool. Every chat request shares the same first ~16
tokens (BOS + role-header boilerplate from the chat template) regardless of
user content, so in practice almost every request hits this shared-header
cache entry and inherits whatever seq_id was last donated to it — which
drifts to arbitrary, non-consecutive values (observed: IDs cycling at 29 and
31, with nothing below 29 admitted for extended stretches) as the cache
churns. llama.cpp's `split_equal` requires **strictly** consecutive
`+1`-incrementing seq_ids to merge sequences into one ubatch — sorting
ascending (which `do_decode_batch` already does) cannot repair a gap; only a
genuinely dense set of IDs can satisfy it. A set like `{0, 1, 29, 31}` still
splits into multiple ubatches even sorted.

**This means the fix's real-world benefit is workload-dependent**: it fully
holds for traffic with low prefix-cache reuse (varied prompts/no shared
system prompt), which is what the earlier one-shot benchmarks happened to
exercise (short runs, cache still mostly cold). It degrades toward the
pre-fix baseline under heavy reuse — which, notably, is not just a synthetic
benchmark artifact: **any real deployment where multiple concurrent
conversations share a common system prompt hits this same pattern**, since
that shared prefix is exactly the kind of content the block-level cache is
designed to reuse.

**Not fixed here** — see the next section: the obvious fix (migrate via
`llama_memory_seq_cp`) was attempted and **does not work** — it crashes the
server. A real fix needs a different mechanism.

## Attempted fix: migrate cache-hit requests to a fresh seq_id via `llama_memory_seq_cp` — crashes, reverted

The natural fix for the limitation above is to give a prefix-cache-hit
request a **fresh**, densely-allocated `seq_id` (via `Scheduler::
try_pop_fresh_seq_id`) instead of the donated one, copying the cached
prefix's KV data across with `llama_memory_seq_cp` before this step's
prefill/decode runs — mirroring how `Model::roll_context` already does FFI
work at the engine layer. This was fully implemented (new
`batch::PrefixHitMigration`, `Scheduler::try_pop_fresh_seq_id`/
`finalize_seq_migration`, a migration loop in `run_loop` calling the
already-existing `Model::copy_sequence_range`) and passed `cargo fmt`/
`clippy`/the full stub test suite, including a new unit test exercising the
migration end-to-end at the scheduler level.

**It crashed the server on the very first prefix-cache hit** when validated
against a real ROCm build under `scripts/repeat_bench.sh`'s sustained-load
test (the same test that surfaced the original limitation):

```
/app/vendor/llama.cpp/src/llama-kv-cache.cpp:518: GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers") failed
```

Root cause, confirmed by reading `vendor/llama.cpp/src/llama-kv-cache.cpp`
directly (`llama_kv_cache::seq_cp`, ~line 463): fox's KV cache is
**non-unified** (`n_stream = n_seq_max > 1`, one stream per seq_id — this is
the same setting the seq_id-ordering fix above depends on for
`split_equal` to batch concurrent sequences at all). When the source and
destination seq_ids live in different streams (`s0 != s1` — true for any
migration to a genuinely different seq_id, by construction), `seq_cp`
takes the "cross-stream" path, which only supports copying the **entire**
KV buffer (`p0`/`p1` must span `[0, get_size())`, i.e. the full `n_ctx`,
not just the `cached_tokens` prefix) — `GGML_ASSERT(is_full)` rejects
anything narrower, including the exact "copy just the cached prefix"
partial range this fix needs. The same-stream fast path (cheap metadata-only
remap, no assert) exists but is unreachable here: two different seq_ids in
a non-unified cache are never in the same stream by construction, so a
migration to a fresh id always takes the cross-stream path. This is not a
fox bug or a version-specific quirk — it's how `seq_cp` is documented to
behave for split/non-unified caches in fox's vendored llama.cpp commit, and
the comment `// TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]` right above it
suggests upstream is aware partial cross-stream copies aren't supported yet
either. Confirmed via the "verify against upstream before calling something
structural" lesson from this doc's own earlier retraction — this time the
upstream check confirmed the limitation is real, not a fox misuse.

**Reverted in full** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`,
`src/scheduler/schedule.rs`, `src/engine/run.rs`) — the crash makes this
strictly worse than the known limitation it tried to fix, so nothing from
this attempt should ship. `Model::copy_sequence_range`/`supports_seq_copy`
themselves are untouched and still exist as a capability probe only (as
before this attempt); they should not be invoked as a real per-request KV
migration mechanism against fox's current non-unified KV cache without a
different approach to the copy itself.

**What a real fix would need instead** (not attempted yet): since
`llama_memory_seq_cp` cannot do a partial cross-stream copy, options are
(a) avoid the copy entirely — treat a prefix-cache hit whose donated
`seq_id` falls outside the pool's current dense/low range as a miss instead
(recompute the ~16-token prefix rather than reuse it), trading a small,
bounded amount of recompute for keeping seq_ids dense; (b) a full-buffer
`seq_cp` (accepting the cost of copying the entire per-stream KV buffer,
not just the cached prefix) if that cost turns out to be acceptable at this
model's context size — unmeasured, and likely not acceptable at large
`n_ctx`; (c) switch to a unified KV cache (`n_stream = 1`) for the
non-batched dimension, if `split_equal`'s consecutive-seq_id requirement
turns out not to actually need `n_stream > 1` — unconfirmed, needs reading
`llama-batch.cpp`/`llama-kv-cache.cpp` more closely than this session did.
None of these were evaluated in depth; this is a genuine open design
problem, not a known-good approach waiting to be typed in.

## The fix that actually closed it: `kv_unified` (2026-08-01, third pass)

Alternative (c) above turned out to be correct, and its stated uncertainty
("if `split_equal`'s consecutive-seq_id requirement turns out not to
actually need `n_stream > 1`") resolves cleanly by reading the two files it
names. The requirement isn't a property of `split_equal` that fox must
satisfy — it's a property of *which splitter runs at all*:

```cpp
// llama-kv-cache.cpp, llama_kv_cache::init_batch
auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch)
                            : balloc.split_equal(n_ubatch, true);
```

`n_stream` is `unified ? 1 : n_seq_max`. So a unified KV cache doesn't make
fox *satisfy* the consecutive-ID rule — it takes the code path where the
rule does not exist. `split_simple` has no equivalent of `split_equal`'s
`batch.seq_id[i][0] == last_seq_id + 1` guard, so it folds the whole decode
batch into one ubatch regardless of which IDs the scheduler happens to
hold. Prefix-cache hits can donate whatever `seq_id` they like.

**The change** is two lines: `ctx_params.kv_unified = true` in both
`LlamaCppModel::load()` and `::new_context()`
(`src/engine/model/llama_cpp/mod.rs`).

### Measuring it without patching the vendor

Verifying this by throughput alone is hopeless on this hardware — the
run-to-run spread swamps the effect (one config measured [72.3, 154.6] t/s
across 5 repetitions). So this pass measured the *mechanism* instead: the
actual width of each decode ubatch, which is deterministic and needs one
short run.

llama.cpp already traces this under `LLAMA_BATCH_DEBUG=1`, but fox installs
a `noop_log` callback that drops llama.cpp's log entirely — which is why
earlier passes resorted to patching the vendored source and rebuilding.
Added instead: **`FOX_LLAMA_LOG=1` forwards llama.cpp's log to stderr**
(same file, next to `noop_log`). Combined:

```bash
LLAMA_BATCH_DEBUG=1 FOX_LLAMA_LOG=1 fox serve --model-path <model.gguf>
# then, from the log:  grep -A4 'equal_seqs   = 0' | grep n_tokens
```

No submodule edits, nothing to remember to revert.

### Result

Same protocol both times: one warmup round, then 3 sustained rounds of
`fox-bench --concurrency 4 --requests 40 --max-tokens 128` against one
long-lived server (the sustained-load shape that surfaced the limitation in
the first place), counting decode ubatch widths.

| Decode ubatch width | Before (non-unified) | After (`kv_unified`) |
|---|---|---|
| 1 token | 5498 | 46 |
| 2 | 1257 | 116 |
| 3 | 379 | 410 |
| 4 (full) | 1444 | 7070 |
| **weighted average** | **1.74 / 4** | **3.90 / 4** |

The before-column's 1.74 independently reproduces the ~1.64 that
`rocprofv3` measured at the kernel level in the first pass — same
fragmentation, measured two unrelated ways. After the change, **zero**
ubatches take the `split_equal` path (`equal_seqs = 1` count: 0),
confirming the splitter switch rather than inferring it.

ROCm throughput (`scripts/repeat_bench.sh`, 5 repetitions, same container
lifetime), for the target machine:

| | median | range |
|---|---|---|
| Before | 110.9 t/s | [72.3, 154.6] |
| After | **158.2 t/s** | **[155.0, 158.9]** |

The collapsing range matters as much as the median: the wild run-to-run
variance previously attributed to thermal/iGPU noise was substantially this
fragmentation, appearing or not depending on which seq_ids the prefix cache
happened to be recycling. fox is now above Ollama's 144-155 t/s on this
benchmark and at ~91% of vanilla `llama-server`'s 173 t/s.

### Cost

**No extra memory.** The KV allocation is identical either way — only its
shape changes (`llama_kv_cache: size` line, same model, `--max-context-len 4096`):

| | total | cells | seqs/streams |
|---|---|---|---|
| Before | 4224.00 MiB | 4096 (per stream) | 33/33 |
| After | 4224.00 MiB | 135168 (shared) | 33/1 |

The shared pool is in fact strictly more flexible: a single long
conversation can exceed the per-stream ceiling when other slots are idle,
which the non-unified layout cannot do.

Verified with `make e2e` (22 passed, 0 failed — including 4-way concurrency,
context rolling past `n_ctx`, embeddings, and mid-stream disconnect) plus
the full stub suite and `cargo test --release --lib` (347 tests).

**Side note**: this also retires the crashing `llama_memory_seq_cp`
migration from the previous section as unnecessary — and would have
unblocked it anyway, since the `GGML_ASSERT(is_full)` that killed it only
guards the *cross-stream* path; with `n_stream = 1` every `seq_cp` is
same-stream, where partial ranges are supported.

### Architecture coverage (verified 2026-08-02)

The concern with a unified KV cache is that it changes the KV buffer's
*shape*, so architectures that don't use a plain dense per-sequence KV are
where it would break. Each was run under `kv_unified` with a single request
plus 4 concurrent ones (the multi-sequence path the change actually
affects):

| Model | Architecture | Result |
|---|---|---|
| `llama-3.2-1b-instruct-q8_0` | dense, no SWA | OK — incl. a 7378-token prompt, correctly recalling a fact from it |
| `mamba-130m-hf.Q8_0` | recurrent / SSM (no KV cache at all) | OK — 4/4 concurrent |
| `Gemma-3-1B-it-…_Q4_k_m` | SWA (`n_swa = 512`) | OK — 4/4 concurrent |
| `gemma-4-E2B-it-Q4_K_M` | SWA | OK — 4/4 concurrent |
| `DeepSeek-V2-Lite.Q4_K_M` | MLA (latent KV) | OK — 4/4 concurrent |

**One pre-existing failure found, not caused by this change**: the Gemma-3
model above returns `completion_tokens = 1` (empty content) for the same
7378-token prompt that llama-3.2 handles correctly. Reproduced identically
on a build with `kv_unified` removed, so it predates this work — see
"Pre-existing issues found while verifying" below.

## `n_threads`: fox was inheriting llama.cpp's 4-thread default (2026-08-02)

Found while chasing the residual gap to `llama-server`. fox never set
`llama_context_params::n_threads`, so it inherited `GGML_DEFAULT_N_THREADS`
= 4 — a value `ggml.h` itself marks `// TODO: better default` — on every
machine regardless of core count. `llama-server` does not inherit it; it
resolves `n_threads` from `common_cpu_get_num_math()`.

Sweep on this 24-logical-core box (concurrency 4, 40 requests, 128 max
tokens, one server at a time):

| threads | t/s |
|---|---|
| 4 (the inherited default) | 80.7 |
| 8 | 123.4 |
| 12 (physical cores) | 123.2 |
| 24 (logical cores) | 103.4 |

Fixed in `cf5cd47` by resolving to *physical* cores, mirroring
`common_cpu_get_num_physical_cores()` (count distinct `thread_siblings`
masks on Linux). Physical rather than logical is what the sweep supports —
SMT siblings share execution units and 24 threads measurably regresses.
`FOX_N_THREADS` overrides it.

| backend | before | after |
|---|---|---|
| CPU | ~79 t/s | **121.5 t/s** (range [119.8, 124.7]) |
| ROCm | 158.2 t/s | 158.6 t/s (unchanged) |

(For what fox's 121.5 t/s should be compared against, see "Where the TTFT gap
actually is" below — the fair reference is a `GGML_NATIVE=OFF` build of
`llama-server` at 151.1 t/s, not the 153.4 t/s natively-optimised one.)

ROCm being flat is expected — compute runs on the GPU there. This is a
CPU-backend fix, and CPU-only users were the ones silently paying for it.

**Methodology warning, learned the hard way**: benchmark only ONE server at
a time. ggml's thread pool spin-waits rather than sleeping, so an *idle*
second server still burns cores, and the distortion scales with each
server's thread count — i.e. it punishes exactly the variable under test.
A head-to-head with both up read fox 79.2 vs llama-server 151.2; measuring
each alone gave 121.5 vs 153.4. The first measurement after this fix looked
like a *regression* (76.6 t/s) purely because of this.

## Verified against vanilla `llama-server` on ROCm (2026-08-02)

Earlier passes cited "173 t/s" for `llama-server` as the ceiling without
this session having measured it. Now measured directly, with
`Dockerfile.llama-server-rocm` (committed): upstream's own server built from
**the same vendored llama.cpp commit** with **the same ggml/HIP flags
`build.rs` uses**, so the comparison differs only in the serving layer.
Confirmed on-GPU via its own log (`ROCm0 — AMD Radeon 890M`, layers assigned
to ROCm0) — necessary, because with `GGML_BACKEND_DL` a `libggml-hip.so`
that fails to dlopen falls back to CPU silently.

Each measured alone, `scripts/repeat_bench.sh`, 5 repetitions:

| | median | range |
|---|---|---|
| `llama-server` | 178.1 t/s | [177.7, 182.1] |
| fox | 158.3 t/s | [157.6, 161.6] |

**Real remaining gap: 11.1%** (the previously-cited ~9% was approximately
right). Splitting it by concurrency localises it:

| | concurrency 1 | concurrency 4 | TTFT (P50) |
|---|---|---|---|
| fox | 52.8 t/s | 158.3 t/s | 42 ms |
| `llama-server` | 55.5 t/s | 178.1 t/s | 24 ms |
| gap | **4.9%** | **11.1%** | **+18 ms** |

So roughly half of it (~5%) is a flat per-request/per-token cost already
present with a single stream, and the other half only appears under
concurrency, consistent with the measured 3.90/4 average ubatch width (not
4.00) plus per-step scheduling overhead. The flat half was investigated
directly and is **not** where it looked: see "Where the TTFT gap actually
is" below — it is not admission latency, not the HTTP layer and not
sampling, but `llama_decode` itself plus the cost of fox's portable-binary
build flags.

None of these are defects like the ones above; they are the structural cost
of fox having its own scheduling layer. Closing them is fine-grained
profiling work with much lower return than the fixes in this document.

Incidentally, `llama-server`'s log reports `n_threads = 12 (n_threads_batch
= 12) / 24` — the same physical-core count fox now derives, independently
confirming the heuristic in the section above.

## Where the TTFT gap actually is (2026-08-02)

The `llama-server` comparison above showed fox's TTFT at 42 ms vs 24 ms
(ROCm) / 35.9 ms vs 24.4 ms (CPU), and the obvious suspicion was fox's own
request lifecycle — HTTP parsing, chat templating, scheduler admission
latency. **Measured, and it is not.** Temporary `Instant`-based phase logs in
`chat_completions` and `run_loop` (reverted, never committed), CPU backend,
22-token prompt, `max_tokens=1`:

| phase | cumulative |
|---|---|
| model resolved | 110 µs |
| chat template + tokenize | 270 µs |
| submitted to scheduler | **290 µs** |
| `run_prefill` returns | ~32 000 µs |
| first token at the handler | ~34 000 µs |

**99.1% of the time is inside `run_prefill`** — i.e. one `llama_decode` of a
22-token prompt. Everything fox owns above the engine costs 290 µs combined.
Scheduler admission specifically is ~24 µs (visible in ordinary INFO logs:
"request admitted to waiting queue" → "request admitted to batch").

Two candidate explanations were tested:

- **KV cache size** — fox sizes its context for `n_seq_max` sequences
  (`4096 × 33 = 135168` cells) where `llama-server` uses 16384. Re-ran with
  `--max-batch-size 4` (so `4096 × 5`): `run_prefill` unchanged at 32-35 ms.
  **Not the cause.**
- **`GGML_NATIVE=OFF`** — fox *must* build with it, since it is incompatible
  with `GGML_BACKEND_DL` (`build.rs`), which is what lets one binary carry
  every backend. Rebuilt `llama-server` with fox's exact flags: its
  single-request latency goes **24.4 ms → 28.8 ms**, suggesting ~4.4 ms of
  the gap is the price of the portable-binary architecture. **Treat this as
  unconfirmed**: a later alternating A/B (see "Recovering most of it" below)
  showed fox's default CPU backend performs the same as a hand-tuned
  AVX-512 variant on this host, so `GGML_NATIVE=OFF` evidently does not cost
  what a single before/after pair implied. Both numbers here came from
  non-alternating runs, which this session repeatedly proved unreliable.

Notably this flag costs *latency* but barely any *throughput*: the same
`GGML_NATIVE=OFF` build still sustains 151.1 t/s (range [144.2, 152.7])
versus 153.4 with `NATIVE=ON`. So the corrected apples-to-apples CPU
comparison is **fox 121.5 vs llama-server 151.1 t/s** — the earlier
"121.5 vs 153.4" slightly overstated the gap by comparing against a
natively-optimised build fox cannot ship.

**Remaining unexplained: ~7 ms**, inside `llama_decode` itself, with fox and
`llama-server` running the same commit and the same flags. Localising that
needs a profiler; `perf` is unusable on this machine (installed, but no
build for kernel 6.17.0-1028 — needs `linux-tools-6.17.0-1028-oem`, which
requires root).

### Recovering most of it: `FOX_CPU_ALL_VARIANTS=1`

The portability/performance trade-off above turns out not to be forced.
`GGML_CPU_ALL_VARIANTS` builds the CPU backend once per instruction-set tier
as separate `.so` files and lets ggml pick the best the host supports at
runtime — and it **requires** `GGML_BACKEND_DL`, which fox already sets. One
portable binary, still gets AVX-512.

Added as an opt-in build flag (`build.rs`): `FOX_CPU_ALL_VARIANTS=1 cargo
build --release`. It produces 15 `libggml-cpu-*.so` on x86 (x64, sse42,
haswell, skylakex, zen4, sapphirerapids, …); on this Ryzen AI 9 HX 370 the
runtime picks `libggml-cpu-zen4.so` (AVX-512 + VBMI + VNNI + BF16, matching
`/proc/cpuinfo` exactly), confirmed via `FOX_LLAMA_LOG=1`:

```
load_backend: loaded CPU backend from .../libggml-cpu-zen4.so
```

**On this machine it is not measurably faster than fox's existing default.**
A first comparison suggested ~20% (generic 51.4 ms vs zen4 39.7 ms), but that
compared runs taken at different times. Redone properly — one binary, only
the available `.so` swapped between arms, alternating, same session:

| round | default (`libggml-cpu.so`) | zen4 |
|---|---|---|
| 1 | 39.4 ms | 40.5 ms |
| 2 | 40.4 ms | 37.3 ms |
| 3 | 40.8 ms | 39.1 ms |

~3% apart with fully overlapping values: **no effect**. The earlier 51.4 ms
figure was drift, not the generic backend being slow.

**Why it still exists**: the instruction tier genuinely matters — the same
A/B against the *baseline* `x64` variant is stark and perfectly consistent:

| round | `x64` (pure baseline) | zen4 |
|---|---|---|
| 1 | 107.8 ms | 40.6 ms |
| 2 | 105.9 ms | 38.3 ms |
| 3 | 108.8 ms | 37.4 ms |

**2.7× .** So the useful correction is this: *`GGML_NATIVE=OFF` does not mean
fox falls back to that baseline*. Whatever CMake selects by default for
`libggml-cpu.so` already performs like the tuned variant on this CPU, so the
"~4.4 ms is the price of the portable binary" reading from the section above
overstates it — that difference is something else, still unexplained. The
flag remains available for hosts where the default build *does* land on a
poor tier (worth checking with `FOX_LLAMA_LOG=1`, which prints the loaded
`.so`), but it should not be enabled expecting a win.

Build time goes 12s → 45s for the llama.cpp step, which is the other reason
it is opt-in.

**Caution when measuring this**: absolute numbers drift substantially across
a long session (this machine's default build read 35.9 ms early in the day
and 51.4 ms hours later, unchanged). Only alternating, same-session A/B
comparisons mean anything here — a single before/after pair is worthless, as
this section's own first attempt demonstrates.

**One dead end worth recording** so nobody re-walks it: `sample_token`'s
no-truncation path (the OpenAI defaults `top_k=0`/`top_p=1.0`) looked like an
obvious culprit and a fast path for it measured **3.21 ms → 0.69 ms per call,
4.6×**, in an isolated probe. End-to-end throughput before/after: **121.5 →
120.3 t/s**, i.e. nothing. The probe's synthetic logits were unrepresentative
— with real model logits the adaptive candidate pool exits on its first
iteration, so the expensive branch is almost never taken. Reverted rather
than shipped: it also changed which token a fixed seed draws.

## Config-matched comparison, and the context-size trade-off (2026-08-02)

Every fox-vs-llama-server number before this point compared servers configured
differently — fox at `n_seq_max=33`, `n_ctx=135168`, `n_batch=4096`,
`kv_unified=true` against llama-server at 4 / 16384 / 2048 / false — and
charged the whole difference to fox. Redone with the configurations matched
(fox `--max-context-len 4096 --max-batch-size 4`; llama-server
`-c 20480 --parallel 4 -kvu -b 4096`), on ROCm, one server at a time,
alternating, via `scripts/ab_bench.sh`:

| metric | fox | llama-server | gap |
|---|---|---|---|
| TTFT | 28.16 ms (range [27.92, 28.20]) | 23.45 ms ([23.24, 23.82]) | **16.7%** |
| throughput | 158.4 t/s ([157.4, 159.2]) | 176.0 t/s ([175.8, 176.2]) | **11.1%** |

Both disjoint, both stable. **The throughput gap is unchanged from the
unmatched comparison (11.1% either way), so configuration was never its
cause.** The TTFT gap, however, was substantially fox's own configuration:
fox's default measured 42 ms and drops to 28 ms purely by matching.

### What in the config costs TTFT: `n_ctx`, not `n_seq_max`

Isolated by holding total context constant (~20k) and varying only the
sequence count — `--max-context-len 4096 --max-batch-size 4` (5 seqs) versus
`--max-context-len 620 --max-batch-size 32` (33 seqs):

| | TTFT |
|---|---|
| 5 seqs, ctx 20k | 28.01 ms |
| 33 seqs, ctx 20k | 28.03 ms |

INCONCLUSIVE, +0.1% — `n_seq_max` is free. The cost is the total `n_ctx`,
which fox derives as `effective_max_ctx * n_seq` (so 33x the per-sequence
context by default).

### The trade-off runs both ways

Same two context sizes, measured for throughput instead:

| | TTFT | throughput |
|---|---|---|
| ctx 20k | **28.2 ms** | 154.6 t/s |
| ctx 135k (fox's default shape) | 42 ms | **161.8 t/s** |

A large context **costs 33% TTFT and buys 4.7% throughput**. So fox's default
is not simply wrong — it is tuned for sustained throughput at the expense of
first-token latency. That is a defensible choice; it is just nowhere stated as
a choice, and users who care about interactive latency have no hint that
`--max-batch-size` is the knob that controls it.

Worth revisiting deliberately: a deployment serving one interactive user wants
the opposite default from one serving batch traffic.

## Pre-existing issues found while verifying (neither caused by this work)

Both were confirmed to reproduce on a build with the `kv_unified` change
removed, i.e. they predate it. Recorded here because this is where they
surfaced, not because they belong to this investigation.

1. **Long prompts on Gemma-3 return a single token.** A 7378-token prompt
   to `Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q4_k_m`
   yields `completion_tokens = 1` and empty content, while short prompts to
   the same model work fine and the same prompt on `llama-3.2-1b` returns a
   correct 15-token answer. llama.cpp logs no error. Unclear whether it's
   fox, llama.cpp's SWA handling, or this particular community merge —
   undiagnosed. Reproduce with the request body used above (400 `Fact N:`
   lines, `max_tokens = 150`, `temperature = 0`).
2. **A golden test fails on the 1B model.**
   `golden_chunked_prefill_matches_single_shot` panics with `non-empty
   logits` (`golden.rs:201`) under
   `FOX_GOLDEN_MODEL=llama-3.2-1b-instruct-q8_0.gguf`. The other 11 golden
   tests pass. Possibly the test assumes a different model (its own header
   suggests a Gemma GGUF), possibly a real chunked-prefill bug —
   undiagnosed.

## `n_batch`/`n_ubatch` experiment (tried, reverted, unrelated finding)

Separately from the residual above, capping `n_batch`/`n_ubatch` at 2048
**alone** (without `--max-batch-size`) was tried earlier as its own
experiment and gave a real, if modest, **+7-13%** (54.4 t/s vs the
then-baseline 48-51 t/s) — reverted anyway, because it has a correctness
risk: `do_get_embeddings` (`batch.rs`) submits an embedding request's
**entire** input as one unchunked `llama_decode` call, and for a
non-causal (encoder-style, BERT-family) embedding model,
`llama-context.cpp` asserts `n_ubatch >= n_tokens` — an embedding input
longer than 2048 tokens on such a model would crash where today (uncapped
`n_batch`) it doesn't. Gating this safely needs either new FFI plumbing
(llama.cpp exposes no way to query causal/non-causal before creating a
context) or a GGUF-metadata heuristic — real work, not obviously justified
by 7-13% alone. If someone wants this independently of the residual above,
the code to restore is: cap `n_batch`/`n_ubatch` at
`effective_max_ctx.min(2048).max(max_batch_size)` in both
`LlamaCppModel::load()` and `::new_context()`
(`src/engine/model/llama_cpp/mod.rs`), gated on the model being causal.

## What's next

1. ~~**Highest priority**: fix the prefix-cache/seq_id interaction.~~
   **Done** — alternative (c), the unified KV cache, closed it; see "The fix
   that actually closed it: `kv_unified`" above. The other two alternatives
   (skip-cache-on-stale-id, full-buffer copy) are moot and were never
   implemented.
2. Close the remaining **11.1%** gap to `llama-server` (178.1 vs 158.3 t/s,
   both measured directly — see "Verified against vanilla `llama-server`"
   above). Now localised: ~5% is a flat per-request cost visible even at
   concurrency 1, the rest appears only under concurrency, and TTFT is
   +18 ms. So the leads are fox's own request lifecycle — sampling, the mpsc
   token channel, SSE, and scheduler admission latency — not llama.cpp
   internals and not the batching layer (ubatch width is already 3.90/4).
3. Profile `llama-server` itself with `rocprofv3` to confirm it hits
   `ncols_dst=4` consistently, as a sanity check against fox's now-fixed
   dispatch pattern. This is now much easier: `Dockerfile.llama-server-rocm`
   builds it from a ROCm dev base that can host the profiler, unlike
   Ollama's minimal image which ships none.
4. Re-run a `rocprofv3` capture on fox's own fixed build to directly
   confirm `ncols_dst=4` now dominates (this doc's fix is verified via
   aggregate throughput; a kernel-level reconfirmation would be the last
   piece of direct evidence, blocked this session by the ROCm runtime image
   missing `libdw.so.1`, a `rocprofv3` dependency not installed in
   `Dockerfile.rocm`'s minimal runtime stage).
5. The independent 7-13% `n_batch`/`n_ubatch` win in the section above, if
   someone builds the causal/non-causal model detection it needs.

## Benchmarking methodology: use `scripts/repeat_bench.sh`, not one-off runs

Single ad-hoc `fox-bench` invocations (fresh container, one comparison) are
what produced every number in this doc until the prefix-cache finding above
— and they're exactly what missed that degradation, since a short one-shot
run never gives the prefix cache time to saturate with donated seq_ids the
way a real, sustained server process does. `scripts/repeat_bench.sh` (new,
committed) runs N repetitions against **already-running** servers, with a
discarded warmup request per engine, alternating which engine goes first
each round (cancels thermal/cache ordering bias), and drops (with a loud
warning, retried once first) any repetition that comes back with request
errors instead of silently averaging in a result computed on a smaller
sample. It reports median + [min, max], not a single number — use it for
any future fox-vs-X comparison on this hardware; a single run here isn't
trustworthy enough to draw conclusions from, as this whole investigation
kept demonstrating.

### `scripts/ab_bench.sh` — when you are comparing two *builds*

`repeat_bench.sh` deliberately does not start or stop servers, which leaves
four ways to produce a confident-looking wrong answer. Every one of them
produced a wrong answer in this investigation:

| hazard | what it produced here |
|---|---|
| two servers up at once (ggml's pool spin-waits, so an *idle* server still burns cores — and it penalises whichever arm uses more threads, i.e. the variable under test) | fox 79 vs llama-server 151 t/s; measured alone, 121 vs 153 |
| comparing runs from different moments | "`FOX_CPU_ALL_VARIANTS` improves TTFT 20%" — it improves nothing |
| a build change that silently did not apply (stale `libggml-cpu-*.so` are loaded regardless of what was just compiled; a `libggml-hip.so` that fails to dlopen falls back to CPU with no error) | nearly benchmarked a CPU build against a GPU one |
| declaring a winner from overlapping ranges | the same 20% claim, retracted in `e3b447a` |

`ab_bench.sh` closes all four: it runs **exactly one arm at a time** (and
refuses to start if something is already on the port), **alternates** A/B and
B/A each round, **prints what each arm actually loaded** (via `FOX_LLAMA_LOG`)
and warns when both arms load the same thing, and returns **INCONCLUSIVE**
unless the two arms' ranges are disjoint. It also warns when an arm varies
more than 10% against *itself*, which is this machine's usual state.

```bash
./scripts/ab_bench.sh \
  --a-label before --a-cmd './target/release/fox serve --model-path M --port 8097' \
  --b-label after  --b-cmd './target/release/fox serve --model-path M --port 8097' \
  --prep-b 'cargo build --release'  \
  --url http://localhost:8097 --model llama-3.2-1b-instruct-q8_0 \
  --rounds 3 --metric ttft        # or --metric throughput
```

`--prep-a`/`--prep-b` run before each start of that arm — rebuild, swap `.so`
files, set an env var — so the arms genuinely differ. Both arms should use the
same port: only one runs at a time, and sharing the port makes that structural.

Rule of thumb from this investigation: **a single before/after pair on this
hardware is worthless**. If a change matters, it survives alternation; if it
does not survive alternation, it was drift.

## Audit of this document's claims (2026-08-02)

This investigation produced several wrong results before producing right ones,
so every quantitative claim above was re-checked and graded. The grading
criterion is the one the investigation itself arrived at: **a number is only
trustworthy if it came from a repeated, alternating comparison, or from a
mechanism that does not depend on timing at all.**

**Re-verified by fresh alternating A/B (`ab_bench.sh`, after its own bugs were
fixed):**

| claim | as committed | on re-measurement |
|---|---|---|
| fox vs `llama-server`, config-matched | 11.1% gap | **9.9%** (fox 160.6 [160.1, 161.4] vs 176.5 [174.7, 177.3]) |
| `n_threads` fix | +54% | **+50.5%** (83.2 [82.5, 84.9] → 125.2 [123.4, 125.6]) |

Both hold. Note the config-matched number originally came from a run predating
the fix to `ab_bench.sh`'s process handling; it was re-measured with a watchdog
confirming zero container overlap.

**Solid without re-measurement, because the evidence is a mechanism or a test,
not a timing:**

- `kv_unified` — decode ubatch width 1.74 → 3.90 of 4, counted from llama.cpp's
  own trace. Deterministic, and independently corroborated by the earlier
  `rocprofv3` figure of ~1.64. Plus `make e2e` 22/22.
- The concurrent-load SIGSEGV — cause read directly from a gdb backtrace;
  fix verified over 85 consecutive runs against a prior 1-in-15 failure rate.
- The golden-test fix — the bug (`all()` over an empty vec is vacuously true) is
  visible in the source; 12/12 pass after.
- Architecture coverage and the registry repair — pass/fail checks, re-runnable.

**Weak — directionally probably right, magnitude not established:**

- The concurrency-1 split of the gap (fox 52.8 vs llama-server 55.5 t/s, "~5%
  flat cost") is a single measurement per side.
- The `n_ctx` trade-off (33% TTFT vs 4.7% throughput) was measured with
  `ab_bench.sh` before its process-handling fix. Direction is consistent across
  both metrics; treat the exact percentages as provisional.
- "~4.4 ms of the gap is `GGML_NATIVE=OFF`" — already marked unconfirmed above,
  and partly contradicted by the later finding that fox's default CPU backend
  matches a hand-tuned AVX-512 variant on this host.

**Retracted:** `FOX_CPU_ALL_VARIANTS` improving TTFT ~20% (commit `5dd75b3`,
retracted in `e3b447a`) — it was drift, and an alternating A/B shows no effect.

## Where to look

| Concern | File |
|---|---|
| `AMDGPU_TARGETS` passthrough | `build.rs` (ROCm/HIP auto-detection block) |
| ROCm Docker build | `Dockerfile.rocm` |
| **Reference `llama-server`** — same vendored commit, same ggml/HIP flags | `Dockerfile.llama-server-rocm` |
| **Thread count** — never inherit ggml's 4-thread default | `src/engine/model/llama_cpp/mod.rs` (`resolve_n_threads`, `FOX_N_THREADS`) |
| Repeated/statistically-sound benchmarking (servers you already run) | `scripts/repeat_bench.sh` |
| **A/B of two builds** — owns server lifecycle, alternates, verifies what loaded, refuses to call overlapping ranges a win | `scripts/ab_bench.sh` |
| **The main fix** — dense/ascending `seq_id` allocation | `src/scheduler/mod.rs` (`seq_id_pool`, now a min-heap) |
| **The main fix** — batch emitted in ascending `seq_id` order | `src/engine/model/llama_cpp/batch.rs` (`do_decode_batch`) |
| **The closing fix** — unified KV cache, so `split_simple` runs instead of `split_equal` | `src/engine/model/llama_cpp/mod.rs` (`ctx_params.kv_unified` in `load()` and `new_context()`) |
| **Measuring ubatch widths** — forward llama.cpp's own log (pairs with `LLAMA_BATCH_DEBUG=1`) | `src/engine/model/llama_cpp/mod.rs` (`FOX_LLAMA_LOG`, next to `noop_log`) |
| **Resolved limitation** — prefix-cache hits inherit a stale seq_id (now harmless) | `src/scheduler/schedule.rs` (prefix-hit admission path, `req.kv_seq_id = hit.seq_id`), `src/scheduler/mod.rs` (`try_insert_prefix`) |
| **Ruled-out fix** — `seq_cp` can't do a partial cross-stream KV copy | `vendor/llama.cpp/src/llama-kv-cache.cpp:463` (`llama_kv_cache::seq_cp`, `is_full` assert at line 518) |
| **The sampling fix** — adaptive candidate selection | `src/engine/model/sampling.rs` (`sample_token`) |
| **The sampling fix** — skip logits copy when unneeded | `src/engine/model/mod.rs` (`needs_logits`), `src/engine/run.rs`, `src/engine/model/llama_cpp/batch.rs` |
| OpenAI vs Ollama sampling defaults (`top_k=0` vs `40`) | `src/api/shared/sampling_defaults.rs` |
| Scheduler admission order (verified correct — ~3.85/4 batch width) | `src/scheduler/schedule.rs`, `src/engine/run.rs` (`run_loop`) |
| llama.cpp's split selection and its `sequential`/ascending-seq_id requirement | `vendor/llama.cpp/src/llama-kv-cache.cpp:725`, `vendor/llama.cpp/src/llama-batch.cpp` (`split_equal`) |
| llama.cpp's async decode + deferred sync | `vendor/llama.cpp/src/llama-context.cpp` (`decode()`, `graph_compute()`, `llama_get_logits_ith()`) |

---

# Vulkan follow-up (2026-08-02): fox vs llama-server, after the 0.19 work

The measurements above are ROCm and predate 0.19's KV-reuse rework. This section
re-measures on **Vulkan** (Radeon 890M, RADV GFX1150) against `llama-server` built
from the **same vendored llama.cpp** with the **same toolchain**
(`Dockerfile.vulkan` and `Dockerfile.llama-server-vulkan`, both Ubuntu 24.04 +
glslc + SPIRV-Headers), so what is left is the serving layer rather than a version
or compiler difference. Model: `llama-3.2-1b-instruct-q8_0`, 4 concurrent clients,
16 requests × 128 tokens, one server at a time, arms alternated.

## The first number was wrong, and this document had already said why

The first run used `fox-bench`, which cannot set `top_k`. So fox ran at its OpenAI
default of `top_k = 0` while `llama-server` ran at its own default of `40` — the
exact asymmetry "The real root cause: fox's own sampling defaults" above documents.
That is not two speeds for the same work; it is two different amounts of work. It
produced **90%**, and it was reported before the asymmetry was noticed.

Re-run with the byte-identical request body posted to both:

| | fox | llama-server | fox / llama-server |
|---|---|---|---|
| `top_k = 0` | 161.1 t/s | 187.4 t/s | **86%** |
| `top_k = 40` | **175.1 t/s** | 181.9 t/s | **96%** |

Medians of 3 alternating rounds; fox's ranges at the two settings are disjoint
([160.4, 161.7] vs [173.6, 175.4]).

**So the gap is 4%, not 10%.** Six of those ten points were the benchmark comparing
two different workloads.

## What the two settings say

- **fox gains 8.7% moving from `top_k = 0` to `40`.** That is the residual cost of
  softmaxing 128,256 entries instead of 40, *after* the adaptive-candidate fix above
  already removed the full sort. The fix bounded the cost; it did not eliminate it.
- **`llama-server` *loses* ~3% at `top_k = 40`** (187.4 → 181.9). Its untruncated path
  is the faster one, which is where fox's remaining 4% lives: llama.cpp handles the
  no-truncation case better than fox does, not the truncated one.

Two candidates for that 4%, neither profiled — stated as hypotheses so nobody mistakes
them for findings:

1. `sampling.rs`'s `let mut logits = logits.to_vec()` copies the full 512 KB logit
   vector per token per request. `needs_logits` already avoids a *second* copy for the
   logprobs path, but this one is unconditional.
2. fox samples on the host; llama.cpp can sample on the backend. On unified iGPU memory
   this matters less than on a discrete GPU, but it is still a round trip per token.

Both weigh more on the `top_k = 0` path, which is consistent with where the gap sits.
**Profile before acting on either** — everything in this document that was fixed by
guessing first had to be retracted later.

## Practical reading

fox is also markedly steadier: its three rounds spanned 1.6 t/s against
`llama-server`'s 12.3. And none of 0.19's work shows up in this benchmark at all — it
is single-turn decode with a short prompt, while 0.19 improved prefill and KV reuse
(34× mean / 94× p90 on a multi-conversation working set, 3.4× on `n=4`). The honest
summary is that decode throughput was already where it was, and 0.19 did not cost any
of it.

The `top_k = 0` default remains **deliberate** (`/v1/*` mirrors OpenAI, which has no
`top_k`). It costs 8.7% for callers who do not set it. Changing the default would alter
output for every existing caller silently, so it is documented rather than changed.


---

# The workload where fox wins: concurrent burst behind a shared prompt (2026-08-02)

The section above closes with "none of 0.19's work shows up in this benchmark at all"
— single-turn decode with a short prompt cannot see prompt reuse, because there is no
prompt worth reusing. That is a statement about the benchmark, not about fox, and
leaving it there would have been half the story. This section is the other half.

`scripts/ab_shared_prefix.sh` + `scripts/bench_burst.py` measure the agent/RAG shape:
N clients arrive **together**, each carrying the same long system prompt and a
different short question. Two bursts per run, and the pair is the point — `cold` with
nothing cached, `warm` with the previous sequences now idle and holding the prefix.

## Result

3 rounds, 8 clients, 1856-token shared prompt, 64 output tokens, 4096 ctx/sequence on
both sides, Radeon 890M / Vulkan, both servers built from the same vendored llama.cpp,
one server at a time, arms alternated:

| | fox | llama-server | |
|---|---|---|---|
| **COLD** TTFT p50 | **1129 ms** | 4550 ms | **fox 4.03×** |
| ranges | [1114, 1130] | [4526, 4573] | disjoint |
| whole-burst wall | 2.65–3.01 s | 8.82–9.13 s | |
| `cached_tokens` | **12908** | 0 | |
| **WARM** TTFT p50 | **50 ms** | 190 ms | **fox 3.80×** |
| ranges | [49, 53] | [186, 192] | disjoint |
| `cached_tokens` | 14840 | 14840 | both reuse |

12908 is exactly 7 × 1844: seven of the eight arrivals copied the shared prefix from a
live sibling rather than prefilling it.

## Why llama-server cannot do this

Predicted from reading it, then confirmed by its `cached_tokens = 0`.
`get_available_slot()` skips slots where `is_processing()` is true — in **both** the
prompt-similarity pass (`server-context.cpp:1609`) and the LRU fallback (`:1652`). Its
parent/child fork path asserts the same (`:2303`). So when N requests sharing a system
prompt arrive at once, no slot is idle, nothing is inheritable, and the prompt is
prefilled N times.

fox copies from a *live* sequence: under `kv_unified`, `seq_cp` shares cells rather
than duplicating the buffer, so a request may copy from a sibling that is already
decoding. One prefill instead of eight.

## Read the warm row before quoting the cold one

The warm row is the honest floor. Both servers reuse an idle prefix — that is table
stakes, not a differentiator — and fox's 3.8× there comes from the slot table
(token-exact LCP, and parking the *generated* tokens rather than only the prompt), a
separate mechanism from the cold gap.

Neither number is the whole picture without the decode-bound control above, where fox
sits at **96%** of llama-server. A benchmark that reports only the favourable workload
is marketing. Both are in the repo; run both.

Also unchanged: the block accounting. Each request still reserves its own fox block
budget, so the ~80% VRAM saving projected before this work did **not** materialise —
the compute is shared, the admission budget is not. llama.cpp's cells *are* shared by
the metadata-only `seq_cp`, so GPU memory is not actually duplicated; what over-counts
is fox's admission arithmetic, which makes fox admit less concurrency than the hardware
would hold. Closing it needs ref-counted blocks with a CoW path that copies llama.cpp
KV, which today it does not.

## The stale-binary trap

This benchmark's first run reported the **opposite** cold result — llama-server 1.40×
ahead, fox `cached_tokens` 0 — and it was written up as refuting the hypothesis. It was
not. The fox arm was a prebuilt Vulkan bundle timestamped 31 minutes *before* the
feature commit. The benchmark faithfully measured a build that did not contain what was
being measured.

What made it convincing was that the warm row still looked right: the slot table it
depends on predated that bundle. A partially-correct result reads as a real finding in
a way that a totally broken one never would.

What broke it open was a unit test reproducing the exact arrival pattern (all clients
submitted before the first `schedule_step`,
`scheduler::tests::simultaneous_burst_behind_one_prompt_reuses_it`). It **passed** —
and the scheduler doing the right thing in the stub is what redirected the search from
the code to the binary. **If an arm shows no effect at all, check the binary's
timestamp against the commit before believing the result.**

One more sizing trap, found the same way: the first draft used a prompt 3× longer than
intended. llama-server returned `400`; fox **silently rolled the context window**, and
rolling sets `rolled_tokens`, which disables reuse. That would have read as "fox cannot
reuse prompts". The driver now reports *measured* prompt tokens and the harness warns
when they exceed the per-sequence context.

## How it scales, and what the block accounting did (and did not) change

Re-measured after the block accounting was fixed — the shared prefix charged once at
admission rather than reserved in full and handed back:

| clients | COLD TTFT p50 | WARM TTFT p50 | whole-burst wall |
|---|---|---|---|
| 8  | fox 1129 ms / ls 4514 ms — **4.00×** | fox 52 ms / ls 193 ms — 3.71× | 2.7 s / 8.8 s |
| 16 | fox 1402 ms / ls 8064 ms — **5.75×** | fox 59 ms / ls 362 ms — 6.13× | 3.8 s / 16.2 s |

All ranges disjoint. At 16 clients, `cached_tokens` is 27660 = 15 × 1844: fifteen of
sixteen arrivals copied the prefix.

**Doubling the concurrency costs fox 24% more cold TTFT (1129 → 1402 ms) and
llama-server 79% (4514 → 8064 ms).** The prefill work fox adds per extra client is one
short suffix; llama-server adds a whole prompt.

The 8-client row is unchanged from before the accounting fix (4.00× vs 4.03×), and that
is the expected result rather than a disappointment: at 8 clients the pool was never the
constraint, so a change to how blocks are *budgeted* has nothing to move. Compute
sharing and budget sharing are separate wins that show up under separate pressures —
quoting the accounting fix as the cause of a TTFT number would be attributing it to the
wrong change.

## Flag audit: was `llama-server` given its best configuration? (2026-08-03)

Published before this check, the concurrent-burst numbers had an obvious line of attack:
that `llama-server` was run near-default while fox was not. Two of its flags could
plausibly have closed the gap, so both were checked in the source and one was measured.

**`--slot-prompt-similarity` defaults to `0.10`** (`common/common.h:671`), the same value
fox defaults to. Its longest-common-prefix slot affinity was therefore *active*
throughout, which is also why its warm run reports the same `cached_tokens` fox does.
Nothing was disabled by omission.

**`--cache-reuse` defaults to `0`, disabled** (`common/common.h:620`), and the original
runs did not set it. Measured with it on, same harness, same model, 8 clients behind the
1856-token prompt:

| | cold TTFT p50 | warm TTFT p50 | cold `cached_tokens` |
|---|---|---|---|
| `--cache-reuse 0` (as published) | 4376 ms | 189 ms | 0 |
| `--cache-reuse 256` | 4367 ms | 186 ms | 0 |

Unchanged, inside the noise. The flag addresses a different situation: reusing a cache
across a *deleted gap in the middle* of a prompt via KV shifting. It cannot help here,
because the problem is not that reuse is refused — it is that at the moment eight
requests arrive together there is no idle sequence holding the prefix, and
`get_available_slot()` will not consider a busy one.

So the comparison stands with the reference server configured in its favour. That is the
claim worth making, and it is only worth making because the check was run rather than
argued.

## vLLM does run on this iGPU (2026-08-03)

Recorded because the prediction was wrong, and a wrong prediction that gets checked is
worth more than a right one that does not.

The expectation was that vLLM could not run here. This machine is a Radeon 890M —
gfx1150, RDNA 3.5, integrated — and vLLM's ROCm builds target gfx90a, gfx942 and the
discrete RDNA3 parts (gfx1100-1102). gfx1150 is not among them, so the plan was to
record "not supported on this hardware" as a scope limit.

It runs. `rocm/vllm:latest` with `HSA_OVERRIDE_GFX_VERSION=11.0.0`, which presents the
iGPU as a discrete RDNA3 card:

```
torch 2.9.0a0+git1c57644   available: True   count: 1
device name: (empty)
Available KV cache memory: 34.30 GiB
GPU KV cache size: 2,997,008 tokens
init engine (profile, create kv cache, warmup model) took 7.61 seconds
ENGINE OK
' Kaitlin and I am a 17 year old female. I have'
```

Two things to carry into any comparison rather than trip over later.

`torch.cuda.get_device_name(0)` returns an **empty string**. ROCm sees the device and
allocates against it, but does not recognise the target well enough to name it. Anything
that keys behaviour off the device name will misbehave, and it is a reminder that the
override is a workaround, not support.

The 5.22 tok/s in that run is **not** vLLM's speed and must not be quoted as such: it was
a 0.5B model under `enforce_eager=True`, which disables CUDA graphs and Inductor
compilation. That gate existed to answer "does it run", and it does. Measuring it fairly
means dropping `enforce_eager` and giving it the same tuning effort as every other engine.

Consequence for scope: the three-way comparison — fox, llama.cpp, vLLM, plus Ollama — is
achievable on this machine. It should be, with the override documented as part of vLLM's
configuration, since a reader with the same hardware needs it too.
