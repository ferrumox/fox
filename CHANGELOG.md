# Changelog

All notable changes to ferrumox are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.22.1] - 2026-08-22

### Documentation

- **The v0.22.0 release notes described `--mtp-model` as drafting badly. It does not
  draft at all.** The entry was written from the commit that added the flag and missed
  that a later commit on the same branch had superseded its numbers, so three claims went
  out wrong: that acceptance is 4-17%, that the main cause had been found and fixed, and
  that "a second desync remains". There was no first one. The head's `llama_decode`
  returns `-1` on every call, so it returns a frozen candidate set — the same three
  tokens at every step, blind to prompt and position — and the ~2.5% the server reports
  is how often that fixed list coincides with the target's output, not draft quality.
  `llama-server` reaches ~60% with the same head file. The `save_state`/`restore_state`
  pair fox wraps around `mtp_propose` is dead code: that implementation does not override
  `get_state`/`set_state`, so the dispatcher returns false, and removing it changed
  nothing. `STATUS.md` item 17 had this already and the CHANGELOG contradicted it inside
  the same release. Corrected in the 0.22.0 entry above, in `--help`, and recorded here;
  the text published under the `v0.22.0` tag cannot be recalled.

- **`src/seq.rs` claimed a guarantee its own FFI helper contradicts.** The module header
  said a bare integer is a type error "wherever a sequence is expected" and that
  `a4171eb` "is now unwriteable", while `set_batch_row` in the same change says a raw
  store into `batch.seq_id` stays expressible because no newtype reaches through a
  bindgen pointer. The honest one is `set_batch_row`'s, and the header now says so
  explicitly, naming the claim it replaces.

### Changed

- **`main` and `develop` share history again.** `main` was an orphan branch — its own
  root commit, no ancestor in common with `develop`, so `git merge-base main develop`
  returned nothing and each release snapshot was a single-parent commit built by copying
  a tree. That made `main` unmergeable and threw away the link between a published
  version and the commit it came from. Fixed by grafting, not rewriting: from this
  release the snapshot is a two-parent commit whose first parent is `main`'s tip — so
  `git log --first-parent main` still shows only the release chain — and whose second is
  the `release:` commit on the working branch. Every published tag keeps its SHA, and the
  snapshot's tree is unchanged. `CONTRIBUTING.md` carries the new recipe and the two
  checks to run before tagging.

- **A llama.cpp sequence id is now a type, not an `i32`.** llama.cpp addresses its KV
  cells by `(seq_id, pos)`, and fox passed that id around as a bare integer. Two of its
  twelve verified KV lifecycle bugs came from exactly that, both in the *sequence*
  lifecycle rather than the block pool — which is also why the typed-block refactor that
  motivated the audit was rejected. `SeqId` has two named `pub(crate)` constructors:
  `slot(i)` for the range the scheduler's slot table owns, and `dedicated(raw)` for the
  two sequences that live outside it (the embeddings sequence, and the draft model's own
  context). Sixty signatures take it, no raw `seq_id: i32` survives outside the FFI
  layer, and `Option<SeqId>` replaces the `-1` sentinel `kv_seq_id` used to carry, so
  "no sequence" is a variant rather than a magic number a comparison can miss — at a
  cost of four bytes per slot, since a plain `i32` newtype has no niche.

  **It does not make `a4171eb` impossible, and the module says so.** `llama_batch` is a
  bag of bindgen pointers and no newtype reaches through one, so a raw store into
  `batch.seq_id` stays expressible. What stops it is that `set_batch_row` is the only
  sanctioned write path and it takes a `SeqId`: writing that literal again means
  hand-rolling pointer arithmetic beside a helper that already exists, which shows up in
  a diff instead of hiding inside a comment claiming the slot is dedicated. An invisible
  literal becomes a visible act; a runtime bug does not become a compile error.

  No performance change. The reasoning, the twelve-bug classification and the decision
  rule agreed before it started are in `docs/design/kv-typed-classification.md`.

---

## [0.22.0] - 2026-08-22

### Added

- **`--n-gpu-layers`, so a model that does not fit can still use the GPU.**
  `n_gpu_layers` was hard-coded to `-1` — every layer on the device — which is correct
  on an iGPU, where unified memory means all layers always fit, and wrong on a discrete
  card with a fixed VRAM budget. A model one gigabyte too large was refused outright
  instead of running the layers that do fit on the GPU and the rest on the CPU, which
  `llama-server` has always done with `-ngl`. Measured on an RTX 5060 Ti 16 GB:
  Qwen3.8-27B-Q4_K_M needs 17402 MiB of weights against 15712 MiB free and would not
  load at all; with `--n-gpu-layers` it serves. Only activations cross PCIe per token, a
  few KB, so even a narrow link does not penalise the split — the cost is that CPU-side
  layers run at system-RAM bandwidth. Verified on Vulkan and CPU as well as CUDA: `0`
  keeps all 29 layers of Qwen2.5-7B on the CPU, `16` splits 16/29 onto a Radeon 890M.
  The draft model deliberately keeps `-1`: it is only worth having if it is much faster
  than the target, so offloading part of it would defeat speculation rather than
  economise on VRAM.

- **`--mtp-model`: MTP speculative decoding, EXPERIMENTAL and opt-in.** Qwen3.5/3.8 ship
  a trained NextN head as a separate small GGUF, and llama.cpp already drives it in
  `common/speculative.cpp`, so fox wraps that driver rather than porting its 444 lines of
  hidden-state carryover — 15 s of extra compile time and no new dependencies. Four entry
  points live in `src/llama-ext.h`, which declares itself staging and exports C++-mangled
  symbols, so `csrc/mtp_shim.cpp` wraps them in `extern "C"`: an upstream signature
  change then fails compiling that one file, with the real signature in the error,
  instead of surfacing as an unresolved mangled symbol at link time. `FOX_NO_MTP=1` drops
  the whole thing.

  **It does not work, and this entry corrects the one that shipped in the v0.22.0 release
  notes.** That text said acceptance was 4-17% and that the main cause had been found and
  fixed with a second desync remaining. All three claims are wrong: they were taken from
  the commit that added the flag, and a later commit on the same branch superseded them.

  What is true is that the head has **never drafted**. Its `llama_decode` returns `-1` on
  every call, so it returns a frozen, context-blind candidate set — asked to count from
  one to twenty it proposes the same three tokens at every step, identical across
  positions and prompts. The ~2.5% the server reports is therefore not draft quality at
  all: it is how often a fixed list happens to coincide with the target's output.
  `llama-server` reaches ~60% with the very same head file, so neither the head nor its
  quantization is at fault. The `save_state`/`restore_state` pair fox wraps around
  `mtp_propose` is dead code — that implementation does not override
  `get_state`/`set_state`, so the dispatcher returns false — and removing it left
  acceptance unchanged. `STATUS.md` item 17 carries the mechanism and the two hypotheses
  already eliminated.

  It is deliberately **not** documented under `docs/cli/`, which is what makes it exempt
  from the Tier 1 flag promise in `COMPATIBILITY.md`: it may change shape or disappear in
  any release while the desync is open. `--help` says the same.

### Fixed

- **One NaN logit made greedy sampling pick an arbitrary token.** `sample_greedy`
  compared with `partial_cmp(..).unwrap_or(Ordering::Equal)` and fed that to
  `max_by`, which replaces its accumulator whenever the comparison is not `Greater`.
  An incomparable NaN therefore reported `Equal` and always displaced the running
  maximum: `[5.0, NaN]` returned the NaN, and `[5.0, NaN, 1.0]` returned `2` —
  neither the maximum nor the NaN. The NaN did not merely win, it wiped the best
  candidate so far and everything after it competed from scratch, so a single NaN
  anywhere in a 128K-wide vector was enough to emit an arbitrary token on the
  `temperature <= 0` path, which is exactly the path callers use when they want
  determinism. NaN logits are reachable through fp16 overflow, a bad quantisation or
  corrupted KV. Now an explicit scan that skips NaN and warns with the count; the tie
  rule is preserved exactly (`>=` keeps the last maximum, as `max_by` did), and an
  all-NaN vector returns 0 like the empty case.

- **The sampler was never compiled, let alone tested, in CI.** `sampling` was gated
  behind `#[cfg(not(fox_stub))]` and `make ci` runs everything with
  `FOX_SKIP_LLAMA=1`, so the module that decides which token every request emits was
  excluded from every clippy and test run CI has ever done — which is how the NaN bug
  above survived. The gate followed the module's only caller (`llama_cpp::batch`);
  the module itself depends on `std::cmp::Ordering` and `rand` and nothing else.
  Ungating it costs a `dead_code` allow in stub builds and buys 33 existing tests in
  CI. The first clippy pass over this code flagged `!(l > threshold)` in
  `select_top_n`; that negation is deliberate NaN handling and is now documented
  rather than "fixed" — `l <= threshold` is false for NaN, which would admit one into
  the candidate pool.

- **Prompt reuse corrupted hybrid/recurrent models (Qwen3.5, Qwen3.8, Qwen3-Next,
  Falcon-H1, Jamba).** After the first request, a repeated prompt was served from a
  sequence that had never actually been rewound to the divergence point, and replies
  came back truncated or as a bare EOS. Measured on Qwen3.8-27B: six identical
  requests returned 64, 1, 1, 1, 1, 64 tokens.

  The reuse decision trusted `Model::trim_sequence`'s return value to refuse an
  impossible rollback. It does not: `llama_memory_recurrent::seq_rm` only range-checks
  the distance while the sequence's tail cell is live, and after an earlier full clear
  invalidates it, a partial rollback skips the check and reports success without
  rewinding anything.

  Reuse is now bounded before it is accepted, via `Model::rollback_budget()` —
  `None` for attention caches, `Some(n_rs_seq)` for recurrent/hybrid ones. Two limits,
  because they differ: a slot decoding in place keeps its per-token state snapshots and
  gets the full budget, while a checkpoint restored with `state_seq_load` gets zero
  (the blob carries the recurrent state but not the snapshot history a rewind indexes
  into). Attention caches are unaffected — the check is a strict no-op for them.

  Multi-turn reuse is preserved: a 173-token third turn still cached 151 tokens.
  `--kv-reuse false`, the workaround, is no longer needed.

- **`top_p: 1.0` cost 4.5x in the sampler.** An unrestricted `top_p` truncates nothing,
  but the adaptive candidate pool tested its requirement as `covered >= top_p` — and a
  float sum of probabilities never reaches exactly 1.0. So the pool grew 64 → 256 → …
  through the whole vocabulary, paying a selection pass and an allocation per step, and
  the result was then sorted in full: 248,320 entries per token on Qwen3.8-27B.

  Measured on that model, decode: `top_p: 1.0` **5.6 → 25.2 tok/s**; `min_p: 0.05` with
  `top_p: 1.0` (the worst case, which also hit the full sort) **5.2 → 25.7 tok/s**.
  Unaffected controls stay put: handler defaults 25.0, `top_p: 0.95` 25.6.

  Three changes, all semantics-preserving except where noted: the pool goes straight to
  the whole vocabulary when nothing truncates rather than growing into it; `top_p >= 1.0`
  no longer holds the growth loop open, so `min_p` alone sizes the pool; and the
  descending sort is skipped when no truncation step will consume its ordering.

  Behaviour note: with a fixed `seed` and no truncation active, a seed no longer maps to
  the same token it did before — the draw now walks the pool in vocabulary order. The
  distribution is identical and seeded runs remain reproducible.

  This is why `fox bench` reported ~6 tok/s: it hardcodes `top_p: 1.0` and `top_k: 0`.
  It now reports 24.9, against Ollama's 26.6 and llama-bench's 27.35 on the same model.

- `golden.rs` did not compile: both `LlamaCppModel::load` call sites were missing the
  `split_mode` argument. Pre-existing on this branch and invisible to `make ci`'s
  stub-built jobs — exactly the breakage `check-real` exists to catch, which is only
  useful if it is actually run.

- **Tool calls already in the conversation were replayed in a syntax no model was
  trained on.** A past call was flattened back into the prompt as
  `[tool_call: name({...})]`. Models read their own history and imitate what they find
  there, so from the second round trip onwards the model stopped emitting `<tool_call>`
  blocks and answered with that literal text; the parser then had nothing to parse and
  the client got prose where it expected a tool call. Past calls now render in whatever
  format the model's template declares — `<tool_call>` blocks for Hermes/Qwen,
  `[TOOL_CALLS][…]` for Mistral, the previous generic text where the model has no native
  format, which is the right shape there because it is what fox's own injected tool
  listing asks for. The Hermes rendering is byte-for-byte what the Qwen3 template emits
  for the same message, so a replayed turn is indistinguishable from a fresh one. Found
  by running a three-step agent: the first two calls worked, the third came back as text.

- **Native tool-use templates never rendered, so `tools` silently vanished from the
  prompt.** `json` is not one of minijinja's default features, so `tojson` was unknown to
  the environment fox builds chat templates in. Every native tool-use template — Qwen,
  Hermes, Mistral — lists its tools with `{{ tool | tojson }}`, so the render failed,
  `render_chat_jinja` returned `None`, and the caller fell back to llama.cpp's built-in
  format, which drops `tools` entirely. The failure was invisible: a request produced a
  prompt of exactly the same length whether it carried one tool or six, and
  `message.tool_calls` came back null with `finish_reason: "stop"`, with nothing in the
  logs to say why. Measured with Qwen3-8B-Q4_K_M, `prompt_tokens` for the same request
  went from 28 → 46 → 46 to 28 → 147 → 465 as the tool description grew. Both failure
  paths now warn and say what the consequence is, and template handling moved to
  `engine::model::chat_template`, which carries no llama.cpp dependency and so is tested
  in CI.

- **The prompt cache picked its checkpoint by insertion order.** N clients behind one
  shared system prompt leave N checkpoints that all cover exactly that prefix and differ
  only in the question after it. They tie on match length, so `take_best`'s `max_by_key`
  kept whichever was stored first — and they are not interchangeable: whatever the chosen
  entry holds beyond the match must be rolled back after restoring, and on a
  hybrid/recurrent cache that rollback is refused past `--rs-rollback` snapshots, turning
  the restore into a full re-prefill. The entry with the least dead tail now wins, with a
  third tie-break on index so a full tie resolves the same way every run. On Qwen3.8-27B
  with 8 clients over 5 identical runs, `cached_tokens` was 1470 or 1260 and refused
  trims 2 or 4 depending on the run; both are now identical in 5/5.

- **Recurrent and hybrid models reserved gigabytes of sequence state nobody used.**
  `n_seq = max_batch_size.max(4) + 1` is cheap insurance on a standard model, where a
  spare sequence costs a few MiB of KV. It is ruinous where every sequence carries a full
  fixed-size recurrent state, which is reserved whether or not anything uses it. On
  Qwen3.8-27B that is 748 MiB per sequence, so `--max-batch-size 1` and `--max-batch-size
  4` reserved the same ~3.7 GB — and that reservation was the memory standing between fox
  and eleven more layers of GPU residency on a 16 GB card. Recurrent state 3740 → 1496
  MiB, GPU layers 40 → 51, throughput 6.5 → 8.6 t/s: arms alternated one server at a
  time, disjoint ranges, with Ollama held as a drift control on the same model (8.7 → 8.8
  across the two runs). Standard models keep the floor and the dense path is unchanged —
  Qwen2.5-7B measures 214.8 t/s against 216.0/218.7 before, inside noise.

- **`golden.rs` did not compile.** Both call sites were missing argument #10 of
  `LlamaCppModel::load`, so neither `cargo check --all-targets` nor `make golden` could
  run. Invisible to every job that sets `FOX_SKIP_LLAMA=1`, which is all of `make ci`
  except `check-real` — exactly the breakage that target's own comment describes, and it
  only helps if it is actually run.

### Testing

- **The benchmark harness answers the question it claims to answer, and its
  environment gate stopped crying wolf.** `ab_bench.sh` fingerprinted which backend `.so`
  got dlopen-ed, which is not the question: `libggml-cuda.so` loads even when
  `CUDA_VISIBLE_DEVICES` is empty and contributes no device, so a CUDA arm and a Vulkan
  arm at 53 vs 216 t/s fingerprinted identically. It now reports which devices llama.cpp
  actually placed layers on. `check_bench_env.sh` used `pgrep -f`, which matches the
  invoking shell's own command line, and reported fox and ollama as competitors when
  nothing was running — three times in one session; it uses `pgrep -x` now. The gate also
  checks the governor, what the CPU reaches *under load* (an idle reading says nothing on
  a powersave laptop), free RAM, and anything else holding the GPU, and `--sample` emits
  one line to record with every measurement so a number always travels with the state it
  was taken in.

- **Ollama's `delta.reasoning` counts toward TTFT.** Ollama spells the reasoning delta
  `reasoning`, not `reasoning_content`, and sends `content: ""` alongside it — falsy, so
  it did not rescue the lookup. Without it the first visible token was never seen and TTFT
  measured the whole request: Qwen3.8-27B multi-turn read as 46 s per turn for Ollama
  against 0.7 s for fox, a 4× "win" that was really prefill plus 96 decoded tokens with
  no token ever counted. Any new engine goes in that list before its numbers are quoted.

- e2e check 17: an identical prompt ×6 with EOG left **armed**. Check 1 covers the same
  lifecycle but pins `min_tokens`, which suppresses EOG — and an immediate EOG is the
  symptom, so it could never have caught this. The check asserts replies reach
  `max_tokens` rather than merely exceeding one token; verified against the pre-fix
  binary, which returned 24, 8, 3, 24, 1, 24 and would have passed a ">1 token" bar.
- `scheduler::tests::{out_of_budget,in_budget,unbounded}_rollback_*`: deterministic
  cover for the budget arithmetic, which the e2e cannot provide on the dense model the
  suite normally runs.

---

## [0.21.0] - 2026-08-08

Three changes about being able to tell what the server is doing: a number for the
work its scheduler avoids, a document for what its interface promises, and a
`model` label so its metrics say which model an observation belongs to. The last
one renames every metric, which is why this is a minor rather than a patch.

And then a fourth, found by pointing a client at a server with real models on it:
the model listing endpoints were unusable on any machine with more than a few
gigabytes of GGUFs. Fixing that changes what `digest` means, which is Tier 1, so
it lands here rather than in a patch.

### Changed

- **BREAKING (Tier 2): the Prometheus metrics move from `ferrumox_*` to `fox_*`.**
  The binary, the CLI, the docs and everything a user types say `fox`; the metrics
  endpoint was the only place that said `ferrumox`. Renaming breaks every existing
  dashboard, which is exactly why it happens now: after 1.0 the prefix is frozen and the
  inconsistency would be permanent.

  Migration: `s/ferrumox_/fox_/` in dashboards, alerts and recording rules. All thirteen
  names change prefix only — type and meaning are identical.

  Per [`COMPATIBILITY.md`](COMPATIBILITY.md) this is Tier 2: observable, changed on a
  minor bump with a CHANGELOG entry and no promised deprecation window.

- **Every metric now carries a `model` label.** Fox serves several models at once with
  `--max-models`, and until now nothing on `/metrics` said which one was responsible: a
  saturated KV cache, a deep queue and a bad p99 all looked like properties of the server
  rather than of one model inside it.

  The label **could not be added without a cap**. Model names are whatever the client asks
  for — `fox pull` accepts arbitrary HuggingFace repos — so the label set is influenced
  from outside the server, and an unbounded one turns `/metrics` into a memory leak that
  every scrape then has to serialise. The cap is 32 distinct values per process; past that
  everything collapses into `model="<other>"`, with a warning emitted once per process.
  Serving is never degraded by this.

  The cap counts models **ever seen**, not loaded at once: a load/evict/load cycle reuses
  its slot instead of consuming a new one, so nobody can walk the limit upward by churning
  models.

  Evicting a model retires its series. Counters could have been left alone — a monotonic
  total that stops advancing is still true — but the gauges could not:
  `fox_kv_cache_usage_ratio` for an evicted model would sit at its last value forever, and
  a dashboard would go on reporting a full KV cache for a model that no longer holds a
  single block.

  `scripts/e2e_smoke.py` read two of these metrics by line prefix and now sums the series
  instead of keeping the last, which with more than one model loaded would have reported a
  single model's drafting.

  A side effect of labelling, checked against a real server: **a metric no longer appears
  on `/metrics` until its first observation.** Previously, unlabelled, all thirteen were
  registered at startup and always emitted at zero. Now a series exists once something
  touches it, so a freshly started server shows only the three gauges the engine loop
  refreshes, and `fox_requests_total` does not appear until the first request finishes.
  This is normal Prometheus behaviour for labelled metrics, but a panel that assumed
  "always present, possibly zero" now gets an empty result: use `or vector(0)` in those
  queries.

- **BREAKING (Tier 1): the `digest` in `/api/tags`, `/api/ps` and `/api/show` is derived
  from the model file's name, size and mtime, not from its contents.** It is still
  `sha256:<hex>` and still changes whenever the file is replaced, but it is now an opaque
  identifier rather than a content hash.

  This is the fix for the hang below, not a cosmetic change: a digest that is a content
  hash cannot be produced without reading every byte of the file, and there is nowhere on
  a listing request to put that work. Ollama can report a real hash because its blobs are
  content-addressed and hashed once at pull; fox stores plain GGUF files in a directory
  that users also drop models into by hand.

  Nothing in fox resolves a model by digest — it identifies, it does not address — and
  `/api/pull` already emitted `sha256:<filename>` rather than a content hash, so no fox
  client could have been verifying one. Per [`COMPATIBILITY.md`](COMPATIBILITY.md),
  changing a field's meaning is Tier 1 and belongs on a minor bump with an entry saying
  so. This is that entry.

### Fixed

- **`GET /api/tags` no longer hangs with a core pinned at 100%.** It computed the SHA-256
  of every `.gguf` in the models directory before it could answer: measured on a 27 GB
  directory, 51.6 s for the first call. `/api/ps` and `/api/show` did the same.

  What turned a slow endpoint into an apparently dead server is that the digest cache was
  only written once a hash *finished*, and identical work in flight was never shared. Each
  retry — a `curl` re-run, an Open WebUI refresh, a page reload — started a full re-hash of
  the whole directory on another blocking thread. Retrying, the natural response to no
  response, is what saturated the CPU. `GET /` stayed instant throughout, because it
  touches no disk, which made the server look up rather than stuck.

  The same directory now answers in 22 ms, and eight concurrent requests complete in 30 ms
  total. `/api/ps` additionally re-read the models directory once per resident model; it
  now reads it once.

- **`GET /health` no longer loads a model to answer.** It called `get_or_load`, so the
  liveness probe blocked for as long as a multi-gigabyte load took — `curl -m 3` against
  a server whose model was not resident simply timed out — and it could *cause* a load,
  which under the default `--max-models 1` evicts whatever is serving traffic. An
  orchestrator polling `/health` during startup gets a timeout and restarts the process
  before it ever finishes loading: the probe becomes the outage. Found while verifying an
  unrelated fix on a server holding a 10 GB model.

  It now reports residency instead of establishing it, answering in 6 ms, and does not
  count as a use — a probe that refreshed the LRU would keep a model resident forever and
  `--keep-alive-secs` would never fire. The response gains `model_loaded`, because
  otherwise "not loaded yet" and "loaded and idle" are the same body; that state was
  nearly unreachable while the handler loaded on demand and is now the normal one before
  the first request.

- **Diffusion models are refused at load instead of served as gibberish.** LLaDA, Dream,
  RND1 and the rest do not generate left to right — they unmask a sequence over a fixed
  number of steps, which is why llama.cpp ships a separate `diffusion` tool for them.
  fox's decode loop is autoregressive, and loading one anyway did not fail: it produced
  replies with mask tokens (`<|mask_start|>`) embedded in them, fragments out of order,
  duplicated spans and truncation. Reported as an output-formatting bug, which is exactly
  what it looks like from the client side. `llama_model_is_diffusion()` is now checked
  after load and the model is rejected with an explanation.

- **An unrecognised `--type-kv` or `--split-mode` value is an error, not a shrug.** Both
  parsers answered anything they did not recognise with the default and no message, so
  `type_kv = "turbo3"` in `config.toml` quantised nothing, said nothing, and left the
  operator believing the setting had applied. A config file has no completion and no type
  checking; this warning was the only feedback available and it was not being given.
  Rejected at startup now, before anything loads, naming the accepted values.

- **A missing GPU dependency no longer stops fox compiling at all.** `build.rs` enabled
  the Vulkan backend on the strength of any single signal — `VULKAN_SDK` set, or `glslc`
  on `PATH`, or `vulkan.h` present. ggml-vulkan then opens with
  `find_package(Vulkan COMPONENTS glslc REQUIRED)` and `find_package(SPIRV-Headers CONFIG
  REQUIRED)`, and a missing one of those is a fatal CMake error rather than a fallback.
  So detecting *half* a toolchain did not produce a CPU build, it failed the whole cargo
  build — and with no way to turn Vulkan off, the user could not build fox at all. Reported
  from Windows with a LunarG SDK 1.3.246, which ships `glslc` and the loader but no
  `SPIRV-HeadersConfig.cmake`.

  All three pieces are now checked before the backend is switched on, on Linux and
  Windows alike, and a partial toolchain produces a CPU build plus a warning naming what
  is missing and how to install it. `FOX_NO_VULKAN=1` forces it off, `FOX_FORCE_VULKAN=1`
  forces it on — the second doubling as the way to re-run the check, since installing a
  package does not invalidate a build script.

  `GGML_VULKAN=OFF` is now passed explicitly instead of being left unset. CMake caches the
  switch in `target/`, so once any build had configured with it ON, every later build in
  that tree inherited ON regardless — someone who hit this and then fixed their toolchain
  would have kept failing, with `cargo clean` and a full llama.cpp rebuild as the apparent
  only cure.

- **Releases are published with notes.** `softprops/action-gh-release` was only ever
  handed files, so every release page was a bare list of six assets and nothing else.
  Two of those assets are tarballs differing by a `-vulkan` suffix, which meant the page
  never said that a GPU build existed or which file to take — and the one *without* a
  suffix reads as the default. It cost someone a bug report: `libggml-vulkan.so` looked
  missing from the release, when it was in the other tarball all along.

  The body is now this version's CHANGELOG section (`scripts/release_notes.py`) plus a
  table of which download is which, how to verify it, and the note that `fox probe`
  reports the backend actually in use. Written by a job that runs after both builds, so
  the two matrix jobs cannot race over it.

  Found alongside it: outside a tag push, `github.ref_name` is a *branch*, so the manual
  `workflow_dispatch` re-run — which exists because GitHub silently fires no workflow when
  more than three tags arrive at once — built tarballs named after a branch and uploaded
  them to the wrong ref. The recovery path was broken in the situation it was written for.
  Both jobs now resolve the tag once and the build checks out that tag.

- **`modified_at` reported the wrong date.** The RFC 3339 formatter computed the calendar
  by approximation — `year = 1970 + days/365`, `month = day_of_year/30 + 1`, 30-day months
  — so it ignored leap years and drifted within every year: a file touched on 2025-08-04
  was reported as 2025-08-20, and the error grows. It now uses an exact civil-from-days
  conversion, checked against the real calendar day by day across 200 years. Affects
  `modified_at` in `/api/tags` and `general.modified_at` in `/api/show`.

### Added

- **`COMPATIBILITY.md`** — what fox promises not to break, and what it does not. Fox is a
  drop-in replacement for Ollama and an OpenAI-compatible server, so its interface is
  consumed by Open WebUI, Continue.dev, the `ollama` CLI and the `openai` SDKs: software
  fox does not control, which breaks silently and remotely when a response shape changes.
  The CHANGELOG records what happened; the document of what is promised was missing.

  The axis is not public versus private but **who breaks**. Tier 1 (contract): the 29 HTTP
  routes, the user-facing subcommands, the documented flags, `config.toml`, `aliases.toml`,
  the model store, and the release tarball layout — `fox` is linked `RPATH=$ORIGIN`, so
  moving the `.so` files breaks every existing install, and that is Tier 1 even though it
  is not code. Tier 2 (observable): metrics, `/slots`, `/props`, the body of `/health`, and
  the diagnostic subcommands. Tier 3 (no commitment): undocumented environment variables,
  the crate — which is not published to crates.io — the pinned llama.cpp commit, and
  `perf-budgets.json`.

  It states what does **not** count as breaking, which matters just as much: additions, and
  any change in speed, batching or scheduling decisions with identical output. The
  performance budgets watch those numbers; they do not promise them.

  Three findings from reading the code to write it, all on the 1.0 checklist:
  `/api/version` reports fox's own version rather than an Ollama one, so a client gating on
  a version comparison compares against the wrong number; the metrics prefix was
  `ferrumox_` while everything user-facing said `fox`; and no metric said which model was
  responsible even though fox serves several at once. The last two are resolved above, in
  this same version.

  `scripts/check_docs_flags.py` now covers `COMPATIBILITY.md`: the flags and variables it
  names are checked against `fox --help` and the source. A policy promising stability for a
  misspelled flag promises nothing. The cost is that a deprecation example can no longer
  invent a flag; that is noted in the script.

- **Performance budgets for the scheduler** — `perf-budgets.json` at the repository root,
  generated and enforced by `scheduler::budgets`. Five scenarios (8- and 16-client bursts
  behind a shared prompt, admission pressure, a 3-turn chat, and a control of 4 disjoint
  prompts), each in two arms: fox as shipped, and with `--kv-reuse` off.

  These are **counts, not times**: prompt tokens handed to the model, peak KV blocks,
  prefix hits and admitted requests, measured by driving `schedule_step` with no model
  behind it. Deterministic on any machine, so the check is exact equality and an
  *improvement* fails as loudly as a regression — the number belongs in the commit that
  earned it. A millisecond gate on a shared runner would flake, and a check that flakes
  gets ignored.

  Runs inside `cargo test --all`, so it is already in `make ci` and in CI without touching
  a workflow. To re-record after an intended change: `make budgets`.

  What it is **not**: a regression net the existing tests lacked. Three regressions were
  injected into the copy-from-a-live-sequence path — disabling it outright, removing the
  donor deferral, and copying only half the shared prefix while leaving `prefix_hits`
  intact — and the existing suite caught all three. What the file adds is the aggregate
  magnitude, which no unit test expresses, at a scale (8 and 16 requests) none of them
  reaches.

  Building it produced a measurement that did not exist before: with a 512-block pool the
  no-reuse arm admits 14 of 16 requests and the reuse arm admits 16 of 16. That is the
  README's "sharing widens concurrency" claim, until now unmeasured anywhere. It is
  pinned by `shared_prefix_admission_pressure_16_clients`.

---

## [0.20.5] - 2026-08-04

`fox run` has been sending the model malformed prompts since it was written. Fixing
that, and turning the chat session into one you can actually use.

### Fixed

- **`fox run` tokenised its prompts as raw text, so the model never saw turn markers.**
  The command rendered the chat template and then passed the result to `tokenize()`,
  which is the *raw text* tokenizer: `add_special = true`, `parse_special = false`. The
  template's `<start_of_turn>` / `<end_of_turn>` went in as literal text instead of the
  control tokens they are, and a second BOS was prepended on top of the one the template
  already emits. The HTTP handlers have always used `build_prompt_tokens`, which picks
  the flags to match how the prompt was rendered; the CLI reimplemented the same two
  steps by hand and got them wrong.

  The visible symptom was an empty reply. Seeing a conversation with no turn structure,
  the model answered often enough by writing a literal `<start_of_turn>model` — the
  token ids it generated (`236820`, `3041`, `236779`) are exactly the ones the broken
  tokenisation produces for that string. The output filter recognised the pattern,
  correctly held the text back, and reported `StopSequence`, so nothing reached the
  screen. Both the filter and the engine were doing their job.

  On gemma-3-1b, four rounds of each scenario: empty replies went from 1-of-3 (no
  cancellation involved) and 3-of-4 (after a cancelled turn) to **0-of-4 in both**. The
  same prompt now tokenises to 27 tokens where it took 38.

  This was never only about empty replies — every `fox run` session was degraded, and
  the empty ones were just where it became impossible to miss.

- **The same mistake in `fox bench`, `fox bench-kv`, `fox bench-spec` and
  `fox bench-prefill`** — every CLI command that builds a chat prompt had it. A sweep of
  the whole tree found the fourth; the first three were found by reading the code around
  the first. Their reported prompt lengths change as a result, so numbers from these
  commands are **not comparable across this version**. The four-engine benchmark is
  unaffected: it drives the server over HTTP.

  The one remaining `apply_chat_template` + `tokenize` pair is the `Model` trait's own
  default `build_prompt_tokens`, which is correct: a backend with no Jinja template has
  no special tokens to parse, and `LlamaCppModel` overrides it.

- **An empty reply was reported as a full context window, and the conversation was
  wiped.** The diagnosis was a guess and usually a wrong one — the window sat at 400 of
  32768. It now checks the context before claiming that, says what actually happened
  otherwise, and keeps the history either way.

### Added

- **Line editing and history in the chat session** (rustyline). The terminal was left in
  canonical mode, where the line discipline does not interpret arrow keys: pressing Up
  to recall the previous message typed a literal `^[[A`, and a typo could only be fixed
  by backspacing to it. History now persists between sessions in
  `~/.config/ferrumox/chat_history`, and bracketed paste means a multi-line paste is no
  longer submitted a line at a time.

- **Ctrl+C stops the reply instead of killing the session.** It only becomes a signal
  during generation — while the editor is reading, the terminal is raw with `ISIG` off
  and rustyline sees the byte — so the two cases do not collide. Dropping the token
  receiver is what cancels the work: the engine's `send()` fails, and it preempts the
  request and frees the KV blocks. What was generated is kept, trimmed back to the last
  sentence or word so the history does not end mid-token.

- **`/help` and `/clear`.** The banner named two of the five commands, which left
  `/clear` with no way to be discovered.

- **`scripts/check_prompt_tokenization.py`, wired into `make ci` and CI.** It fails when
  a function renders a chat template and then calls `tokenize()` on the result, which is
  the shape of the bug above. Verified both ways: clean on the fixed tree, and it does
  report the defect when it is deliberately put back. The mistake survived for as long as
  it did because it is invisible to every test fox has — `make e2e` passed 22 of 22 with
  it present, since those tests go over HTTP.

### Changed

- `fox run` is described as what it does: *"Chat with a model in the terminal, or answer
  one prompt and exit"*. It has opened an interactive session since it was written, and
  the help said only "single-shot inference".

---

## [0.20.4] - 2026-08-04

The binary published in 0.20.3 did not start. This fixes the packaging and makes a
release prove it runs before it is published.

### Fixed

- **The release tarball was missing `libmtmd.so.0`, so the installed binary died on
  startup:**

  ```
  error while loading shared libraries: libmtmd.so.0: cannot open shared object file
  ```

  The workflow bundled `libggml*` and `libllama*` by name. `libmtmd` arrived with vision
  support in 0.17 and was never added, and no published release had contained the vision
  code until 0.20.3 — so the omission only became fatal when it shipped.
  `Dockerfile.vulkan` had always used `lib*.so*`; the release workflow had drifted from
  it. Now it globs the same way.

- **A release now has to run before it is published.** After packaging, the workflow
  unpacks its own tarball somewhere else and executes `fox --version`. Packaging had
  only ever been verified by reading the glob, which is how a release that cannot start
  reached users. This was found by installing 0.20.3 with its own advertised one-liner
  and running the result — the check now does that automatically.

---

## [0.20.3] - 2026-08-04

The advertised way to install fox did not work. This release makes it work, and is the
first whose tag actually publishes an installer.

### Fixed

- **`curl … /releases/latest/download/install.sh | sh` returned 404.** The release
  workflow uploaded the tarball and its checksum and nothing else, so the one-liner
  printed in the README since the beginning could never have worked. `install.sh` and
  `install.ps1` now ship as assets of every release.
- **The Vulkan build was unreachable from the installer.** Each release carries two
  tarballs — CPU and Vulkan — and `install.sh` did not contain the word "vulkan": it
  always asked for the plain name. The variant that matters on the AMD/Intel iGPUs fox
  targets could not be installed by the documented method. It now detects `/dev/dri` and
  picks the Vulkan build, with `--vulkan` / `--cpu` to override.
- **The installer offered platforms nobody builds.** macOS and aarch64 were mapped to
  filenames no workflow produces, so those users got a bare 404 from `curl`. It now says
  which platforms are published and how to build for the rest. `install.ps1` targeted
  `x86_64-pc-windows-msvc`, which is not built at all — it now inspects the release's
  assets, explains WSL2 and building from source, and will start working by itself the
  day a Windows build is published.
- **The published checksum was never verified.** Every tarball ships a `.sha256` beside
  it and nothing read it. It is checked now, and a mismatch aborts.
- **Documentation advertised five direct downloads that do not exist** —
  `fox-linux-x86_64`, `fox-macos-arm64`, `fox-macos-x86_64`, `fox-windows-x86_64.exe`
  and its `.zip`. README, `installation.md`, `quickstart.md` and `index.md` now describe
  the assets a release really publishes, under their real names.

### Added

- **`make soak`** — sustained mixed traffic against a real server, ending in a verdict.
  `make e2e` is 22 checks over two minutes and everything below it starts from a fresh
  process, so a leak, a KV pool that never returns, or latency drift are all invisible.
  Traffic mixes multi-turn conversations, one-off prompts, and clients that hang up
  mid-stream — that last shape is where the three prefix-cache lifecycle bugs of 0.15.1
  lived, and no request-shaped test can see it.

  It checks that the KV floor does **not grow between two load cycles**, rather than
  that it returns to zero. A drained fox legitimately holds KV: parked sequences keep
  theirs on purpose, so a single measurement cannot tell parking from a leak. Measured
  across two runs: 1579 and 1823 requests, zero failures, RSS +0.6%, KV floor 0.0234 →
  0.0164.
- Tests for the prefill-checkpoint plumbing shipped untested in 0.20.2: that a
  checkpoint can only ever capture the prompt boundary, and that a checkpoint tying the
  live slot still wins when the KV cannot roll back.

---

## [0.20.2] - 2026-08-03

Hybrid and recurrent models reuse prompts for the first time. Qwen3.5 — which
`registry.json` recommends — went from `cached_tokens: 0` for an entire conversation to
reusing everything up to the last user message.

### Fixed

- **A conversation on a hybrid model re-read itself from scratch on every turn.**
  Measured on Qwen3.5-9B before this: 20 slot hits, 20 **refused** trims,
  `cached_tokens` 0 throughout. The prefix was reusable; the *route* to it was not. fox
  reached a past position by trimming back from where generation stopped, and that
  rollback spans the whole generated reply — which recurrent state refuses beyond
  `--rs-rollback` snapshots, and the window that would cover it costs ~453 MB per
  snapshot.

  Restoring a serialised state has no such limit: it writes the sequence outright. fox
  already knew how to serialise a sequence, but only when evicting a slot — by which
  time the blob holds prompt *and* reply, reproducing the problem it would solve.
  Checkpointing at the **end of prefill** makes the blob cover exactly the prompt, so
  the next turn restores it and trims two tokens instead of sixty-six.

  A second condition had to go with it: the host-RAM cache was consulted only when it
  **strictly beat** the live slot. The two tie at 155 tokens, so a usable checkpoint sat
  in RAM unread while the engine took a slot whose offer it could not honour. Where the
  KV cannot roll back, a tie now goes to the cache.

  Measured after, 3 rounds, 4 conversations × 6 turns, Qwen3.5-9B: later-turn TTFT
  **1144 → 750 ms**, `cached_tokens` **0 → 257**, refused trims **20 → 2**. In the same
  run `llama-server` sits flat at 3200 ms (1.1× from turn 0 to the rest, against fox's
  3.7×), so fox is **4.3× ahead** where before it had no reuse at all.

### Changed

- **`--cache-ram` is no longer off by default for models whose KV cannot be rolled
  back.** They get 2048 MB of host RAM implicitly, logged at startup with the reason,
  and an explicit `--cache-ram` always wins. For these architectures it is not a tuning
  option but the only route to prompt reuse. Dense models are untouched — they reach the
  same prefix by trimming, for free.

### Known limits of this fix

Stated because they are measured, not guessed:

- **Turn 0 costs ~500 ms more** — serialising the state, ~53 MB per sequence on this
  model. Repaid on the following turn and every turn after.
- **The advantage grows with conversation length; there is no crossover up to 5151
  prompt tokens.** Later-turn TTFT stays flat at 400-600 ms whatever the history size,
  because the restore cost barely moves while the prefill avoided scales with the
  prompt:

  | prompt | checkpoint | turn 0 | later turns | gain |
  |---|---|---|---|---|
  | 159 tok | 58 MB | 1228 ms | 586 ms | 2.1× |
  | 991 | 85 MB | 5325 ms | 396 ms | 13.4× |
  | 2591 | 135 MB | 14821 ms | 466 ms | 31.8× |
  | 5151 | 215 MB | 28856 ms | 582 ms | **49.6×** |

  (An earlier draft of this entry claimed the advantage narrowed, from a 549 → 961 ms
  drift inside one 4-conversation run. That was contention between conversations, not a
  trend against length, and measuring it properly refuted it.)

- **The limit is memory, not time.** A checkpoint is **~50 MB fixed plus ~32 KB per
  token** (exact fit across the four sizes above). The 2048 MB default therefore holds
  ~34 checkpoints of a short conversation but only ~9 at 5000 tokens, and behaviour once
  the budget evicts is still untested. Scaling the default with the context length rather
  than fixing it is the obvious next step.
- **2 of the original 20 refused trims remain**, unexplained. Small, but a loose end
  rather than a rounded-down zero.
---

## [0.20.1] - 2026-08-03

Benchmark harness and design notes only. **No engine changes** — the binary behaves
exactly as 0.20.0. Cut separately because the measurements it records retract four
published numbers, and a retraction is worth a version people can point at.

### Fixed

- **The benchmark drivers could not see a reasoning model's output.** `llama-server`
  streams a reasoning model's tokens as `reasoning_content`; the drivers read only
  `delta.content`, saw an empty stream, and reported the *total* request time as
  time-to-first-token. Everything measured on Qwen3.5-9B against `llama-server` was
  therefore wrong in both directions: "`llama-server` loses the warm burst 3×" and "fox
  wins multi-turn 7.9×" are both **retracted**. Its real prefill rate is 67 tok/s, the
  same as fox's. The tell was printed in the same table — `ITL p50 0.0 / p99 0.0`, no
  inter-token gaps at all — and was read as an oddity rather than as a broken
  instrument.
- **A stale binary could be benchmarked with the evidence on screen.** The harness
  printed the bundle's timestamp and the commit next to each other and left the reader
  to compare them; a bundle 14 minutes older than the commit that changed a default was
  measured and the result read as a finding until `strings` settled it. It now compares
  them and says so loudly.
- **A shell variable collision ran one engine out of three and printed a table anyway.**
  The multi-turn read loop used a variable named `n`, which the arm-rotation logic also
  used; after the first engine the rotation divided by zero. It did not abort — it
  produced a well-formatted single-column table with correct numbers, which is exactly
  what a result looks like.

### Added

- **`scripts/bench_multiturn.py`** — the workload behind "conversations get faster over
  time", which had no number under it until now. Real conversations: each turn carries
  the previous prompt plus the model's *actual* reply plus a new message. Written to be
  able to disappoint, and it did: fox goes 372 → 53 ms per turn but `llama-server` goes
  383 → 87, because between turns the sequence is idle and inheriting an idle slot is
  what it does well. **The honest figures for multi-turn are 1.64× over `llama-server`
  and 4.9× over Ollama — not the 3.9× the concurrent burst produces.** Two different
  claims; the docs blur them.
- **TTFT decomposition** (`docs/design/benchmark-plan-2026-08.md`). fox looked 2.7×
  faster than `llama-server` on hybrid multi-turn while reusing *nothing*, which is an
  advantage with no mechanism. Fitting TTFT against prompt length (R² > 0.995) shows
  prefill per token is a wash — `llama-server` is 10% *faster* — and the whole
  difference is fixed cost per request: **110 ms against 540 ms**. Crossover around
  **1900 tokens**: fox wins short prompts, `llama-server` wins long ones. This also
  refutes the "fox interleaves chunked prefill better" explanation offered earlier —
  going from 1 to 4 clients multiplies the slope by 4.04 for fox and 4.25 for
  `llama-server`, so neither overlaps prefill across requests.

---

## [0.20.0] - 2026-08-03

Prompt reuse now works on hybrid and recurrent models, where it had been silently off.
Minor rather than patch because of one line in particular: `--rs-rollback` defaults to
`4`, so **a hybrid model allocates ~1.8 GB more than it did in 0.19.1**. On a
memory-tight box that is a difference you want to read before upgrading, not discover.

### Added

- **`--rs-rollback <N>`** (`FOX_RS_ROLLBACK`, `rs_rollback` in `config.toml`, default
  `4`) — recurrent-state snapshots kept per sequence so a hybrid or recurrent model can
  roll its KV cache back far enough to reuse a prompt prefix. Dense models ignore it and
  allocate nothing; llama.cpp clamps it to `0` for architectures without rollback
  support.

  The cost is not proportional to the number. Measured on Qwen3.5-9B with 8 concurrent
  sequences, **~453 MB per snapshot**:

  | `--rs-rollback` | extra memory | covers |
  |---|---|---|
  | `0` | none | nothing — no prompt reuse on these models |
  | `4` (default) | ~1.8 GB | multi-turn chat: the next turn contains the previous reply, so the rollback is one token |
  | `64` | ~30 GB | re-sending an identical prompt, where a whole reply must be rolled back |

  Raise it only for workloads that re-send prompts verbatim.

### Fixed

- **Hybrid and recurrent models could not reuse prompts at all, and `llama-server`
  could.** On Qwen3.5-9B with 8 clients behind a shared 1856-token prompt, fox's warm
  TTFT was **42923 ms with `cached_tokens: 0`** against `llama-server`'s 13264 ms with
  14680 — the reference reusing on the same architecture, the same llama.cpp and the
  same GGUF. `registry.json` recommends this family (`qwen3.5`, `qwen3.5:9b`), so fox's
  main differentiator was off on the models its own catalogue leads with.

  Four separate gates had to be opened, and three were only found by measuring against
  a real model — the unit tests passed and the log reported the capability as enabled
  while reuse stayed at zero:

  1. **One capability where there were two.** Inheriting the KV a sequence already
     holds copies nothing and is legal on hybrids; copying a prefix out of *another*
     live sequence needs `seq_cp` and is not. Both hung off one flag, so hybrids lost
     the cheap kind too. Now `Model::supports_slot_reuse()` is separate from
     `supports_seq_copy()`, and the scheduler carries both.
  2. **Finished sequences were never parked.** `logits.rs` parked a completed sequence
     only when the model supported *copying*. With nothing resident, every reuse path
     downstream was dead no matter what it was permitted to do.
  3. **`trim_sequence` discarded llama.cpp's result.** A partial `seq_rm` on a
     recurrent cache legitimately fails outside its snapshot window
     (`llama-memory-recurrent.cpp:181`) and mutates nothing. Ignoring that left a
     request skipping a prefix that was no longer there. It now returns the result and
     the engine re-prefills on refusal — slower, never wrong.
  4. **`n_rs_seq` defaulted to `0`.** The snapshot window itself is `0` in
     `llama_context_default_params` (`llama-context.cpp:3457`) and fox inherited it, so
     every partial rollback failed. `QWEN35` is in `llm_arch_supports_rs_rollback`: the
     architecture was never the obstacle.

  Result on Qwen3.5-9B, 8 clients, warm burst, with a window sized for the workload
  (`--rs-rollback 64`): TTFT **42923 → 638 ms**, `cached_tokens` **0 → 14856**, trims
  refused **8 of 8 → 0**. Against `llama-server` on the same model that is 13296 → 638
  ms, i.e. fox goes from losing 3.2× to winning 20.8×.

  **At the shipped default it does not, and this workload is the reason to say so.**
  Re-sending an identical prompt requires rolling back the whole generated reply, which
  needs a window of that size; `--rs-rollback 4` cannot and falls back to a full
  prefill. Measured with defaults: fox warm **39970 ms, `cached_tokens` 0** against
  `llama-server`'s **13214 ms** — still 3.0× behind. Out of the box this release
  *unblocks* prompt reuse on hybrid models; it does not deliver it for prompts sent
  verbatim twice unless you raise the window and pay the memory.

  **The case the default was chosen for does not hold either, now that it has been
  measured.** The reasoning was that a multi-turn conversation needs a rollback of one
  token, because the next turn contains the previous reply. `scripts/bench_multiturn.py`
  refutes it on this model: 4 conversations × 6 turns produced **20 slot hits and 20
  refused trims**, `cached_tokens` 0 throughout. The refusals report `keep_from=155`
  against a turn-0 prompt of ~157 tokens, i.e. the shared prefix ends at the first user
  message and the assistant reply does not match at all — the parked sequence holds the
  raw generated tokens while the next request carries that reply re-wrapped by the chat
  template. Where those two do not coincide, the required rollback is the whole reply,
  not one token.

  So the premise was wrong, not the arithmetic. On Llama-3.2-1B the chain does line up
  (fox reuses 342 tokens per turn), so this is model- and template-dependent and not a
  blanket property. **`--rs-rollback 4` should be read as "enough for models whose
  template makes a conversation a token-exact prefix chain", which is not all of them,
  and is not the catalogue's own Qwen3.5.** Left at 4 rather than raised because the
  window that would cover it is the reply length, at ~453 MB per snapshot.

- **Prompt reuse was decided without checking whether the model could perform it**,
  which aborted the process instead of degrading. Three entry points — slot affinity,
  copy-from-a-live-sibling, and the `n>1`/`best_of` fork — all reached
  `llama_memory_seq_cp` without consulting `supports_seq_copy()`, so the engine could
  log `prefix caching disabled` while the scheduler went on skipping prefill, ending in
  `GGML_ASSERT(is_full)` at `llama-kv-cache.cpp:518`. The only guard that existed
  (`batch.rs:261`) tested `llama_memory_can_shift()`, which the codebase itself
  documents as the wrong predicate because it returns true for recurrent models. The
  check now sits on `allow_reuse`, which feeds all three.

### Performance

- **Decode-bound throughput: 45 → 47 tok/s per request at 4 clients** (aggregate 170 →
  175, ranges disjoint over 3 alternating rounds), closing the gap to `llama-server`
  from 1.10× to 1.06×.

  Profiling found the sampler's candidate selection, not the copy the older notes
  blamed: fox spent **6.6% of wall time in `quicksort::partition`** against
  `llama-server`'s 1.4% in `llama_token_data_array_partial_sort_inplace`. Per token
  *per sequence* it allocated a 128256-element index vector (1 MB) only to truncate it
  to `n`, and partitioned it with a comparator that dereferenced into a separate 512 KB
  logits array on every comparison. `select_top_n` now keeps a sorted buffer of at most
  `n` entries and streams the logits once — one `f32` compare against a running
  threshold in the common case, sequential, no indirection.

  Validated end-to-end rather than by micro-benchmark: this repo has a precedent of a
  4.6× sampling micro-benchmark win producing zero real throughput. The gain shrinks as
  models grow — on a dense 7B the aggregate figure reaches parity with `llama-server`
  either way — so it is worth most on small models.

### Internal

- **`make ci` and the pre-push hook now type-check against a real llama.cpp build.**
  Both ran entirely with `FOX_SKIP_LLAMA=1`, which never compiles the llama.cpp module,
  so adding a parameter to `LlamaCppModel::load()` left eight call sites broken with
  every check green. CI's `golden` job would have caught it; the local gate would not,
  and the hook's banner claimed it mirrored CI. New `make check-real`,
  `FOX_SKIP_REAL_CHECK=1` to skip, and `cargo check --all-targets` added to the
  `golden` job so binaries and `tests/` are covered too.
- **Benchmark harness**: `scripts/bench_engines.sh` runs fox, `llama-server` and Ollama
  across two backends and four workloads (burst, decode, saturation sweep, noisy
  neighbour), one server alive at a time with arm order rotated per round; plus
  `bench_decode.py`, `bench_noisy.py`, `probe_cached_tokens.py`, `bench_vllm.sh` and
  `try_ollama_rocm.sh`. Findings, including the ones that went against fox, are in
  `docs/design/benchmark-plan-2026-08.md`.

---

## [0.19.1] - 2026-08-03

A correctness pass over everything a reader sees before they run anything, plus one
build fix. No engine changes.

### Fixed

- **Starting the server printed a screenful of noise before it had served
  anything.** Three separate causes, all in the first thing a new user sees. The
  human log format was `tracing`'s `.pretty()`, built for reading a debug session:
  it prints an indented `at src/file.rs:line` under every event and a blank line
  between them, so each entry took three lines. Underneath that, the stop-token log
  dumped the model's entire special-token list — 247 `<|reserved_special_token_N|>`
  entries for Llama 3, which buried every other line. And `model ready` was logged
  twice per model, once after the weights loaded and once after the engine was
  built, which read as two models loading. Startup is now nine lines. The token
  list moved to `debug`, the count stays at `info`, and `--json-logs` still carries
  source locations as fields for anything machine-read. The `--max-models 1` notice
  stays — making that trade-off visible was a deliberate fix — but says it in one
  line instead of a paragraph, since it fires on the default configuration and
  therefore greets everyone.

- **`FOX_SKIP_LLAMA` did not invalidate the build script, so real builds silently ran
  as stubs.** `build.rs` declared `cargo:rerun-if-env-changed` for
  `FOX_CPU_ALL_VARIANTS` and nothing else. Cargo therefore never re-ran the script when
  `FOX_SKIP_LLAMA` changed: after any stub build, a plain `cargo test` reused the stub
  artifacts and kept compiling with `cfg(fox_stub)` set. It did not fail — it tested the
  stub model, which is exactly what the real-model suites exist to catch. **99 tests
  were being skipped in silence** (425 in a real build, 326 in a stub one), and
  `make golden` could have been running the golden net against the stub. All six
  environment variables the script reads are now declared. Found because a test count
  dropped from 425 to 326 with no code change; all 425 pass in a real build, so nothing
  had rotted while they were unrun.

- **The ROCm FP8 guard patch is applied by the build instead of by hand.** The fix
  existed only as an uncommitted edit inside the `vendor/llama.cpp` working tree.
  Commit `79935f7` recorded the intent; the change itself was never tracked. A fresh
  `git clone --recurse-submodules` did not get it, so the ROCm build failed for
  everyone except the machine where someone had made the edit manually. `build.rs`
  now applies it before configuring cmake, idempotently, and warns rather than fails
  if upstream moves the code — silently building without the fix being the failure
  mode worth avoiding. Verified by reverting the file to its upstream state and
  rebuilding.

- **Published performance figures that were never measured.** The same two numbers
  (87 ms TTFT, 312 tok/s) appeared in four places — the README table, the README's
  sample `fox-bench` output, `docs/index.md`, and `docs/benchmarks.md` — attributed
  to an RTX 3090 in one and an RTX 4060 in another. The same figures cannot come from
  both; they were placeholders that stayed and then corroborated each other. Replaced
  with the measured `llama-server` comparison, including the workload where fox loses
  by 4% and a sentence saying that workload will not get faster. Per-hardware tables
  are relabelled as estimates nobody has run for this project.

- **Five pages announced v1.0.0** while `Cargo.toml` said 0.19.0 and this project had
  already retracted a premature 1.0.

- **Two false claims in the README.** "The fastest local LLM server" — fox wraps
  llama.cpp, runs the same kernels for a lone decoding request, and measures 96% of
  `llama-server` there. And "single static binary" — the bundle is the binary plus the
  ggml backend libraries beside it, loaded at runtime, which is exactly the mechanism
  that lets one build cover CPU, CUDA, ROCm, Vulkan and Metal.

- **The FAQ claimed production deployments** without naming one. Replaced with what can
  be shown: pre-1.0 with a retraction in its history, a stable HTTP API, shipped Docker
  and systemd units, and an end-to-end suite against a real model gating each release.

- **`docs/features.md` documented a mechanism that no longer exists.** "Block-level
  prefix caching" described block-aligned reuse keyed by hash; 0.19 replaced it with
  per-sequence resident token lists and token-exact matching that also reuses generated
  tokens. Documentation describing a removed design is worse than none, because a
  reader tunes against it.

### Added

- **`fox pull` downloads sharded GGUFs.** Large models are published split across
  `name-00001-of-00014.gguf` … and llama.cpp loads the whole set when handed the first
  part, so fetching one part left an unusable file. Several of the most-downloaded
  models on HuggingFace — Kimi K3, DeepSeek V4, GLM 5.2, MiniMax M3 — were therefore
  unreachable through `fox pull` and could not be listed in the catalogue at all.
  Handed any part, fox now resolves the whole set, downloads each part, and reports
  the first one, which is what llama.cpp must be pointed at.

  Two details came from checking real repositories rather than assuming the layout.
  Parts are nested in a per-quant subdirectory (`UD-IQ1_S/Model-UD-IQ1_S-00001-of-…`),
  so the destination's parent directory has to be created; and a repository commonly
  holds several differently-sized sets at once, so grouping keys on the split count as
  well as the name — mixing two sets yields a model that cannot load.

  The four models this unblocks are in the catalogue, each labelled with its real
  size: DeepSeek V4 Flash (82 GB), MiniMax M3 (90 GB), GLM 5.2 (217 GB) and Kimi K3
  (594 GB). Those figures are the *smallest complete set published* for each, at IQ1_S,
  so the description says outright that this is the set that loads rather than the best
  the model does. Listing them without that would be worse than omitting them: a
  `fox pull` sitting next to 2 GB entries should not quietly start a 594 GB download.

- **25 current models in the built-in catalog, taking it from 18 entries to 43.** The
  catalog was not broken — all 18 existing entries still resolve on HuggingFace — but
  its selection had aged, so `fox pull` offered a 2024/2025 line-up and the README's
  worked example pulled a model from 2024. Added across roles rather than by
  popularity: Qwen3.5 4B and 9B, Qwen3.6 35B-A3B and 27B, Gemma 4 26B-A4B, 12B and
  E4B, Qwen3-Coder 30B-A3B, Qwen3-VL 4B, Ornith 1.0 9B, EmbeddingGemma 300M, Qwen3
  Embedding 0.6B, and two rerankers. Existing entries are kept, since removing them
  would break anyone's `fox pull llama3.2`.

  Two gaps this closes are features that shipped without anything to run them on:
  `/rerank` and `--reranking` landed in 0.19 with no reranker in the catalog at all,
  and the mixture-of-experts entries are the first models here that give `--moe-cpu`
  something to demonstrate.

  Vendors beyond Qwen and Gemma: IBM Granite 4.1, AI2 Olmo 3, TII Falcon-H1R, Apertus,
  NVIDIA Nemotron 3 Nano Omni, Tencent Hunyuan MT, and Ornith. Falcon-H1R is worth
  singling out — it is hybrid attention/state-space, so prompt reuse is disabled for it
  and it will report no cached tokens, which makes it the one entry that exercises that
  path from the catalog.

  Some current families are missing for a reason rather than by oversight: Kimi K3,
  DeepSeek V4, GLM 5.2, MiniMax M3 and Inkling all publish as multi-part GGUFs, and
  neither `registry.json` (one `recommended` filename) nor `fox pull` handles sharded
  downloads. That is a real gap in fox, not a shortage of models.

  Every repository, filename, projector and size was verified against the HuggingFace
  API rather than written from memory, and sizes are the real byte counts. The whole
  catalog is re-checked at 43 of 43 resolving, with no duplicate aliases.

- **The 16 `fox serve` flags that were never documented** — `--speculative`,
  `--spec-ngram`, `--spec-draft-len`, `--draft-model`, `--lora-modules`, `--mmproj`,
  `--reranking`, `--kv-reuse`, `--slot-prompt-similarity`, `--cache-ram`,
  `--repeat-last-n`, `--tool-call-parser`, `--max-queue-depth`, `--max-prefill-chunk`,
  `--context-shift`, `--context-keep`. The page listed 23 of 39. Nothing documented was
  wrong; the gap was omission. Defaults appear only where they were checked against the
  binary, because a wrong default costs more than a missing one.

- **The endpoints shipped in 0.19 in the README's API table** — `/infill`, `/rerank`,
  `/tokenize`, `/detokenize`, `/apply-template`, `/props`, `/slots`, `/lora-adapters`.
  A third of the release was invisible to anyone reading the README.

- **A methodology section in `docs/benchmarks.md`** on how these benchmarks produced
  confident wrong answers before they produced right ones: a binary predating the
  feature under test, a metric that could not fall when sharing worked, and an
  oversized prompt that fails differently on each server.

---

## [0.19.0] - 2026-08-03

### Fixed

- **The intermittent `make e2e` failure, found and fixed — it was the test.** Checks 1
  and 9 sent `max_tokens: 12` with no `temperature` (so fox's stochastic `0.8`
  default) and no `min_tokens`, then asserted `finish == "length"`. Nothing stopped
  the model emitting EOS before the twelfth token, which yields `"stop"` and fails.
  Measured rather than waited for: 600 requests in check 9's concurrent shape
  produced 2 early stops (0.33% each), which across the suite's 7 such requests is
  ~1 failing run in 43 — against the 1-in-~52 actually observed. Adding
  `min_tokens: 12` makes `finish == "length"` a fact about the engine instead of a
  coin flip, without weakening what the checks are for (a request that dies after
  its prefill token still reports `n < 12`). Verified under identical concurrent
  conditions: 600/600 `length`, zero short.

- **A model loaded from outside `models_dir` was unreachable by its own name.**
  `--model-path` pointing anywhere other than `models_dir` produced a server that
  advertised the model in `/health` and `/props` under a name every request path
  then answered `404` for. Two things combined: `resolve_model_name` only ever
  scans `models_dir`, and `get_or_load` resolved *before* checking what was already
  resident — so a model that was loaded and serving could not vouch for its own
  name.

  The registry now records where each model it loads came from and consults that
  right after alias resolution, before the directory scan. Deliberately not
  forgotten on unload: the path is still the right answer for that stem, and
  dropping it would make an evicted `--model-path` model unreachable again at the
  first keep-alive expiry — the same bug, resurfacing later and harder to attribute.

- **`/api/show` answered from the filename, not the model.** `parameters` and
  `template` were empty strings and `parameter_size` was the literal `"unknown"`,
  because every field was parsed out of the file stem. When the model is resident it
  now answers for itself: real dimensions, effective context, and an **exact**
  parameter count from `llama_model_n_params`. It still does not load a model to
  answer — `GET /props` covers the resident model unconditionally, and loading here
  would evict the serving model under `--max-models 1`.

  (An intermediate version derived the parameter count from
  `n_embd`/`n_layer`/`vocab_size` and reported **1.6B for llama-3.2-1b**, which is
  really 1.24B: the estimate ignored grouped-query attention, which shrinks K and V,
  and double-counted tied input/output embeddings. Replacing `"unknown"` with a
  confident wrong number is worse than either, so the estimate was dropped for the
  exact value.)

- **Ollama `keep_alive` was parsed and never applied.** A client asking to keep a
  model warm, or to drop it promptly, got a `200` and no effect — and the test
  only asserted that `200`, so the field being inert was invisible. Now honoured
  per request: a duration string (`"5m"`, `"30s"`, `"500ms"`) or a number of
  seconds sets that model's idle TTL, a negative value pins it against timed
  eviction, and `0` sets a zero TTL so the next eviction pass drops it. The
  eviction task now runs even when the server-wide `--keep-alive-secs` is `0`,
  since a request can set a TTL where the server has none — otherwise the field
  would have stayed inert in exactly that configuration. Deliberate divergence
  from Ollama: `keep_alive: 0` unloads on the next eviction tick (within 60s)
  rather than the instant the response ends, which also means it can never kill
  the request that asked for it (`is_busy` guards the pass).

- **Ollama options silently dropped by serde** (`num_ctx`, `num_keep`,
  `typical_p`, `mirostat`, `mirostat_tau`, `mirostat_eta`, `penalize_newline`)
  are now declared and reported. fox does not implement them — `num_ctx` would
  need a model reload, `num_keep` exists only as the server-wide
  `--context-keep`, and the samplers are absent for the reasons in
  `docs/design/llama-server-gap-analysis.md` §3 — so setting one logs a warning
  naming it, instead of vanishing. They are not rejected: clients send Ollama's
  defaults on every request, and a 400 would break them.

- **JSON-Schema→GBNF dropped optional properties.** A schema like
  `{properties: {a, b}, required: [a]}` produced a grammar that *forbade* `b`
  entirely, so guided decoding could never emit a declared optional field. This
  was a correctness bug, not the documented "simplification": the grammar was not
  merely stricter than the schema, it contradicted it. Optional properties are now
  emitted as genuinely optional members. Two limits stay, both documented at the
  call site: optional properties may only appear in declaration order (modelling
  every permutation is exponential; llama.cpp's own converter has the same
  limitation), and an *absent* `required` still means "all properties required".
  An explicitly empty `"required": []` is now honoured literally, so an
  all-optional object is expressible — previously the two were collapsed.

- **`/v1/completions` ignored almost every parameter and returned the wrong
  shape.** The handler hard-coded `None` for `top_p`, `top_k`, `stop`, `seed`,
  `logprobs`, `logit_bias` and both penalties, so they were accepted and silently
  did nothing; and it returned a `chat.completion` object, while clients of the
  legacy endpoint read `choices[].text`. (`CompletionResponse`/`CompletionChoice`
  were declared in the types module and never constructed.) All sampling
  parameters are now threaded through, and the response — streaming and
  non-streaming — is rewritten to the `text_completion` shape. `echo` and
  `suffix` are rejected with a 400 rather than silently ignored, since a caller
  cannot otherwise detect that they had no effect.

- **Ollama responses reported no prefill/decode split.** `load_duration` and
  `prompt_eval_duration` were the literal constant `0`, and `total_duration` and
  `eval_duration` were the same wall clock, on both `/api/generate` and
  `/api/chat`, streaming and not. All four are now measured: model-load time,
  submission-to-first-token, and first-token-to-last. `prompt_eval_duration`
  includes scheduler queueing as well as prefill compute — real latency the
  client paid, and fox has no cheaper place to separate the two. The same split
  is now in the per-request `done` log line (`prefill_ms`, `decode_ms`).

- **`stream_options.include_usage` was parsed and ignored** — usage always rode
  the final chunk. An explicit `false` now suppresses it. Omitting
  `stream_options` keeps the previous always-attach behaviour, so no existing
  caller changes.

### Changed

- **`n>1` / `best_of` no longer prefills the same prompt N times.** Branch 0
  prefills; the rest wait for it and copy its KV
  (`llama_memory_seq_cp`) instead of recomputing the identical thing. Measured on
  an 801-token prompt with `n=4`, alternating arms, one server at a time:
  **6.60/6.63 s → 2.01/1.96 s (3.4×)**. The server log shows one full admission and
  three branches reporting `cached_tokens: 800` of 801.

  The wait is what makes it correct: a branch is held back until its parent is
  actually decoding, since there is nothing to copy before that. Deferred branches
  are re-queued rather than left at the queue head — they are not blocked on
  *capacity*, so stalling everything behind them would be a self-inflicted
  head-of-line block. If the parent never materialises (finished, failed, never
  admitted) the branch falls back to an ordinary full prefill, so this is a speed
  optimisation with no correctness edge of its own; slot affinity usually finds the
  parent's parked KV anyway.

  Multimodal and LoRA branches are excluded: multimodal counts positions from image
  chunks while the resubmission boundary counts `prompt_tokens` (empty for them), so
  the two would disagree; a LoRA branch must not inherit KV computed under a
  different adapter. Branches allocate their own block budget rather than sharing —
  llama.cpp shares the cells itself under `kv_unified`, so fox's accounting is
  merely conservative, and the dormant copy-on-write path stays dormant.

  Output is unaffected: `n=4` still returns four distinct completions.

### Added

- **`GET` / `POST /lora-adapters` — inspect and re-scale adapters at runtime.**
  fox could load adapters with `--lora-modules` and select one by naming it in the
  `model` field, but there was no way to see what was loaded or to change an
  adapter's strength without restarting — the point of a scale being a number
  rather than a rebuild. `GET` lists `{id, name, path, scale}` with the scale
  currently in effect; `POST` takes `[{id|name, scale}]`.

  Only the *default* scale is mutable, and that is sufficient: a request's
  `LoraSelection` already carries its own scale down to `llama_set_adapters_lora`,
  so overriding what gets copied into it is the whole mechanism and the model's
  adapter handles stay immutable after load. A body naming one valid and one
  unknown adapter is resolved before anything is applied, so it changes **nothing**
  rather than leaving the server in a state the caller neither asked for nor can
  infer from the error. Changes apply to subsequent requests; a generation already
  in flight keeps the scale it was admitted with, since the adapter set is a
  property of the batch being decoded.

  Verified against a real adapter: listing, setting by name and by id, the
  all-or-nothing rejection (scale unchanged at 0.9 after a partially-invalid body),
  and the new scale reaching the server.

- **`--cache-ram <MiB>` — host-RAM prompt cache.** Complements `--kv-reuse`: a slot
  keeps a conversation warm by holding GPU blocks, this keeps one warm without
  holding any. A reclaimed sequence is serialised to host memory
  (`llama_state_seq_get_data_ext`) and restored later
  (`llama_state_seq_set_data_ext`) instead of being re-prefilled.

  Ordering in the engine is load-bearing: **saves → clears → restores → trims**. A
  save must read the sequence before the clear wipes it; a restore must land after
  the clears (its destination may itself have just been reclaimed) and before the
  trims, which bound the *restored* state at the new request's divergence point. A
  failed restore resets the request to prefill from token 0 rather than letting it
  read cells that were never written — slower, never wrong.

  The FFI round-trip is verified against a real model: a saved state restored into a
  *different* sequence predicts the identical token with logits matching to <1e-3,
  and restoring over a dirty destination is correct because the load clears first.

  **Not a general speedup, and defaults to `0`.** Reclamation only triggers when the
  block pool is exhausted *and* the claiming request needs more blocks than the slot
  it inherits — neither holds under sequential single-client traffic, where every
  request shares the chat-template prefix and LCP affinity routes them all onto one
  slot whose blocks they inherit unchanged. It earns its keep under concurrent,
  distinct conversations that exhaust the pool. See
  `docs/design/llama-server-gap-analysis.md` §1.C.

- **`GET /props` and `GET /slots`** — server and per-sequence introspection.
  `/props` reports architecture, backend, allocated vs trained context,
  dimensions, and capability flags (`supports_thinking`, `supports_vision`,
  `supports_infill`, `supports_kv_reuse`) **read from the loaded model**, never
  inferred from its filename. It never triggers a load: under the default
  `--max-models 1` that would evict whatever is serving traffic, so `model` is
  simply `null` when nothing is resident. `/slots` reports each sequence as
  `free`/`processing`/`idle` with its resident token count, blocks charged and idle
  time — which is what makes the KV reuse work observable. Slot *contents* are
  deliberately not exposed: a parked sequence is another user's conversation, and
  llama-server redacts the same fields.

- **`POST /infill`** — fill-in-the-middle completion, what editor plugins call to
  fill in code at the cursor. Emits the FIM token layout the model was trained on
  (`[SUF] suffix [PRE] prefix [MID]`; suffix first, so the model reads what it must
  join up to before it starts writing) and accepts `input_extra` for repo-level
  context. A model whose vocabulary has no FIM tokens is **rejected with an
  explanation** rather than answered: prompting a chat model for infill produces
  fluent text that ignores the suffix, and the caller has no way to tell.
  Verified against a real FIM-capable model — the completion joined the suffix
  exactly.

- **`POST /rerank` and `/v1/rerank`** — score documents against a query with a
  reranker model, plus **`--reranking`** to load one. Accepts both the Jina/Cohere
  (`documents`) and TEI (`texts`) spellings, plus `top_n` and `return_text`, and
  preserves each result's original index across sorting.

  `--reranking` creates the model's context with `RANK` pooling, which is what
  makes the relevance head readable. **This cannot be auto-detected**: a reranker
  GGUF does not reliably carry a `<arch>.pooling_type` key — `jina-reranker-v1-tiny-en`
  has none — so llama.cpp's `UNSPECIFIED` fallback resolves to `NONE`. llama-server
  takes a flag for exactly the same reason (`arg.cpp:3067-3070`). Without it,
  `llama_get_embeddings_seq` returns `NULL`, and that `NULL` is the signal used to
  reject the request rather than answer it with a number derived from a mean-pooled
  vector, which would look like a ranking and rank nothing.

  Verified end to end against a real reranker: querying "What is the capital of
  France?" over four documents ranked Paris first, the Eiffel Tower second and
  bananas last, with original indices preserved; `top_n` and the TEI spelling both
  behave.

- **`POST /tokenize`, `/detokenize`, `/apply-template`** — llama-server's tokenizer
  utilities (`server-context.cpp:4899-4956, 4846-4856`). No inference involved, just
  the loaded model's vocabulary and chat template; clients use them to count tokens
  before sending a request and to debug template rendering. fox already had every
  underlying piece (`InferenceEngine::tokenize`, `build_prompt_tokens`) and simply
  never routed them. `/tokenize` supports `with_pieces`, reporting raw bytes for a
  token that holds only part of a multi-byte codepoint rather than lossily decoding
  it. `/apply-template` renders through the *same* path a real request takes and
  then detokenizes, so what it returns is literally what the model would receive,
  control tokens included.

- **Raw GBNF `grammar` request field** on `/v1/chat/completions` and
  `/v1/completions`, mirroring llama-server. The engine has had full GBNF support
  since 0.14, but the only way to reach it was `response_format`/`format`, which can
  only describe JSON. Setting both `grammar` and `response_format` is a 400 rather
  than a silent precedence rule.

- **`usage.prompt_tokens_details.cached_tokens`** on the OpenAI surface — how many
  prompt tokens were served from resident KV instead of being re-prefilled. This is
  the only way a client can observe the KV-reuse rework. The field is omitted
  entirely when nothing was cached, so responses are unchanged when there is nothing
  to report.

- **`top_n_sigma` and `min_keep` samplers**, on both API surfaces (and
  `/v1/completions`). `top_n_sigma` keeps only tokens within N standard
  deviations of the top logit — unlike `top_p` the cutoff lives on the logit
  scale, so it is invariant under `temperature` (there is a test for exactly
  that). `min_keep` floors how few candidates any truncation step may leave.
  Both default to off. `typical_p`, `mirostat`, XTC and DRY remain unimplemented:
  they need distribution-wide state that conflicts with fox's adaptive candidate
  pool, which is a design question rather than a missing line of code — see
  `docs/design/llama-server-gap-analysis.md` §3.

- **`repeat_last_n` on the Ollama surface** (`options.repeat_last_n`), which
  upstream Ollama supports and fox previously dropped silently.

### Changed

- **KV reuse reworked: sequences now remember what they hold** (`--kv-reuse`,
  `--slot-prompt-similarity`). fox's prefix cache was a `LruCache` of donated
  whole-block prompt prefixes, and it had three structural limits: it held
  `max_batch_size/4` entries (**8** at defaults, with no flag to raise it); reuse
  was aligned to `block_size`, so a prompt matching 31 of 32 tokens reused 16;
  and **only the prompt was cached — the generated reply was always discarded**,
  which is why multi-turn chat, whose next prompt contains the previous reply,
  could never hit past the previous prompt's end.

  Replaced with llama-server's slot model (`server-context.cpp:1586-1694`,
  `:3166-3243`): every sequence permanently records the tokens resident in its
  KV, prompt *and* generation. A finished request parks its sequence instead of
  freeing it, admission picks the sequence sharing the longest common prefix with
  the incoming prompt, and reuse is token-exact. Idle sequences are reclaimed
  LRU under block pressure — not preemption: they belong to requests that already
  finished and whose output the client already has, and `Busy` slots are never
  touched, so the never-preempt-on-admission invariant is unchanged.

  Measured on CPU/zen4 with llama-3.2-1b-instruct-q8_0, alternating arms, one
  server at a time. Reuse on vs off, repeated ~3.5k-token prompt: **6.1×** median
  TTFT (4760 → 782 ms, disjoint ranges). Old build vs new on a 12-conversation
  working set — bigger than the old 8-entry cache, smaller than the new 32-slot
  table: median 52.1 → 37.6 ms, but the real number is the **tail**, mean
  1247.6 → 36.3 ms (**34×**) and p90 3650.5 → 38.7 ms (**94×**), because the old
  cache evicted a third of the working set every pass and each eviction cost a
  full re-prefill. See `docs/design/llama-server-gap-analysis.md`.

  `--kv-reuse false` restores the previous behaviour exactly, and is the baseline
  arm for reproducing the measurement.

  **Known consequence: greedy output drifts more often under concurrent load.**
  Measured at `temperature: 0`, 4 concurrent clients, 10 rounds against a sequential
  baseline: 2/10 rounds differed with `--kv-reuse false`, 10/10 with it on. The
  nondeterminism is **pre-existing** — the control arm has reuse disabled and still
  drifts, because concurrent requests are batched together by arrival timing and
  llama.cpp does not guarantee bit-identical logits across batch compositions. Reuse
  amplifies it by collapsing prefill, so requests spend far more of their life
  decoding alongside each other. It is **not** incorrect KV: *sequential* reuse is
  byte-identical across repeats (verified with `cached_tokens` up to 396), which is
  the same code path with the same cache state. A caller needing reproducible greedy
  output needs a serialised request stream — `seed` does not help, since the variation
  is in the forward pass rather than the sampler. See
  `docs/design/llama-server-gap-analysis.md` §1, which also records a single
  unreproduced `make e2e` failure (1 in ~52 runs, zero in the 51 since) whose cause
  remains unknown and which the drift above is the wrong magnitude to explain.

- **Concurrent requests now copy a shared prefix from a *live* sequence.** Slot
  affinity can only inherit an *idle* sequence, so N requests arriving together
  behind one system prompt could reuse nothing from each other — each prefilled the
  shared prompt. A busy sequence cannot be inherited without stealing a live
  request's KV, but it *can* be copied from: under `kv_unified`, `seq_cp` shares
  llama.cpp's cells rather than duplicating the buffer. Requests behind a donor that
  is still prefilling are deferred and re-queued rather than left at the queue head,
  since they are blocked on a sibling and not on capacity. Measured against
  `llama-server` (both from the same vendored llama.cpp, Radeon 890M / Vulkan, 3
  rounds, disjoint ranges): **4.0× faster cold TTFT at 8 concurrent clients behind a
  1856-token system prompt, 5.75× at 16**, with the whole-burst wall clock at 3.8 s
  against 16.2 s. Doubling the clients costs fox 24% more cold TTFT and
  `llama-server` 79%. `llama-server` cannot do this by construction: its
  `get_available_slot()` skips `is_processing()` slots in both its similarity pass
  and its LRU fallback, so its concurrent arrivals report `cached_tokens` 0.

- **A shared prefix is charged to the block budget once, not once per sharer.**
  Sharing the prefill left the accounting duplicated: each sharer skipped the
  prefill and still reserved its own blocks for the positions it had just copied.
  Blocks are an admission budget rather than addresses, so this wasted no GPU memory
  — llama.cpp's cells really are shared — but it made fox admit less concurrency
  than the hardware holds. Pool occupancy on 6 concurrent clients behind a 673-token
  prompt: **282 → 72 blocks**. The reservation is now sized before allocating, so
  the capacity check agrees with reality instead of turning a burst away for
  capacity it was never going to hold. Only *whole* blocks are shared: the block
  straddling the divergence point stays private, which is what guarantees a shared
  block never receives a write — and is why the decode path deliberately has no
  copy-on-write pass.


### Added

- **`--repeat-last-n` / `FOX_REPEAT_LAST_N` — bounded penalty window.** The
  repetition, frequency and presence penalties previously scanned *every* token
  generated so far on every sampling step. Two consequences, both fixed by
  bounding the window: `apply_frequency_presence_penalty` rebuilt a full
  `HashMap` over the whole history per token, making the penalty pass
  `O(generated²)` per request; and the Ollama surface's `repeat_penalty = 1.1`
  default kept penalising tokens from thousands of positions back, degrading
  long outputs. Semantics follow llama.cpp's `repeat_last_n`: `-1` = whole
  history, `0` = disabled, `n` = last `n`. Overridable per request via
  `repeat_last_n` (`/v1/*`, a fox extension) and `options.repeat_last_n`
  (`/api/*`, which upstream Ollama supports and fox previously dropped
  silently). **Defaults to `-1`, so output is bit-identical to before unless
  the knob is set** — llama.cpp defaults to `64`, but adopting that would have
  silently changed output for every existing caller. One deliberate divergence
  from llama.cpp, documented at the call site: fox's window covers only
  *generated* tokens, never the prompt, which is what fox has always done.

## [0.18.0]

### Added

- **LoRA adapter support** (`--lora-modules <name>=<path>[:<scale>][,...]` /
  `FOX_LORA_MODULES`) — loads one or more named LoRA adapters onto the primary
  model at startup; a client selects an adapter per-request the same way it
  selects a model, by passing the adapter's name as the `model` field
  (mirrors vLLM's `--lora-modules name=path` convention). Since
  `llama_set_adapters_lora` is a property of the whole `llama_context`, not of
  a sequence, requests are grouped by adapter selection and processed as
  separate sub-batches — the same approach llama.cpp's own reference server
  uses. Prefix caching is skipped for any request carrying an adapter
  selection (KV computed under one adapter is invalid for another). Verified
  end-to-end against a real base model + a real, independently-trained
  reasoning-style adapter: 24/24 e2e checks pass, including the adapter
  measurably changing output and an interleaved base/adapter/base/adapter
  request sequence never corrupting context state. v1 scope: one base model
  per `--lora-modules` set (no cross-base-model LoRA), no per-sequence mixed-
  adapter batching (not a fox limitation — llama.cpp has no kernel for it), no
  hot-reload. See `docs/design/lora-support.md`.

- **Multiple completions per request** (`n`, `best_of` on
  `/v1/chat/completions` and `/v1/completions`) — `n` (1–8) returns that many
  independent completions in `choices[]`; `best_of` (≥ `n`) samples more
  candidates than returned and keeps the `n` with the highest total
  log-likelihood. Each choice is a fully independent generation over the same
  prompt (fan-out, not a shared-prefill fork), so branches naturally diverge
  under `temperature > 0`; an explicit `seed` is perturbed per branch so it
  doesn't collapse `n` completions into identical copies. Streaming interleaves
  all branches into one SSE stream, tagged by `index`, merged via
  `tokio_stream::StreamMap`. `best_of > n` is rejected with `stream: true`
  (matches OpenAI's own restriction — ranking needs the full completion
  before anything can be shown). Verified end-to-end against a real model:
  `n: 3` returns 3 correctly-indexed choices with correctly summed usage, no
  regressions across the rest of the e2e suite. v1 scope: no KV-level
  shared-prefill forking (each branch reprocesses the prompt independently),
  beam search itself remains unimplemented. See
  `docs/design/n-best-of-support.md`.

- **Correct KV sizing for MLA and recurrent/hybrid models** — the context's
  `n_ctx` was still capped at load time by a hand-rolled positional formula
  (`n_head_kv * head_dim * n_layer`), wrong for MLA (DeepSeek-V2/V3, whose
  compressed latent KV the formula massively over-estimates) and meaningless
  for recurrent/hybrid (Mamba, RWKV, Jamba — no per-token KV at all). Replaced
  with an empirical create-then-shrink-on-failure retry loop: attempt the full
  desired context, halve and retry only on a real `llama_init_from_model`
  failure — the same "observe real failure, retry smaller" approach 0.16
  already shipped for decode-time OOM, applied one layer earlier, uniformly
  across every architecture. Added a lightweight `KvMemoryClass`
  (Standard/Latent/Recurrent) to `ModelInfo`/`fox probe` for observability.
  Verified against real DeepSeek-V2-Lite (MLA) and Mamba (recurrent) models —
  both added to `registry.json`. See `docs/design/mla-recurrent-kv-sizing.md`.

- **Reactive context-rolling on OOM** — closes the last item on the
  vLLM-parity shortlist. When 0.16's decode-time bisection retry bottoms out
  at a single request and it still can't decode, fox now attempts one
  targeted context roll on that request (reusing the existing
  `--context-shift` mechanism) and retries the whole batch once more before
  giving up with `EngineError` — a "further degrade" step beyond just
  shrinking the batch. A typed error carries the failing request id from the
  model layer (which has no scheduler/config access) up to the engine layer
  that does. In practice a narrow safety net for residual cases — verified
  live that the existing proactive context-shift threshold already prevents
  most contention under normal concurrent load. See
  `docs/design/reactive-context-rolling.md`.

### Fixed

- **`fox pull <name>` didn't actually use the curated registry it advertises** —
  `fox models`/`registry.json` map short names/aliases to a specific HF repo +
  recommended file, but `fox pull` never consulted that catalog: it always ran
  a live HuggingFace search by name and hoped the top result happened to match.
  `fox pull` now checks the registry first (exact name, alias, or
  `<name>-<quant>`) and resolves straight to the intended repo/file; falls back
  to the historical live-search behavior unchanged for any name not in the
  registry. Vision entries now print a follow-up hint for their paired mmproj
  file after downloading (not auto-fetched, to avoid a surprise extra
  download for text-only use).

- **Recurrent/hybrid models were silently getting prefix caching enabled**
  when it should have been disabled — found while verifying the KV-sizing fix
  above against a real Mamba model. The existing detection
  (`llama_memory_can_shift`) has, since an upstream llama.cpp change, returned
  `true` for recurrent memory too ("shifting the pos is trivial" for it — a
  cheap-operation signal, not a "safe for fox's block-copy prefix cache"
  signal), silently defeating the v0.3.1 fix this was supposed to be. Replaced
  with `llama_model_is_recurrent`/`llama_model_is_hybrid` — the direct,
  architecture-level llama.cpp APIs for the question actually being asked.

- **`--max-models 1` (the default) and `--swap-fraction` were silent
  footguns** — a second model request evicting the first, or a
  `--swap-fraction` value doing nothing at all, both happened with zero
  feedback. `fox serve` now logs the trade-off explicitly at startup when
  `--max-models` is left at its default, and warns if `--swap-fraction` is
  set to a nonzero value (it remains unimplemented — real CPU↔GPU KV swap is
  blocked on a llama.cpp API that doesn't exist yet). The `max_models=1`
  default itself is intentionally left unchanged: fox has no cross-model VRAM
  accounting yet (the per-load fit check compares against a static,
  whole-GPU figure from startup, never subtracting what other loaded models
  already claim), so raising the default without that accounting would trade
  a churn footgun for a real OOM-crash footgun.

- **A real server crash, found while verifying reactive context-rolling
  against actual concurrent load** — several requests admitted into the same
  prefill step each contributed their own chunk to one shared `llama_decode`
  call, and their combined token count could exceed `n_batch`.
  `--max-prefill-chunk` only capped one request's own chunk, not the sum
  across several concurrently-admitted requests (also reachable with a
  *single* request whenever `max_prefill_chunk` itself exceeds `n_batch` — a
  small `--max-context-len` shrinks `n_batch` below the 512-token default).
  Unlike `ret==1` ("no KV slot"), llama.cpp enforces this via a hard
  `GGML_ASSERT` abort with no graceful return code — a full process crash,
  not a per-request failure, reproduced live with 9 concurrent requests
  against a deliberately small `--max-context-len`. Fixed by allocating the
  real `n_batch` (queried via `llama_n_batch`) across requests in submission
  order before ever building the batch; any request that doesn't fit this
  step simply gets deferred to the next one, the same mechanism a single
  request's own multi-step chunking already relied on. See
  `docs/design/reactive-context-rolling.md`.

### Changed

- **Beam search closed as a deliberate non-goal**, not left as an open
  backlog item — the last row on `vllm-gap-analysis.md`'s vLLM-parity
  shortlist. Investigated rather than assumed: llama.cpp removed its
  `llama_beam_search()` API in 2024 (an ancestor of fox's pinned commit,
  nothing to build on); vLLM itself pulled beam search out of its
  PagedAttention/continuous-batching fast path into a separate,
  offline-batch-oriented API for the same composability reasons fox would
  face; and no major LLM API (OpenAI, Gemini, Claude) exposes real
  token-level beam search today. A real, KV-sharing-efficient
  implementation would need live cross-sequence forking mechanics neither
  fox nor llama.cpp currently offer cleanly; a naive independent-request
  approximation would just be a more expensive, weaker variant of the
  `n`/`best_of` fan-out already shipped in 0.18. No code changed — see
  `docs/design/vllm-gap-analysis.md` §2 for the full reasoning.

## [0.17.0]

fox gets **vision/multimodal input** — the top feature-gap item for the LatAm
go-to-market push (`STATUS.md`, `vllm-gap-analysis.md`). Wraps llama.cpp's `mtmd`
library rather than building a vision pipeline from scratch: it already handles
image decoding and per-architecture position bookkeeping (causal vs M-RoPE)
internally, so the real work was fitting a paired-projector model into a scheduler
built around `prompt_tokens: Vec<i32>` as the single source of prompt content *and*
position count. Verified end-to-end against a real, mainstream model — Google's
Gemma 4 E2B — with 24/24 e2e checks passing, including two different images
back-to-back not cross-contaminating output (the exact failure mode the design
exists to prevent). See `docs/design/vision-support.md` for the full rationale.

### Added

- **Vision / multimodal input** (`--mmproj <name>` / `FOX_MMPROJ`) — loads a paired
  vision-projector GGUF alongside the target model (mirrors `--draft-model`: one
  global pairing, resolved the same way). OpenAI `image_url` (base64 `data:` URI
  only — a remote `http(s)://` URL is rejected with `400`, not fetched, to avoid an
  SSRF surface) and Ollama `images` are both supported on every chat/generate
  endpoint. Image content is spliced into the rendered prompt as a marker literal
  before the chat template runs, then `mtmd_tokenize` splits on it — mirrors
  llama.cpp server's own approach rather than teaching every Jinja template about
  images. A multimodal request's prefill is a single atomic
  `mtmd_helper_eval_chunks` call — it never joins the shared per-step token batch,
  and (by design, not a bolted-on flag) never touches the prefix cache, since
  `prompt_tokens` stays empty for such requests and every prefix-cache path already
  keys off it directly. v1 scope: one global mmproj pairing, no multimodal prefix
  caching, no OOM bisection-retry on this path — see `docs/design/vision-support.md`
  for why each was cut.
- **`gemma4-e2b` in the model registry** — Google Gemma 4 E2B (natively multimodal,
  131K context) + its mmproj, verified end-to-end (24/24 smoke checks, exact
  one-word-correct answers to test images). Recommended vision model; `moondream2`
  is also available as a smaller edge-friendly option (2048 context).

### Fixed

- **A model's `chat_template` metadata that's a bare legacy name (e.g. literally
  `"vicuna"`) instead of real Jinja source no longer silently collapses every
  prompt to that one word.** Some GGUF conversions store a pre-Jinja template-name
  hint in `tokenizer.chat_template` (meant for llama.cpp's own name-based
  classifier), and `render_chat_jinja` trusted it as Jinja unconditionally —
  minijinja renders a string with no `{{`/`{%` tags as itself, so the *entire*
  chat prompt became `"vicuna"` for any model shaped this way, with no error.
  Found via real e2e testing against `ggml-org/moondream2-20250414-GGUF` while
  validating vision support (a pre-existing bug, unrelated to vision itself — it
  affects any request to an affected model). Now requires actual Jinja syntax
  before committing to that path, falling through to the legacy name-based
  classifier (which already handles this convention correctly) otherwise.

## [0.16.0]

fox gets **production hardening + native tool calling**: a request now fails fast
instead of hanging forever when the queue is full or the engine hits an
unrecoverable error; a recoverable `llama_decode` OOM retries by bisecting the
batch instead of killing every request in it; tool calls are parsed in the
model's own native wire format (Hermes/Qwen, Mistral) instead of always falling
back to fox's generic prompt-injected listing; and speculative decoding
generalizes beyond repetitive/context-echoing text via an optional draft model.

### Added

- **Backpressure / fail-fast** (`--max-queue-depth` / `FOX_MAX_QUEUE_DEPTH`,
  default unbounded) — `Scheduler::submit()` is now fallible: a full queue is
  rejected with HTTP `429` instead of queueing forever, and an oversized request
  (bigger than the entire KV pool) is rejected synchronously instead of blocking
  the queue head. A real engine failure now gets a distinct `StopReason::EngineError`
  and an explicit terminal token on the response channel instead of silently
  closing it (which used to read as a fake empty `200`).
- **OOM recovery — batch-size bisection retry** — `do_prefill`/`do_decode` now
  distinguish `llama_decode`'s return codes instead of treating any non-zero as
  fatal: `1` ("no KV slot for the batch") retries by splitting the batch in half
  and decoding each half independently — llama.cpp's own documented mitigation —
  recursing down to a single request before giving up. `2`/`-1`/`< -1` stay
  immediately fatal. Observable via the `ferrumox_decode_bisection_retries_total`
  Prometheus counter plus a `tracing::warn!` per retry.
- **Hermes, Mistral, and Llama3 tool-call parsers** (`--tool-call-parser
  auto|generic|hermes|mistral|llama3`, default `auto`) — `tools` is now threaded
  into the Jinja chat-template render context, so a model whose real template
  natively formats tool calls (Hermes/Qwen `<tool_call>{...}</tool_call>`,
  Mistral `[TOOL_CALLS]`) renders and parses its own format instead of fox's
  generic system-message listing, auto-detected from the model's own template.
  The Mistral parser handles both real-world wire formats (the classic JSON array
  and the newer per-call `name[ARGS]{...}`). Llama3 is explicit-opt-in only — most
  GGUF chat templates for Llama3 models strip the tool-calling block entirely, so
  there's no reliable template signal to auto-detect it by. Models without a
  detected/selected native format keep the original generic prompt-based JSON
  parsing as the fallback.
- **Draft-model speculative decoding** (`--draft-model <name>`) — generalizes
  0.15's n-gram speculation beyond repetitive/context-echoing output via a second,
  smaller resident model proposing tokens for the target to verify. Requires
  `--speculative true` (ignored with a startup warning otherwise); the draft and
  target must share the same tokenizer, checked via a vocab-fingerprint at load
  time and failing loudly on mismatch. Loaded once alongside the target and kept
  resident for the process lifetime — not subject to LRU eviction or VRAM
  budgeting in this release, so both models need to be sized to fit together.
  `--spec-ngram` is ignored in this mode. Landed alongside a `Proposer` trait
  extraction so n-gram and draft-model speculation share the same verify/accept
  machinery.



## [0.15.1]

The bug-hunt release. Exercising a **real server end-to-end on the target machine**
(and then code-hunting the subsystem boundaries that pattern pointed at) surfaced six
pre-existing bugs that 173 unit + 39 integration + 11 golden tests all structurally
missed — every one living at a crossing between subsystems (prefix cache × request
lifecycle, rolling × speculation × KV capacity, embeddings × sequence pool). All six
are fixed, and the layer that found them is now permanent: a strict end-to-end smoke
suite (`make e2e`) that runs in CI on every push and gates every release.

### Fixed

- **Prefix-cache reuse no longer breaks the server** (pre-existing, found by
  exercising a real server end-to-end on the target machine). Three related bugs in
  the same subsystem: (1) a finished request that donated its prompt prefix to the
  cache left its *whole* KV (prompt + generated tokens) in the sequence, so the next
  cache hit re-submitted tokens at occupied positions and `llama_decode` failed;
  (2) the decode/prefill error paths recycled the sequence id without clearing its
  KV, permanently poisoning it — every later request assigned that sequence failed
  too; and (3) after a cache hit, `prefilled_tokens` recorded only the *submitted*
  token count instead of the KV's true length, so the hit request's first decode
  landed `skip` positions short — inside occupied cells — and died after one token.
  Donated sequences are now trimmed to exactly the cached prefix, failed requests
  clear their sequence before the id returns to the pool, and the decode position is
  derived from the KV's total length. Guarded by a new golden
  (`golden_prefix_reuse_after_trim`) and the new end-to-end smoke suite.
- **Context rolling now fires with headroom** — the roll triggered exactly *at*
  `n_ctx`, but the step that would cross the boundary (up to `draft_len + 1` cells
  for a speculative verify batch) failed with "no KV slot" *before* the roll ever got
  its chance, killing long generations right at the context boundary. The roll
  threshold now reserves the largest possible next step, so the window slides just
  before the boundary instead of the request dying on it. Found by the new
  context-fill e2e check on real hardware.
- **Rolled generations no longer donate to the prefix cache** (silent-corruption
  class, found by code-hunting subsystem boundaries). Rolling removes the oldest KV
  cells and shifts the survivors down, so a rolled request's cells at positions
  `[0, cached)` are mid-generation tokens — NOT the prompt prefix the cache key
  promises. Donating them would make the next cache hit condition its generation on
  garbage, with no visible error. Requests with `rolled_tokens > 0` now skip donation.
- **Embeddings no longer share a KV sequence with generation.** `do_get_embeddings`
  hardcoded sequence 0 and wiped it after every call — but the scheduler's pool hands
  out ids 0..max_batch, so under full concurrency an embedding request would clobber
  (then erase) a live generation's KV. Embeddings now use a dedicated slot allocated
  beyond the pool (`n_seq + 1`).
- **Admission no longer preempts running requests** (two bugs, one root). LIFO
  preemption on admission could (a) resume a preempted request from the bare prompt
  while its position counter still included the tokens already streamed to the client
  — a positional gap in the KV producing a silently corrupted continuation — and
  (b) **livelock**: when a newcomer and a running request couldn't fit together, the
  newcomer evicted the runner *within the same scheduling step it was re-admitted*,
  so neither ever reached the engine again (total starvation, reproduced by unit
  test). Since fox fully reserves a request's blocks at admission (prompt +
  max_new_tokens), running requests never grow and admission preemption was
  unnecessary to begin with: a request that doesn't fit now simply waits (FIFO), and
  a request larger than the entire pool is rejected instead of blocking the queue
  head forever.
- **Concurrent requests for a cold model no longer load it twice.** Two simultaneous
  requests for an unloaded model both passed the "already loaded?" check and each
  loaded the full GGUF (transient double VRAM — an OOM risk on iGPUs), and the second
  registry insert dropped the first entry, aborting its in-flight generations. Loads
  are now single-flight (serialized with a re-check).
- **Eviction no longer kills models with requests in flight.** `last_used` marks
  request *start*, so a long generation looked idle and the keep-alive sweep (default
  300s) would evict — and thereby abort — it mid-generation; LRU eviction at capacity
  had the same flaw. Both eviction paths now skip busy models (active or queued
  requests).

### Changed

- **`InferenceEngine::new` takes an `EngineOptions` struct** (prefill chunking,
  context shift, speculation) instead of a growing tail of positional
  `Option` arguments; oversized war-story comments trimmed to their load-bearing
  constraint; shared sampler-parameter construction deduplicated.

### Added

- **`make e2e` smoke suite** (`scripts/e2e_smoke.sh`) — starts a real server with
  a real model and drives it over HTTP across multiple requests: the prefix-cache
  donate→hit lifecycle, guided decoding, logprobs, sampling controls, the Ollama
  surface, speculation, streaming (SSE + NDJSON), four concurrent clients, a
  context-window fill that forces rolling mid-generation, a re-request after a rolled
  generation, a mid-stream client disconnect, and embeddings alongside chat. Runs in
  CI's golden job on every push; it covers the cross-request layer that
  unit/golden/stub tests structurally cannot reach (which is exactly where all six
  bugs above were hiding).

## [0.15.0]

fox gets **speculative decoding**. On single-request decode steps it guesses the next
few tokens by matching the recent output against the request's own history (n-gram /
prompt-lookup — no draft model, no extra memory) and verifies all the guesses in one
forward pass. The output is provably unchanged — a fixed sampler produces byte-identical
text with speculation on or off; only speed changes. Measured on a real model:
**1.78× faster at 98% draft acceptance on repetitive output** (code edits, JSON, RAG),
0.92× at 9% on free-form prose — which is why `--speculative` ships off by default.
Acceptance is observable in Prometheus, and a new `fox bench-spec` quantifies the
trade-off on any model. See `docs/design/speculative-decoding.md`.

### Added

- **Speculative decoding — n-gram / prompt-lookup** (`--speculative` /
  `FOX_SPECULATIVE`, default off; `--spec-ngram`, default 2; `--spec-draft-len`,
  default 4) — on single-request decode steps, fox guesses the next few tokens by
  matching the recent output against the request's own history and verifies all the
  guesses in **one** forward pass, committing however many the model agrees with.
  Output is provably unchanged — every committed token is a genuine model sample, so a
  fixed sampler produces byte-identical text with speculation on or off (golden
  `golden_speculative_matches_greedy`); only speed changes. Fastest on
  context-echoing output (code edits, JSON, RAG, repetition). Needs no draft model and
  no extra memory. Skipped while a request uses guided decoding; multi-request batches
  decode normally. Flagship of the 0.15 work (`docs/design/speculative-decoding.md`).
- **Speculation observability** — Prometheus counters
  `ferrumox_spec_tokens_proposed_total` / `ferrumox_spec_tokens_accepted_total` and the
  `ferrumox_spec_acceptance_ratio` gauge report how well drafting is working on a live
  server.
- **`fox bench-spec`** — runs the same greedy generation with speculation off and on
  (repetitive and prose workloads), reports tok/s, acceptance and speedup, and verifies
  the off/on outputs are byte-identical — the exactness invariant checked end-to-end
  (`docs/cli/bench-spec.md`).

## [0.14.0]

fox gains **structured and controllable output**. Guided decoding constrains generation
to a grammar via llama.cpp's core GBNF sampler, so `response_format` / Ollama `format`
now *guarantee* valid (and schema-conforming) JSON instead of hoping for it — with the
JSON-schema→GBNF converter written in Rust. Alongside it, the OpenAI chat endpoint
exposes token `logprobs`/`top_logprobs`, and the sampler grows `min_p`, `logit_bias`
and `min_tokens` — knobs that were previously accepted and silently dropped. Every piece
rides the regression net (golden tests on a real model + stub unit/integration tests).
See `docs/design/structured-output.md`.

### Added

- **Guided decoding / structured output** — fox can now *constrain* generation to a
  grammar instead of hoping the model produces valid JSON. Set OpenAI
  `response_format` (`{"type":"json_object"}` or `{"type":"json_schema","json_schema":
  {"schema":…}}`) or Ollama `format` (`"json"` or a JSON-schema object), and every
  sampled token is masked to the grammar-legal set via llama.cpp's core GBNF sampler
  before fox's sampler picks within it — so the output always parses. JSON-schema is
  converted to GBNF in Rust (`type`, `properties`+`required`, `items`, `enum`, nesting).
  A schema fox can't convert is a `400`, never a silent unconstrained fallback.
  Verified on a real model (golden `golden_grammar_constrains_output`,
  `golden_json_schema_constrains_to_valid_json`). First item of the 0.14
  structured-output work (`docs/design/structured-output.md`).
- **Token log-probabilities** — the OpenAI chat endpoint now honours `logprobs` and
  `top_logprobs` (0–20). Each generated token reports its natural-log probability plus
  the most-likely alternatives (`choices[].logprobs.content[]`), on both streaming
  chunks and non-streaming responses. Computed from the logits fox already produces, so
  no extra inference cost; the log-softmax core is unit-tested. logprobs reflect the
  model's raw distribution (before any guided-decoding grammar mask).
- **More sampling controls** — `min_p` (drop tokens below `min_p × max_prob`; OpenAI
  and Ollama `options.min_p`), `logit_bias` (per-token additive bias, OpenAI; ±100
  bans/forces a token), and `min_tokens` (suppress end-of-generation until at least N
  tokens are produced; OpenAI). All are honoured instead of being silently dropped.

## [0.13.0] - 2026-07-21

fox becomes a real server under concurrent, long-prompt, long-conversation load. The
three serving-robustness gaps from the 0.12 capabilities checklist are closed
(`docs/design/serving-robustness.md`): **chunked prefill** breaks a long prompt into
per-step chunks so it interleaves with other requests' generation instead of
head-of-line-blocking the engine loop; **context rolling** discards the oldest KV
window when a conversation fills `n_ctx` so generation continues instead of stopping
with `length`; and the **Jinja chat template is compiled once and cached** instead of
re-parsed on every request. A new `fox bench-prefill` quantifies the chunked-prefill
win, and every change rides the 0.12 regression net (golden tests in CI + the scheduler
conservation stress test).

### Added

- **Chunked prefill** (`--max-prefill-chunk` / `FOX_MAX_PREFILL_CHUNK`, default 512) —
  a long prompt is now prefilled in chunks of at most N tokens per scheduler step
  instead of one giant `llama_decode`, so it interleaves with other requests' token
  generation instead of head-of-line-blocking the whole engine loop. A request stays
  `Prefilling` across steps until its prompt is fully in the KV cache, then samples and
  moves to `Decoding`. `0` disables chunking (single-shot). Verified byte-for-byte
  equivalent to single-shot on a real model (`golden_chunked_prefill_matches_single_shot`)
  plus a scheduler state-machine unit test. First flagship item of the 0.13
  serving-robustness work (`docs/design/serving-robustness.md`).
- **`fox bench-prefill`** — a validation benchmark for chunked prefill. It submits a
  long prompt and a concurrent short request, then reports the short request's *worst
  stall* (largest gap between its tokens, including time-to-first-token) for each
  `--max-prefill-chunk` value. Chunking bounds that stall to one chunk's prefill;
  single-shot (`--chunks 0`) balloons it to the full long-prompt prefill, so one run
  quantifies the win (`docs/cli/bench-prefill.md`).
- **Context rolling on full** (`--context-shift` / `FOX_CONTEXT_SHIFT`, default on;
  `--context-keep` / `FOX_CONTEXT_KEEP`, default 0) — when a conversation fills the
  context window, fox now discards the oldest KV window (preserving the first
  `--context-keep` head tokens) and shifts the rest down via `llama_memory_seq_rm` +
  `llama_memory_seq_add`, so generation continues instead of stopping the request with
  `length`. Recurrent/hybrid models whose KV cache can't shift keep the old
  stop-with-`length` behavior. Verified on a real model
  (`golden_context_shift_continues_past_n_ctx`) plus a scheduler unit test. Second
  serving-robustness item of 0.13 (`docs/design/serving-robustness.md`).

### Changed

- **Chat template compiled once, not per request** — the model's Jinja chat template
  was re-parsed on every chat request. The `minijinja` environment (with the template
  parsed and the pycompat callback installed) is now built lazily and cached on the
  model, so only the render runs per request. Behaviour-identical; covered by a new
  golden test (`golden_chat_template_renders`) that asserts a non-empty, deterministic
  render across calls. First item of the 0.13 serving-robustness work
  (`docs/design/serving-robustness.md`).

## [0.12.0] - 2026-07-06

GPU inference becomes a first-class, reproducible path, and the model-architecture
rework (started in 0.11) is finished off. Vulkan is validated end-to-end on an AMD
Radeon 890M (`gfx1150`, RDNA 3.5) and shipped three ways — a Docker image, a prebuilt
release tarball, and a `make vulkan` bundle — with fox now reporting the active
backend at startup. On the correctness side, P4 (API consistency) lands, and the
rework's regression net is wired up for real: golden tests run in CI against a live
model, and a stress test settles the last open question (§7) by proving the prefix
cache doesn't leak.

### Added

- **`Dockerfile.vulkan`** — a reproducible Vulkan build for AMD/Intel iGPUs and any
  Vulkan-capable GPU (no CUDA/ROCm). Validated end-to-end on an AMD Radeon 890M
  (`gfx1150`, RDNA 3.5): coherent output, GPU-accelerated, both by extracting the
  binary to run natively and by running the image with `--device /dev/dri`. The image
  ships the Mesa Vulkan driver and falls back to CPU when no GPU is present.
  CONTRIBUTING documents the GPU-build story, including the exact toolchain
  (`glslc`, `glslang-tools`, `libvulkan-dev`, `spirv-headers`) and the
  build-in-container / run-on-host split.
- **fox reports the active compute backend at startup.** `fox run`, `fox serve` (in
  the log) and `fox probe` now show whether inference runs on the GPU (e.g.
  `Vulkan0 — AMD Radeon 890M`) or the CPU, read from the ggml device registry —
  closing the "is it actually using my GPU?" gap. Exposed on `ModelInfo.backend`.
- **Prebuilt Vulkan binary in releases** — `release.yml` now builds a
  `x86_64-unknown-linux-gnu-vulkan` tarball (on Ubuntu 24.04) alongside the CPU one,
  so GPU users get a ready-to-run binary. The Vulkan tarball needs glibc 2.39+ and a
  Vulkan driver (Mesa RADV/ANV, etc.) at runtime.
- **`make vulkan`** — builds the `Dockerfile.vulkan` image and extracts the bundle
  (`fox`, `fox-bench`, `libggml-vulkan.so`) to `./fox-vulkan/`, so you get a
  GPU-enabled binary that runs natively on any host with a Vulkan driver — no build
  toolchain needed on the host.
- **Golden tests now run in CI** — a new `golden` job builds llama.cpp for real (the
  only CI job that does; the rest stay on the fast stub) and runs the golden suite
  against a tiny GGUF (Qwen2.5-0.5B) on CPU: `ModelInfo` invariants, non-degenerate
  embeddings, and tokenize round-trips on emoji/CJK. The model and the llama.cpp build
  are cached so it only pays the full cost when either changes. This wires up the
  regression net that P0 built but only ran locally.
- **Prefix-cache leak stress test** (`scheduler::tests::stress_prefix_cache_no_leak`) —
  settles the last open question of the model-architecture rework (§7). It drives 400
  admit/finish/cache/hit/refuse-when-full cycles and asserts, after every step, that
  every seq_id and KV block is owned by exactly one of {pool, running request, cache
  entry} — never dropped, never duplicated — and that allocation returns to zero after
  draining. Confirms the prefix cache does **not** leak (the initial automated flag was
  a false positive). Adds `KVCacheManager::allocated_blocks()` for the assertion.

### Changed

- **Sampling defaults centralized** (`src/api/shared/sampling_defaults.rs`) — the
  per-request defaults were duplicated as magic literals across the OpenAI and Ollama
  handlers. They now live in one table keyed by API surface, with the cross-surface
  divergence documented as a **deliberate** decision: the OpenAI surface (`/v1/*`)
  mirrors OpenAI (no `top_k`, no repeat penalty) while the Ollama surface (`/api/*`)
  mirrors upstream Ollama (`top_k = 40`, `repeat_penalty = 1.1`). A unit test locks
  the divergence so it can't be "unified" by accident. (Model-architecture rework P4.)

### Fixed

- **API docs listed the wrong sampling defaults.** `docs/api/{openai,ollama}.md`
  claimed `temperature = 1.0` / `top_p = 1.0` (actual: `0.8` / `0.9`) and the Ollama
  page showed `top_k = 0` / `repeat_penalty = 1.0` when fox actually applies Ollama's
  `40` / `1.1`. Corrected, with a note explaining the deliberate `/v1` vs `/api`
  divergence.
- **`--max-models` help now states the default-1 trade-off** — a request for a second
  model evicts the first (logged), which is the safe choice for small-VRAM iGPUs;
  raise it if you have the VRAM.

---

## [0.11.0] - 2026-07-03

Model-architecture correctness rework (see
`docs/design/model-architecture-rework.md`) — makes per-model facts a single
inspectable source of truth and closes several "fix one model, break another" gaps.

### Added

- **`fox probe <model>`** — loads a model and prints its resolved `ModelInfo`
  (architecture, `n_embd`, head counts, `head_dim`, layers, trained context, EOS,
  embedded-template presence, native-thinking/seq-copy, recommended sampling), then
  flags **contradictions** between the model's metadata and the formulas fox uses.
  Unlike `fox show` (which guesses from the filename), probe reads the truth.
- **`ModelInfo`** — one inspectable snapshot of a loaded model's facts, the basis of
  the rework and of `fox probe`.
- **Golden regression tests** (`make golden GOLDEN_MODEL=<path.gguf>`) — real-model
  assertions (ModelInfo invariants, non-degenerate embeddings, tokenize round-trip)
  that lock in the fixes below. Gated to real builds; the stub CI is unaffected.
- **Community-health files** — `CODE_OF_CONDUCT.md`, issue templates (bug/feature)
  and a pull-request template. `Cargo.toml` gains package metadata (`repository`,
  `homepage`, `documentation`, `keywords`, `categories`).

### Changed

- **CI/CD workflows simplified to stop failing.** The Docker and Release workflows
  dropped their fragile multi-platform matrices (arm64/CUDA, ROCm apt bundle,
  aarch64 cross, Windows+Vulkan, macOS) — which failed often — for a single reliable
  `linux/amd64` build; Docker now builds+pushes directly (no push-by-digest/manifest
  merge). The redundant `test-linux-build` workflow (only ran on `-test` tags, pinned
  rotting ROCm versions) was removed. GPU users can use the Docker image or build
  locally; platforms will be re-added once each is verified in isolation.

- **Chat prompts now execute the model's real Jinja template** (via `minijinja` +
  `minijinja-contrib` pycompat) instead of llama.cpp's simplified built-in format,
  and tokenize the result with the template's own BOS (`add_special=false`, no
  double BOS) and real control tokens (`parse_special=true`, not literal text). The
  prompt now matches what each model was trained on. Falls back to the built-in
  format when a model has no embedded template or it fails to render.

- **Thinking/reasoning is now opt-in and correctly detected.** `supports_thinking`
  recognizes models whose chat template exposes an `enable_thinking` toggle
  (Gemma-4, Qwen3), not just the `<think>`-token heuristic — `fox probe` now reports
  `Native thinking: yes` for Gemma. Thinking activates only when the request opts in
  (OpenAI: a `think: true` extension field; Ollama: the existing `think` field) and
  threads `enable_thinking` into the model's Jinja template; the default is off (no
  reasoning latency unless asked). Note: clean separation of reasoning from the
  answer works for both `<think>`-delimited models (Qwen3, DeepSeek-R1) and
  channel-format models (Gemma's `<|channel>`/`<channel|>`): the output filter AND
  the API-layer thinking extraction read each model's reasoning markers via
  `Model::reasoning_delimiters`, detected from the model's OWN chat template through
  a small documented format registry (`REASONING_FORMATS`) — never the model name.
  Supporting a new reasoning format is one registry line plus a golden test.
  Tool-calling through the template remains a follow-up.

### Fixed

- **Embeddings returned an all-zeros vector** for every model. The generation
  context uses `pooling_type = NONE`, so `llama_get_embeddings_seq` returned NULL and
  fox served a zero vector. `/v1/embeddings` and `/api/embed` now mean-pool the
  per-token embeddings (`llama_get_embeddings_ith`) and L2-normalize the result.
- **`n_embd` was reconstructed as `num_heads * head_dim`**, wrong for Gemma/MLA-class
  models (`head_dim != n_embd/n_head`). This produced wrong-length embeddings and an
  out-of-bounds read of the embedding buffer. `n_embd` is now read from
  `llama_model_n_embd` and stored on `ModelConfig`.
- **The KV block pool was sized from an independent formula** that could disagree with
  the backend's real `n_ctx` (and is wrong for shared/SWA KV, MLA and recurrent
  models), letting fox over-claim KV and crash `llama_decode` under load. The serving
  paths now size the pool from `llama_n_ctx` so it follows the backend exactly.
- **`frequency_penalty` / `presence_penalty` were accepted but silently ignored.**
  They are now applied in the sampler with OpenAI semantics
  (`logit -= presence*(seen) + frequency*count`), threaded from the request. Default
  0.0 (disabled).
- **Misleading load-failure message.** `diagnose_load_failure` asserted "not enough
  memory" on *any* failure (its condition was almost always true), even when the real
  cause was a missing compute backend. It now claims OOM only when free memory is
  actually below the model size, and otherwise lists the real possible causes.
- **Image/audio content is no longer dropped silently.** The OpenAI handler now
  warns when a request carries non-text content blocks (fox has no vision/audio
  support). `--swap-fraction` is documented as reserved/not-yet-implemented rather
  than appearing to do something.

---

## [0.10.0] - 2026-06-30

Re-baselines the project after retracting a premature `1.0.0`. The version line continues at `0.10.x` and will reach `1.0.0` only when the engine is proven stable. This release also migrates the vendored llama.cpp to upstream and removes TurboQuant.

### Changed

- **Vendored llama.cpp now tracks upstream** (`ggml-org/llama.cpp` @ `b9842`). Previously the
  submodule pointed at a long-lived fork that carried the TurboQuant patches and had drifted
  ~1,200 commits / 3 months behind upstream. Tracking upstream restores binding/library version
  parity, lets a clean `--recurse-submodules` clone fetch the pinned commit, and unblocks newer
  model architectures (e.g. Gemma 4) without maintaining a fork. The migration required only a
  build-flag fix in `build.rs` (`LLAMA_BUILD_APP/UI=OFF`) — zero FFI changes.

### Removed

- **TurboQuant KV cache quantization** (`turbo2` / `turbo3` / `turbo4`). The fork's custom GGML
  type IDs collided with upstream's (e.g. `Q1_0=41` vs `TURBO3_0=41`), making it impossible to
  follow upstream while keeping TurboQuant. Compatibility with upstream llama.cpp was prioritized.
  KV cache quantization remains available via the standard llama.cpp types `f16`, `q8_0` and
  `q4_0`. Configs or commands referencing `turbo*` KV types must switch to one of these.

---

## [0.9.0] - 2026-03-15

### Added

- **Multi-Model support** — `ModelRegistry` loads and serves multiple models simultaneously
  with LRU eviction.
  - New `src/model_registry.rs`: `ModelRegistry`, `EngineEntry`, `RegistryConfig`.
  - `GET /api/ps` now lists **all** currently-loaded models (previously only the one model).
  - `GET /v1/models` now lists **all** `.gguf` files in `models_dir` (not just the loaded one).
  - Each inference/embedding request is routed to the correct engine based on the `model` field;
    unknown models return HTTP 404.
  - `DELETE /api/delete` now also unloads the model from the registry if it was loaded.

- **`--max-models` flag** (`FOX_MAX_MODELS` env var, default `1`) — maximum number of models
  kept in memory simultaneously; excess models are evicted LRU-first.

- **`--alias-file` flag** (`FOX_ALIAS_FILE` env var) — optional TOML file mapping short names
  to model stems (e.g. `"llama3" = "Llama-3.2-3B-Instruct-f16"`).
  Default path: `~/.config/ferrumox/aliases.toml`.

### Changed

- `AppState` replaces `engine: Arc<InferenceEngine>` with `registry: Arc<ModelRegistry>` +
  `primary_model: String`. Backward-compatible: `fox serve --model-path X.gguf` works unchanged.
- `router()` signature updated accordingly.
- Engine run-loop is now started inside `ModelRegistry::get_or_load` and aborted automatically
  on LRU eviction via `Drop` on `EngineEntry`.

---

## [0.8.0] - 2026-03-15

### Added

- **Embeddings API** — unlocks RAG pipelines (LangChain, LlamaIndex, Open WebUI RAG, etc.)
  - `POST /v1/embeddings` — OpenAI-compatible endpoint; accepts `input` as a string or array
    of strings, returns `data[].embedding` vectors.
  - `POST /api/embed` — Ollama-compatible endpoint; returns `embeddings: [[f32]]`.
  - `InferenceEngine::embed()` async method; `Model::get_embeddings()` + `Model::embedding_dim()`
    trait methods with full `LlamaCppModel` implementation via `llama_set_embeddings` /
    `llama_get_embeddings_seq` FFI and stub fallback.
  - New types: `EmbeddingInput` (untagged enum for String/Vec<String>), `EmbeddingRequest`,
    `EmbeddingObject`, `EmbeddingUsage`, `EmbeddingResponse`, `OllamaEmbedRequest`,
    `OllamaEmbedResponse`.

- **`POST /api/pull` with SSE streaming** — download models from HuggingFace Hub via the
  server API, identical to Ollama's pull flow.
  - Emits newline-delimited JSON events: `pulling manifest` → `downloading` (with `digest`,
    `total`, `completed` bytes) → `verifying sha256 digest` → `success`.
  - Automatically selects Q4_K_M quantization when available, otherwise picks the first GGUF.
  - New `--hf-token` flag on `fox serve` (also `HF_TOKEN` env var) forwarded to pulls.
  - New `AppState.hf_token` field; new file `src/api/pull_handler.rs`.
  - New types: `PullRequest`, `PullStatus`.

- **Release binaries + `install.sh`** — one-command installation.
  - `.github/workflows/release.yml` — triggered on `v*` tags; builds for four targets:
    `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`,
    `aarch64-apple-darwin`. Uploads tarballs as GitHub Release assets.
  - `install.sh` — detects OS + arch, downloads the correct tarball, installs to
    `/usr/local/bin/fox` (configurable via `--prefix`).
  - `fox.service` — systemd unit for running `fox serve` as a daemon.

### Changed

- `Cargo.toml`: version bumped to `0.8.0`.
- `src/api/routes.rs`: `router()` now takes an extra `hf_token: Option<String>` parameter.
- `src/cli/serve.rs`: `ServeArgs` gains `--hf-token` / `HF_TOKEN`.

---

## [0.7.0] - 2026-03-14

### Added

- **Ollama-compatible API layer** (`src/api/routes.rs`, `src/api/types.rs`)
  - `GET /api/tags` — lists all `.gguf` models in `~/.cache/ferrumox/models/` with name,
    size, SHA256 digest, architecture family, quantization level, and `modified_at` timestamp.
    Open WebUI and Continue.dev use this endpoint to discover available models.
  - `GET /api/ps` — returns the currently loaded model with real file size (bytes) and
    SHA256 digest looked up from disk.
  - `POST /api/show` — returns detailed metadata for a named model: architecture family,
    quantization, human-readable size, digest, modification date, and file path.
  - `DELETE /api/delete` — removes a `.gguf` file from the models directory by model name
    or filename. Returns `404` if the model is not found.
  - New response types: `OllamaModel`, `OllamaDetails`, `TagsResponse`, `PsEntry`,
    `PsResponse`, `ShowRequest`, `ShowResponse`, `DeleteRequest`.
  - SHA256 digest computed once per file via `sha2` + `hex` and cached in `AppState`
    (`Arc<Mutex<HashMap<PathBuf, String>>>`). Subsequent requests for the same file return
    instantly.
  - New dependencies: `sha2 = "0.10"`, `hex = "0.4"`.

- **`models_dir` added to `AppState`** (`src/api/routes.rs`, `src/cli/serve.rs`)
  - `router()` now accepts a `models_dir: PathBuf` parameter (default:
    `~/.cache/ferrumox/models`) used by the Ollama-compat handlers.
  - `src/cli/show::parse_architecture` and `parse_quantization` promoted to `pub(crate)`
    so they can be reused by the API layer without duplication.

### Compatibility

With v0.7.0, **Open WebUI** and **Continue.dev** work out of the box by pointing their
Ollama URL to `http://localhost:8080`. No other configuration change is required.

---

## [0.6.0] - 2026-03-13

### Added

- **CLI visual overhaul — minimalista con color** (`src/cli/theme.rs`, all CLI modules)
  - New `src/cli/theme.rs` module centralises all ANSI styling. Respects `NO_COLOR` and
    non-TTY contexts (pipes, CI) — every helper silently falls back to plain text.
  - New direct dependency: `crossterm = "0.28"`.
  - **`fox run` loading spinner** — replaces the static `"Loading model… done."` line with a
    cyan Braille spinner (`indicatif`) that clears itself and prints `  ✓  Model loaded.`
    (bold green) on success.
  - **REPL banner** — after load, prints `🦊  <model name>` (bold white), a dim separator
    and a dim hint line (`/bye o Ctrl+D para salir · N tokens`).
  - **Prompt glyph** — `  ❯ ` (bold cyan) replaces `"You: "`.
  - **Thinking spinner** — a dim Braille spinner labelled `"Thinking…"` runs while the model
    generates; cleared on the first emitted token.
  - **Role label** — `  Fox  ` (bold yellow) is printed once to stderr immediately before the
    first token, producing `  Fox  <streamed response>` inline.
  - **Per-turn timing** — dim `  N tokens · X.Xs` line printed after each assistant turn.
  - **`fox list`** — table header bold, separator dim, SIZE column blue, MODIFIED dim.
  - **`fox ps`** — table header bold, separator dim; STATUS `ok` → bold green; KV cache
    usage colour-coded (green < 50 %, yellow < 80 %, red ≥ 80 %).
  - **`fox show`** — all key/value rows use `theme::print_kv_pair` (key bold+dim, padded).
  - **`fox pull`** — post-download success line uses `  ✓  Saved to …` (bold green); hint
    lines for `fox run` / `fox serve` are dimmed.
  - **`fox serve`** — prints `  🦊  <model>  ·  listening on <addr>` (green) to stderr when
    the server is ready.

- **Interactive REPL mode for `fox run`** (`src/cli/run.rs`)
  - Running `fox run --model-path model.gguf` without a prompt now opens a conversational chat session.
  - Full message history is maintained across turns: each new turn sends the complete history through `apply_chat_template`, giving the model proper context.
  - Exit commands: `/bye`, `/exit`, `exit`, `quit`, or Ctrl+D (EOF).
  - Existing one-shot behavior (`fox run --model-path model.gguf "prompt"`) is fully preserved.
  - The engine loop stays alive across turns; no model reload between messages.

### Changed

- **Project renamed from `ferrum-engine` to `ferrumox`** — the CLI binary is now `fox`, the benchmark binary is `fox-bench`.
  - All environment variables renamed from `FERRUM_*` to `FOX_*` (e.g. `FOX_MODEL_PATH`, `FOX_PORT`).
  - Model cache directory changed from `~/.cache/ferrum/models` to `~/.cache/ferrumox/models`.
  - Prometheus metric names updated from `ferrum_*` to `ferrumox_*`.
  - Build stub flag renamed from `FERRUM_SKIP_LLAMA` to `FOX_SKIP_LLAMA`.
  - Docker image tag changed from `ferrum-engine:latest` to `ferrumox:latest`.

---

## [0.5.1] - 2026-03-12

### Added

- **`--show-thinking` flag for `ferrum run`** (`src/cli/run.rs`, `src/scheduler/batch.rs`, `src/engine/mod.rs`)
  - New `SamplingParams::show_thinking: bool` field (default `false`).
  - When `--show-thinking` is passed, the model's `<think>…</think>` reasoning block is
    forwarded to stdout instead of being silently discarded. The `<think>` and `</think>`
    tags themselves are also emitted so the user sees the complete block.
  - Thinking tokens are still excluded from API responses (`show_thinking = false` in
    `src/api/routes.rs`).
  - `PerRequestState` initialised with `show_thinking` taken from the request's
    `SamplingParams` on first token arrival (`or_insert_with` instead of `or_default`).

### Fixed

- **EOG token detection for multi-token-EOS models** (`src/engine/model.rs`, `src/engine/mod.rs`)
  - `is_eos` was computed as `token_id == self.model.eos_token_id()`, which only matched the
    *primary* EOS token.  Models like Qwen3.5 declare five EOG tokens
    (`<|endoftext|>`, `<|im_end|>`, `<|fim_pad|>`, `<|repo_name|>`, `<|file_sep|>`).
  - New `Model::is_eog_token(token_id) -> bool` method added to the trait and both
    implementations (`LlamaCppModel` delegates to `ffi::llama_vocab_is_eog(vocab, token)`;
    stub returns `token_id == 2`).
  - `handle_logits` now uses `self.model.is_eog_token(token_id)` so any EOG token
    correctly stops generation and produces empty output text.

- **Multi-token `<|im_end|>` leaking into user output** (`src/engine/mod.rs`)
  - Qwen3.5 (and other ChatML models running without forced-greedy sampling) may generate
    `<|im_end|>` as six individual BPE tokens: `<` (27), `|` (91), `im` (316), `_end` (6018),
    `|` (91), `>` (29) instead of the single special token 248046.  The previous per-token
    `raw.contains("<|")` check missed this because no individual fragment matched.
  - **New two-stage output pipeline**:
    1. `apply_output_filter` now returns `(String, bool)` — the emittable text plus a
       `control_stop` flag.  Text is buffered in `state.pending_output` before being
       released; `flush_pending_output` scans for complete control-token patterns
       (`CONTROL_TOKEN_PATTERNS`) and calls `find_holdback_start` to hold back any suffix
       that could still be the beginning of such a pattern.
    2. `find_holdback_start(text)` — returns the index of the first `<` from which *some*
       control-token pattern could start (i.e. the pattern `starts_with` the suffix).
       Everything before that index is safe to emit immediately; the rest stays in
       `pending_output` for the next token.
  - `handle_logits` combines the two stop signals:
    `is_stop_hit = control_stop || user_stop` (where `user_stop` comes from
    `check_stop_sequences` as before).
  - `SPECIAL_TOKEN_PATTERNS` renamed to `CONTROL_TOKEN_PATTERNS` to better reflect their
    role (end-of-turn markers that must never reach the user and must stop generation).
  - `check_stop_sequences` reverted to handle only *user-supplied* stop strings; control
    patterns are fully owned by `apply_output_filter` / `flush_pending_output`.
  - **New unit tests** covering the two-stage pipeline:
    - `test_filter_control_single_token_stopped` — single-token `<|im_end|>` triggers stop.
    - `test_filter_control_multi_token_im_end` — 5-token sequence triggers stop only at `>`.
    - `test_filter_holdback_released_on_non_pattern` — `<x` releases `<` when `x`
      confirms the sequence cannot be a control pattern.
    - `test_filter_text_before_control_token_emitted` — normal text before `<|im_end|>`
      is emitted correctly and the pattern itself stops generation.

---

## [0.5.0] - 2026-03-12

### Added

- **Block-level chain-hash prefix caching** (`src/kv_cache/mod.rs`, `src/scheduler/mod.rs`, `src/engine/mod.rs`)
  - Added `compute_block_hash(parent_hash, tokens) -> u64` and
    `prompt_block_hashes(tokens, block_size) -> Vec<u64>` to `kv_cache/mod.rs`.
    Each block's hash chains the previous block's hash with the block's token IDs
    (same design as vLLM).  Two prompts that share their first N complete blocks
    therefore produce the same chain hash at each of the first N boundaries.
  - `schedule_step` now computes block hashes on admission and searches the cache
    from the longest matching block prefix down to 1 block, enabling partial prefix
    matches: a request whose prompt starts with the same system prompt as a previous
    request reuses those cached blocks even if the rest of the prompt differs.
  - `try_insert_prefix` no longer accepts an external `token_hash` parameter; it
    computes the chain hash internally.  Only the *complete* block prefix of the
    prompt is stored — partial trailing blocks and all generation blocks are freed
    immediately, reducing memory pressure.
  - `PrefixCacheEntry.token_count` removed (derivable as `block_ids.len() × block_size`).
  - New test: `test_prefix_cache_block_level_partial_match` — verifies that request B
    (prompt = shared 16-token prefix + 4 different tokens) gets a prefix hit against
    request A (prompt = the same 16 tokens) and has `skip_prefix_tokens = 16`.

- **True copy-on-write before decode** (`src/kv_cache/mod.rs`, `src/scheduler/mod.rs`, `src/engine/mod.rs`)
  - `KVCacheManager::is_shared(block_id) -> bool` — returns `true` when `ref_count > 1`.
  - `Scheduler::cow_update_page_table(req_id, logical_idx, new_block_id)` — replaces a
    single page-table entry for a running request (called by the engine's CoW path).
  - `InferenceEngine::run_decode` now inspects every block in each request's page table
    before issuing `decode_sync`.  Any block with `ref_count > 1` is privatised via
    `KVCacheManager::copy_on_write`; the new exclusive block ID is written back via
    `cow_update_page_table`.  This guarantees a decoding request never writes into a
    block shared with the prefix cache or another future request.

- **`RequestState::Swapped` + CPU↔GPU swap scaffold** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`, `src/cli/serve.rs`, `src/cli/run.rs`)
  - New `RequestState::Swapped` variant with full documentation on the intended
    semantics and current API limitation (byte-level KV tensor transfer requires
    low-level buffer access not yet exposed by llama.cpp's public API).
  - `Scheduler::swap_out(req_id) -> bool` — transitions a `Decoding` request to
    `Swapped`; caller is responsible for the GPU→CPU KV copy before calling.
  - `Scheduler::swap_in(req_id) -> bool` — transitions a `Swapped` request back to
    `Decoding`; caller is responsible for the CPU→GPU KV copy before calling.
  - `--swap-fraction` flag added to both `ferrum serve` and `ferrum run` (env:
    `FERRUM_SWAP_FRACTION`, default `0.0`).  Accepted but no-op until the llama.cpp
    transfer API is available; enables future configuration files to specify the flag
    without breaking.

### Changed

- `engine/mod.rs` local `hash_tokens` function removed (was using `DefaultHasher`);
  replaced by `kv_cache::compute_block_hash` / `prompt_block_hashes`.
- `DefaultHasher` and `std::hash::{Hash, Hasher}` imports removed from `engine/mod.rs`.

---

## [0.4.0] - 2026-03-11

### Added

- **Unified `ferrum` CLI with subcommands** (`src/cli/`, `src/main.rs`, `Cargo.toml`)
  - The project now ships a single `ferrum` binary (renamed from `ferrum-engine`) with three
    subcommands dispatched via `clap`:
  - `ferrum serve` — start the OpenAI-compatible HTTP server. Accepts all previous flags plus
    `--system-prompt <text>` (injected as the first system message if not already present) and
    `--json-logs`. Logic extracted from `main.rs` into `src/cli/serve.rs`.
  - `ferrum run <prompt>` — single-shot terminal inference: loads the model, runs prefill +
    decode, streams tokens to stdout, then exits. Useful for quick one-off queries without
    running a server. Flags: `--model-path`, `--temperature`, `--top-p`, `--top-k`,
    `--repetition-penalty`, `--seed`, `--max-new-tokens`, `--system-prompt`,
    `--no-system-prompt`, `--ctx-len`, `--gpu-memory-fraction`, `--verbose`.
    Implemented in `src/cli/run.rs`.
  - `ferrum pull <model-id>` — download a GGUF model from HuggingFace Hub. Fetches the model
    file list from the Hub API, presents an interactive selector when multiple GGUF files are
    found (`dialoguer`), downloads with a live progress bar (`indicatif`), and saves to
    `--output-dir` (default: `./models/`). Supports `--hf-token` for private repositories.
    Implemented in `src/cli/pull.rs`.
  - `src/config.rs` deleted; all configuration is now owned by the CLI arg structs.
  - New dependencies: `indicatif = "0.17"`, `dialoguer = "0.11"`.

- **Configurable system prompt for the HTTP server** (`src/api/routes.rs`)
  - `AppState` struct introduced to hold `Arc<InferenceEngine>` + `Option<String>` system
    prompt. `router()` now takes the prompt as a parameter and injects it into every
    `chat/completions` request that doesn't already have a system message.

- **13 sampler unit tests** (`src/engine/model.rs`)
  - `sample_greedy`: argmax correctness, single-token input, tie-breaking behaviour.
  - `apply_repetition_penalty`: positive/negative logit cases, no-op on empty history,
    out-of-range token IDs.
  - `sample_token`: greedy path at temperature ≤ 0, seeded reproducibility, top-K candidate
    restriction (50-sample Monte Carlo), top-P nucleus restriction (dominant token always
    sampled), repetition penalty overrides raw logit ranking.

### Fixed

- **KV cache positional gap on hybrid/recurrent models** (`src/scheduler/batch.rs`, `src/scheduler/mod.rs`, `src/engine/model.rs`, `src/engine/mod.rs`)
  - When a prefix-cache hit was used, the decode step was starting at position
    `prompt_tokens.len() + generated_tokens` instead of the number of tokens actually
    submitted to llama.cpp. For models with a recurrent memory backend (Qwen3.5, Mamba)
    this caused `find_slot: non-consecutive token position` warnings and incoherent output.
  - Fix: added `prefilled_tokens: usize` to `InferenceRequest` (initialised to 0). After
    prefill, `InferenceEngine::run_prefill` calls `Scheduler::set_prefilled_tokens` with the
    actual count. `context_len()` returns `prefilled_tokens + generated_tokens` once set,
    falling back to `prompt_tokens.len()` only before prefill completes.
  - `Model::prefill_sync` / `LlamaCppModel::do_prefill` return type extended from
    `Vec<(u64, Logits)>` to `Vec<(u64, Logits, usize)>` (third element is `tokens_submitted`).

- **Graceful recovery on KV cache exhaustion** (`src/engine/mod.rs`)
  - `llama_decode` returning a non-zero error code (e.g. `init_batch: failed to prepare
    attention ubatches` / `decode: failed to find a memory slot`) previously propagated as a
    hard `anyhow::Error` and crashed the engine loop.
  - `run_prefill` and `run_decode` errors are now caught with `match`; affected requests are
    marked `StopReason::Length` and the engine loop continues. Subsequent requests are
    unaffected.

- **Stale `stop_reason` on preempted-request re-admission** (`src/scheduler/mod.rs`)
  - When a request was LIFO-preempted its `stop_reason` was set to `Some(Preempt)`. On
    re-admission to `Prefilling` the field was never cleared, so the engine could see a
    non-`None` stop reason on a still-active request.
  - Fix: `schedule_step` now sets `req.stop_reason = None` in both the prefix-cache-hit and
    normal admission paths before transitioning to `Prefilling`.

- **CUDA build with non-standard CUDA installations** (`build.rs`)
  - Removed the hard `CUDA_PATH` env-var requirement. `build.rs` now locates `nvcc` via
    `which nvcc`, falling back to `$CUDACXX` and then `/usr/local/cuda/bin/nvcc`. The
    resolved path is passed to CMake as `CMAKE_CUDA_COMPILER`; the parent directory is used
    to derive the CUDA library search paths.

### Changed

- **`ahash` replaces `DefaultHasher` for token hashing** (`src/scheduler/mod.rs`)
  - `hash_tokens` now uses `ahash::AHasher` (initialised from a process-stable
    `OnceLock<ahash::RandomState>`). Faster with better avalanche properties; still
    deterministic within a single run. Dependency added: `ahash = "0.8"`.

- **Prefix cache backed by `lru::LruCache`** (`src/scheduler/mod.rs`)
  - `HashMap<u64, PrefixCacheEntry>` replaced with `lru::LruCache<u64, PrefixCacheEntry>`.
    The LRU ordering is preserved in preparation for future automatic eviction (currently the
    manual capacity check is kept to avoid silent block leaks until the eviction path is
    wired through properly). Dependency added: `lru = "0.12"`.

- `src/config.rs` deleted; server configuration now lives in `src/cli/serve.rs::ServeArgs`.
- `src/api/mod.rs` now re-exports `AppState` alongside `router`.

---

## [0.3.1] - 2026-03-10

### Fixed

- **Crash on hybrid/recurrent models** (`src/engine/model.rs`, `src/engine/mod.rs`)
  - Qwen3.5, Mamba, and other hybrid architectures use `llama_memory_recurrent` instead of
    the standard attention KV cache. Calling `llama_memory_seq_cp` on those models triggered
    `GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers")` inside
    llama.cpp, terminating the process with `SIGABRT` on the second request with an identical
    prompt.
  - Added `Model::supports_seq_copy()` backed by `llama_memory_can_shift()`: returns `true`
    only for full KV cache backends (standard attention-only transformers).
  - `InferenceEngine::new()` stores the result as `supports_prefix_cache` and logs it at
    startup.
  - `do_prefill` now guards `llama_memory_seq_cp` with a `can_shift` check as a second safety
    net.
  - Prefix caching is automatically disabled for incompatible models; all other features
    (stop sequences, metrics, streaming usage) remain fully functional.

- **CUDA build** (`build.rs`, `Cargo.toml`)
  - Removed the optional `cudarc` dependency (only used to query GPU memory, but its
    `build.rs` requires a CUDA-version feature flag that caused `--features cuda` to fail with
    a compile error). Replaced with a `nvidia-smi` subprocess call — no extra dependencies.
  - `build.rs` now links `ggml-cuda`, `libcuda` (driver API), `libcudart`, `libcublas`, and
    `libcublasLt`, searching both `/cuda/lib64` and `/cuda/targets/x86_64-linux/lib` to
    support different CUDA installation layouts.

- **Prefix-cache boundary token position** (`src/engine/model.rs`)
  - `do_prefill` was copying positions `0..skip_prefix_tokens` via `seq_cp` and then
    submitting the last prompt token at the wrong position (`context_len` instead of
    `skip_prefix_tokens - 1`). Changed to copy `0..skip_prefix_tokens-1` and always
    re-submit the boundary token in the batch at the correct position, ensuring valid
    positional encodings and correct logits.

---

## [0.3.0] - 2026-03-10

### Added

- **PageTable — explicit logical→physical block mapping** (`src/kv_cache/mod.rs`)
  - Replaced the flat `kv_block_ids: Vec<BlockId>` field in `InferenceRequest` with a named
    `PageTable` struct. The struct encapsulates the `entries` vector
    (`logical_block_index → physical_block_id`) and exposes `block_ids()`, `len()`,
    `is_empty()`, `clear()`, and `extend()`.
  - Added `ref_count: Vec<AtomicUsize>` to `KVCacheManager` (one entry per physical block).
    `allocate` sets `ref_count = 1`; `free_blocks` decrements and only returns the block to the
    free list when the count reaches zero.
  - `retain_block(id)` — increments ref_count (used when a block is shared for prefix caching).
  - `copy_on_write(id) -> Option<BlockId>` — allocates a new exclusive block and decrements the
    shared one's ref_count; foundational for future true memory-sharing CoW.

- **Prefix caching — skip re-prefill for identical prompts** (`src/scheduler/mod.rs`, `src/engine/mod.rs`, `src/engine/model.rs`)
  - `Scheduler` now embeds a `PrefixCache: HashMap<u64, PrefixCacheEntry>` keyed by
    `hash(prompt_tokens)`. Max capacity = `max_batch_size / 4` entries.
  - When a request finishes (EOS, Length, or StopSequence), `InferenceEngine` calls
    `Scheduler::try_insert_prefix` which atomically transfers the request's `kv_seq_id` and
    `page_table` blocks into the cache (the KV data in llama.cpp is preserved — `clear_sequence`
    is skipped for cached entries).
  - On the next admission of a request with the same prompt hash, `schedule_step` detects the
    hit, transfers the cached blocks to the new request's `PageTable`, allocates only the
    generation blocks, sets `skip_prefix_tokens` and `prefix_seq_id` on the request.
  - `do_prefill` calls `llama_memory_seq_cp(mem, prefix_seq_id, new_seq_id, 0, skip_tokens)`
    inside the blocking task before building the batch, then submits only
    `prompt_tokens[skip_prefix_tokens..]` starting at the correct absolute position.
    After prefill, the engine clears the now-redundant prefix sequence and returns its ID to
    the pool via `Scheduler::return_prefix_seq_id`.
  - New `Model` trait method: `copy_sequence_range(src, dst, token_count)`, backed by
    `llama_memory_seq_cp`; no-op in the stub.
  - Counters `prefix_hits` / `prefix_misses` (atomic) on `Scheduler`; exposed on `InferenceEngine`.

- **Stop sequences** (`src/scheduler/batch.rs`, `src/engine/mod.rs`, `src/api/types.rs`)
  - `SamplingParams` gains `stop: Option<Vec<String>>`.
  - `ChatCompletionRequest.stop` accepts both a JSON string and an array (OpenAI spec). Uses a
    custom `deserialize_stop` Serde helper.
  - `StopReason::StopSequence` variant added.
  - `handle_logits` now runs a rolling-buffer stop-sequence check (last `2 × max_stop_len` chars)
    per request. The stop string is **not** emitted in the output; only the prefix before the
    match is sent to the client (OpenAI behaviour). Detection works across token boundaries.
  - Output filtering (`<think>` suppression, special token stripping) and stop sequence detection
    now share a single lock acquisition on `per_request_state` to prevent deadlocks.

- **Prometheus `/metrics` endpoint** (`src/metrics.rs`, `src/api/routes.rs`, `src/main.rs`)
  - New `GET /metrics` route returning Prometheus text exposition format (version 0.0.4).
  - Metrics registered at startup via `Metrics::new()`:
    - `ferrum_requests_total{finish_reason}` — counter
    - `ferrum_tokens_generated_total` — counter
    - `ferrum_request_latency_seconds` — histogram (10 buckets: 0.05 s … 60 s)
    - `ferrum_kv_cache_usage_ratio` — gauge
    - `ferrum_queue_depth` — gauge
    - `ferrum_active_requests` — gauge
    - `ferrum_prefix_cache_hits_total` — counter
    - `ferrum_prefix_cache_misses_total` — counter
  - Dependency added: `prometheus = "0.14"`.
  - Gauges and counter deltas are refreshed on every engine scheduling step.

- **Streaming `usage` in the final chunk** (`src/api/routes.rs`, `src/api/types.rs`)
  - The last SSE chunk (the one that carries `finish_reason`) now includes a `usage` object
    with `prompt_tokens`, `completion_tokens`, and `total_tokens`, matching the OpenAI
    streaming spec. Intermediate chunks omit the field (`skip_serializing_if = "Option::is_none"`).

### Changed

- `InferenceRequest` fields: `kv_block_ids: Vec<BlockId>` → `page_table: PageTable`;
  new fields `skip_prefix_tokens: usize`, `prefix_seq_id: Option<i32>`,
  `submitted_at: Instant` (for latency metrics).
- `InferenceEngine::new` now accepts `metrics: Option<Arc<Metrics>>`.
- `OutputFilterState` renamed to `PerRequestState`; its `text_buffer` field drives stop
  sequence detection.
- `PLAN.md` updated: Phase 1 marked completed with v0.1.0/v0.2.0 summaries; Phase 2
  progress tracked.

---

## [0.2.0] - 2026-03-09

### Added

- **Real stochastic sampling** (`src/engine/model.rs`)
  - Replaced the deterministic `sample_top_p` (which always returned the same
    token for identical logits) with a weighted random draw from the nucleus.
  - New pipeline per token: repetition penalty → temperature scaling →
    top-K masking → softmax → top-P nucleus truncation → weighted random sample.
  - Added `top_k: u32` (0 = disabled) and `repetition_penalty: f32` (1.0 =
    disabled) parameters — surfaced in `SamplingParams`, `InferenceRequest`,
    `InferenceRequestForModel`, and the OpenAI request types.
  - Added `seed: Option<u64>` for reproducible output: when set, each token
    position uses `StdRng::seed_from_u64(seed ^ token_count)`.
  - Scheduler now tracks `generated_token_ids` per request for repetition
    penalty without storing the full history in the model.
  - Dependency added: `rand = "0.8"`.

- **Docker support**
  - `Dockerfile` — multi-stage build (Rust 1.84 builder + `debian:bookworm-slim`
    runtime); ships both `ferrum-engine` and `ferrum-bench`.
  - `docker-compose.yml` — mounts a local `./models` volume; optional NVIDIA
    GPU passthrough via commented `deploy.resources` section.
  - `.dockerignore` — excludes `target/`, `models/`, `.git/`, editors.
  - `README.md` updated with a new **Docker** section.
  - `Makefile` new targets: `docker` and `docker-run`.

- **Integrated benchmark binary** (`src/bin/bench.rs` → `ferrum-bench`)
  - Launches N concurrent workers, each sending `--requests` SSE chat
    completions to the target server.
  - Reports **TTFT** (P50/P95), **total latency** (P50/P95/P99), aggregate
    **tokens/second**, and total elapsed time.
  - CLI: `--url`, `--model`, `--concurrency`, `--requests`, `--prompt`,
    `--max-tokens`.
  - Dependency added: `reqwest = "0.12"` with `json` + `stream` features.
  - `Makefile` new target: `bench` (configurable via `BENCH_CONCURRENCY`,
    `BENCH_REQUESTS`, `BENCH_PROMPT`, `BENCH_MAX_TOKENS`).
  - `README.md` updated with a new **Benchmark** section.

### Fixed

- **`llama_decode` crash: "non-consecutive token position" / "inconsistent sequence positions"**
  — Two related bugs caused `llama_decode` to return `-1` and kill the engine loop
  when more than one request was ever processed:
  1. **Unstable `seq_id`** — each request was assigned `seq_id = its_index_in_the_current_batch`.
     A request prefilled as `seq_id=0` would end up as `seq_id=1` in the next decode batch
     (if another request joined), so llama.cpp looked up the wrong KV/recurrent-memory slot.
     Fix: each request is now assigned a stable `kv_seq_id` from a pool
     (`0..max_batch_size`) at admission time; the ID does not change until the request
     finishes or is preempted.
  2. **Stale KV state on seq_id reuse** — when a request ended or was LIFO-preempted its
     seq_id was returned to the pool but the llama.cpp memory module still held all its
     cached positions. The next request that received that ID would submit tokens at
     position 0 while the context still recorded the previous occupant's last position
     (e.g. 55), causing an M-RoPE position assertion failure.
     Fix: `Model::clear_sequence(seq_id)` — backed by `llama_memory_seq_rm(mem, seq_id, 0, -1)`
     — is now called (a) in `handle_logits` when a request finishes and (b) in `run_loop`
     for every `seq_id` in `ScheduledBatch::preempted_seq_ids`, before those IDs can be
     handed to a new request.

### Changed

- `SamplingParams` struct introduced in `scheduler/batch.rs` to group all
  sampling hyper-parameters; `InferenceRequest::new` now accepts it instead of
  individual `temperature`/`top_p` arguments.
- `ChatCompletionRequest` in `api/types.rs` exposes `top_k`, `repetition_penalty`,
  and `seed` as optional JSON fields, fully forward-compatible with the
  OpenAI spec.
- `ScheduledBatch` now carries a `preempted_seq_ids: Vec<i32>` field so the engine
  can clear stale KV state immediately after preemption.
- `Scheduler::new` now accepts `max_batch_size: usize` to size the seq_id pool.
- `Model` trait gains a new required method: `clear_sequence(&self, seq_id: i32)`.

---

## [0.1.0] - 2026-03-08

Initial release.

### Added

- **OpenAI-compatible HTTP API**
  - `POST /v1/chat/completions` — streaming (SSE) and non-streaming chat
  - `POST /v1/completions` — text completion (delegates to chat endpoint)
  - `GET /v1/models` — returns the name of the loaded model derived from the file path
  - `GET /health` — KV cache usage, queue depth, active requests
- **Inference engine**
  - llama.cpp FFI backend with GGUF model support
  - Continuous batching scheduler with LIFO preemption
  - Block-based KV-cache memory manager
  - `temperature` and `top_p` sampling wired through the full pipeline (API → scheduler → model)
  - Output filtering: `<think>...</think>` blocks, `<|...|>` special tokens, SentencePiece `▁` word-boundary character
- **Configuration** (CLI flags and environment variables)
  - `--model-path` / `FERRUM_MODEL_PATH`
  - `--max-context-len` / `FERRUM_MAX_CONTEXT_LEN` (default: 4096)
  - `--gpu-memory-fraction` / `FERRUM_GPU_MEMORY_FRACTION` (default: 0.85)
  - `--max-batch-size` / `FERRUM_MAX_BATCH_SIZE` (default: 32)
  - `--block-size` / `FERRUM_BLOCK_SIZE` (default: 16)
  - `--host` / `FERRUM_HOST` (default: 0.0.0.0)
  - `--port` / `FERRUM_PORT` (default: 8080)
  - `--json-logs` / `FERRUM_JSON_LOGS`
- **Operability**
  - Graceful shutdown on SIGTERM and SIGINT
  - `tokio::sync::Notify` wakes the engine loop on new requests (replaces 100 µs polling)
  - Unified per-token scheduler update (`update_after_token`) — single lock acquisition per token
- **MIT + Apache 2.0 dual license**

### Fixed

- SSE stream now correctly emits `finish_reason: "length"` when `max_tokens` is reached (previously the client would hang waiting for more data)
- Panic on `Event::json_data(...).unwrap()` in the SSE path replaced with a safe fallback
- `CString::new(...).unwrap()` panics when role or content strings contain null bytes in `apply_chat_template`
- Partial special tokens (`<|`) no longer leak into the output stream
- `gpu_memory_fraction` out-of-range values now produce a clear error at startup instead of silently misbehaving

### Changed

- Context length is no longer hardcoded to 2048; controlled via `--max-context-len`
- Removed unused runtime dependencies: `thiserror`, `tokenizers` (HuggingFace)
- `KVCacheManager` debug set (`_allocated_set: Mutex<HashSet<BlockId>>`) removed — eliminated a second mutex acquisition on every allocate/free call
- bindgen warnings from llama.cpp FFI bindings suppressed in `ffi.rs`
