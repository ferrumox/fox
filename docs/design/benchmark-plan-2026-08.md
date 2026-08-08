# Four-engine benchmark and white paper — plan and current state

Handoff document. Everything a fresh session needs to continue the comparison without
re-deriving it. Written 2026-08-03, at the end of the 0.19.1 work.

## The goal

A white paper demonstrating what fox does differently, backed by a comparison of fox,
`llama-server`, Ollama and vLLM on this machine.

## Where this landed — the balance, 2026-08-03

Read this before the sections below. They are in the order the work happened, which means
several state conclusions that later sections retract. This is what survived.

### One claim survives, and it is narrow

**fox reuses a shared prefix from a sequence that is still decoding; `llama-server`
cannot, and no flag gives it that.** `get_available_slot()` skips `is_processing()` slots
in both its similarity pass (`server-context.cpp:1609`) and its LRU fallback (`:1652`), so
concurrent arrivals cannot inherit from each other. Measured, 3 rounds, disjoint ranges,
8 clients behind a 1856-token shared prompt:

| | fox | `llama-server` | Ollama |
|---|---|---|---|
| cold TTFT p50 | **1102 ms** | 4327 ms | 5344 ms |
| warm TTFT p50 | **50 ms** | 184 ms | 405 ms |

That is the paper's thesis, and it is the only headline that holds up.

**Multi-turn is a second, weaker claim and must not borrow the first one's number.**
Measured at last (`scripts/bench_multiturn.py`): within a conversation fox goes 372 → 53
ms per turn, but `llama-server` goes 383 → 87 — it reuses here too, because between turns
the sequence is idle and inheriting an idle slot is the thing it does well. So multi-turn
is **1.64× over `llama-server`** and **4.9× over Ollama**, not 3.9×. Two different
workloads, two different numbers; the docs currently blur them.

### Three claims did not survive

| claim | what it actually was |
|---|---|
| "fox degrades 5.5× under a noisy neighbour, `llama-server` 44×" | **a default.** `--max-prefill-chunk 512` against `n_batch 2048`. With `-b 512` the reference stalls 263 ms against fox's 273. |
| "fox is within ~10% of `llama-server` on throughput" | **true only to 16 clients.** The gap is 15% at 32, 22% at 64 and 103% at 128. |
| "the sweep gains at 32 clients" | **a contaminated control.** Duplicate prompts above 16 clients handed fox its own prefix cache; retracted. |
| "`llama-server` loses the 9B warm burst 3×" and "fox wins 9B multi-turn 7.9×" | **a broken instrument.** The drivers ignored `reasoning_content`, so a reasoning model's whole stream was invisible and the total request time was reported as TTFT. Both retracted; re-measured below. |

### The finding that matters most is a trade, not a win

fox's unified KV cache (`kv_unified = true`, `n_stream = 1`) is what makes a partial
`seq_cp` metadata-only — the mechanism behind the one surviving claim. It is *also* the
sole cause of fox collapsing above 64 concurrent requests, because a decode step then
attends over the union of every sequence's cells: cost grows as N·(N·L) instead of N·L.

| | unified KV buys | unified KV costs |
|---|---|---|
| | 5.7× cold TTFT, 117× warm | 6% throughput at 16 clients, 108% at 128 |
| | | a concurrency ceiling at ~64 |

Proved by running the same binary with `FOX_KV_UNIFIED=0`: that arm does not bend at all,
matching `llama-server` at 64 clients and passing it at 128.

**fox's real advantage and its worst weakness are the same design decision.** That is a
more interesting paper than "fox is 4× faster", but it only works if told whole.

### On hybrid models, nothing is settled

The Qwen3.5 family is where fox's prompt reuse was off entirely until 0.20.0, and where
it is still off at the shipped `--rs-rollback 4`: measured, 20 slot hits and 20 refused
trims, `cached_tokens` 0, because the conversation is not a token-exact prefix chain on
this template. fox is nonetheless ~2.7× ahead of `llama-server` on short-prompt
multi-turn there — but for the fixed-cost reason below, not for any reuse. Treat every
hybrid number in this document as provisional.

### Where fox is behind, plainly

- decode throughput: 1.06× behind at 4 clients after the sampler fix (was 1.10×)
- idle stream smoothness: ITL p99 50 ms against `llama-server`'s 21 ms
- memory: ~400 MB more GTT for the same workload
- concurrency: collapses above 64; `llama-server` was still climbing at 128

### The size axis, isolated — Qwen2.5-7B (dense GQA)

Qwen3.5-9B changed size *and* architecture at once, and the architecture dominated (next
section). Qwen2.5-7B-Instruct-Q4_K_M is dense GQA, the same family as Llama-3.2-1B, so it
isolates size. 3 rounds, disjoint ranges, Vulkan, same workloads.

| | 1B | 7B dense |
|---|---|---|
| cold TTFT, fox | 1102 ms | 5130 ms |
| cold TTFT, `llama-server` | 4327 ms | 33207 ms |
| **fox's cold advantage** | 3.9× | **6.5×** |
| warm TTFT, fox vs `llama-server` | 3.7× | 2.3× |
| decode, per request | 1.06× behind | 1.05× behind |
| decode, aggregate | 1.05× behind | **parity** (ranges overlap) |
| noisy neighbour, absolute stall | fox 273 ms / ls 933 ms | fox 1712 ms / ls 5630 ms |
| idle ITL p99 | fox 50 / ls 21 ms | fox 188 / ls 67 ms |

**The signs hold and the main advantage grows with model size** — 3.9× → 6.5× cold, which
follows from prefill costing more on a bigger model, so avoiding it is worth more.
`llama-server` takes **33 seconds** to first token against fox's 5.1, with `cached_tokens`
0 against 12901: the same mechanism as the 1B, more expensive.

Two rows move against fox:

- **The warm advantage narrows** (3.7× → 2.3×). Both engines reuse there; what remains
  weighs relatively less when the model itself is slow.
- **Decode aggregate is now parity** rather than 5% behind — the sampler cost is per token
  and constant while the GPU step lengthens, so its share shrinks. That also means **the
  sampler fix matters less the larger the model**; it was worth most on the 1B.

The noisy-neighbour ratio between engines is unchanged at 3.3× of absolute stall, matching
the 4× prefill-chunk ratio again, so that explanation carries across model sizes. fox's
worse idle jitter carries across too (188 ms vs 67 ms).

### The 9B run found something bigger than a benchmark number

Qwen3.5 is a **hybrid** architecture — `LLM_ARCH_QWEN35` sits in llama.cpp's
`llm_arch_is_hybrid` list (`llama-arch.cpp:946`) alongside `QWEN3NEXT`, `FALCON_H1` and
`JAMBA`. fox's `supports_seq_copy()` returns false for hybrids, so it logs
`prefix caching disabled (model cannot donate KV cells)` and **turns prompt reuse off
entirely**. Measured on Qwen3.5-9B-Q4_K_M, 8 clients, shared 1856-token prompt:

| warm burst | fox | `llama-server` |
|---|---|---|
| TTFT p50 | 43166 ms | **13264 ms** |
| `cached_tokens` | **0** | 14680 |

**fox's headline advantage does not merely shrink here — it inverts.** `llama-server` is
3.25× faster warm because it still reuses and fox does not.

This is not only a paper problem. `registry.json` recommends `qwen3.5` (4B) as "a good
default to try fox with" and `qwen3.5:9b` as the step up — the **same hybrid family**. On
the models fox's own catalogue leads with, its main differentiator is off.

The guard is also broader than the hardware requires, and `llama-server` proves it on the
same llama.cpp: reuse comes in two kinds, and only one needs cross-sequence copying.

| kind | needs | works on hybrids |
|---|---|---|
| inherit a slot's own KV and skip prefill | nothing copied | **yes — `llama-server` does it, 14680 tokens** |
| copy a prefix out of another (live) sequence | `seq_cp` + unified KV | no |

fox gated both on one flag, so a hybrid model lost the cheap, no-copy kind too.

**Fixed, 2026-08-03.** It took four changes, and three of them were only found by testing
against the real model — the unit tests passed and the log said the capability was on
while reuse still did not happen:

1. `Model::supports_slot_reuse()` split out from `supports_seq_copy()`; the scheduler
   carries both flags and `schedule.rs` gates slot affinity on the weak one and the
   live-sibling fork on the strict one.
2. `logits.rs` parked a finished sequence only when the model supported *copying*. With
   nothing parked, everything downstream was dead regardless of what it was allowed to
   do. This was the gate that kept reuse at zero after the first two were fixed.
3. `trim_sequence` returned `()`, discarding llama.cpp's bool. A partial `seq_rm` on a
   recurrent cache legitimately fails outside its snapshot window
   (`llama-memory-recurrent.cpp:181`) and mutates nothing; ignoring that would leave a
   request skipping a prefix that is no longer there. It now returns the result and
   `run.rs` re-prefills on refusal.
4. `n_rs_seq` — the snapshot window itself — defaults to **0** in llama.cpp
   (`llama-context.cpp:3457`), and fox inherited it. With it at 0 every partial rollback
   fails, so the capability was present and switched off. `QWEN35` is in
   `llm_arch_supports_rs_rollback`, so nothing about the architecture prevented this.

Measured on Qwen3.5-9B, 8 clients, warm burst:

| | before | after |
|---|---|---|
| warm TTFT p50 | 42923 ms | **652 ms** (66×) |
| `cached_tokens` | 0 | 14856 |
| trims refused | 8 of 8 | 0 |

The snapshot window is a memory trade and the numbers are steep — ~453 MB per snapshot
at 8 sequences on this model:

| snapshots | GTT | warm TTFT | reuse |
|---|---|---|---|
| 0 | 6.9 GB | 43093 ms | none |
| 4 (default) | 8.7 GB | 40294 ms | none *in this benchmark* |
| 64 | 36.5 GB | 652 ms | full |

The default is 4, not 64. It buys the **multi-turn** case — where the next turn contains
the previous reply, the common prefix runs past it and the rollback is one token — for
~1.8 GB. Covering a *repeated* prompt, which is what `bench_burst.py` sends, needs a
window the size of the reply and costs 30 GB on a laptop; that is `FOX_RS_ROLLBACK`,
opt-in. So the 66× above is what the feature can do, not what the default does, and the
benchmark that produced it is the adverse case rather than the representative one.

Still owed: `FOX_RS_ROLLBACK` is an environment variable only. The repo's convention is
that every knob has a CLI flag, an env var and a config key, and this one has a 30 GB
failure mode, so it should be promoted.

### Multi-turn, measured at last — and the claim is narrower than the docs say

The most-quoted product claim had no number under it until now. `scripts/bench_multiturn.py`
runs real conversations: each turn's prompt is the previous prompt plus the model's
*actual* reply plus a new user message, so the history is a genuine prefix that grows.
Each conversation opens with a unique marker, so this isolates within-conversation reuse
from the shared-system-prompt case the burst benchmark already covers.

4 concurrent conversations, 8 turns, prompt growing 157 → 498 tokens, 3 rounds,
Llama-3.2-1B, ranges disjoint:

| | turn 0 | turns 1-7 | speed-up within a conversation |
|---|---|---|---|
| fox | 372 ms [367, 374] | **53 ms** [49, 59] | 7.0× |
| `llama-server` | 383 ms [381, 385] | 87 ms [71, 108] | 4.4× |
| Ollama | 632 ms [519, 661] | 259 ms [214, 335] | 2.4× |

**"Conversations get faster over time" is true for fox — and also for `llama-server`.**
It reuses here too (`cached_tokens` 350 against fox's 342), because between turns a
conversation's sequence is *idle*, and inheriting an idle slot is exactly what
`llama-server` does well. Its limitation is inheriting from a sequence that is still
decoding, and a sequential conversation never exercises it.

So the honest figures for this workload are **1.64× over `llama-server`** and **4.9× over
Ollama** — not the 3.9× the concurrent burst produces. The burst and the conversation are
different claims and the docs currently blur them.

Which comparison to lead with is a positioning decision, not a measurement one: fox
describes itself as a drop-in replacement for **Ollama**, and against Ollama this is 4.9×.
Against `llama-server` — which has no model management, no pull, no catalogue and no
Ollama API — it is 1.64×. Quoting the larger number against the weaker competitor is fine
if the comparison is named; leaving it unnamed is not.

### What TTFT is actually made of

Prompted by fox looking 2.7× faster than `llama-server` on hybrid multi-turn while
reusing *nothing* — an advantage with no mechanism is a result waiting to be retracted.

Measured without parsing either engine's logs, because only one of them publishes
timings and comparing a published number against an estimate is the asymmetry that has
spoiled half the measurements here. TTFT against prompt length at fixed concurrency,
then a linear fit: the slope is prefill per token, the intercept everything that does not
depend on the prompt. Qwen3.5-9B, R² > 0.995 on all four fits.

| | prefill per token | fixed cost per request |
|---|---|---|
| fox, conc 1 | 2.63 ms | **110 ms** |
| `llama-server`, conc 1 | **2.40 ms** | 540 ms |
| fox, conc 4 | 10.65 ms | **439 ms** |
| `llama-server`, conc 4 | **10.17 ms** | 1739 ms |

**fox's advantage is entirely the fixed per-request cost — 4.9× lower — and
`llama-server` prefills 10% faster per token.** So fox wins short prompts, where the
constant dominates, and loses long ones. The crossover is around **1900 tokens**
(110 + 2.63n = 540 + 2.40n). Multi-turn sits at 150-320 tokens, deep in fox's half,
which is the whole of the 2.7×.

Two things this kills:

- **"fox interleaves chunked prefill across concurrent requests better."** That was my
  explanation and it is wrong: going from 1 to 4 clients multiplies the slope by 4.04 for
  fox and 4.25 for `llama-server`. Neither overlaps prefill across requests.
- Any claim that fox is faster on this model family in general. It is faster on *short*
  prompts on it.

What those 540 ms and 110 ms *are* is still unknown. One decode pass on this model is
~71 ms, so neither figure is "the first token"; there is 40-470 ms of something else in
each engine. Going further needs instrumentation inside both, symmetrically.

### What has not been done

The whole comparison is **Llama-3.2-1B-Q8_0 on one iGPU**, i.e. one size and one
architecture (dense GQA). The Qwen3.5-9B attempt above is not a size comparison — it
changed architecture at the same time, and the architecture dominated the result.

Still unmeasured: RAG and agentic workloads, sliding-window attention, and
MoE. (The dense 7B size axis is now measured — see above.) Nothing here should be published as a
general claim about fox.

The 9B run is also inconclusive as a *performance* comparison for separate reasons worth
recording so nobody re-runs it unchanged: fox's burst ranges spanned [43109, 57537] ms
(33% spread), decode ranges overlapped at every pairing, and the noisy-neighbour driver
reported 0 ms ITL for two engines because its 10 s baseline is too short for a model this
slow — the interactive clients had not produced a token yet. A slow model needs the
baseline window scaled, not the same constants.


## Method, agreed

**Two axes, both reported.**

1. **Config-matched.** Same model, quantisation, context length and sampler settings
   across all four. Isolates the serving layer. This is the number nobody can argue with.
   (It once read "fox sits at 96% of `llama-server`" here; measured properly that is
   1.06× behind at 4 clients and 2.03× behind at 128 — see the balance above.)
2. **Best-effort per engine.** Each engine tuned with its own techniques — the comparison
   a user actually cares about when choosing one.

The discipline that makes axis 2 honest is that **the tuning effort must be equal**. If
fox gets speculative decoding, `llama-server` gets `--draft-model`. If fox quantises its
KV cache, so do the others. Every engine's exact configuration gets published. Tuning one
side and not the others is the failure mode to avoid, and it is easy to fall into by
accident because fox is the one whose flags we know best.

Report the workload where fox loses as prominently as the ones where it wins. A benchmark
page that only reports wins is marketing.

## Hardware

AMD Radeon 890M — **gfx1150**, RDNA 3.5, integrated. 123 GB system RAM, shared with the
GPU. Read from `/sys/class/kfd/kfd/topology/nodes/*/properties`.

## Engine status

| Engine | State | How |
|---|---|---|
| fox | ready | `Dockerfile.vulkan` → bundle; `make vulkan` |
| `llama-server` | ready, **flags audited** | `Dockerfile.llama-server-vulkan`, same vendored llama.cpp |
| vLLM | **serves, measured** | `rocm/vllm:latest` + `HSA_OVERRIDE_GFX_VERSION=11.0.0`; `scripts/bench_vllm.sh` |
| Ollama | **runs on GPU**, verified | `ollama/ollama:rocm` + `OLLAMA_IGPU_ENABLE=1`; `scripts/try_ollama_rocm.sh` |

All four engines run on this hardware. No engine has to be excluded from the comparison.

### Ollama — gate results, 2026-08-03

Run `scripts/try_ollama_rocm.sh`. It imports the same Q8_0 GGUF the published runs use
via a Modelfile rather than `ollama pull`, so axis 1 stays exact — `ollama pull
llama3.2:1b` would bring Q4_K_M and the comparison would no longer be of serving layers.

- **No `HSA_OVERRIDE_GFX_VERSION` needed.** Ollama recognises gfx1150 natively:
  `inference compute … library=ROCm compute=gfx1150 … type=iGPU`. Unlike vLLM, which
  needs the override, this is a difference worth stating in the write-up.
- **`OLLAMA_IGPU_ENABLE=1` is mandatory here.** By default Ollama *finds* the 890M and
  then discards it — `dropping integrated GPU; to enable, set OLLAMA_IGPU_ENABLE=1` —
  and falls back to CPU **without failing**. `ollama ps` reports `100% CPU` and it serves
  normally. Benchmarking that fallback against fox on Vulkan would have produced a huge,
  entirely fake win. With the flag: `100% GPU`.
- That silent fallback is why the gate asks *which processor the model is resident on*
  instead of *did the model load*. Every Ollama arm must assert `100% GPU` from
  `ollama ps` before its numbers count.

**Config-matching knobs for Ollama** — its defaults do not match what the other three
are given, and they are set by env var, not by request:

| knob | Ollama default | must be set to |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `0` → chose **131072** for this model | `CTX_PER_SEQ` (4096) |
| `OLLAMA_NUM_PARALLEL` | `1` → 8 concurrent clients **serialise in a queue** | `CONC` |
| `OLLAMA_FLASH_ATTENTION` | `false` | match the other arms |
| `OLLAMA_KV_CACHE_TYPE` | unset (f16) | match `--kv-cache-type` |

`OLLAMA_NUM_PARALLEL` is the dangerous one: left at 1 it turns a concurrency benchmark
into a queueing benchmark, and the resulting TTFT curve looks exactly like the prefix-reuse
failure the paper is about. It would be a fabricated win in the direction of the thesis.

- The OpenAI-compatible surface works with `bench_burst.py` as written (streaming plus
  `stream_options.include_usage`), but usage carries **no `prompt_tokens_details`**, so
  `cached_tokens` reads 0 for Ollama. That means "not reported", not "no reuse", and the
  table has to say so. TTFT remains directly comparable.

### llama-server flag audit — done, numbers stand

- `--slot-prompt-similarity` defaults to `0.10`, same as fox. Its LCP slot affinity was
  active all along.
- `--cache-reuse` defaults to `0` and was not set in the published runs. Measured with
  `--cache-reuse 256`: cold TTFT 4376 → 4367 ms, warm 189 → 186 ms, cold `cached_tokens`
  0 either way. Inside the noise.

The comparison therefore stands with the reference configured in its favour.

### vLLM caveats

- `torch.cuda.get_device_name(0)` returns an **empty string**. ROCm allocates against the
  device without recognising it. Anything keying off the device name will misbehave.
- Do **not** quote the 5.22 tok/s from the feasibility gate. That was a 0.5B model under
  `enforce_eager=True`, which disables CUDA graphs and Inductor. Drop `enforce_eager` for
  any real measurement.
- The override belongs in vLLM's *documented configuration*, not a footnote — a reader
  with this hardware needs it too.

## Backend topology — no single run holds all four engines

Discovered while wiring the Ollama arm, and it changes the shape of the paper: **there is
no configuration in which all four engines run on the same compute backend against the
same model file.**

| engine | Vulkan | ROCm | consumes the GGUF |
|---|---|---|---|
| fox | yes (`Dockerfile.vulkan`) | yes (`Dockerfile.rocm`) | yes |
| `llama-server` | yes (`Dockerfile.llama-server-vulkan`) | yes | yes |
| Ollama | yes, but only the `:latest` image (0.30.10) | yes, only the `:rocm` image (0.32.5) | yes, via Modelfile |
| vLLM | **no Vulkan path at all** | yes | no — needs its own artifact |

So the bank splits in two, and each half must say what it is:

1. **Vulkan trio** — fox, `llama-server`, Ollama. Same backend, same GPU, same GGUF file.
   This is the config-matched axis, and it is where the serving-layer claim lives.
   `scripts/bench_engines.sh`.
2. **vLLM, separately, on ROCm, with its own model artifact.** Two variables move at
   once against the trio, so its number is not comparable to theirs at the serving-layer
   level and must not be put in the same column. What it *can* answer is the question a
   user actually asks: what does the best-known serving stack do on this hardware.

Reporting vLLM inside the trio's table would publish a backend difference as if it were
an engine difference — the exact failure mode the "equal tuning effort" rule exists to
prevent, arriving through the back door.

Two further caveats the trio table has to carry:

- fox and `llama-server` are built from the **same vendored llama.cpp**; Ollama ships its
  own fork (ggml 0.17 against the vendored 0.15.3). The fox↔`llama-server` comparison
  isolates the serving layer; the Ollama comparison does not isolate it as cleanly.
- fox runs `kv_unified = true`, both others `false`. That is not a tuning knob handed to
  fox — sharing cells via `seq_cp` under a unified KV *is* the mechanism under test — but
  it is a real difference and belongs in the configuration table, not in a footnote.

## Model

`qwen3.5:9b` (`unsloth/Qwen3.5-9B-GGUF`, 5.7 GB) for the main comparison — current, and
small enough that three rounds across four engines finishes.

The measurements above are on **Llama-3.2-1B-Q8_0**, not qwen3.5:9b — it is the model the
earlier fox↔`llama-server` runs used, so reusing it let the new harness be checked
against a known answer before anything new was claimed. Repeat on the larger model before
publishing.

vLLM's artifact question is settled: it does not take the GGUF, and it was given
`unsloth/Llama-3.2-1B-Instruct` safetensors at BF16 (ungated, no HF token needed). That
is a real difference in what is being executed, recorded everywhere its numbers appear.

Note the architecture axis matters and is not covered by one model: sliding-window
attention (Gemma), hybrid attention/state-space (`falcon-h1` and the whole Qwen3.5
family in the catalogue — where fox disabled prompt reuse entirely until 0.20.0), and
MoE all change the prefill/decode balance. A paper
measuring only dense GQA and concluding "4-6×" is refutable with a modern Gemma.

## Workloads

Built already:

- `scripts/ab_bench.sh` — decode-bound throughput, the neutral control.
- `scripts/ab_shared_prefix.sh` + `scripts/bench_burst.py` — concurrent burst behind a
  shared system prompt, cold and warm.

Built since (2026-08-03):

- `scripts/bench_engines.sh` — N engines, two backends, four modes (`burst`, `decode`,
  `sweep`, `noisy`), one server alive at a time, arm order rotated per round.
- `scripts/bench_decode.py` — the neutral decode-bound control.
- `scripts/bench_noisy.py` — noisy neighbour: a long prefill injected into live streams.
- `scripts/bench_vllm.sh` — vLLM on its own terms, separate table.
- `scripts/probe_cached_tokens.py` — tells "did not reuse" apart from "does not report".
- `scripts/try_ollama_rocm.sh` — Ollama feasibility gates, including GPU residency.

Still to build, in the order they are worth doing:

1. **Multi-turn chat** — reuses most of the burst driver, and backs the most-quoted
   product claim ("conversations get faster").
2. **RAG, cache-hostile** — shared system prompt, different retrieved context per query.
   Deliberately adverse to fox. Publishing where the advantage narrows is what makes the
   rest credible.
3. **Agentic** — long prefix, short fast turns, parallel sub-agents. Where fox should win
   most, and where n-gram speculative decoding should pay.
4. Code/FIM (`/infill`) and structured output (validity of produced JSON, not just speed).

KPIs worth adding next, in order of what they would reveal:

- **Extend the sweep to 64 and 128.** The current one stops before fox and `llama-server`
  bend, so no maximum can be quoted from it.
- **Goodput under an SLO** (fraction of requests meeting TTFT and ITL targets at each
  concurrency) — derivable from data already collected, no new runs.
- **Energy per 1000 tokens.** `power1_average` is exposed under the GPU's hwmon (~40 W
  idle here). For a product that runs on a laptop this is a differentiator nobody publishes.
- **Cold start and reload cost.** Ollama unloads after 5 minutes by default; fox has an
  LRU with `--keep-alive-secs`. A mid-session reload is invisible in every throughput table.
- **Reproducibility under concurrency.** fox is known to drift at `temperature=0` under
  concurrent load. Whether the other three do too decides if that is a property of
  continuous batching or a fox defect — it is currently an untested assumption.

Use cases still unmeasured: model switching (two models alternating — the most common
local setup), mid-generation cancellation, long-context single prompts (prefill-only,
where fox's cache cannot help), batch embedding, and offline bulk processing.

## Benchmarking discipline — non-negotiable, learned the hard way

- **One server at a time.** ggml's thread pool spin-waits; an idle second server burns
  cores and skews the arm under test.
- **Alternate arms each round**, 3+ rounds, report median and range, and say plainly when
  ranges overlap.
- **Check the binary's timestamp against the commit.** A stale bundle once produced a
  confident, plausible, completely wrong result, and it was convincing because half the
  table still looked right.
- **Check the metric can move.** Pool usage read as the sum of per-slot block counts
  cannot fall when sharing works — it hid a real win across two measurements. Use
  `/slots`' `kv_blocks_used`.
- **Report measured prompt tokens.** An oversized prompt fails differently per engine:
  `llama-server` returns 400, fox rolls the context window and silently disables reuse.
- Kill servers and delete downloaded models after **every** test, not at the end.
- **Vary every input the workload claims is unrelated.** `bench_decode.py` handed out 16
  prompts with `i % 16`, so above 16 clients the "nothing to reuse" control was feeding
  fox byte-identical prompts. It biased exactly one engine, in the direction of the
  hypothesis, at exactly the concurrencies making the strongest claims.
- **Any before/after on a sweep needs both arms inside one alternating run.** Two
  post-fix sweeps at 16 clients differed by 5.7% between sessions while within-run ranges
  were ±1%.
- **Before publishing an advantage, try to hand it to the reference with a flag.** The
  noisy-neighbour result was `--max-prefill-chunk 512` against `n_batch 2048`, and `-b
  512` erased it. If a one-flag change closes the gap, it was never a design difference.
- **Check where the benchmark's own scratch space lives.** `$OUT` here defaults under
  `/tmp`, which on this machine is a **62 GB tmpfs — RAM, shared with the GPU**. Each
  Ollama arm copies the whole GGUF into its blob store, so three 9B runs silently held
  33 GB of the memory being measured and the next run failed to start. The fix is the
  store living on disk and the script deleting it, not a printed reminder: a cleanup rule
  that depends on remembering is not a rule. (Re-measured with RAM free, fox's 7B cold
  TTFT moved 5130 → 5062 ms, so the affected runs stand — but that was luck, not design.)
- **Read every field an engine might put the output in.** `llama-server` streams a
  reasoning model's tokens as `reasoning_content`, not `content`. Drivers reading only
  `content` saw an empty stream and reported the *total* request time as TTFT — which is
  how "`llama-server` loses the 9B warm burst 3×" and "fox wins 9B multi-turn 7.9×" both
  got published before being retracted. The tell was in the same table: `ITL p50 0.0 /
  p99 0.0`, i.e. no inter-token gaps at all, read as an oddity instead of as a broken
  instrument.
- **Discard the first round of every arm.** Not just of the engine you happen to know
  has a warm-up cost. The first run of *any* arm pays a cold page cache for the model
  file and the first fault-in of its KV reservation; measured 2026-08-04, GPU occupancy
  in round 1 was 17% for fox and 33% for `llama-server` against ~70-84% afterwards, and
  several sweep levels failed to complete outright. The resulting outlier was read as
  "the `llama-server` arm is unstable" for most of a day, and its ranges were declared
  unpublishable on that basis. `scripts/bench_engines.sh` now runs an unrecorded warm-up
  round by default (`WARMUP=0` to skip). After it, occupancy is 58/58/58 and 56/58/57,
  and the 16-client range narrowed from [300, 436] to [376, 389].

  The lesson had already been learned and written down — "discard vLLM's first start,
  the torch.compile cache is cold" — and applied to that one engine instead of to the
  harness. A rule filed under one engine's name is a rule you will re-learn.
- **When a result is surprising, suspect the instrument first.** Three times in one day a
  striking number was a measurement fault: a bundle 14 minutes older than the commit, a
  shell variable collision that silently ran one engine of three, and the field above.
  None of them failed loudly; all three produced well-formatted, plausible tables.
- **Prefer absolute latencies to ratios against each engine's own baseline.** The
  noisy-neighbour "factor" rewarded whichever engine had rougher idle streams: at a
  matched chunk `llama-server` scored a worse factor while stalling less.

## Results — Vulkan trio, 2026-08-03

`scripts/bench_engines.sh`, 3 rounds, arm order rotated each round so every engine leads
exactly once. 8 clients, 1856-token shared system prompt, 64 output tokens, 4096 ctx per
sequence, Llama-3.2-1B-Q8_0, Vulkan on the 890M. One server alive at a time.

| workload | fox | `llama-server` | Ollama |
|---|---|---|---|
| cold TTFT p50 | **1102 ms** | 4339 ms | 5377 ms |
| cold range | [1100, 1119] | [4312, 4341] | [5137, 5392] |
| cold burst wall | **3.00 s** | 8.43 s | 9.18 s |
| warm TTFT p50 | **50 ms** | 184 ms | 400 ms |
| warm range | [48, 53] | [184, 191] | [390, 411] |
| `cached_tokens`, cold | 12908 | 0 | not reported |

All ranges disjoint. fox is 3.94× `llama-server` and 4.88× Ollama cold; 3.68× and 8.00×
warm. The fox↔`llama-server` figures reproduce the earlier separate run (1129/4550 cold,
50/190 warm) on a freshly rebuilt bundle, which is the harness agreeing with itself.

Two things this table does **not** establish, and the write-up must not let it imply:

- **Ollama's warm TTFT is the odd number here**, 2.2× `llama-server`'s despite both being
  llama.cpp underneath, and nothing measured so far explains it. Its config was verified
  from its own log — `n_ctx = 32768`, `n_ctx_seq = 4096`, 8 slots, `flash_attn = auto`,
  matching the `llama-server` arm exactly — so it is not the obvious misconfiguration.
  Ollama ships a different llama.cpp (ggml 0.17 vs the vendored 0.15.3). Until the cause
  is found this is an observation, not a mechanism, and should be published as one.
- Nothing about **decode throughput**, which is the workload where fox has historically
  sat *below* `llama-server` at 96%. See the control below.

### The neutral control — where fox loses

`MODE=decode scripts/bench_engines.sh`, same rounds and rotation, 4 clients, 4 unrelated
short prompts, 128 output tokens each. Nothing to reuse, so this is the sampling and
batching path with prefill out of the picture. All three engines produced exactly 128
tokens per request, so they did the same work.

| metric | fox | `llama-server` | Ollama |
|---|---|---|---|
| per-request decode p50 | 45.3 tok/s | **49.6 tok/s** | 45.2 tok/s |
| range | [45.2, 45.4] | [49.0, 49.8] | [44.7, 46.0] |
| aggregate | 170.3 tok/s | **185.5 tok/s** | 158.3 tok/s |
| range | [170.0, 171.8] | [183.4, 186.9] | [155.3, 160.3] |

`llama-server` wins this one: **1.09× per request, 1.09× aggregate, ranges disjoint.**
fox and Ollama tie on the per-request rate (their ranges overlap, so no winner), but fox
finishes the batch 1.08× sooner on the aggregate with disjoint ranges — same per-stream
speed, better batching.

Note the gap against `llama-server` measured this way is **8-9%, not the 4%** quoted
elsewhere in this document. Different workload — 4 clients × 128 tokens here — so both
can be true, but the paper must quote the figure with the workload attached and should
not lead with the smaller one.

This is the table that has to sit next to the burst results, at the same prominence.
fox's case is "much faster when there is a prefix to share, slightly slower when there
is not", and stating the second half is what makes the first half credible.

### Both backends, measured — there is no single winner

Decided to publish both rather than pick one. `BACKEND=vulkan|rocm scripts/bench_engines.sh`,
3 rounds each. Cold-burst TTFT p50:

| engine | Vulkan | ROCm | |
|---|---|---|---|
| fox | **1121 ms** | 2391 ms | Vulkan 2.1× |
| `llama-server` | **4327 ms** | 11315 ms | Vulkan 2.6× |
| Ollama | 5344 ms | **4645 ms** | ROCm 1.15× |

Warm TTFT reverses it — ROCm wins for all three (fox 48 vs 49, `llama-server` 140 vs 180,
Ollama 370 vs 405 ms) — and decode leans slightly Vulkan. So the older
`rocm-benchmarking-2026-08.md:107` line, "ROCm is ~15% faster than Vulkan", holds **only
for decode**; on cold prefill Vulkan is 2-2.6× better for both llama.cpp-derived engines.
Any backend recommendation has to name the workload.

Practical asymmetry worth stating alongside the numbers: gfx1150 is not officially
supported by ROCm. Both ROCm images compile for gfx1100 and `HSA_OVERRIDE_GFX_VERSION`
misrepresents the card to the runtime. Vulkan needs none of that and also runs on Intel
and NVIDIA. A 15% decode win does not buy that fragility for a default.

### Saturation curves — SUPERSEDED (contaminated control, ceiling not reached)

> Kept for the record only. The control was contaminated above 16 clients and the sweep
> stopped before either engine bent. Use "Saturation, to 128 clients" further down.

`MODE=sweep`, decode workload at concurrency 1→32, 3 rounds. Aggregate tok/s and the
scaling efficiency against a single client:

| conc | fox (Vulkan) | `llama-server` (Vulkan) | Ollama (Vulkan) | fox (ROCm) | `llama-server` (ROCm) | Ollama (ROCm) |
|---|---|---|---|---|---|---|
| 1 | 53 | 54 | 48 | 52 | 54 | 48 |
| 4 | 170 | **192** | 158 | 174 | 184 | 129 |
| 8 | 249 | **277** | 248 | 277 | 299 | 221 |
| 16 | 376 | **429** | 337 | 416 | 434 | 140 |
| 32 | 584 | **663** | 460 | 496 | 496 | 133 |
| efficiency @32 | 35% | 38% | 30% | 30% | 28% | 9% |

Read honestly, three things come out of this:

- **The sweep never found fox's or `llama-server`'s knee on Vulkan.** Both were still
  climbing at 32, so "peak at concurrency 32" is the sweep's ceiling, not the engine's.
  Extend to 64 and 128 before quoting a maximum. Reporting the ceiling as a peak would
  be the same error class as a silent truncation.
- **`llama-server` leads the decode sweep at every level.** Consistent with the neutral
  control; fox's advantage is not throughput.
- **Ollama on ROCm collapses past 8 clients** — 221 tok/s at 8, then 140 at 16 and 133 at
  32, with efficiency down to 9% and ITL p99 at 204 ms. On Vulkan it scales normally to
  32. Something in the ROCm path degrades under concurrency; not root-caused, and it
  should be reproduced before it goes in a paper.

### Noisy neighbour — the workload where the gap is largest

`MODE=noisy`: 4 interactive clients streaming short chats continuously, then one ~4000-token
prompt injected. Everything is measured as inter-token latency inside the injection
window, because all three engines produce identical average throughput over the run — the
damage is a freeze in somebody else's stream, and an average cannot see it.

| | ITL p99 before | during | factor | long prefill |
|---|---|---|---|---|
| **Vulkan** | | | | |
| fox | 51 ms | **278 ms** | **5.5×** | 1972 ms |
| `llama-server` | 21 ms | 940 ms | 43.8× | 1760 ms |
| Ollama | 40 ms | 1059 ms | 26.3× | 2194 ms |
| **ROCm** | | | | |
| fox | 60 ms | **664 ms** | **11.1×** | 4819 ms |
| `llama-server` | 23 ms | 2329 ms | 100.8× | 4618 ms |
| Ollama | 46 ms | 900 ms | 19.5× | 1894 ms |

**RETRACTED as an architectural claim — it is a default.** See "the noisy-neighbour
advantage is a flag" below. The numbers stand; the interpretation does not.

This is the largest separation any workload here produces, and it is the one a user feels
most directly. But it comes with a finding that must be published next to it:

**fox has the worst baseline jitter of the three.** 51-60 ms ITL p99 at rest against
`llama-server`'s 21-23 ms. fox does not win by having a smoother stream; it wins by not
freezing the stream when a long prefill arrives. Stating only the factor would be
misleading — a reader who measures idle jitter would find fox 2.4× worse and conclude the
whole table was cooked.

### The noisy-neighbour advantage is a flag, not a design

Traced by arithmetic first: the stall an interactive stream suffers is the **prefill chunk
size × per-token prefill cost**. fox chunks at 512 tokens (`--max-prefill-chunk`, default
512); llama.cpp fills `n_batch = 2048` per `llama_decode` (`server-context.cpp:3051`).
Both interleave decode tokens with prefill — only the chunk differs, by 4×.

Tested from both directions, 3 rounds each:

| arm | ITL p99 before | during | factor | long prefill |
|---|---|---|---|---|
| fox, chunk 512 (default) | 50 ms | 273 ms | 5.5× | 1964 ms |
| `llama-server`, n_batch 2048 (default) | 21 ms | 933 ms | 44.3× | 1759 ms |
| **`llama-server` with `-b 512`** | 21 ms | **263 ms** | 12.4× | 1896 ms |
| **fox with chunk 2048** | 50 ms | **976 ms** | 19.5× | 1801 ms |

At a matched chunk the two engines stall **the same amount**: 263 ms against fox's 273 ms.
Give fox llama.cpp's chunk and it degrades to 976 ms, indistinguishable from
`llama-server`'s 933 ms. The advantage is entirely `--max-prefill-chunk 512` being a
quarter of `n_batch = 2048`, and `-b 512` hands it to the reference for free. Publishing
"fox degrades 5.5× where `llama-server` degrades 44×" as an architectural property would
have been exactly the failure the equal-tuning-effort rule exists to prevent — this time
in fox's favour.

It is not free for either: the smaller chunk costs ~8% on the long prefill itself (fox
1964 vs 1801 ms; `llama-server` 1896 vs 1759 ms). That is the real trade — interactive
smoothness against prefill throughput — and it is a flag both engines expose.

**The headline metric was also wrong.** The "factor" is a ratio against each engine's own
baseline, so it rewards an engine for having *worse* idle jitter. At a matched chunk
`llama-server` shows a worse factor (12.4× vs 5.5×) while suffering a *smaller* absolute
stall (263 vs 273 ms) — purely because its baseline is 21 ms against fox's 50 ms. Report
the **absolute stall**; the ratio flatters whoever starts out rougher.

### The KPIs that were missing

Two were added because their absence was hiding real behaviour, not to pad the table.

**Inter-token latency.** Adding ITL to the *existing* burst workload revealed what TTFT
alone had been reporting as a mere 4× difference: in the cold burst, `llama-server`'s ITL
p99 is **872 ms on Vulkan and 2304 ms on ROCm**, against fox's 74-76 ms. The interference
effect was in the data all along and no reported metric could see it.

**GPU memory, split VRAM/GTT.** This iGPU carves out only 2 GB as VRAM; everything else
lands in GTT, system RAM mapped for the GPU. Peaks above an idle baseline, burst workload:

| | VRAM | GTT | GPU busy |
|---|---|---|---|
| fox (Vulkan) | 237 MB | 2481 MB | 66% |
| `llama-server` (Vulkan) | 298 MB | 2069 MB | 84% |
| Ollama (Vulkan) | 238 MB | 2192 MB | 77% |
| fox (ROCm) | 2 MB | 2834 MB | 71% |
| `llama-server` (ROCm) | 3 MB | 2516 MB | 92% |

On ROCm the VRAM figure does not move at all. A VRAM-only memory column would have read
as "nothing allocated" for every engine on that backend.

**fox uses ~400 MB more GTT than `llama-server`, consistently**, while keeping the GPU
busy 66-71% against its 84-92%. Less redundant work, more memory held. Both halves belong
in the table; the second is the cost of the first.

The occupancy number doubles as the assertion that replaced a log that does not exist:
`llama-server` never states its backend anywhere, and this machine has a documented case
of `libggml-hip.so` failing to dlopen and the server falling back to CPU silently. Reading
the driver catches that for every engine. A busy percentage near zero aborts the row.

### The decode deficit: two hypotheses tested, both wrong

fox sits ~10% below `llama-server` on decode-bound throughput. Two explanations were
tested directly rather than argued about. Neither survived.

**Hypothesis 1 — `kv_unified = true` costs decode throughput.** Tested with
`FOX_KV_UNIFIED=0`, a runtime switch so both arms come from **one binary** (building two
would put the build into the comparison). Arm `fox-seq` in `scripts/bench_engines.sh`.

| decode, conc 4 | per request | aggregate |
|---|---|---|
| fox (unified) | 45 tok/s | 169.6 |
| fox-seq (not unified) | 46 tok/s | 172.8 |
| `llama-server` | **50 tok/s** | **184.6** |

Turning unified KV off recovers **2%**, not 10. It is not the lever. **Prediction was
wrong and is retracted.**

What the same run *did* price is the trade itself, and it is lopsided:

| | fox | fox-seq |
|---|---|---|
| cold burst TTFT p50 | **1108 ms** | 6300 ms |
| warm burst TTFT p50 | **51 ms** | 5987 ms |

Unified KV buys 5.7× cold and 117× warm TTFT for 2% of decode. Caveat that must travel
with those numbers: `fox-seq` has **no prompt reuse at all** — without a unified cache
fox cannot reuse, whereas `llama-server` without one still reuses from *idle* slots. So
this prices "fox with vs without its prefix cache", not "unified vs non-unified with
equivalent reuse". Only the 2% decode figure isolates unified KV cleanly, because that
workload has nothing to reuse either way.

Also from that run: `fox-seq` degrades 6.3× under a noisy neighbour against fox's 5.5×.
fox's resistance to prefill interference is therefore **not** coming from its prefix
cache — it survives with the cache off. That mechanism is still unidentified.

**Hypothesis 2 — fox fragments its decode batches.** Precedent existed: seq_id
fragmentation was once measured at 1.74 of a possible 4. Read from llama.cpp itself via
`LLAMA_BATCH_DEBUG=1` (never by patching `vendor/`); in decode each sequence contributes
exactly one token, so a ubatch's `n_tokens` *is* the batch fill.

fox at 4 clients: **3.89 of 4**, with 248 of 262 steps completely full. Essentially no
fragmentation; it accounts for ~3% at most. **Also not the lever.**

(`llama-server`'s equivalent trace was not captured — its log never reached DEBUG level
even at `-lv 4`. fox's own figure is enough to rule out gross fragmentation on fox's
side, but the comparison is one-sided and should be completed.)

**Found by profiling: it is the sampler's candidate selection.** `perf record` on both
servers under the same decode workload, 4 clients, one server at a time. (The `perf` on
`PATH` is a stub with no binary for this kernel; `/usr/lib/linux-tools-6.8.0-136/perf`
samples user space fine, which is all this needs.)

| self cost | fox | `llama-server` |
|---|---|---|
| waiting on the GPU fence | 78.7% | 82.8% |
| **sorting** | **6.61%** `quicksort::partition` | **1.39%** `llama_token_data_array_partial_sort_inplace` |
| sampler proper | 2.33% `sample_token` | 2.63% `common_sampler_sample` |
| output filter | 0.93% | — |
| **total CPU outside the GPU wait** | **~9.9%** | **~4.0%** |

The ~5.9% difference is the size of the unexplained gap. The mechanism is at
`src/engine/model/sampling.rs:189`, executed once per token **per sequence**:

```rust
let mut idx: Vec<usize> = (0..logits.len()).collect();   // 128256 × 8 B = 1 MB, per token
idx.select_nth_unstable_by(k - 1, |&a, &b| {
    logits[b].partial_cmp(&logits[a])                    // indirect: chases a 512 KB array
```

Two costs `llama-server` does not pay: a **1 MB allocation per token**, and a comparator
that **dereferences into a separate logits array** on every comparison while permuting the
index array. llama.cpp keeps `llama_token_data` (id and logit adjacent) contiguous and
partial-sorts it in place, so its comparisons read the value they are sorting by.

This also explains the shape of the curve, which nothing else did. The GPU decode step is
weight-bound and barely grows from 1 to 4 sequences, while this CPU cost is paid once per
sequence and grows linearly. So its *share* rises with concurrency: ~2% at 1-2 clients,
~10% from 4 upward — exactly the measured step.

Note what is **not** implicated: `logits.to_vec()`, which the older docs blamed, really is
~0.5%. The copy was never the problem; the selection over the copy is.

**Fixed and validated, 2026-08-03.** `select_top_n` (`src/engine/model/sampling.rs`)
keeps a sorted buffer of at most n entries and streams the logits once: the common case
per element is one `f32` compare against a running threshold, sequential, no indirection,
no allocation proportional to the vocabulary.

Validated end-to-end over 3 rounds, not with a micro-benchmark — this repo has a
precedent of a 4.6× sampling micro-benchmark win producing zero real throughput.

| decode, conc 4 | before | after |
|---|---|---|
| fox per request | 45 tok/s [45, 46] | **47 tok/s [47, 47]** |
| fox aggregate | 170 tok/s | **175 tok/s** |
| gap vs `llama-server` | 1.10× | **1.06×** |

No regression in the burst workload: cold TTFT 1108 → 1100 ms, warm 51 → 47 ms, and
`cached_tokens` identical at 12908/14840, so prefix reuse is untouched.

**The sweep-based claims first published for this fix were withdrawn** — see "the
neutral control was not neutral" below. Only the conc-4 decode figures above survive,
because they come from arms alternating inside one run with disjoint ranges. Validating
the fix at higher concurrency needs an old-sampler arm inside the same run, the way
`fox-seq` was done; comparing sweeps across sessions cannot carry it.

The new unit tests **do not run in CI**: the whole sampling module is
`#[cfg(not(fox_stub))]` and CI runs with `FOX_SKIP_LLAMA=1` (331 tests there against 430
in a real build). A sampler regression would not be caught by `make ci`.

### The neutral control was not neutral above 16 clients

`bench_decode.py` held 16 prompts and handed them out with `i % 16`, so from 32 clients
upward two clients got **byte-identical prompts** — precisely what fox's prefix cache
exists to reuse. The control turned into the favourable workload at exactly the
concurrencies where the sweep was making its strongest claims. Fixed by putting the
client index first, so two clients share one token instead of a whole prompt.

The bias was real and one-sided, which is itself a demonstration of the paper's thesis:

| conc 32, aggregate | duplicate prompts | unique prompts |
|---|---|---|
| fox | 641 | **570** (−11%) |
| `llama-server` | 664 | 673 (+1%) |

fox gained 11% from the duplicates and `llama-server` nothing, because `llama-server`
cannot reuse from a live sibling and fox can. **Every sweep figure at 32 published before
this fix is inflated in fox's favour and is retracted**, including "the deficit at 32
goes from 13.5% to 3.5%" — measured cleanly the deficit at 32 is **15%**.

A second lesson from the same comparison: two post-fix runs at 16 clients gave 423 and
400 tok/s, a 5.7% spread between sessions, while the within-run ranges were ±1%. Sweep
numbers are **not comparable across sessions** at better than ~6%, so any A/B on them
must run both arms inside one alternating run.

### Saturation, to 128 clients — fox has a ceiling and `llama-server` does not

3 rounds, alternating arms, unique prompts, Vulkan.

| conc | fox | range | `llama-server` | range |
|---|---|---|---|---|
| 1 | 53 | [53, 53] | 54 | [54, 54] |
| 4 | 176 | [174, 176] | 190 | [187, 190] |
| 8 | 250 | [249, 255] | 275 | [269, 278] |
| 16 | 400 | [399, 404] | 432 | [431, 435] |
| 32 | 570 | [570, 570] | 673 | [673, 675] |
| 64 | **610** | [604, 617] | 782 | [780, 789] |
| 128 | **416** | [414, 416] | **843** | [680, 871] |

**fox peaks at 64 clients and then collapses**: 610 → 416 tok/s, scaling efficiency down
to 6%, and ITL p99 at **400 ms** against `llama-server`'s 124 ms. `llama-server` never
bends inside this range — it is still climbing at 128, so its own knee is beyond what was
measured (and its 128 range, [680, 871], is wide enough that the level is unstable).

At 128 clients `llama-server` serves **2.03× fox's throughput**. This is a far more
important result than the sampler fix.

**Cause found: the unified KV cache.** Same sweep with the `fox-seq` arm
(`FOX_KV_UNIFIED=0`, same binary), 3 rounds, ranges disjoint against fox at every level:

| conc | fox | fox-seq | `llama-server` | cost of unified KV |
|---|---|---|---|---|
| 16 | 392 [391, 394] | 417 [415, 420] | 435 [431, 439] | 6% |
| 32 | 568 [561, 572] | 644 [641, 656] | 658 [647, 665] | 13% |
| 64 | 611 [611, 616] | 778 [771, 790] | 777 [772, 787] | 27% |
| 128 | **422** [420, 424] | **876** [869, 876] | 845 [794, 858] | **108%** |

`fox-seq` does not bend at all: it matches `llama-server` at 64 (778 vs 777) and passes it
at 128 (876 vs 845). **The entire collapse is the unified KV cache**, and nothing else in
fox is implicated — scheduler, admission budget and sampler are common to both arms.

The mechanism follows from `n_stream = 1`: with one shared cell pool, a decode step
attends over the union of every sequence's cells, so cost grows as N·(N·L) instead of
N·L. Measured decode-step time confirms the shape — doubling clients from 64 to 128
multiplies fox's step time by **2.93** and `llama-server`'s by 1.85.

ITL p99 at 128 tells the same story from the user's side: fox 391 ms, fox-seq 176 ms,
`llama-server` 125 ms.

**This corrects an earlier conclusion in this document.** "Turning unified KV off recovers
2%, it is not the lever" was measured at **4 clients**, where it is true. As a general
statement it was wrong: the cost is not a fixed percentage but a curve — 2% at 4 clients,
108% at 128.

So fox's central design choice is now priced on both sides, and the trade is sharp rather
than free:

| unified KV buys | unified KV costs |
|---|---|
| 5.7× cold TTFT, 117× warm (prefix reuse from a *live* sibling) | 6% throughput at 16 clients, 108% at 128 |
| the noisy-neighbour advantage is **not** among them — `fox-seq` degrades 6.3× vs fox's 5.5×, so that comes from somewhere else | a concurrency ceiling at ~64 |

The obvious follow-up is a design question, not a measurement: the mode is currently
compile-time-fixed, and `FOX_KV_UNIFIED` exists only as a measurement switch. Choosing it
per load — unified while concurrency is low and prefixes are shared, non-unified above the
knee — is not implemented and would need the switch to be safe to flip on a live model,
which it is not today.

It also bounds every "fox is within X% of `llama-server`" claim in this document to
**concurrency ≤ 16**. Above that the gap widens: 15% at 32, 22% at 64, 103% at 128.

Memory at 128 clients: `llama-server` peaks at **+17.5 GB of GTT**, which is the KV cache
for 128 × 4096 tokens. It fits only because this machine shares 123 GB of system RAM with
the GPU.

### vLLM — its own section, 2026-08-03

`scripts/bench_vllm.sh`, 3 rounds, server restarted per round. `rocm/vllm:latest`
(v0.11.2.dev), `HSA_OVERRIDE_GFX_VERSION=11.0.0`, `--max-model-len 4096 --max-num-seqs 8
--enable-prefix-caching --gpu-memory-utilization 0.55`, BF16 safetensors
(`unsloth/Llama-3.2-1B-Instruct`), ROCm. Startup 40-46 s per round. Same clients, same
workloads as the trio.

| workload | vLLM |
|---|---|
| cold TTFT p50 | 1995 ms, range [1975, 2058] |
| warm TTFT p50 | 669 ms, range [654, 711] |
| decode per request | 19.3 tok/s |
| decode aggregate | 75.6 tok/s |

**Do not put this column next to the trio's.** Backend and weight format both differ.
The decode figure in particular is mostly explained by the weights, not the serving
layer: BF16 moves roughly twice the bytes per token that Q8_0 does, and decode on this
iGPU is memory-bound, which is about the whole of the 45 → 19 tok/s difference. Saying
"fox decodes 2.3× faster than vLLM" from this table would be quoting a quantisation
choice as an engine result.

The first vLLM run of the day reported a **2758 ms** cold TTFT against the 1995 ms
measured here. The difference is `torch.compile`'s cache being cold on the very first
start; it is discarded rather than averaged in, and any future run should throw away its
first start for the same reason. Nothing equivalent applies to the other three.

### `cached_tokens` reads 0 for two different reasons — checked, not assumed

`scripts/probe_cached_tokens.py` sends the same prompt twice, streamed and non-streamed,
and reports whether `prompt_tokens_details` comes back at all.

| engine | `prompt_tokens_details` | so its 0 means |
|---|---|---|
| fox | present (12908 cold, 14840 warm) | real reuse, measured |
| `llama-server` | present (0 cold, 14840 warm) | real: none cold, full warm |
| Ollama | **absent**, both streamed and not | not reported |
| vLLM | **absent**, both streamed and not | not reported |

Both engines that report nothing show a large warm TTFT drop (Ollama 5377 → 400 ms, vLLM
1995 → 669 ms), so they are reusing prefixes and simply not exposing the counter.
Publishing their 0 in a "cached tokens" column would state the opposite of what happened,
which is why the column has to carry the distinction rather than the number alone.

## Results so far — SUPERSEDED, kept for provenance

> The first fox↔`llama-server` run, from before the harness was generalised. Its TTFT
> figures were reproduced within noise by `scripts/bench_engines.sh`; its "96%" throughput
> line did not survive (see the balance at the top). Numbers below are the originals.

fox vs `llama-server`, Vulkan, Llama-3.2-1B-Q8_0, both from the same vendored llama.cpp,
3 rounds, disjoint ranges:

| workload | fox | llama-server |
|---|---|---|
| 8 clients, shared 1856-token prompt, cold TTFT p50 | **1129 ms** | 4550 ms |
| 16 clients, cold TTFT p50 | **1402 ms** | 8064 ms |
| 16 clients, whole-burst wall | **3.8 s** | 16.2 s |
| 8 clients, warm TTFT p50 | **52 ms** | 193 ms |
| 4 clients, short unrelated prompts, throughput | 96% | baseline |

Doubling the clients costs fox 24% more cold TTFT and `llama-server` 79%.

The mechanism, which is the paper's actual thesis: `get_available_slot()` skips
`is_processing()` slots in both its similarity pass (`server-context.cpp:1609`) and its
LRU fallback (`:1652`), so concurrent arrivals cannot reuse from each other. fox copies a
shared prefix out of a sequence that is still decoding.

## Also outstanding, unrelated to the benchmark

- Merge to `develop` by milestones — **`main` is a squashed release-snapshot branch with
  no common ancestor**, so it is not the merge target. Three decisions are the user's:
  whether to integrate `origin/develop`'s 4 divergent commits, whether to push, and
  whether to create/push `v0.14.0`…`v0.19.1` tags (pushing tags triggers release
  workflows).
- Branch is `feature/0.19`; version is 0.19.1.
- `--moe-cpu` has no demo or guide, now that the catalogue has MoE models.
- The 4% decode gap: profile before acting. The `logits.to_vec()` the docs blamed is
  ~0.5% by arithmetic, so it is not the lever.
