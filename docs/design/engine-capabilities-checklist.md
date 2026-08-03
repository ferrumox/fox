# Engine capabilities checklist

A master taxonomy of what a *useful* LLM inference engine has to account for —
**backends, model architectures, quantization, inference features, serving, and the
cross-cutting properties that keep it maintainable** — annotated with fox's current
standing. Use it as the expansion map: harden the 🎯 subset on the target machine first,
then widen.

See also: [`STATUS.md`](../../STATUS.md) (per-feature status),
[model-architecture rework](model-architecture-rework.md) (the plan that closes most ❌/⚠️).

**Legend:** ✅ fox has it · ⚠️ partial / with caveats · ❌ no · 🚧 llama.cpp has it, fox
doesn't wire it · 🎯 matters for the current target machine (AMD Ryzen + Radeon 890M; CPU/Vulkan)

---

## 1. Backends / hardware

| | Backend | Notes |
|---|---------|-------|
| 🎯✅ | CPU (x86 AVX2/AVX512/AMX, ARM NEON/SVE) | guaranteed baseline; this PC's Ryzen has AVX512 |
| ✅ | CUDA (NVIDIA) | |
| ⚠️ | ROCm/HIP (AMD) | wired for dGPU; **fragile on the 890M iGPU (gfx1150)** — avoid for "stable" |
| 🎯✅ | Vulkan | best GPU path for the Radeon 890M; needs `glslc`+headers so `build.rs` enables it |
| ✅ | Metal (Apple) | |
| 🚧 | SYCL (Intel), CANN (Huawei), MUSA, OpenCL/Adreno, RPC (distributed) | in llama.cpp, not exposed by fox |
| ✅ | Runtime backend selection (`GGML_BACKEND_DL`, CUDA→ROCm→Vulkan→Metal→CPU) | |

**Per-backend concerns:** flash-attention support, `head_dim` constraints, fp16 vs fp32
accumulate, real *free* memory (not total), clean fallback if a backend fails mid-load.

## 2. Model architectures

| | Class | fox |
|---|-------|-----|
| ✅ | Dense (Llama, Mistral, Phi) | |
| ✅ | GQA/MQA (`n_head_kv < n_head`) | `embedding_dim` bug fixed (= `n_embd`, 0.11) |
| ⚠️ | Non-standard head_dim + softcapping (Gemma 2/3) | needs FA=AUTO + head_dim from metadata (patched) |
| ⚠️ | Sliding-window / local attention (Gemma, Mistral, Phi3) | llama.cpp handles it; fox's paged KV doesn't model it |
| ⚠️ | MoE (Mixtral, Qwen-MoE, DeepSeek-MoE) | load + CPU offload; approximate sizing |
| ✅ | MLA / latent KV (DeepSeek-V2/V3) | sizing fixed (0.18) via empirical create-then-shrink retry, no per-token formula; verified against real DeepSeek-V2-Lite — see `mla-recurrent-kv-sizing.md` |
| ✅ | Recurrent/hybrid (Mamba, RWKV, Jamba) | sizing fixed (0.18, same mechanism); prefix-cache-disable detection also fixed (0.18) — was silently wrong (`llama_memory_can_shift`), now `llama_model_is_recurrent`/`llama_model_is_hybrid`; verified against a real Mamba model. **0.20.0**: disabling it was too broad — these models keep *slot* reuse (nothing is copied), only cross-sequence `seq_cp` stays off, and prompt reuse needs `--rs-rollback > 0` |
| ❌ | Encoder-decoder (T5) | |
| ⚠️ | Embeddings (BERT, nomic) | dimension + all-zeros bugs fixed (0.11, golden-verified); always mean-pooled + L2 — dedicated-model pooling (CLS) not auto-detected |
| ❌ | Vision / multimodal (llava, qwen-vl, gemma3-vision) | image blocks silently dropped |
| ⚠️ | RoPE scaling (linear/NTK/YaRN, long context) | llama.cpp handles; fox doesn't expose/validate |

**Concern:** each family introduces *one* parameter that breaks assumptions (softcap,
sliding window, MLA, state-space). This is why a single source of truth (`ModelInfo`) matters.

## 3. Quantization

| | Kind | fox |
|---|------|-----|
| ✅ | GGUF weights: K-quants (Q2_K…Q8_0), legacy, IQ (imatrix) | any model llama.cpp loads |
| ✅ | KV cache: f16 / q8_0 / q4_0 | standard llama.cpp KV types; TurboQuant removed (upstream migration) |
| ❌ | Non-GGUF (AWQ/GPTQ/FP8/bnb safetensors) | out of scope (GGUF engine) |

## 4. Inference features

**Sampling:** ✅ temp, top_p, top_k, seed, repetition_penalty, frequency/presence_penalty
(additive, OpenAI semantics) · ✅ min_p, logit_bias, min_tokens (0.14) · ❌ typical_p, mirostat.

**Decoding / scheduling:**

| | Feature | fox |
|---|---------|-----|
| ✅ | Continuous batching; paged KV + ref-count + CoW; automatic prefix caching; text stop sequences | |
| 🎯✅ | Chunked prefill | `--max-prefill-chunk` (default 512): a long prompt is prefilled in chunks across scheduler steps, interleaved with other requests' decode |
| ✅ | Speculative decoding (draft / n-gram / EAGLE) | n-gram/prompt-lookup (0.15) + draft-model (0.16), both exact + golden-verified via `--speculative`/`--draft-model`; EAGLE-style trained draft heads ❌ |
| ✅ | Guided/structured decoding (grammar / JSON-schema) | GBNF-constrained via `response_format`/`format` (0.14, golden-verified); regex ❌ |
| ✅ | Tool/function calling | Hermes + Mistral parsers auto-detected from the model's own template (0.16, `tools` threaded into the Jinja context); Llama3 parser explicit-opt-in only (`--tool-call-parser llama3` — unreliable template auto-detection in practice); generic prompt-based JSON remains the fallback otherwise |
| ⚠️ | `n>1` / `best_of` / beam search; logprobs / echo | logprobs/top_logprobs ✅ (0.14); `n`/`best_of` ✅ (0.18, independent fan-out); beam search closed as a deliberate non-goal (0.18) — see `vllm-gap-analysis.md`; echo ❌ |
| ⚠️ | Context management: RoPE scaling partial; **context-shift/rolling** on full (`--context-shift`, shiftable caches) ✅; RoPE scaling still not exposed | |
| ❌ | LoRA / adapters (incl. multi-LoRA) | |
| ⚠️ | Thinking/reasoning (`<think>` separation) | real per-model detection via the Jinja template's `enable_thinking` + a small `REASONING_FORMATS` registry (0.11); an unlisted family still falls back to the `<think>` heuristic |

**Correctness:** ✅ tokenization BPE/SPM/Unigram + add_special/BOS · ✅ chat templates —
rendered via real Jinja (`minijinja`, fixed 0.11, see [`STATUS.md`](../../STATUS.md) for the
resolved finding); falls back to llama.cpp's legacy engine only when a model has no embedded
template. `tools` is threaded into the render (0.16), so native tool-formatting macros
(Hermes/Qwen tool-use templates) are exercised when present · ✅ EOG/control tokens from
vocab (⚠️ some hardcoded literals) · ✅ multi-token UTF-8 reassembly · ✅ seeded determinism.

## 5. Serving / API / runtime

✅ OpenAI + Ollama compat · ✅ SSE/NDJSON streaming · ✅ embeddings (mean-pool + L2; ⚠️ CLS not auto-detected) · ✅ multi-model
+ LRU + keep-alive (⚠️ `max_models=1` default) · ✅ disconnect cancellation · ✅
preemption/queueing · ✅ auth + CORS · ✅ Prometheus + logs + health.
❌ consistent defaults across both APIs · ⚠️/❌ rate-limit/backpressure/max-queue · ❌ OOM
recovery (retry, degrade context).

## 6. Model management / distribution

✅ HF pull + curated registry + aliases (⚠️ ambiguous resolution) · ⚠️ real Modelfile (only
`FROM`) · ✅ single static binary · ✅ build with backend detection (⚠️ needs submodule +
toolchain) · ✅ config flags/env/file · ✅ Docker/systemd/installers.

## 7. Cross-cutting — what makes an engine *maintainable* (not just featureful)

- ✅ **Single source of truth per model** (`ModelInfo`) — landed (P0); a lighter
  subset of the doc's original proposal (no full `ArchClass`, but includes the
  0.18 `KvMemoryClass`).
- ⚠️ **Per-architecture regression net** (golden tests + CI) — a generic
  (any-model) golden harness exists; still no dedicated per-architecture-class
  fixture matrix in CI (MoE/MLA/recurrent are verified manually per-feature,
  not gated in CI).
- ❌→planned **Explicit support contract** (what's supported, at what level).
- ⚠️ **Fail loudly, not silently** (several silent failures today).
- ✅ **Observability of derived facts** (`fox probe`) — landed (P0).

**Note:** this checklist predates the 0.11 P0/P1 rework commits and the 0.16-0.18
feature work (draft-model speculation, tool-call parsers, vision, LoRA, `n`/`best_of`,
MLA/recurrent sizing) — several rows above are stale. [`STATUS.md`](../../STATUS.md)
is the current, actively-maintained source of truth; this file is kept for its
original architecture-coverage framing, not as an up-to-date inventory.

---

## The subset that matters NOW (target machine: AMD Ryzen + Radeon 890M)

1. **Correct CPU backend** (deterministic baseline) → then **Vulkan** for the 890M (FA=AUTO already helps).
2. **The models actually used**: small/medium **dense/GQA** (Llama-3.2, Qwen2.5, Gemma-3). No MoE/MLA/recurrent yet.
3. **Core correctness** on this hardware: tokenizer, chat template, sampling, stop, UTF-8.
4. **Stability with 1–2 clients** before high concurrency (chunked prefill can wait).

Everything else (MoE, MLA, recurrent, vision, distributed, speculative/guided decoding) is
a later **expansion phase**, once this base is solid.
