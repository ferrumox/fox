# fox vs vLLM — gap analysis

What separates fox from [vLLM](https://github.com/vllm-project/vllm), the reference
high-throughput serving engine. The point of this doc is **not** to make fox into vLLM
— it's to be honest about which gaps are *worth closing* and which are consequences of
a deliberate architectural choice.

## The load-bearing difference

**vLLM is a native PyTorch/CUDA engine with custom kernels; fox wraps llama.cpp.** vLLM
owns its attention kernels (PagedAttention, FlashInfer), its CUDA graph capture, its
tensor-parallel all-reduce. fox rides llama.cpp's `llama_decode` — fox's scheduler
decides *what* to batch, but the compute is llama.cpp's.

That single fact sets the ceiling: **on a datacenter GPU, fox's tokens/s is bounded by
llama.cpp, not by fox.** Chasing vLLM's raw throughput there is not a feature backlog,
it's a rewrite. So the gaps below are tagged:

- **achievable** — wireable inside the wrapper model, often because llama.cpp already
  has the primitive; these are real backlog items.
- **structural** — tied to owning the kernels / being a datacenter engine; out of fox's
  niche (single static binary, CPU + consumer iGPU via Vulkan, Ollama-compatible,
  instant start).

**Legend:** ✅ has it · ⚠️ partial / with caveats · ❌ no

---

## 1. Throughput / batching core

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Continuous batching | ✅ fused in kernel | ✅ at scheduler level | — |
| PagedAttention | ✅ custom CUDA kernel | ⚠️ paged **accounting**; the attention itself is llama.cpp's | structural |
| Automatic prefix caching | ✅ | ✅ | — |
| Chunked prefill | ✅ | ✅ (0.13) | — |
| CUDA graphs / `torch.compile` | ✅ | ❌ (llama.cpp's domain) | structural |
| FlashAttention / FlashInfer selection | ✅ | ⚠️ FA=AUTO via llama.cpp | structural |

Behaviourally (interleaving, queueing, fairness) fox can match vLLM; on raw GPU
throughput it cannot, and that's fine — different hardware target.

## 2. Advanced decoding — **the highest-ROI gaps**

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Guided / structured decoding (JSON-schema, regex, grammar) | ✅ (outlines / xgrammar) | ❌ prompt-only | **achievable** |
| logprobs / prompt_logprobs / echo | ✅ | ❌ | **achievable** |
| Speculative decoding (draft / n-gram / EAGLE / Medusa) | ✅ | ❌ | **achievable** |
| `n>1` / best_of / beam search | ✅ | ❌ | achievable |

**llama.cpp has native GBNF grammar support** → structured/JSON decoding is the single
biggest impact for the least effort. Speculative decoding is the largest *latency* win
that is actually within reach (llama.cpp has draft-model and n-gram primitives).

## 3. Sampling

fox has: temperature, top_p, top_k, seed, repetition/frequency/presence penalties.

Missing vs vLLM (all **achievable** — pure sampling logic): **min_p, typical_p,
mirostat, logit_bias, min_tokens.**

## 4. LoRA / adapters

vLLM serves multi-LoRA with per-request hot-swap. fox: ❌. llama.cpp supports LoRA, so
this is **achievable** at medium effort.

## 5. Model architectures

| Class | vLLM | fox |
|---|---|---|
| Dense / GQA | ✅ | ✅ solid |
| MoE (Mixtral, DeepSeek-MoE, Qwen-MoE) | ✅ optimized | ⚠️ loads + CPU offload, approximate sizing |
| MLA / latent KV (DeepSeek V2/V3) | ✅ | ❌ positional sizing wrong |
| Vision / multimodal (LLaVA, Qwen-VL) | ✅ | ❌ image blocks silently dropped |
| Embeddings (BERT, nomic) | ✅ | ✅ dim + pooling fixed (0.11); mean-pool only, CLS not auto-detected |
| Encoder-decoder (T5) | ✅ | ❌ |
| Recurrent / hybrid (Mamba, RWKV) | ✅ | ⚠️ detected, prefix-cache off |

See [`engine-capabilities-checklist.md`](engine-capabilities-checklist.md) §2 for the
per-architecture detail and [`model-architecture-rework.md`](model-architecture-rework.md).

## 6. Quantization

vLLM: GPTQ, AWQ, FP8, INT8, bitsandbytes, Marlin kernels, KV-cache fp8.
fox: **GGUF only** (K-quants / legacy / IQ) + KV f16 / q8_0 / q4_0.

This is **not a real gap** — GGUF is exactly right for fox's CPU/consumer niche, and
non-GGUF safetensors formats are out of scope by design (fox is a GGUF engine).

## 7. Scale / parallelism (mostly structural)

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| Tensor parallel (kernel-level all-reduce) | ✅ | ⚠️ layer/row split via llama.cpp | structural |
| Pipeline parallel / multi-node / distributed | ✅ | ❌ | structural |
| Disaggregated prefill/decode (P/D) | ✅ | ❌ | structural |

Datacenter-scale features, outside fox's single-node niche.

## 8. Serving robustness (achievable, medium value)

| Capability | vLLM | fox | Kind |
|---|---|---|---|
| OOM recovery (retry, degrade context) | ✅ | ❌ | achievable |
| Backpressure / rate-limit / max-queue | ✅ | ⚠️/❌ | achievable |
| Request priority (priority preemption) | ✅ | ⚠️ LIFO preemption only, no priority | achievable |
| KV offload / swap to CPU | ✅ | ⚠️ `--swap-fraction` placeholder, unimplemented | achievable |
| Tool calling with per-model parsers | ✅ | ⚠️ generic prompt-based | achievable |

fox already has: continuous batching, disconnect cancellation, LIFO preemption,
context rolling (0.13), OpenAI + Ollama compat, Prometheus metrics, auth, health.

---

## Prioritized shortlist (best ROI given the llama.cpp wrapper)

Shipped since this analysis was written:

- ✅ **Guided / structured decoding via GBNF** (0.14) — `response_format` / Ollama
  `format`, JSON-schema→GBNF in Rust.
- ✅ **logprobs / top_logprobs** (0.14).
- ✅ **min_p, logit_bias, min_tokens** (0.14).
- ✅ **Embeddings** were already fixed back in 0.11 (correct `n_embd` length, mean-pool +
  L2, non-degenerate — golden-verified); the only remaining nuance is that dedicated
  embedding models' native pooling (CLS) isn't auto-detected (fox always mean-pools).

Still open, in priority order:

1. **Speculative decoding (n-gram / draft)** — the largest real latency win still in
   reach; llama.cpp has the draft/n-gram primitives.
2. **OOM recovery + backpressure / max-queue** — makes it a real server under overload.
3. **Per-model tool-call parsers** — today's tool calling is generic prompt-based.

## What NOT to chase (outside the niche)

Disaggregated serving, pipeline / multi-node, kernel-level tensor parallelism,
FP8 / AWQ / GPTQ safetensors, CUDA graphs. That is vLLM-in-a-datacenter. fox's niche —
**one static binary, CPU + consumer iGPU (Vulkan), Ollama-compatible, instant start,
low memory** — is a place vLLM doesn't play. Winning there beats losing the throughput
race on an H100.
