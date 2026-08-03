# Speculative decoding — the full ladder (roadmap)

0.15 shipped the *ground floor* of speculative decoding (n-gram / prompt-lookup). This
doc maps **every rung above it**, with an honest viability analysis of each — so when
the roadmap wants "more speed", the options and their real costs are already thought
through. Viability is always judged against fox's two structural facts:

1. **fox wraps llama.cpp's core** — anything that needs custom kernels, custom attention
   masks, or graph surgery is gated on what `llama.h` exposes (verified below against
   the vendored copy).
2. **fox's niche is a single binary on consumer hardware** — a technique that needs
   per-model trained artifacts the GGUF ecosystem doesn't distribute is dead on arrival
   regardless of elegance.

The one design decision that makes this ladder climbable at all: 0.15 split speculation
into **proposer** (guess tokens) and **verifier** (one multi-token `llama_decode`,
accept-on-sampled-match, `seq_rm` the rejects, exactness invariant, metrics, bench).
The verifier, engine wiring, and validation harness are *proposer-agnostic* — every
level below is "swap the proposer", not "rebuild the machinery".

---

## Level 1 — n-gram / prompt-lookup ✅ (shipped, 0.15)

**Proposer:** copy what followed the last occurrence of the current suffix in the
request's own history. **Cost:** zero (no model, no memory). **Win:** 1.78× at 98%
acceptance on context-echoing output; ~0.9× on free prose (9% acceptance) — hence off
by default.

Remaining headroom *within* this level (cheap, incremental):
- **Adaptive draft length** — shrink `draft_len` after misses, grow after hits
  (llama.cpp's `common/ngram-cache` does a weighted version of this). Cuts the prose
  penalty toward ~1.0× so speculation could eventually default on.
- **Multi-ngram / cache across requests** — index n-grams once instead of rescanning,
  and share across a conversation's turns. Only worth it if profiling shows the scan
  matters (it's O(seq) per step today).

## Level 2 — draft model (classic speculative) — **VIABLE NOW, the natural 0.16+ theme**

**Proposer:** a small model of the same family (e.g. Qwen-0.5B for Qwen-7B,
Llama-3.2-1B for Llama-8B) autoregressively predicts `draft_len` tokens; the target
verifies them in one pass. Unlike n-gram, the draft *predicts* rather than *copies* —
it works on *any* text.

**Why it's viable:** needs only core primitives fox already uses — load a second GGUF
(`LlamaCppModel::load`, same code path), run its decode loop for the draft, feed the
verify batch the 0.15 machinery already builds. No new FFI surface at all.

**Expected win:** 60–80% acceptance on general text with a well-matched pair → 1.5–2.5×
reported in the literature and by llama.cpp / vLLM users. Honest caveat for fox's
niche: the economics assume the target is **memory-bandwidth-bound** and the draft is
≥8–10× smaller. On the 890M/CPU with a 7–8B target and a 0.5–1B draft that holds, but
the draft's own compute eats part of the win — expect the low end of the range, and
`bench-spec` (already built) measures it per pair.

**The real work** (why it's a theme, not a patch):
- **Registry/memory management** — a second resident model per engine: loading,
  eviction pairing, VRAM/RAM budgeting (`--draft-model <name>`).
- **Vocab compatibility** — draft and target must share the tokenizer/vocab, or
  drafted ids are meaningless. Needs a hard load-time check (compare vocab hashes),
  failing loudly per the project's rules.
- **Draft scheduling** — the draft decodes `draft_len` tokens sequentially per step;
  batching it sensibly against the target's step matters for the win.
- Proposer trait: `propose(seq) -> Vec<i32>` — n-gram and draft-model become two
  implementations; config picks (`--speculative ngram|draft`).

## Level 3 — MTP / NextN (the model drafts for itself) — **VIABLE SOON; watch item**

**Proposer:** models trained with multi-token-prediction heads (DeepSeek-V3, GLM-4.5)
ship extra "NextN" layers *inside the GGUF* that predict token t+2 cheaply. The draft
is a second, tiny forward through those layers — no second model file, no vocab
mismatch possible, acceptance is high because the heads were trained with the model.

**Why it's more viable than expected:** verified against the vendored llama.cpp — this
is **already in the core**, not `common/`: `llama_context_params.ctx_type` with
`LLAMA_CONTEXT_TYPE_MTP` (create a second *context* over the same weights that runs the
NextN path), `llama_model_n_layer_nextn()`, and the `nextn.*` tensor family in
`llama-arch/llama-model`. That means the fox-side work is close to the draft-model
level's: create the MTP context alongside the main one, use it as the proposer, feed
the 0.15 verifier.

**Why it's not the next step anyway:**
- Only models *trained* with NextN carry the layers — today that's DeepSeek-V3-class
  and GLM-4.5-class (large / MoE), which don't fit fox's target machine. No small
  dense model ships MTP heads yet.
- The core API is new (upstream added it for GLM-4.5/DeepSeek in 2025); stability and
  per-arch coverage need to mature.

**Trigger to act:** a small/medium model fox's users actually run ships NextN layers in
GGUF (the industry direction — MTP training is spreading), or upstream promotes the API
as stable. Then this likely *leapfrogs* level 2 for those models: draft-quality
proposals with zero extra memory.

## Level 4 — self-speculation / layer-skip (Kangaroo, LayerSkip, SWIFT) — **NOT VIABLE in a wrapper**

**Proposer:** run only the first K layers of the target as the draft (early exit).

**Why not:** verified — `llama.h` exposes no per-decode early-exit or layer-range API;
implementing it means building a custom compute graph per model, which is exactly the
"own the kernels" line fox deliberately doesn't cross (see the vLLM gap analysis).
Additionally, good acceptance needs models trained/calibrated for early exit; stock
GGUF models aren't. **Trigger to reconsider:** upstream ever exposing an early-exit
context flag — unlikely, since MTP (level 3) solves the same problem better for models
trained for it.

## Level 5 — EAGLE / Medusa (trained heads + tree verification) — **NOT VIABLE today; ecosystem-gated**

**Proposer:** small trained heads predict several futures as a **tree** of candidates;
the target verifies the whole tree in one pass with a tree-attention mask; the best
path wins. State of the art: 2.5–4× on general text (EAGLE-2/3).

**Why not, in two independent parts:**
1. **The weights don't exist in fox's ecosystem.** EAGLE/Medusa heads are per-model
   trained artifacts distributed as HF/PyTorch checkpoints for specific models; the
   GGUF ecosystem has no standard for packaging or converting them. Without weights,
   the best tree engine does nothing. This is the hard gate, and it's external.
2. **Tree verification is at the edge of the core API.** A tree *can* be emulated with
   the batch API (a token may belong to multiple `seq_id`s; branches = sequences,
   `seq_rm` the losers — same primitives as 0.15), but real tree-attention masks within
   one sequence aren't exposed, and the seq-per-branch emulation multiplies KV traffic,
   eating the speedup it's meant to buy.

**Trigger to reconsider:** GGUF standardizes speculative-head tensors (the way it
absorbed LoRA and control vectors) *and* llama.cpp grows tree-mask support. Both are
upstream events fox can watch for free. Notably, if MTP-in-GGUF (level 3) becomes
common first, EAGLE's advantage narrows — they compete for the same niche.

---

## Summary — what to do and when

| Level | Technique | General-text win | Extra memory | Blocked on | Verdict |
|---|---|---|---|---|---|
| 1 | n-gram (0.15) | ~1× (1.8× repetitive) | none | — | ✅ shipped; cheap tuning headroom left |
| 2 | **draft model** | 1.5–2.5× | small model (0.5–1B) | nothing — core-only | **next real step (0.16+ candidate)** |
| 3 | MTP / NextN | ~2× (for MTP models) | none | small MTP models in GGUF; API maturity | watch item — may leapfrog level 2 |
| 4 | layer-skip | 1.3–1.8× | none | no core API; per-model calibration | skip — superseded by MTP |
| 5 | EAGLE/Medusa | 2.5–4× | trained heads | GGUF weight standard + tree masks | ecosystem-gated; revisit on upstream events |

The through-line: **fox climbs when the ingredient is in llama.cpp's core or in the
GGUF file itself** (levels 1, 2, 3), and stays put when the ingredient is a custom
kernel or an artifact the ecosystem doesn't distribute (levels 4, 5). Same rule that
decided GBNF (core → shipped) vs. `json_schema_to_grammar` (common/ → rewrote in Rust)
vs. EAGLE (nowhere → wait).

All levels reuse the 0.15 verifier, engine wiring, metrics and `bench-spec` unchanged —
each new level must pass the same two gates: **byte-identical output** (the exactness
golden) and **a measured win on `bench-spec`**.
