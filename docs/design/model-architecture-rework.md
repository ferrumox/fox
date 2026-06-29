# Design — Model-architecture correctness & the "whack-a-mole" problem

Status: **Draft / for discussion**
Scope decision: **maximum** — target parity with every architecture class llama.cpp
supports (dense, GQA, MoE, MLA, recurrent/hybrid, embeddings), not just the models
we use today.
Author: design doc requested 2026-06-29.

This document is the "think it from zero" plan. It does **not** propose code yet — it
defines the problem precisely, the target design, and a migration path where every step
is guarded so we never regress a model family while fixing another.

> Current per-feature status (what works / what doesn't) lives in
> [`STATUS.md`](../../STATUS.md). This doc is the *plan*; STATUS.md is the *inventory*.

---

## 1. Problem statement

Fox currently bakes per-architecture knowledge into **formulas, hardcoded token
literals, and heuristics scattered across the FFI, engine, and API layers.** Each new
model family hits a different scattered assumption, so fixing one model tends to break
another. Concretely, the same fact is derived in more than one place and the derivations
disagree:

| Fact | Place A | Place B | Disagreement |
|------|---------|---------|--------------|
| `head_dim` | `n_embd / n_head` in `load()` (`engine/model/llama_cpp/mod.rs`) — wrong for Gemma (256≠224/288), DeepSeek-MLA | `config.head_dim` in `new_context()` | two derivations, one wrong |
| KV `bytes_per_token` | `load()` hardcodes **f16** (`2 * n_head_kv * head_dim * 2 * n_layer`) | `kv_cache/mod.rs` and `new_context()` use `kv_type_bytes(type_k/type_v)` | with quantized/turbo KV the budget is wrong |
| `embedding_dim` | `num_heads * head_dim` (`batch.rs`, `mod.rs`) | should be `n_embd` | wrong for GQA; a `head_dim` fix makes it worse |

Beyond duplicated numbers, three more sources of fragility:

- **Hardcoded family literals**, repeated in different layers: control patterns
  `<|im_end|>`/`<|endoftext|>`… (`engine/output_filter.rs`), `<think>`/`</think>` (in
  the output filter *and* in the API layer), `SPM_SPACE = U+2581`, the fallback
  chat-template name list (`vocab.rs`), and the tool-call wire format
  `[tool_call: name(args)]` (`api/shared/inference.rs`).
- **Capability detection by heuristic**: "supports thinking?" tokenizes the string
  `"<think>"` and checks for ≤2 tokens. Fragile; false results are inevitable.
- **Divergent defaults across the two APIs**: same model, different output depending on
  endpoint — OpenAI `top_k=0, repeat_penalty=1.0` vs Ollama `top_k=40,
  repeat_penalty=1.1`.

And the reason none of this can be fixed safely today: **there is no per-architecture
regression test.** Every change is a bet.

> Note on "maximum scope": recurrent/hybrid (Mamba, RWKV, Jamba) and MLA (DeepSeek-V2/V3)
> are the families that most violate fox's implicit assumptions, because their KV/state
> memory does **not** follow `n_head_kv * head_dim * n_layer`. Committing to them forces
> the design to stop computing memory itself and instead trust llama.cpp — which is the
> right thing for every family anyway.

---

## 2. Goals / non-goals

**Goals**
1. One source of truth for every model fact and capability, populated once at load.
2. Correct KV/state sizing for **all** architecture classes, including MLA and recurrent.
3. Capabilities (thinking, template, stop/control tokens, seq-copy) derived from the
   model, not from hardcoded literals.
4. A regression net (golden tests + a diagnostic command) that makes any future
   architecture regression visible immediately and blocks it in CI.
5. Consistent, documented behavior across the OpenAI and Ollama surfaces.

**Non-goals (for this rework)**
- Vision/multimodal input (still explicitly unsupported; must fail loudly, not silently).
- Grammar/constrained decoding for JSON mode (remains prompt-only for now; tracked separately).
- New sampling algorithms or performance work. This rework is correctness + safety only.

---

## 3. Scope: the support contract (matrix)

We commit to a published matrix. Each class names what is structurally different and
what the regression net must assert. "Verified" = has golden tests in CI.

| Class | Examples | Structurally special | Must assert |
|-------|----------|----------------------|-------------|
| Dense | Llama, Mistral, Phi | baseline | tokenize↔detok round-trip, greedy output, KV sizing |
| GQA | Llama-3, Qwen2.5 | `n_head_kv < n_head` | `embedding_dim == n_embd`; KV uses `n_head_kv` |
| Non-standard head_dim | Gemma-2/3 | `head_dim` from metadata (256), logit softcap → no flash-attn | head_dim source; coherent output with FA=AUTO |
| MoE | Mixtral, DeepSeek-MoE | expert tensors; `--moe-cpu` offload | loads with/without offload; expert regex matches |
| MLA (latent KV) | DeepSeek-V2/V3 | compressed KV ≠ `n_head_kv*head_dim` | KV budget comes from llama.cpp, not our formula |
| Recurrent/Hybrid | Mamba, RWKV, Jamba | **no positional KV**; `can_shift=false` | prefix cache disabled; no seq_cp; sizing deferred |
| Embeddings | nomic-embed | pooled output, dim = `n_embd` | embedding length == `n_embd`; non-zero vector |

The matrix lives in code (an enum/table that `ModelInfo` resolves into) **and** in
`docs/` so users know what "supported" means.

---

## 4. Design

### 4.1 Single source of truth: `ModelInfo`

A struct built **once**, right after `llama_model_load_from_file`, before any context is
created. It is the only place allowed to derive model facts. Everything downstream
(context params, KV manager, scheduler, sampling, output filter, API) reads from it.

```
struct ModelInfo {
    // identity
    arch: ArchClass,            // Dense | Gqa | Moe | Mla | Recurrent | Embedding (+raw arch string)
    arch_name: String,          // GGUF general.architecture, verbatim

    // dimensions — each field documents its SOURCE
    n_embd: usize,              // llama_model_n_embd  (NOT num_heads*head_dim)
    n_head: usize,              // llama_model_n_head
    n_head_kv: usize,           // llama_model_n_head_kv
    head_dim: usize,            // GGUF <arch>.attention.key_length, fallback n_embd/n_head
    n_layer: usize,             // llama_model_n_layer
    n_ctx_train: u32,           // llama_model_n_ctx_train
    vocab_size: usize,

    // memory model — the key abstraction for "maximum scope"
    kv_model: KvModel,          // memory CLASS only (Positional | Latent | Recurrent | None) — NOT a sizing formula; governs paging/prefix decisions. Actual size comes from llama.cpp (§4.2)

    // capabilities — from model, not literals
    has_kv_cache: bool,         // false for recurrent → disables paging/prefix/seq_cp
    supports_seq_copy: bool,    // llama_memory_can_shift
    chat_template: TemplateSource, // builtin | named-fallback | none
    eog_tokens: Vec<i32>,       // every token where llama_vocab_is_eog
    control_tokens: Vec<i32>,   // every token where llama_vocab_is_control
    stop_strings: Vec<String>,  // text forms of the above (today's stop_tokens())
    reasoning: ReasoningSupport,// detected think markers (token ids), or None
    is_spm: bool,               // SentencePiece vs BPE → governs U+2581 handling
}
```

Principles:
- **Prefer the llama.cpp API; fall back to GGUF metadata; never invent.** Where neither
  is available, record `Unknown` and let callers decide — do not silently guess.
- `head_dim`, `embedding_dim` and friends have **exactly one** definition here, read from
  metadata / the llama.cpp API (never derived by formula). **KV memory sizing is
  deliberately NOT one of these fields — fox does not compute it at all; see §4.2.**

### 4.2 fox does not size the KV cache — it asks llama.cpp

**The single most load-bearing rule of the rework.** Today fox computes
`bytes_per_token = n_head_kv * head_dim * n_layer * …` and uses it to size both llama.cpp's
`n_ctx` and its own paged block pool. **There is no formula that is correct across modern
architectures** — and, crucially, this is *not* limited to exotic models. Even a mainstream
dense model breaks it.

Evidence — `gemma-4-E2B`, read straight from its GGUF metadata (verified 2026-06-29):

```
gemma4.attention.key_length       = 512   (global-attention layers)
gemma4.attention.key_length_swa   = 256   (sliding-window layers — a DIFFERENT head_dim!)
gemma4.attention.shared_kv_layers = 20    (KV shared across 20 of 35 layers)
gemma4.attention.head_count_kv    = 1     (MQA)
gemma4.feed_forward_length        = [array of 35]   (per-layer FFN sizes)
```

There is no single `head_dim` (it is 512 *and* 256, depending on layer type), and the real KV
footprint is far smaller than `n_head_kv · head_dim · n_layer · tokens` because 20 layers
share their KV. Any hand-rolled formula over-counts wildly. The same is true for MLA
(compressed latent KV), recurrent/hybrid (state-space, no per-token KV), and whatever ships
next. Reading the GGUF tells you the formula is wrong — it does **not** hand you a corrected
formula, because combining these keys correctly *is the model's forward-pass logic*.

**Design — fox never computes KV size, for any architecture:**

- llama.cpp owns the KV/state memory. Pick `n_ctx` from the user/trained limit, create the
  context, and **read back the actual capacity llama.cpp allocated** (context introspection /
  `llama_state_get_size` / memory APIs) instead of estimating it from dims.
- fox's paged block pool **follows** llama.cpp's real capacity — it never leads. This
  eliminates the whole "fox thinks there's room, `llama_decode` returns nonzero" class of
  hangs/crashes under load.
- `has_kv_cache = false` (recurrent/hybrid) → disable paging / prefix cache / seq-copy entirely.
- There is **no** `KvModel::PositionalKv`-with-our-own-formula case. The formula is deleted
  for everyone; the per-architecture branching collapses into "ask llama.cpp."

> **The lesson Gemma 4 taught us:** reading the GGUF is *necessary but not sufficient*. fox
> reads the file for **identity and formatting** (arch, template, special tokens) and
> delegates **sizing and the forward pass** to llama.cpp — the one component that already
> parses every key and knows how to combine them. "Derive, don't branch; and delegate the
> math you can't own."

### 4.3 Capabilities from the model, not literals

- **Stop/control/EOG**: already partly done (`stop_tokens()` enumerates control+EOG).
  Promote this into `ModelInfo.control_tokens` / `eog_tokens` and have the output filter
  consume **token ids** (and their detokenized text) from there. Delete the hardcoded
  `CONTROL_TOKEN_PATTERNS` list, or keep it only as a last-ditch fallback for quants with
  broken metadata, clearly labeled.
- **Thinking/reasoning**: replace the "≤2 tokens" heuristic with explicit detection of
  the model's reasoning token ids (look up `<think>`/`</think>` or arch-specific markers
  in the vocab and store the ids). The output filter matches ids, not substrings, so it
  is robust to tokenization. If a model has no such tokens → `ReasoningSupport::None`.
- **Chat template**: **HIGH PRIORITY — see [`STATUS.md`](../../STATUS.md) finding
  (2026-06-29).** fox currently applies templates via llama.cpp's legacy C engine, which
  does *not* execute Jinja: the model's real template (Gemma 4 `enable_thinking`, Qwen3,
  tool-format macros) is discarded for a simplified built-in format, so **thinking and
  native tool-calling are lost**. The rework must adopt a real Jinja engine (llama.cpp
  `minja`/`--jinja`, or `minijinja` in Rust) and thread `enable_thinking`/tools. Record
  which engine/template path was used in `ModelInfo.chat_template` so `fox probe` shows it.
- **SPM vs BPE**: detect once (`is_spm`); only then apply the `U+2581 → space`
  substitution. Avoids corrupting BPE output that legitimately contains that codepoint.

### 4.4 API-layer consistency

- Centralize sampling defaults in one table keyed by API surface, and **document the
  divergence deliberately** (Ollama mirrors upstream Ollama; OpenAI mirrors OpenAI) or
  unify them. Either is fine — but it must be a decision, not an accident, and visible
  in `fox probe`/docs.
- Tool-call and JSON-mode parsing stays prompt-based for now (non-goal to add grammars),
  but the wire format and the parser live next to each other with a shared constant, and
  unknown-tool handling is documented (currently: treated as text).
- Multimodal content must **fail loudly** (or warn in the response) instead of silently
  dropping image blocks.

---

## 5. Regression safety net (built first)

The net is the precondition for touching anything. Three layers:

1. **`fox probe <model>` (a.k.a. `fox doctor`)** — loads the model, prints the full
   `ModelInfo`, and flags internal contradictions, e.g. "head_dim: 256 (from
   metadata); n_embd/n_head would give 224 — mismatch, metadata wins" or "KvModel=Latent
   (MLA): self-sizing disabled". This turns every derived fact into something inspectable
   and every regression into a visible diff.

2. **Golden tests per architecture class** — one tiny GGUF per class (or recorded
   fixtures where a real tiny model doesn't exist), run under `FOX_SKIP_LLAMA` off in a
   GPU-or-CPU CI job:
   - tokenize → detokenize round-trips on tricky text (emoji, CJK, leading spaces);
   - greedy generation with a fixed seed → assert the **exact** first N tokens;
   - assert the `ModelInfo` numbers (head_dim, embedding_dim, bytes/token, n_ctx);
   - embeddings: assert length == `n_embd` and vector is non-degenerate.
   These run in CI and are the contract behind the support matrix.

3. **Load-time invariant assertions** — cheap self-checks that turn silent corruption
   into a clear error: assert fox's paged block pool capacity matches the KV capacity
   **reported by llama.cpp** (never a hand formula); assert `embedding_dim == n_embd`; for
   `has_kv_cache == false` assert paging/prefix are disabled.

CI matrix: at minimum a CPU job that runs the golden suite on the smallest GGUF of each
class. Document how to add a model to the matrix (it should be: drop a fixture + one
table row).

---

## 6. Migration plan (each phase non-breaking, behind the net)

- **P0 — Net first.** Implement `fox probe` and the golden harness against *current*
  behavior. Capture today's outputs as baselines. No behavior change. This is also how we
  retroactively validate the two patches already applied (flash-attn AUTO, head_dim from
  metadata).
- **P1 — Introduce `ModelInfo`** and route the duplicated numbers (head_dim,
  bytes_per_token, embedding_dim) through it. Reabsorb the existing Gemma/flash-attn
  patches here. Golden tests must stay green for dense/GQA/Gemma.
- **P2 — Delete fox's KV sizing; read it from llama.cpp** (§4.2) for *all* architectures —
  not just MLA/recurrent. Gemma 4 (SWA + shared-KV) proves mainstream dense models need this
  too. fox's block pool follows llama.cpp's reported capacity. Add MLA/recurrent golden
  fixtures. This is where "maximum scope" is actually earned.
- **P3 — Capabilities from model.** Move control/think/template/SPM off literals onto
  `ModelInfo`. Golden tests for stop-sequence and thinking behavior per class.
- **P4 — API consistency + footguns.** Decide sampling-default policy; validate `turbo*`
  preconditions at startup; fix `max_models`/`swap_fraction`/silent-multimodal; confirm
  the prefix-cache eviction question under a stress test (left open below).

Each phase is shippable on its own and gated by the net from P0.

---

## 7. Open questions / risks

- **Prefix-cache eviction**: a read of `schedule.rs` suggests `pop`-before-`put` and the
  `len >= max` guard keep blocks/seq-ids balanced (i.e. *no* leak), contradicting an
  initial automated flag. Unresolved by reading alone — **P0's stress test settles it.**
- **Tiny GGUF fixtures**: do suitable miniature models exist for MLA / recurrent, or do
  we record fixtures / synthesize? Affects P2 cost.
- **CI hardware**: golden generation tests ideally want a GPU runner for the real
  backends; a CPU-only matrix still catches sizing/tokenizer/capability regressions.
- **Sampling defaults**: unify across APIs vs match each ecosystem — product decision.
- **llama.cpp introspection**: confirm which exact APIs expose real KV/state size at the
  pinned submodule commit (`bc05a68`); the `KvModel` defer-to-llama.cpp design depends on
  it. Verify before P2.

---

## Appendix A — current-issue → fix mapping (file:line)

| # | Issue | Location | Resolved by |
|---|-------|----------|-------------|
| 1 | `head_dim = n_embd/n_head` wrong (Gemma/MLA) | `engine/model/llama_cpp/mod.rs` (`resolve_head_dim`, already patched to read metadata) | §4.1 `ModelInfo.head_dim` |
| 2 | KV `bytes_per_token` hardcodes f16 in `load()` (and no single formula is correct anyway — Gemma 4 SWA/shared-KV) | `engine/model/llama_cpp/mod.rs` (load path) | §4.2 (delete the formula; read KV size from llama.cpp) |
| 3 | `embedding_dim = num_heads*head_dim` | `engine/model/llama_cpp/batch.rs`, `mod.rs` | §4.1 (`== n_embd`) + P3 invariant |
| 4 | Flash-attn forced ENABLED (Gemma softcap garbage) | `mod.rs` (already patched to AUTO) | §3 Gemma row, validated by golden test |
| 5 | Hardcoded control patterns | `engine/output_filter.rs` | §4.3 control/EOG from model |
| 6 | `<think>` substring + ≤2-token heuristic | `output_filter.rs`, `mod.rs`, `api/shared/inference.rs` | §4.3 reasoning token ids |
| 7 | `U+2581` applied unconditionally | `mod.rs`, `engine/logits.rs` | §4.3 `is_spm` gate |
| 8 | Sampling defaults diverge by API | `api/v1/chat.rs` vs `api/shared/inference.rs` | §4.4 |
| 9 | `turbo*` KV needs FA + head_dim%128, unvalidated | `model_registry/config.rs`, `cli/serve.rs` | P4 startup validation |
| 10 | `max_models=1` default; `swap_fraction` unused; silent multimodal drop | `cli/serve.rs`; `api/types/v1.rs` | P4 |
| 11 | Recurrent/MLA sizing via positional formula | `mod.rs`, `kv_cache/mod.rs` | §4.2 |
| 12 | Prefix-cache eviction cleanup (suspected leak) | `scheduler/schedule.rs`, `scheduler/mod.rs` | P0 stress test (open) |
