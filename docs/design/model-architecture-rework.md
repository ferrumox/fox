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
    kv_model: KvModel,          // PositionalKv { per_token_bytes_fn } | LatentKv | RecurrentState | None

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
- `head_dim`, `embedding_dim`, and `bytes_per_token` have **exactly one** definition
  here. The current Gemma `head_dim` fix and the f16/quantized KV inconsistency both
  dissolve into this.

### 4.2 The memory model is per-architecture (`KvModel`)

This is the load-bearing change for maximum scope. Today fox computes a single
`bytes_per_token = n_head_kv * head_dim * … ` and uses it to size both llama.cpp's
`n_ctx` and its own paged block pool. That formula is **only valid for positional KV
(dense/GQA/MoE)**. It is wrong for:

- **MLA**: KV is a compressed latent; real per-token cost is far smaller → fox
  over-reserves, under-provisions context, or mismatches llama.cpp's real `n_ctx`.
- **Recurrent**: there is no per-token KV at all; memory is fixed state-space size →
  the formula is meaningless and the paged KV cache / prefix cache do not apply.

Design:
- `KvModel::PositionalKv` keeps the existing (now type-correct) `kv_type_bytes`-based
  formula — the only class where fox may size memory itself.
- `KvModel::LatentKv` and `RecurrentState`: **do not self-size.** Defer to llama.cpp:
  pick `n_ctx` from the trained/user limit and let llama.cpp allocate; fox's block pool
  either tracks llama.cpp's actual capacity or is bypassed. Cross-check against
  `llama_state_get_size` / context introspection rather than a hand formula.
- Where fox and llama.cpp must agree on capacity, make llama.cpp the authority and have
  fox's bookkeeping follow, not lead. (Prevents "fox thinks there's room, llama_decode
  returns nonzero" hangs/crashes under load.)

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
- **Chat template**: keep llama.cpp's builtin template as the primary; the named
  fallback list stays but is data, not control flow. Record which path was used in
  `ModelInfo.chat_template` so `fox probe` can show it.
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
   into a clear error: e.g. for `PositionalKv`, assert fox's computed block capacity is
   ≤ llama.cpp's actual `n_ctx`; assert `embedding_dim == n_embd`; assert
   `head_dim * n_head` relationship only where it should hold.

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
- **P2 — `KvModel` per class.** Add MLA + recurrent handling (defer-to-llama.cpp sizing,
  invariant cross-checks). Add their golden fixtures. This is where "maximum scope" is
  actually earned.
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
| 2 | KV `bytes_per_token` hardcodes f16 in `load()` | `engine/model/llama_cpp/mod.rs` (load path) | §4.2 `KvModel::PositionalKv` |
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
