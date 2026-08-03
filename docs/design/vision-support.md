# Design — Vision / multimodal support

Status: **Shipped (0.17)**

> Current per-feature status lives in [`STATUS.md`](../../STATUS.md); the
> comparison vs Ollama/vLLM lives in
> [`vllm-gap-analysis.md`](vllm-gap-analysis.md). This doc records *why* image
> support was built the way it was, for whoever extends it next (M-RoPE/tiling
> models, per-request mmproj, multimodal prefix caching).

## Why this shape

fox wraps llama.cpp's `mtmd` library (`vendor/llama.cpp/tools/mtmd/`) rather than
building a vision pipeline from scratch. `mtmd` already ships everything that looked
like the hard part going in:

- **Image decoding**: `mtmd_helper_bitmap_init_from_buf` decodes raw encoded bytes
  (bundled `stb_image` — jpg/png/bmp/gif/…) directly. fox never touches image bytes
  itself beyond base64-decoding the request payload.
- **Position bookkeeping across architectures**: `mtmd_helper_eval_chunks` /
  `mtmd_helper_eval_chunk_single` transparently branch between normal causal
  positions and M-RoPE (Qwen2-VL/Qwen3-VL, HunyuanVL, …) internally — a caller
  never hand-rolls per-architecture position math. This is *why* fox didn't need to
  scope out M-RoPE models: the integration cost is the same either way.

The real cost turned out to be architectural, not FFI: fox's scheduler and KV
manager are built around `prompt_tokens: Vec<i32>` as the single source of both
prompt *content* and *position count* (`schedule.rs`'s block accounting, prefix-cache
hashing, and `do_prefill`'s chunk slicing all read it directly). Image content has no
token ids. The design below scopes deliberately around that coupling instead of
rewriting it.

## Key decision: multimodal prefill is a separate, atomic path

A multimodal request's prompt is tokenized up front (API layer, `prepare_multimodal_prompt`
→ `Model::tokenize_multimodal` → `mtmd_tokenize`), producing a `MultimodalChunks`
handle (`engine/model/mod.rs`) instead of a `Vec<i32>`:

1. **`prompt_tokens` stays empty** for such a request (`InferenceRequest::with_multimodal`).
   Every place that today reads `prompt_tokens.len()` to mean "how many KV positions
   will this consume" was **not** touched to special-case multimodal — instead,
   `InferenceRequest::n_positions()` (`multimodal.map(|m| m.n_positions()).unwrap_or(prompt_tokens.len())`)
   is the one new call `blocks_needed()`/admission uses for sizing. Everything else
   (prefix-cache block hashing, prefix donation on completion) already keys off
   `prompt_tokens` directly, and an empty vec makes both a no-op for free — a
   multimodal request never matches or donates a prefix-cache entry, without a
   separate `skip_prefix_cache` flag.
2. `do_prefill` splits multimodal requests out **before** building the shared token
   batch and handles each via `mtmd_helper_eval_chunks` against the model's
   `llama_context`, holding `_ctx`'s lock for the whole call. This is atomic and
   **not interleaved** with any other request's decode/prefill step during that
   window — mtmd owns the context for as long as its internal `llama_decode` calls
   take. This mirrors llama.cpp's own server, which doesn't interleave multimodal
   encode with other slots either. No fox-level chunking (`max_prefill_chunk` is
   ignored for this path — the whole prompt is one call) and no OOM
   bisection-retry (a failure surfaces as a normal `EngineError`).
3. Once the multimodal turn is resident in KV, decode proceeds through the
   ordinary, completely unmodified token-based continuous-batching/sampling/
   streaming pipeline.

**Alternative considered and rejected**: represent each image as a run of
placeholder token ids inside `prompt_tokens`, so the existing length-based
machinery "just works" unmodified. Rejected because two *different* images would
hash-collide in the prefix cache under the same placeholder id unless the actual
pixel data were folded into the hash — solvable, but strictly more code than the
empty-`prompt_tokens` approach for a v1, and it would have required touching the
prefix-cache hash function itself (`kv_cache/mod.rs`), which the chosen design
leaves completely alone.

## The marker: flatten before the template runs

Image content blocks are spliced into the flattened `(role, content)` string as a
literal marker (`MEDIA_MARKER = "<__fox_media__>"`, `engine/model/mod.rs`) **before**
the chat template ever renders (`MessageContent::as_text_with_media_marker`,
`api/types/v1.rs`) — mirroring llama.cpp server's own approach
(`server-common.cpp`'s `media_marker` content-part rewrite) rather than teaching
every Jinja template about images. `mtmd_tokenize` then splits the *rendered* prompt
string on marker occurrences and matches them against the bitmaps in order. This
means zero changes were needed to `render_chat_jinja` — it already only needs a
plain string per message.

## v1 scope cuts (deliberate, not oversights)

- **Base64 `data:` URIs only** — an OpenAI `image_url` pointing at a remote
  `http(s)://` URL is rejected with `400`, not fetched. fox has no outbound-fetch
  code path today and this avoids adding an SSRF surface for a feature that isn't
  the common case in a local-first server.
- **One global mmproj pairing** (`--mmproj`, mirrors `--draft-model`) — resolved
  once via `ModelRegistry::resolve_model_name` alongside whichever model is
  currently loaded, not a per-registry-entry or per-request mapping. Fine for
  `fox run`/single-vision-model `fox serve`; a multi-vision-model deployment
  would need this generalized.
- **No multimodal prefix caching** — two requests sharing the exact same image +
  prefix currently re-encode it every time. Not attempted here: caching mtmd's
  computed image embeddings keyed by content hash is a real, separable follow-up.
- **No reactive OOM mitigation** on this path (see point 2 above) — consistent
  with the MLA/recurrent KV-sizing gap already tracked in `STATUS.md` as accepted
  correctness debt, not silently swept under this feature.
- **`fox run` has no `--mmproj` flag** — only `fox serve`/`ModelRegistry` wires it
  (the CLI single-shot path loads `LlamaCppModel` directly, bypassing the registry
  entirely; adding the flag there is a small, isolated follow-up).

## Where to look

| Concern | File |
|---|---|
| mmproj loading, `mtmd_context` lifecycle | `engine/model/llama_cpp/mod.rs` (`load()`, `Drop`, `supports_vision`) |
| `mtmd_tokenize` glue | `engine/model/llama_cpp/vocab.rs` (`tokenize_multimodal_impl`) |
| Atomic prefill via `mtmd_helper_eval_chunks` | `engine/model/llama_cpp/batch.rs` (`do_prefill_multimodal`) |
| `MultimodalChunks` owned handle (type-erased for `fox_stub`) | `engine/model/mod.rs` |
| Request-side plumbing, `n_positions()` | `scheduler/batch.rs` |
| API-layer marker splicing, base64/data-URI decode | `api/types/v1.rs` (OpenAI `image_url`), `api/v1/chat.rs`, `api/ollama/chat.rs`, `api/ollama/generate.rs` (Ollama `images`), `api/shared/inference.rs` (`prepare_multimodal_prompt`) |
| Real end-to-end check (cross-request image isolation) | `scripts/e2e_smoke.py` check 14, opt-in via `E2E_MMPROJ=/path/to/mmproj.gguf` |

## Verified against a real model (2026-08-01)

Ran the full e2e suite against `ggml-org/moondream2-20250414-GGUF` (phi2, 1.8B,
2048 trained ctx) + its paired mmproj. Check 14 passes cleanly: both
`/v1/chat/completions` (`image_url`) and `/api/chat` (`images`) return coherent,
correct answers (a red test image → "vibrant red color", a different blue image
right after → "flat blue background", no cross-contamination), and a remote
`image_url` is rejected with `400` as designed.

This run also caught a real, **pre-existing, non-vision bug**: this GGUF's
`tokenizer.chat_template` metadata is the bare string `"vicuna"` (a legacy
template-name hint, not real Jinja source). `render_chat_jinja` trusted it
anyway — minijinja renders a string with no `{{`/`{%` tags as itself, so every
prompt silently collapsed to the one word `"vicuna"`. Fixed by requiring actual
Jinja syntax before committing to that path (see the "GGUF chat template" row in
`STATUS.md`). Exactly the class of bug the project's e2e-over-unit-tests
philosophy exists to catch — every unit/golden test still passed throughout.

Separately, checks 9/10/11 (a concurrency edge case, context-fill rolling, and
post-roll continuation) fail on this specific model **with or without**
`--mmproj` — confirmed by re-running the suite text-only. This model's very
small trained context (2048) doesn't fit those checks' assumptions well; it's
unrelated to vision and not something this session changed or fixed.

**Re-verified the same day against `bartowski/google_gemma-4-E2B-it-GGUF`**
(natively multimodal, 131K trained context) + its mmproj — a mainstream,
recognizable model rather than an edge case: **24/24 e2e checks pass, zero
failures**, including checks 9/10/11 above (confirming those were specific to
moondream2's tiny context, not a fox bug) and check 14 (`Red`/`Blue` — exact,
one-word-correct answers to both `image_url` and `images` inputs, prefix-cache
isolation across different images holds). No fox changes were needed to make
this model work — it's the `gemma4-e2b` registry entry (`registry.json`) and
the recommended vision model for anyone validating this feature.
