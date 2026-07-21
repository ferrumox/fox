// Golden tests — the real-model regression net (design doc §5.2, P0).
//
// These assert model-facing invariants against an ACTUAL loaded model, so a
// future architecture change that silently corrupts sizing, embeddings, or the
// tokenizer round-trip fails here instead of in production.
//
// They compile only in a real build (`not(fox_stub)`) and run only when
// FOX_GOLDEN_MODEL points at a GGUF; otherwise each test skips (and passes), so
// the stub CI is unaffected. Run locally, e.g.:
//
//   FOX_GOLDEN_MODEL=~/.cache/ferrumox/models/gemma-4-E2B-it-Q4_K_M.gguf \
//     cargo test golden -- --nocapture
//
// CI: the `golden` job in .github/workflows/ci.yml runs this on a tiny GGUF
// (Qwen2.5-0.5B) on CPU — the one job that builds llama.cpp for real. To extend
// coverage to another architecture class (MoE / MLA / recurrent / embeddings),
// cache another tiny GGUF in that job and pass it through, per the support matrix.

use crate::engine::model::{InferenceRequestForModel, LlamaCppModel, Logits, Model};
use crate::model_registry::kv_type;

/// Load the model named by FOX_GOLDEN_MODEL, or `None` (→ the test skips).
fn golden_model() -> Option<LlamaCppModel> {
    let path = std::env::var("FOX_GOLDEN_MODEL").ok()?;
    if path.trim().is_empty() {
        return None;
    }
    let model = LlamaCppModel::load(
        std::path::Path::new(&path),
        1,          // max_batch_size
        Some(2048), // max_context_len — keep the golden KV small
        24 * 1024 * 1024 * 1024,
        0.9,
        kv_type::F16,
        kv_type::F16,
        0,
        0,
        &[],
        false,
    )
    .expect("FOX_GOLDEN_MODEL failed to load");
    Some(model)
}

macro_rules! golden {
    () => {
        match golden_model() {
            Some(m) => m,
            None => {
                eprintln!("SKIP: set FOX_GOLDEN_MODEL=<path.gguf> to run golden tests");
                return;
            }
        }
    };
}

/// Every dimensional fact must be present and self-consistent, and
/// `embedding_dim()` must equal the metadata `n_embd` (regression guard: it used
/// to be reconstructed as `num_heads * head_dim`, wrong for Gemma/MLA).
#[test]
fn golden_model_info_invariants() {
    let m = golden!();
    let info = m.model_info();

    assert!(info.n_embd > 0, "n_embd must be > 0");
    assert!(info.n_head > 0, "n_head must be > 0");
    assert!(info.n_head_kv > 0, "n_head_kv must be > 0");
    assert!(info.head_dim > 0, "head_dim must be > 0");
    assert!(info.n_layer > 0, "n_layer must be > 0");
    assert!(info.vocab_size > 0, "vocab_size must be > 0");
    assert!(info.n_ctx_train > 0, "n_ctx_train must be > 0");
    assert!(
        info.n_head_kv <= info.n_head,
        "n_head_kv ({}) must not exceed n_head ({})",
        info.n_head_kv,
        info.n_head
    );

    assert_eq!(
        m.embedding_dim(),
        info.n_embd,
        "embedding_dim() must equal metadata n_embd (not num_heads*head_dim)"
    );
}

/// Embeddings must have length `n_embd` and be non-degenerate (regression guard:
/// they used to be an all-zeros vector because the context uses pooling=NONE).
#[test]
fn golden_embeddings_nondegenerate() {
    let m = golden!();
    let info = m.model_info();

    let tokens = m
        .tokenize("The quick brown fox jumps over the lazy dog.")
        .unwrap();
    let emb = m.get_embeddings(&tokens).unwrap();

    assert_eq!(
        emb.len(),
        info.n_embd,
        "embedding length must equal n_embd ({})",
        info.n_embd
    );
    let nonzero = emb.iter().filter(|&&x| x != 0.0).count();
    assert!(
        nonzero > 0,
        "embedding must be non-degenerate (was all-zeros before the pooling fix)"
    );
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        norm.is_finite() && norm > 0.0,
        "embedding L2 norm must be finite and > 0 (got {norm})"
    );
}

/// The chat template must render to a non-empty prompt, and — because the compiled
/// `minijinja::Environment` is cached after the first call — rendering the same
/// messages twice must be byte-for-byte identical. Guards the template-cache change.
#[test]
fn golden_chat_template_renders() {
    let m = golden!();
    let messages = vec![
        (
            "system".to_string(),
            "You are a helpful assistant.".to_string(),
        ),
        ("user".to_string(), "Say hi in one word.".to_string()),
    ];

    let first = m
        .build_prompt_tokens(&messages, false)
        .expect("build_prompt_tokens should succeed");
    assert!(
        !first.is_empty(),
        "rendered chat prompt tokenized to nothing"
    );

    // Second call hits the cached environment — output must be identical.
    let second = m
        .build_prompt_tokens(&messages, false)
        .expect("second build_prompt_tokens should succeed");
    assert_eq!(
        first, second,
        "cached template render must be deterministic across calls"
    );
}

/// Chunked prefill (0.13, S2) must produce the SAME result as single-shot prefill.
/// Prefilling a prompt in small chunks across steps places the exact same tokens at
/// the exact same positions in the KV cache, so the final-position logits must pick
/// the same next token. Run on two separate sequences so they don't interfere.
#[test]
fn golden_chunked_prefill_matches_single_shot() {
    let m = golden!();
    let prompt = m
        .tokenize("The quick brown fox jumps over the lazy dog and keeps on running through the forest at dawn.")
        .unwrap();
    assert!(
        prompt.len() > 8,
        "need a multi-chunk prompt to be meaningful"
    );

    let mk_req = |seq_id: i32, prefill_pos: usize| InferenceRequestForModel {
        id: 1,
        prompt_tokens: prompt.clone(),
        last_token: None,
        generated_tokens: 0,
        max_new_tokens: 1,
        context_len: 0,
        kv_seq_id: seq_id,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: None,
        generated_token_ids: vec![],
        skip_prefix_tokens: 0,
        prefix_seq_id: None,
        prefill_pos,
    };

    // argmax of the final-position logits — robust to tiny fp reduction-order diffs.
    let argmax = |l: &Logits| -> usize {
        l.values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .expect("non-empty logits")
    };

    // Single-shot prefill on sequence 0.
    let single = m.do_prefill(&[1], &[mk_req(0, 0)], 0).unwrap();
    let single_logits = single[0].logits.clone().expect("single-shot completes");

    // Chunked prefill (4 tokens/step) on sequence 1, looping until the cursor reaches
    // the prompt end. Only the final step carries logits.
    let mut pos = 0usize;
    let mut chunked_logits = None;
    let mut steps = 0;
    while pos < prompt.len() {
        let out = m.do_prefill(&[1], &[mk_req(1, pos)], 4).unwrap();
        pos = out[0].prefill_pos;
        if let Some(l) = &out[0].logits {
            chunked_logits = Some(l.clone());
        }
        steps += 1;
        assert!(steps <= prompt.len(), "chunk loop failed to advance");
    }
    let chunked_logits = chunked_logits.expect("chunked prefill eventually completes");

    assert!(steps > 1, "chunk size 4 should take several steps");
    assert_eq!(
        argmax(&single_logits),
        argmax(&chunked_logits),
        "chunked prefill must pick the same next token as single-shot"
    );
}

/// Context rolling (0.13) must let a sequence keep decoding past its `n_ctx`.
/// We prefill a short prompt, then decode token-by-token; when the KV window fills,
/// we roll it (discard the oldest half after a small head, shift the rest down) and
/// keep going. Every `llama_decode` after the roll must still succeed and produce
/// finite logits — proving the shifted KV is a valid, continuable state.
#[test]
fn golden_context_shift_continues_past_n_ctx() {
    let m = golden!();

    // A tiny working window so we hit the limit after a handful of decodes.
    let n_ctx: usize = 48;
    let n_keep: usize = 4;

    let prompt = m.tokenize("The lazy dog").unwrap();
    assert!(prompt.len() < n_ctx, "prompt must fit the tiny window");

    let mk_req = |last: Option<i32>, ctx_len: usize| InferenceRequestForModel {
        id: 1,
        prompt_tokens: prompt.clone(),
        last_token: last,
        generated_tokens: 0,
        max_new_tokens: 1,
        context_len: ctx_len,
        kv_seq_id: 0,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: None,
        generated_token_ids: vec![],
        skip_prefix_tokens: 0,
        prefix_seq_id: None,
        prefill_pos: 0,
    };

    // Prefill the prompt on seq 0.
    let pre = m.do_prefill(&[1], &[mk_req(None, 0)], 0).unwrap();
    let mut next = pre[0].logits.clone().unwrap().sampled_token;
    // Live length of the sequence in the KV cache (== next write position).
    let mut live = prompt.len();
    // Absolute count of tokens generated (drives context_len before subtracting rolls).
    let mut rolled = 0usize;

    // Decode well past n_ctx — without rolling this would fail once live == n_ctx.
    for step in 0..(n_ctx * 3) {
        if live >= n_ctx {
            let n_discard = ((live - n_keep) / 2).max(1);
            m.roll_context(0, n_keep, n_discard)
                .expect("shiftable KV must roll");
            live -= n_discard;
            rolled += n_discard;
        }

        // context_len passed to the model = live length (raw count minus rolled).
        let ctx_len = live + 1; // this token will be written at position `live`
        let out = m.do_decode(&[1], &[mk_req(Some(next), ctx_len)]).unwrap();
        let logits = &out[0].1;
        assert!(
            logits.values.iter().all(|v| v.is_finite()),
            "logits after roll must be finite (step {step}, rolled {rolled})"
        );
        next = logits.sampled_token;
        live += 1;
    }

    assert!(rolled > 0, "the loop must have triggered at least one roll");
}

/// tokenize → detokenize must reconstruct tricky text. Uses the raw-byte piece
/// path so multi-token UTF-8 sequences (emoji, CJK) survive; normalizes the SPM
/// word-boundary marker so the comparison works for SentencePiece models too.
#[test]
fn golden_tokenize_roundtrip() {
    let m = golden!();

    for text in [
        "Hello, world!",
        "café ☕ 日本語 test",
        "emoji 🦊🎉 mixed",
        "numbers 12345 and symbols #@%",
    ] {
        let tokens = m.tokenize(text).unwrap();
        assert!(!tokens.is_empty(), "tokenize({text:?}) produced no tokens");

        let mut bytes = Vec::new();
        for t in &tokens {
            bytes.extend(m.token_to_piece_bytes(*t));
        }
        let out = String::from_utf8_lossy(&bytes).replace(super::SPM_SPACE, " ");

        // BOS/leading-space handling varies by tokenizer, so assert containment
        // of the trimmed original rather than exact equality.
        let want = text.trim();
        assert!(
            out.contains(want),
            "roundtrip lost content for {text:?}: detokenized {out:?}"
        );
    }
}
