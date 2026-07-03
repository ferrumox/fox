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

use crate::engine::model::{LlamaCppModel, Model};
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
        ("system".to_string(), "You are a helpful assistant.".to_string()),
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
