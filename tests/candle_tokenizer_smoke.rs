//! Validate the candle tokenizer + chat template stack against a real GGUF
//! model on disk. The tests are `#[ignore]` so the suite does not depend on
//! the user having a particular model downloaded; run with:
//!
//!   cargo test --test candle_tokenizer_smoke --features backend-candle -- --ignored
//!
//! The model path is resolved relative to `~/.cache/ferrumox/models/`. If the
//! file is missing the test panics with a descriptive message rather than
//! silently passing.

#![cfg(feature = "backend-candle")]

use std::path::PathBuf;

use ferrumox::candle::chat_template::apply_chat_template;
use ferrumox::candle::tokenizer::{load_vocab, ByteBpeTokenizer};

fn model_path(filename: &str) -> PathBuf {
    let home = std::env::var("HOME").expect("HOME must be set");
    let p = PathBuf::from(home).join(".cache/ferrumox/models").join(filename);
    assert!(
        p.exists(),
        "Test fixture missing: {}. Download the model first or skip with --ignored.",
        p.display()
    );
    p
}

#[test]
#[ignore = "requires Llama-3.2-3B-Instruct GGUF on disk; run with --ignored"]
fn loads_llama_3_2_vocab_with_expected_size() {
    let vocab = load_vocab(&model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
        .expect("vocab should load");
    assert_eq!(vocab.size(), 128_256, "Llama 3.2 vocab is 128_256 tokens");
    assert!(!vocab.merges.is_empty(), "Llama 3 ships BPE merges");
    assert!(
        vocab.specials.bos.is_some(),
        "Llama 3.2 declares a BOS token"
    );
    assert!(
        vocab.specials.eos.is_some(),
        "Llama 3.2 declares an EOS token"
    );
    assert_eq!(vocab.model_kind, "gpt2", "Llama 3 uses GPT-2 BPE in GGUF");
    assert!(
        vocab.chat_template.is_some(),
        "Llama 3.2 ships a chat template"
    );
}

#[test]
#[ignore = "requires Llama-3.2-3B-Instruct GGUF on disk"]
fn round_trips_plain_english_through_byte_bpe() {
    let vocab = load_vocab(&model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
        .expect("vocab should load");
    let tok = ByteBpeTokenizer::new(vocab);
    for case in [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "function add(a, b) { return a + b; }",
    ] {
        let ids = tok.encode(case);
        assert!(
            !ids.is_empty(),
            "encoder produced zero tokens for '{case}'"
        );
        let back = tok.decode(&ids);
        assert_eq!(back, case, "round-trip diverged for '{case}'");
    }
}

#[test]
#[ignore = "requires Llama-3.2-3B-Instruct GGUF on disk"]
fn round_trips_unicode_through_byte_bpe() {
    let vocab = load_vocab(&model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
        .expect("vocab should load");
    let tok = ByteBpeTokenizer::new(vocab);
    for case in ["¡Hola, mundo!", "λx.x²", "🦊 fox"] {
        let ids = tok.encode(case);
        let back = tok.decode(&ids);
        assert_eq!(back, case, "Unicode round-trip diverged for '{case}'");
    }
}

#[test]
#[ignore = "requires Gemma-4-E2B GGUF on disk"]
fn loads_gemma_4_vocab_and_round_trips() {
    let vocab = load_vocab(&model_path("gemma-4-E2B-it-Q4_K_M.gguf"))
        .expect("vocab should load");
    // Gemma 3/4 use a 256k SentencePiece-derived vocab exposed as BPE in GGUF.
    assert_eq!(vocab.size(), 262_144, "Gemma 4 vocab is 262_144 tokens");
    assert!(
        vocab.specials.eos.is_some(),
        "Gemma 4 declares an EOS token"
    );
    assert!(
        vocab.chat_template.is_some(),
        "Gemma 4 ships a chat template"
    );

    let tok = ByteBpeTokenizer::new(vocab);
    for case in ["Hello, world!", "λx.x²"] {
        let ids = tok.encode(case);
        let back = tok.decode(&ids);
        assert_eq!(back, case, "Gemma round-trip diverged for '{case}'");
    }
}

#[test]
#[ignore = "requires Qwen3.5-2B GGUF on disk"]
fn loads_qwen_3_5_vocab_and_round_trips() {
    let vocab =
        load_vocab(&model_path("Qwen3.5-2B-Q4_K_M.gguf")).expect("vocab should load");
    assert!(vocab.size() > 100_000, "Qwen 3.x vocab is ≥100k tokens");
    assert!(
        vocab.specials.eos.is_some(),
        "Qwen 3.5 declares an EOS token"
    );

    let tok = ByteBpeTokenizer::new(vocab);
    for case in ["Hello, world!", "你好世界"] {
        let ids = tok.encode(case);
        let back = tok.decode(&ids);
        assert_eq!(back, case, "Qwen round-trip diverged for '{case}'");
    }
}

#[test]
#[ignore = "requires Llama-3.2-3B-Instruct GGUF on disk"]
fn renders_chat_template_with_assistant_marker() {
    let vocab = load_vocab(&model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
        .expect("vocab should load");
    let messages = vec![
        ("system".to_string(), "You are helpful.".to_string()),
        ("user".to_string(), "Say only OK.".to_string()),
    ];
    let prompt = apply_chat_template(&vocab, &messages, true).expect("template renders");

    // Llama 3 chat templates use the explicit role-block markers shown below.
    // We check for a few of them; we don't pin the entire prompt because the
    // template can ship with date stamps or system-prompt prefaces that vary
    // between model revisions.
    for marker in [
        "<|start_header_id|>system<|end_header_id|>",
        "<|start_header_id|>user<|end_header_id|>",
        "<|start_header_id|>assistant<|end_header_id|>",
        "Say only OK.",
    ] {
        assert!(
            prompt.contains(marker),
            "rendered prompt missing marker '{marker}'\n{prompt}"
        );
    }
}
