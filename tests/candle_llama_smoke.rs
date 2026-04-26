//! End-to-end smoke test for the candle Llama wrapper.
//!
//! Loads a real GGUF model, runs a single forward pass, and checks that the
//! greedy-argmax token is well-formed. The test is `#[ignore]` so the
//! standard suite does not depend on having Llama-3.2-3B downloaded.
//!
//! Run with:
//!   cargo test --test candle_llama_smoke --features backend-candle -- --ignored --nocapture

#![cfg(feature = "backend-candle")]

use std::path::PathBuf;

use candle_core::Device;

use ferrumox::candle::llama_arch::LlamaArch;
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
#[ignore = "loads ~2GB Llama-3.2-3B; run explicitly with --ignored"]
fn forwards_a_single_prompt_and_picks_a_valid_token() {
    let path = model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf");

    // Tokeniser is independent from the model — needed to pick an input.
    let vocab = load_vocab(&path).expect("vocab loads");
    let vocab_size = vocab.size();
    let tok = ByteBpeTokenizer::new(vocab);

    // Forward on CPU so the test is deterministic and does not require CUDA
    // to be online for CI.
    let arch = LlamaArch::from_gguf(&path, Device::Cpu).expect("model loads on CPU");

    let prompt_ids = tok.encode("Hello");
    assert!(!prompt_ids.is_empty(), "encoder produced zero tokens");
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&t| t as i32).collect();
    eprintln!("prompt tokens ({}): {:?}", prompt_i32.len(), prompt_i32);

    let logits = arch.forward(&prompt_i32, 0).expect("forward runs");
    assert_eq!(
        logits.len(),
        vocab_size,
        "logits row should match vocab size"
    );

    // Pick greedy. The argmax is deterministic on CPU; we don't pin it to a
    // specific token because the choice depends on the model revision, but
    // we do require it to be a valid id.
    let (best_id, best_score) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bs), (i, &s)| {
            if s > bs {
                (i, s)
            } else {
                (bi, bs)
            }
        });
    assert!(best_id < vocab_size, "argmax index in range");
    assert!(best_score.is_finite(), "logit score is finite");

    eprintln!(
        "argmax: id={best_id} score={best_score:.4} text={:?}",
        tok.decode(&[best_id as u32])
    );
}
