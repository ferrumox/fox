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

/// Argmax helper used by every smoke test below.
fn greedy(logits: &[f32]) -> i32 {
    let (idx, _) = logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bs), (i, &s)| {
            if s > bs { (i, s) } else { (bi, bs) }
        });
    idx as i32
}

#[test]
#[ignore = "loads ~2GB Llama-3.2-3B; run explicitly with --ignored"]
fn forward_at_position_zero_resets_the_kv_cache() {
    // Stronger guarantee than `clear_sequence` doc: prove the reset by running
    // the *same* prompt before and after a "conversation switch" and checking
    // both produce identical logits. If the KV cache leaked across calls the
    // second forward would see a contaminated context and the argmax would
    // diverge.
    let path = model_path("Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    let arch = LlamaArch::from_gguf(&path, Device::Cpu).expect("model loads on CPU");

    let prompt = [9906_i32]; // "Hello"

    let logits_first = arch.forward(&prompt, 0).expect("first forward runs");
    let argmax_first = greedy(&logits_first);
    eprintln!("first forward: argmax={argmax_first}");

    // Advance the conversation a few decode steps so the KV cache fills with
    // tokens that are NOT the same as `prompt`. After this, the cache is
    // dirty with whatever the model generated.
    let mut next = argmax_first;
    let mut pos = prompt.len();
    for _ in 0..3 {
        let l = arch.forward(&[next], pos).expect("decode runs");
        next = greedy(&l);
        pos += 1;
        eprintln!("decode step pos={pos}: token={next}");
    }

    // Now reset by forwarding `prompt` again at position 0.
    let logits_after_reset = arch
        .forward(&prompt, 0)
        .expect("reset forward runs");
    let argmax_after_reset = greedy(&logits_after_reset);
    eprintln!("after reset: argmax={argmax_after_reset}");

    assert_eq!(
        argmax_first, argmax_after_reset,
        "Re-running the same prompt at position 0 must yield the same argmax. \
         Divergence ⇒ KV cache leaked across the synthetic 'conversation switch'."
    );

    // Stronger check: top-3 logits should match exactly.
    let top3 = |v: &[f32]| -> Vec<(usize, f32)> {
        let mut indexed: Vec<(usize, f32)> = v.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        indexed.into_iter().take(3).collect()
    };
    let t1 = top3(&logits_first);
    let t2 = top3(&logits_after_reset);
    for ((i1, s1), (i2, s2)) in t1.iter().zip(t2.iter()) {
        assert_eq!(i1, i2, "top-3 token IDs diverged after reset");
        assert!(
            (s1 - s2).abs() < 1e-3,
            "top-3 logit scores diverged after reset: {s1} vs {s2}"
        );
    }
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
