//! End-to-end smoke test for the GGUF probe + arch table pipeline.
//!
//! Builds a synthetic GGUF file on disk (header + metadata only — no tensor
//! data) for a handful of representative architectures and verifies that the
//! probe reads them correctly and that the arch table picks a sensible
//! backend.

use std::io::Write;
use std::path::PathBuf;

use ferrumox::model_registry::{
    probe, recommend_backend, render_diagnostic, BackendHint, DiagnosticLevel, Modality,
    ModelDiagnostic,
};
use tempfile::TempDir;

const GGUF_MAGIC: u32 = 0x4655_4747;

#[derive(Default)]
struct GgufFixture {
    payload: Vec<u8>,
    kv_count: u64,
}

impl GgufFixture {
    fn new() -> Self {
        Self::default()
    }

    fn string(mut self, key: &str, value: &str) -> Self {
        self.write_string(key);
        self.payload.extend_from_slice(&8u32.to_le_bytes());
        self.write_string(value);
        self.kv_count += 1;
        self
    }

    fn u32(mut self, key: &str, value: u32) -> Self {
        self.write_string(key);
        self.payload.extend_from_slice(&4u32.to_le_bytes());
        self.payload.extend_from_slice(&value.to_le_bytes());
        self.kv_count += 1;
        self
    }

    fn string_array(mut self, key: &str, n: usize) -> Self {
        self.write_string(key);
        self.payload.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        self.payload.extend_from_slice(&8u32.to_le_bytes()); // STRING elements
        self.payload.extend_from_slice(&(n as u64).to_le_bytes());
        for i in 0..n {
            let token = format!("t{i}");
            self.write_string(&token);
        }
        self.kv_count += 1;
        self
    }

    fn write_string(&mut self, s: &str) {
        self.payload
            .extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.payload.extend_from_slice(s.as_bytes());
    }

    fn write_to(self, dir: &TempDir, file: &str) -> PathBuf {
        let path = dir.path().join(file);
        let mut bytes = Vec::with_capacity(self.payload.len() + 24);
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&self.kv_count.to_le_bytes());
        bytes.extend_from_slice(&self.payload);
        let mut f = std::fs::File::create(&path).expect("create fixture");
        f.write_all(&bytes).expect("write fixture");
        path
    }
}

#[test]
fn llama_fixture_recommends_llama_cpp_backend() {
    let dir = tempfile::tempdir().unwrap();
    let path = GgufFixture::new()
        .string("general.architecture", "llama")
        .u32("llama.embedding_length", 4096)
        .u32("llama.attention.head_count", 32)
        .u32("llama.attention.head_count_kv", 8)
        .u32("llama.context_length", 8192)
        .u32("llama.feed_forward_length", 14336)
        .string_array("tokenizer.ggml.tokens", 32000)
        .write_to(&dir, "llama.gguf");

    let profile = probe(&path).expect("probe should succeed");
    assert_eq!(profile.architecture, "llama");
    assert_eq!(profile.head_dim, 128);
    assert_eq!(profile.vocab_size, 32000);

    let (hint, notes) = recommend_backend(&profile);
    assert_eq!(hint, BackendHint::LlamaCpp);
    // Grouped-query attention info note expected.
    assert!(notes.iter().any(|n| n.message.contains("grouped-query")));
}

#[test]
fn gemma4_fixture_recommends_candle_backend() {
    let dir = tempfile::tempdir().unwrap();
    let path = GgufFixture::new()
        .string("general.architecture", "gemma4")
        .u32("gemma4.embedding_length", 3584)
        .u32("gemma4.attention.head_count", 7)
        .u32("gemma4.attention.head_count_kv", 7)
        .u32("gemma4.attention.key_length", 512)
        .u32("gemma4.context_length", 131072)
        .u32("gemma4.feed_forward_length", 14336)
        .string_array("tokenizer.ggml.tokens", 256000)
        .write_to(&dir, "gemma4.gguf");

    let profile = probe(&path).expect("probe should succeed");
    assert_eq!(profile.head_dim, 512);
    assert!(profile.quirks.nonstandard_head_dim);

    let (hint, _notes) = recommend_backend(&profile);
    assert_eq!(hint, BackendHint::Candle);
}

#[test]
fn vision_fixture_overrides_arch_preference() {
    let dir = tempfile::tempdir().unwrap();
    let path = GgufFixture::new()
        .string("general.architecture", "llama")
        .u32("llama.embedding_length", 4096)
        .u32("llama.attention.head_count", 32)
        .u32("clip.vision.embedding_length", 1024)
        .string_array("tokenizer.ggml.tokens", 32000)
        .write_to(&dir, "vision.gguf");

    let profile = probe(&path).expect("probe should succeed");
    assert_eq!(profile.quirks.multimodal, Some(Modality::Vision));

    let (hint, notes) = recommend_backend(&profile);
    assert_eq!(hint, BackendHint::Candle);
    assert!(notes
        .iter()
        .any(|n| n.message.contains("vision") && n.level == DiagnosticLevel::Warn));
}

#[test]
fn diagnostic_render_includes_recommendation_section() {
    let dir = tempfile::tempdir().unwrap();
    let path = GgufFixture::new()
        .string("general.architecture", "llama")
        .u32("llama.embedding_length", 4096)
        .u32("llama.attention.head_count", 32)
        .string_array("tokenizer.ggml.tokens", 32000)
        .write_to(&dir, "render.gguf");

    let profile = probe(&path).unwrap();
    let (hint, notes) = recommend_backend(&profile);
    let rendered = render_diagnostic(&ModelDiagnostic {
        profile,
        hint,
        notes,
    });
    assert!(rendered.contains("Recommendation"));
    assert!(rendered.contains("llama-cpp"));
}

#[test]
fn non_gguf_file_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("garbage.gguf");
    std::fs::write(&path, b"not a gguf at all").unwrap();
    assert!(probe(&path).is_err());
}
