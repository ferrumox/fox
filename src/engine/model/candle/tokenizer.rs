//! GGUF-backed byte-level BPE tokenizer.
//!
//! Loads the vocabulary, merges and special-token IDs from a GGUF metadata
//! block (see [`gguf_metadata`]) and exposes encode / decode that mirror the
//! conventions used by Llama 3, Gemma and Qwen — namely, a GPT-2-style
//! `bytes_to_unicode` mapping followed by greedy BPE merges.
//!
//! Pre-tokenisation is intentionally *not* the full GPT-2 regex split. Each
//! input is mapped byte-to-codepoint as one stream and merges are applied
//! over the entire stream. That is correct for round-trip fidelity (decode
//! ∘ encode = identity for any UTF-8 input) and matches llama.cpp output for
//! single-line prompts. Multi-line prompts may differ from the reference
//! tokenisation; tightening to the full regex is left to a later pass.
//!
//! Other tokenizer families (SentencePiece-Unigram for the original Llama 1/2,
//! WordPiece for BERT) are not implemented here — `Vocab::model_kind` lets
//! callers detect them and refuse to instantiate the BPE tokenizer.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use super::gguf_metadata::{self, GgufMetadata, MetadataError};

/// Subset of `tokenizer.ggml.*` metadata that the runtime cares about for
/// generation control (start of sequence, end of sequence, end of turn, …).
#[derive(Debug, Default, Clone)]
pub struct SpecialTokens {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
    pub pad: Option<u32>,
    /// "End of turn" — distinct from sequence-end EOS in chat-tuned models
    /// (Llama 3 uses `<|eot_id|>` here while keeping `<|end_of_text|>` as EOS).
    pub eot: Option<u32>,
}

/// Vocabulary materialised from a GGUF file.
#[derive(Debug, Clone)]
pub struct Vocab {
    pub tokens: Vec<String>,
    /// Per-token type tag from `tokenizer.ggml.token_type`. 1=normal,
    /// 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte. Empty when
    /// the GGUF does not provide types — callers should treat all tokens as
    /// normal in that case.
    pub token_types: Vec<i32>,
    /// BPE merge rules in priority order. Each entry has the form
    /// `"left right"` — the same string the GGUF stores.
    pub merges: Vec<String>,
    pub specials: SpecialTokens,
    /// Value of `tokenizer.ggml.model` (e.g. `"llama"`, `"gpt2"`). Empty when
    /// absent.
    pub model_kind: String,
    /// Raw chat template string, when present (`tokenizer.chat_template`).
    pub chat_template: Option<String>,
}

impl Vocab {
    pub fn size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn id_of(&self, token: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == token)
            .map(|i| i as u32)
    }
}

#[derive(Debug)]
pub enum TokenizerError {
    Metadata(MetadataError),
    MissingTokens,
    InconsistentTypes { tokens: usize, types: usize },
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(e) => write!(f, "GGUF metadata error: {e}"),
            Self::MissingTokens => {
                write!(f, "GGUF is missing 'tokenizer.ggml.tokens' — cannot build a vocabulary")
            }
            Self::InconsistentTypes { tokens, types } => write!(
                f,
                "tokenizer.ggml.token_type has {types} entries but vocab has {tokens}"
            ),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Metadata(e) => Some(e),
            _ => None,
        }
    }
}

pub fn load_vocab(path: &Path) -> Result<Vocab, TokenizerError> {
    let meta = gguf_metadata::load(path).map_err(TokenizerError::Metadata)?;
    vocab_from_metadata(&meta)
}

pub fn vocab_from_metadata(meta: &GgufMetadata) -> Result<Vocab, TokenizerError> {
    let tokens = meta
        .string_arrays
        .get("tokenizer.ggml.tokens")
        .cloned()
        .ok_or(TokenizerError::MissingTokens)?;

    let token_types: Vec<i32> = meta
        .int_arrays
        .get("tokenizer.ggml.token_type")
        .map(|a| a.iter().map(|&v| v as i32).collect())
        .unwrap_or_default();
    if !token_types.is_empty() && token_types.len() != tokens.len() {
        return Err(TokenizerError::InconsistentTypes {
            tokens: tokens.len(),
            types: token_types.len(),
        });
    }

    let merges = meta
        .string_arrays
        .get("tokenizer.ggml.merges")
        .cloned()
        .unwrap_or_default();

    let specials = SpecialTokens {
        bos: meta
            .uints
            .get("tokenizer.ggml.bos_token_id")
            .map(|&v| v as u32),
        eos: meta
            .uints
            .get("tokenizer.ggml.eos_token_id")
            .map(|&v| v as u32),
        unk: meta
            .uints
            .get("tokenizer.ggml.unknown_token_id")
            .map(|&v| v as u32),
        pad: meta
            .uints
            .get("tokenizer.ggml.padding_token_id")
            .map(|&v| v as u32),
        eot: meta
            .uints
            .get("tokenizer.ggml.eot_token_id")
            .map(|&v| v as u32),
    };

    let model_kind = meta
        .string("tokenizer.ggml.model")
        .unwrap_or("")
        .to_string();
    let chat_template = meta.string("tokenizer.chat_template").map(String::from);

    Ok(Vocab {
        tokens,
        token_types,
        merges,
        specials,
        model_kind,
        chat_template,
    })
}

// ---------------------------------------------------------------------------
// Byte-level BPE
// ---------------------------------------------------------------------------

/// Byte-level BPE encoder/decoder.
pub struct ByteBpeTokenizer {
    vocab: Vocab,
    token_to_id: HashMap<String, u32>,
    merge_ranks: HashMap<(String, String), u32>,
    byte_to_unicode: [char; 256],
    unicode_to_byte: HashMap<char, u8>,
}

impl ByteBpeTokenizer {
    pub fn new(vocab: Vocab) -> Self {
        let (b2u, u2b) = build_byte_unicode_maps();
        let mut token_to_id = HashMap::with_capacity(vocab.tokens.len());
        for (i, t) in vocab.tokens.iter().enumerate() {
            token_to_id.insert(t.clone(), i as u32);
        }
        let mut merge_ranks = HashMap::with_capacity(vocab.merges.len());
        for (rank, merge_str) in vocab.merges.iter().enumerate() {
            // Each merge is `"<left> <right>"` separated by a single ASCII
            // space. `split_once` is exact and avoids surprises if the right
            // side contains spaces (it never does in practice).
            if let Some((left, right)) = merge_str.split_once(' ') {
                merge_ranks.insert((left.to_string(), right.to_string()), rank as u32);
            }
        }
        Self {
            vocab,
            token_to_id,
            merge_ranks,
            byte_to_unicode: b2u,
            unicode_to_byte: u2b,
        }
    }

    pub fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Encode `text` to a sequence of vocabulary IDs.
    ///
    /// Algorithm: bytes → byte-encoded codepoints → greedy BPE merges driven
    /// by the merge-rank table → vocabulary lookup. Tokens that fail to
    /// resolve fall back to `specials.unk`; if no UNK is defined they are
    /// silently dropped (matching llama.cpp behaviour for byte-fallback
    /// vocabs that should never have unknowns in the first place).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut pieces: Vec<String> = text
            .as_bytes()
            .iter()
            .map(|&b| self.byte_to_unicode[b as usize].to_string())
            .collect();

        // Greedy merge: in each iteration, find the adjacent pair with the
        // lowest rank and combine it. Stops when no adjacent pair has a
        // known merge.
        loop {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..pieces.len().saturating_sub(1) {
                let pair = (pieces[i].clone(), pieces[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    match best {
                        None => best = Some((i, rank)),
                        Some((_, br)) if rank < br => best = Some((i, rank)),
                        _ => {}
                    }
                }
            }
            match best {
                Some((idx, _)) => {
                    let merged = format!("{}{}", pieces[idx], pieces[idx + 1]);
                    pieces.splice(idx..=idx + 1, std::iter::once(merged));
                }
                None => break,
            }
        }

        let mut ids = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match self.token_to_id.get(&piece) {
                Some(&id) => ids.push(id),
                None => {
                    if let Some(unk) = self.vocab.specials.unk {
                        ids.push(unk);
                    }
                }
            }
        }
        ids
    }

    /// Decode a sequence of vocabulary IDs back to a UTF-8 string.
    ///
    /// Codepoints in the BPE alphabet (the printable subset chosen by
    /// `bytes_to_unicode`) round-trip back to their original bytes. Special
    /// tokens whose textual form contains characters outside that alphabet
    /// (e.g. `<|begin_of_text|>`) pass through unchanged as UTF-8.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len() * 2);
        for &id in tokens {
            let Some(token) = self.vocab.tokens.get(id as usize) else {
                continue;
            };
            for ch in token.chars() {
                if let Some(&b) = self.unicode_to_byte.get(&ch) {
                    bytes.push(b);
                } else {
                    let mut buf = [0u8; 4];
                    let s = ch.encode_utf8(&mut buf);
                    bytes.extend_from_slice(s.as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// GPT-2's `bytes_to_unicode`: a reversible mapping that turns every byte
/// 0..=255 into a printable Unicode scalar. Bytes that already correspond to
/// printable ASCII / Latin-1 codepoints map to themselves; the rest are
/// shifted into the 256.. range to avoid clashing with any byte that maps to
/// itself.
fn build_byte_unicode_maps() -> ([char; 256], HashMap<char, u8>) {
    let mut printable = [false; 256];
    for b in b'!'..=b'~' {
        printable[b as usize] = true;
    }
    for b in 0xA1u8..=0xACu8 {
        printable[b as usize] = true;
    }
    // Inclusive range to 0xFF; iterate as u32 to avoid u8 overflow at the upper bound.
    for b in 0xAEu32..=0xFFu32 {
        printable[b as usize] = true;
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    let mut next_extended: u32 = 256;

    for b in 0u32..=255u32 {
        let ch = if printable[b as usize] {
            char::from_u32(b).expect("ASCII / Latin-1 codepoints are valid")
        } else {
            let c = char::from_u32(next_extended).expect("extended range never crosses surrogates");
            next_extended += 1;
            c
        };
        encoder[b as usize] = ch;
        decoder.insert(ch, b as u8);
    }

    (encoder, decoder)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small synthetic vocab + merges that we control end-to-end.
    /// The vocab covers all 256 byte-encoded codepoints (so any UTF-8 byte
    /// input can round-trip) plus three merged tokens for the word "hi".
    fn synthetic_vocab() -> Vocab {
        let (b2u, _) = build_byte_unicode_maps();
        let mut tokens: Vec<String> = (0..256).map(|i| b2u[i].to_string()).collect();
        // Merges: "h" + "i" → "hi"; " " + "h" → " h"; " h" + "i" → " hi"
        let space_h = format!("{}h", b2u[b' ' as usize]);
        let space_hi = format!("{}hi", b2u[b' ' as usize]);
        tokens.push("hi".to_string());
        tokens.push(space_h.clone());
        tokens.push(space_hi.clone());

        let merges = vec![
            "h i".to_string(),
            format!("{} h", b2u[b' ' as usize]),
            format!("{} hi", b2u[b' ' as usize]),
        ];

        Vocab {
            tokens,
            token_types: vec![],
            merges,
            specials: SpecialTokens {
                bos: None,
                eos: None,
                unk: None,
                pad: None,
                eot: None,
            },
            model_kind: "gpt2".to_string(),
            chat_template: None,
        }
    }

    #[test]
    fn byte_unicode_maps_are_bijective() {
        let (enc, dec) = build_byte_unicode_maps();
        for b in 0u8..=255 {
            let c = enc[b as usize];
            let back = dec.get(&c).copied().expect("char must map back to a byte");
            assert_eq!(back, b, "byte 0x{:02x} did not round-trip", b);
        }
        // No two bytes share the same encoded char.
        let mut seen = std::collections::HashSet::new();
        for &c in enc.iter() {
            assert!(seen.insert(c), "duplicate codepoint in byte_to_unicode");
        }
    }

    #[test]
    fn round_trip_ascii_text() {
        let tok = ByteBpeTokenizer::new(synthetic_vocab());
        let cases = ["", "h", "hi", "hi there", "Hello, world!", "hi hi hi"];
        for case in cases {
            let ids = tok.encode(case);
            let back = tok.decode(&ids);
            assert_eq!(back, case, "round-trip failed for '{case}'");
        }
    }

    #[test]
    fn round_trip_utf8_bytes() {
        let tok = ByteBpeTokenizer::new(synthetic_vocab());
        // Greek + emoji via raw UTF-8 bytes.
        for case in ["α", "λ ω", "🦊 fox"] {
            let ids = tok.encode(case);
            let back = tok.decode(&ids);
            assert_eq!(back, case, "round-trip failed for '{case}'");
        }
    }

    #[test]
    fn applies_merges_greedily_to_lowest_rank() {
        let tok = ByteBpeTokenizer::new(synthetic_vocab());
        let ids = tok.encode("hi");
        // With "h"+"i" → "hi" merged, encode("hi") should be exactly one token.
        assert_eq!(ids.len(), 1);
        assert_eq!(tok.decode(&ids), "hi");
    }

    #[test]
    fn merges_cross_byte_alphabet_boundaries() {
        let tok = ByteBpeTokenizer::new(synthetic_vocab());
        // " hi" should be one token (" h" then " hi" merges chain).
        let ids = tok.encode(" hi");
        assert_eq!(ids.len(), 1, "expected ' hi' to merge into one token");
        assert_eq!(tok.decode(&ids), " hi");
    }

    #[test]
    fn vocab_lookup_helpers() {
        let v = synthetic_vocab();
        let hi_id = v.id_of("hi").unwrap();
        assert_eq!(v.token(hi_id), Some("hi"));
    }
}
