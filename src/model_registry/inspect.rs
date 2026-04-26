//! Pre-load GGUF inspection.
//!
//! Reads the GGUF file header and the metadata key-value section without
//! touching tensor data or the GPU. Produces a `ModelProfile` describing the
//! architecture, attention shape and quirks that downstream code uses to pick
//! the right backend and emit actionable diagnostics before attempting to load
//! the model.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

/// Magic bytes at the start of every GGUF file: ASCII "GGUF" read as a
/// little-endian `u32`. The byte sequence on disk is `0x47 0x47 0x55 0x46`,
/// which packs into `0x4655_4747`.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF format versions this probe knows how to walk.
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Common attention head dimensions seen in mainstream transformer models.
/// Anything else is flagged as `nonstandard_head_dim` so that the backend
/// router can prefer an implementation that supports it natively.
const COMMON_HEAD_DIMS: &[u32] = &[64, 80, 96, 128];

/// Multi-modal capability inferred from architecture name or metadata keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Vision,
    Audio,
}

/// Architecture-level traits that influence backend selection.
#[derive(Debug, Clone, Default)]
pub struct ArchQuirks {
    /// `head_dim` is not in `COMMON_HEAD_DIMS` (e.g. Gemma uses 256/512).
    pub nonstandard_head_dim: bool,
    /// Memory backend is recurrent or hybrid (no `seq_cp` support).
    pub hybrid_memory: bool,
    /// Mixture-of-experts model — value is the number of experts.
    pub moe_experts: Option<u32>,
    /// Multi-modal model — vision or audio inputs supported.
    pub multimodal: Option<Modality>,
}

/// Snapshot of a GGUF model derived from its metadata, never from filename.
#[derive(Debug, Clone)]
pub struct ModelProfile {
    pub architecture: String,
    pub embedding_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub head_dim: u32,
    pub context_length: u32,
    pub vocab_size: u32,
    pub ff_length: u32,
    pub gguf_version: u32,
    pub tensor_count: u64,
    pub quirks: ArchQuirks,
}

/// Errors returned by [`probe`]. Wraps both I/O failures and structural issues
/// in the GGUF header so callers can distinguish "this isn't a GGUF file" from
/// "this GGUF file is malformed".
#[derive(Debug)]
pub enum ProbeError {
    Io(std::io::Error),
    NotGguf,
    UnsupportedVersion(u32),
    Truncated,
    InvalidValueType(u32),
    MissingKey(String),
    InvalidUtf8,
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error reading GGUF file: {e}"),
            Self::NotGguf => write!(f, "file does not start with GGUF magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::Truncated => write!(f, "GGUF header is truncated"),
            Self::InvalidValueType(t) => write!(f, "unknown GGUF metadata value type {t}"),
            Self::MissingKey(k) => write!(f, "required metadata key '{k}' is missing"),
            Self::InvalidUtf8 => write!(f, "metadata string is not valid UTF-8"),
        }
    }
}

impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

/// Inspect a GGUF model file and return its profile.
///
/// Only the header and the metadata KV section are read; tensor info and
/// tensor data are skipped via byte arithmetic, so probe cost is bounded by
/// the metadata size (typically <100 KiB).
pub fn probe(path: &Path) -> Result<ModelProfile, ProbeError> {
    let file = File::open(path).map_err(ProbeError::Io)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);

    let header = read_header(&mut reader)?;
    let kv = read_kv_section(&mut reader, header.metadata_kv_count)?;
    profile_from_kv(header, kv)
}

// ---------------------------------------------------------------------------
// Header + KV walking
// ---------------------------------------------------------------------------

struct RawHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

/// All metadata variants the GGUF spec defines. Several variants carry
/// payloads that the current consumers do not read — they are kept for spec
/// completeness so the parser can walk every well-formed file end-to-end.
#[derive(Debug)]
#[allow(dead_code)]
enum MetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    /// Arrays are not materialised. We retain only the element type and length
    /// because the only piece of information we currently need from any GGUF
    /// array is `tokenizer.ggml.tokens.len()` for vocab size.
    Array { element_type: u32, len: u64 },
}

fn read_header<R: Read>(r: &mut R) -> Result<RawHeader, ProbeError> {
    let magic = read_u32(r)?;
    if magic != GGUF_MAGIC {
        return Err(ProbeError::NotGguf);
    }
    let version = read_u32(r)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(ProbeError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(r)?;
    let metadata_kv_count = read_u64(r)?;
    Ok(RawHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

fn read_kv_section<R: Read + Seek>(
    r: &mut R,
    count: u64,
) -> Result<HashMap<String, MetadataValue>, ProbeError> {
    let cap = std::cmp::min(count, 1024) as usize;
    let mut map = HashMap::with_capacity(cap);
    for _ in 0..count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;
        let value = read_value(r, value_type)?;
        map.insert(key, value);
    }
    Ok(map)
}

fn read_value<R: Read + Seek>(r: &mut R, tag: u32) -> Result<MetadataValue, ProbeError> {
    match tag {
        0 => Ok(MetadataValue::U8(read_u8(r)?)),
        1 => Ok(MetadataValue::I8(read_u8(r)? as i8)),
        2 => Ok(MetadataValue::U16(read_u16(r)?)),
        3 => Ok(MetadataValue::I16(read_u16(r)? as i16)),
        4 => Ok(MetadataValue::U32(read_u32(r)?)),
        5 => Ok(MetadataValue::I32(read_u32(r)? as i32)),
        6 => Ok(MetadataValue::F32(f32::from_bits(read_u32(r)?))),
        7 => Ok(MetadataValue::Bool(read_u8(r)? != 0)),
        8 => Ok(MetadataValue::String(read_string(r)?)),
        9 => {
            let element_type = read_u32(r)?;
            let len = read_u64(r)?;
            skip_array_payload(r, element_type, len)?;
            Ok(MetadataValue::Array { element_type, len })
        }
        10 => Ok(MetadataValue::U64(read_u64(r)?)),
        11 => Ok(MetadataValue::I64(read_u64(r)? as i64)),
        12 => Ok(MetadataValue::F64(f64::from_bits(read_u64(r)?))),
        _ => Err(ProbeError::InvalidValueType(tag)),
    }
}

/// Skip the payload of a metadata array without allocating its contents.
///
/// Fixed-width element types are skipped via a single seek; strings are walked
/// one length-prefixed entry at a time. Nested arrays are not supported by the
/// GGUF spec and would surface as `InvalidValueType` here.
fn skip_array_payload<R: Read + Seek>(
    r: &mut R,
    element_type: u32,
    len: u64,
) -> Result<(), ProbeError> {
    let fixed_width = match element_type {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        8 => None,
        _ => return Err(ProbeError::InvalidValueType(element_type)),
    };
    match fixed_width {
        Some(width) => {
            let bytes = width.saturating_mul(len);
            r.seek(SeekFrom::Current(bytes as i64))
                .map_err(ProbeError::Io)?;
        }
        None => {
            for _ in 0..len {
                let s_len = read_u64(r)?;
                r.seek(SeekFrom::Current(s_len as i64))
                    .map_err(ProbeError::Io)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// KV map -> ModelProfile
// ---------------------------------------------------------------------------

fn profile_from_kv(
    header: RawHeader,
    kv: HashMap<String, MetadataValue>,
) -> Result<ModelProfile, ProbeError> {
    let architecture = match kv.get("general.architecture") {
        Some(MetadataValue::String(s)) => s.clone(),
        _ => return Err(ProbeError::MissingKey("general.architecture".to_string())),
    };

    let arch_key = |suffix: &str| format!("{architecture}.{suffix}");

    let embedding_length = require_u32(&kv, &arch_key("embedding_length"))?;
    let head_count = require_u32(&kv, &arch_key("attention.head_count"))?;
    let head_count_kv = read_u32_or(&kv, &arch_key("attention.head_count_kv"), head_count);
    let head_dim = read_u32_or_else(&kv, &arch_key("attention.key_length"), || {
        if head_count > 0 {
            embedding_length / head_count
        } else {
            0
        }
    });
    let context_length = read_u32_or(&kv, &arch_key("context_length"), 0);
    let ff_length = read_u32_or(&kv, &arch_key("feed_forward_length"), 0);
    let moe_experts = read_u32_opt(&kv, &arch_key("expert_count")).filter(|&n| n > 0);

    let vocab_size = match kv.get("tokenizer.ggml.tokens") {
        Some(MetadataValue::Array { len, .. }) => *len as u32,
        _ => read_u32_or(&kv, &arch_key("vocab_size"), 0),
    };

    let multimodal = detect_modality(&architecture, &kv);
    let nonstandard_head_dim = head_dim != 0 && !COMMON_HEAD_DIMS.contains(&head_dim);
    let hybrid_memory = is_hybrid_arch(&architecture);

    Ok(ModelProfile {
        architecture,
        embedding_length,
        head_count,
        head_count_kv,
        head_dim,
        context_length,
        vocab_size,
        ff_length,
        gguf_version: header.version,
        tensor_count: header.tensor_count,
        quirks: ArchQuirks {
            nonstandard_head_dim,
            hybrid_memory,
            moe_experts,
            multimodal,
        },
    })
}

fn require_u32(kv: &HashMap<String, MetadataValue>, key: &str) -> Result<u32, ProbeError> {
    read_u32_opt(kv, key).ok_or_else(|| ProbeError::MissingKey(key.to_string()))
}

fn read_u32_or(kv: &HashMap<String, MetadataValue>, key: &str, default: u32) -> u32 {
    read_u32_opt(kv, key).unwrap_or(default)
}

fn read_u32_or_else<F: FnOnce() -> u32>(
    kv: &HashMap<String, MetadataValue>,
    key: &str,
    default: F,
) -> u32 {
    read_u32_opt(kv, key).unwrap_or_else(default)
}

fn read_u32_opt(kv: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    match kv.get(key)? {
        MetadataValue::U8(v) => Some(*v as u32),
        MetadataValue::U16(v) => Some(*v as u32),
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::U64(v) => Some(*v as u32),
        MetadataValue::I8(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::I16(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::I64(v) if *v >= 0 => Some(*v as u32),
        _ => None,
    }
}

fn detect_modality(architecture: &str, kv: &HashMap<String, MetadataValue>) -> Option<Modality> {
    let lower = architecture.to_lowercase();
    if lower.contains("llava")
        || lower.contains("vision")
        || lower.contains("-vl")
        || lower.ends_with("vl")
    {
        return Some(Modality::Vision);
    }
    if lower.contains("audio") || lower.contains("whisper") {
        return Some(Modality::Audio);
    }
    if kv
        .keys()
        .any(|k| k.starts_with("clip.") || k.starts_with("vision."))
    {
        return Some(Modality::Vision);
    }
    if kv.keys().any(|k| k.starts_with("audio.")) {
        return Some(Modality::Audio);
    }
    None
}

fn is_hybrid_arch(architecture: &str) -> bool {
    let lower = architecture.to_lowercase();
    matches!(lower.as_str(), "rwkv" | "jamba")
        || lower.contains("mamba")
        || lower.contains("hybrid")
}

// ---------------------------------------------------------------------------
// Primitive readers (all little-endian per GGUF spec)
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> Result<u8, ProbeError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, ProbeError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, ProbeError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, ProbeError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, ProbeError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(map_io)?;
    String::from_utf8(buf).map_err(|_| ProbeError::InvalidUtf8)
}

fn map_io(e: std::io::Error) -> ProbeError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        ProbeError::Truncated
    } else {
        ProbeError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Builder for synthetic GGUF byte streams used in tests. Writes a valid
    /// header (version 3) followed by the requested KV entries — no tensor
    /// section is required since `probe` stops after the metadata block.
    struct GgufBuilder {
        bytes: Vec<u8>,
        kv_count: u64,
    }

    impl GgufBuilder {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                kv_count: 0,
            }
        }

        fn push_str(&mut self, key: &str, value: &str) -> &mut Self {
            self.write_string(key);
            self.bytes.extend_from_slice(&8u32.to_le_bytes()); // STRING
            self.write_string(value);
            self.kv_count += 1;
            self
        }

        fn push_u32(&mut self, key: &str, value: u32) -> &mut Self {
            self.write_string(key);
            self.bytes.extend_from_slice(&4u32.to_le_bytes()); // UINT32
            self.bytes.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn push_u64(&mut self, key: &str, value: u64) -> &mut Self {
            self.write_string(key);
            self.bytes.extend_from_slice(&10u32.to_le_bytes()); // UINT64
            self.bytes.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        /// Push an array of strings; only the length is examined by `probe`.
        fn push_string_array(&mut self, key: &str, items: &[&str]) -> &mut Self {
            self.write_string(key);
            self.bytes.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
            self.bytes.extend_from_slice(&8u32.to_le_bytes()); // element type STRING
            self.bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                self.write_string(item);
            }
            self.kv_count += 1;
            self
        }

        fn write_string(&mut self, s: &str) {
            self.bytes
                .extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.bytes.extend_from_slice(s.as_bytes());
        }

        fn finish(self, tensor_count: u64) -> Vec<u8> {
            let mut out = Vec::with_capacity(self.bytes.len() + 24);
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&tensor_count.to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.bytes);
            out
        }
    }

    fn probe_bytes(bytes: Vec<u8>) -> Result<ModelProfile, ProbeError> {
        let mut cursor = Cursor::new(bytes);
        let header = read_header(&mut cursor)?;
        let kv = read_kv_section(&mut cursor, header.metadata_kv_count)?;
        profile_from_kv(header, kv)
    }

    #[test]
    fn magic_constant_matches_ascii_gguf_bytes() {
        // Guard against a constant typo that would silently pass synthetic
        // fixtures (because the builder uses the same constant) yet fail
        // against any real GGUF file on disk.
        let bytes = b"GGUF";
        let actual = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(actual, GGUF_MAGIC);
    }

    #[test]
    fn detects_llama_arch_and_derived_head_dim() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "llama")
            .push_u32("llama.embedding_length", 4096)
            .push_u32("llama.attention.head_count", 32)
            .push_u32("llama.attention.head_count_kv", 8)
            .push_u32("llama.context_length", 8192)
            .push_u32("llama.feed_forward_length", 14336)
            .push_string_array("tokenizer.ggml.tokens", &["a", "b", "c"]);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.architecture, "llama");
        assert_eq!(p.head_count, 32);
        assert_eq!(p.head_count_kv, 8);
        assert_eq!(p.head_dim, 128); // 4096 / 32
        assert_eq!(p.vocab_size, 3);
        assert!(!p.quirks.nonstandard_head_dim);
        assert!(!p.quirks.hybrid_memory);
        assert!(p.quirks.multimodal.is_none());
    }

    #[test]
    fn detects_nonstandard_head_dim_via_key_length() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "gemma3")
            .push_u32("gemma3.embedding_length", 3584)
            .push_u32("gemma3.attention.head_count", 7)
            .push_u32("gemma3.attention.key_length", 256);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.head_dim, 256);
        assert!(p.quirks.nonstandard_head_dim);
    }

    #[test]
    fn detects_moe_expert_count() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "mixtral")
            .push_u32("mixtral.embedding_length", 4096)
            .push_u32("mixtral.attention.head_count", 32)
            .push_u32("mixtral.expert_count", 8);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.quirks.moe_experts, Some(8));
    }

    #[test]
    fn detects_hybrid_memory_for_mamba() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "mamba")
            .push_u32("mamba.embedding_length", 2048)
            .push_u32("mamba.attention.head_count", 16);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert!(p.quirks.hybrid_memory);
    }

    #[test]
    fn detects_vision_modality_from_clip_keys() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "llava")
            .push_u32("llava.embedding_length", 4096)
            .push_u32("llava.attention.head_count", 32)
            .push_u32("clip.vision.embedding_length", 1024);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.quirks.multimodal, Some(Modality::Vision));
    }

    #[test]
    fn rejects_non_gguf_files() {
        let bytes = b"NOT_GGUF_at_all".to_vec();
        let err = probe_bytes(bytes).unwrap_err();
        assert!(matches!(err, ProbeError::NotGguf));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let err = probe_bytes(bytes).unwrap_err();
        assert!(matches!(err, ProbeError::UnsupportedVersion(99)));
    }

    #[test]
    fn missing_architecture_key_is_reported() {
        let b = GgufBuilder::new();
        let err = probe_bytes(b.finish(0)).unwrap_err();
        assert!(matches!(err, ProbeError::MissingKey(k) if k == "general.architecture"));
    }

    #[test]
    fn vocab_size_falls_back_to_explicit_key() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "qwen2")
            .push_u32("qwen2.embedding_length", 4096)
            .push_u32("qwen2.attention.head_count", 32)
            .push_u32("qwen2.vocab_size", 152064);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.vocab_size, 152064);
    }

    #[test]
    fn u64_metadata_is_narrowed_to_u32() {
        let mut b = GgufBuilder::new();
        b.push_str("general.architecture", "llama")
            .push_u32("llama.embedding_length", 4096)
            .push_u32("llama.attention.head_count", 32)
            .push_u64("llama.context_length", 32768);
        let p = probe_bytes(b.finish(0)).unwrap();
        assert_eq!(p.context_length, 32768);
    }
}
