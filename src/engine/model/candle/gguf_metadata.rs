//! Materialised view of a GGUF metadata block.
//!
//! Phase A's `inspect::probe` only walks scalar values it cares about —
//! arrays are skipped without allocation. The candle backend needs the
//! arrays themselves (vocab, merges, token types, scores), so this module
//! reads the same metadata block end-to-end and returns it as typed maps.
//!
//! Like the tensor-index reader, this lives behind the `backend-candle`
//! feature so the base build keeps zero dependencies on candle metadata
//! parsing.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Typed view of a GGUF metadata block. Each map is keyed by the metadata
/// key as it appears in the file (e.g. `"general.architecture"`).
#[derive(Debug, Default)]
pub struct GgufMetadata {
    pub architecture: String,
    pub gguf_version: u32,
    pub strings: HashMap<String, String>,
    pub uints: HashMap<String, u64>,
    pub ints: HashMap<String, i64>,
    pub floats: HashMap<String, f64>,
    pub bools: HashMap<String, bool>,
    pub string_arrays: HashMap<String, Vec<String>>,
    pub int_arrays: HashMap<String, Vec<i64>>,
    pub float_arrays: HashMap<String, Vec<f32>>,
}

impl GgufMetadata {
    pub fn arch_key(&self, suffix: &str) -> String {
        format!("{}.{}", self.architecture, suffix)
    }

    /// Convenience: look up a uint by the architecture-prefixed key.
    pub fn arch_uint(&self, suffix: &str) -> Option<u64> {
        self.uints.get(&self.arch_key(suffix)).copied()
    }

    /// Convenience: look up a string by its full key.
    pub fn string(&self, key: &str) -> Option<&str> {
        self.strings.get(key).map(String::as_str)
    }
}

#[derive(Debug)]
pub enum MetadataError {
    Io(std::io::Error),
    NotGguf,
    UnsupportedVersion(u32),
    Truncated,
    InvalidValueType(u32),
    InvalidUtf8,
    MissingArchitecture,
    UnreasonableArrayLength(u64),
}

impl fmt::Display for MetadataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error reading GGUF metadata: {e}"),
            Self::NotGguf => write!(f, "file does not start with GGUF magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::Truncated => write!(f, "GGUF metadata block is truncated"),
            Self::InvalidValueType(t) => write!(f, "unknown GGUF metadata value type {t}"),
            Self::InvalidUtf8 => write!(f, "GGUF metadata string is not valid UTF-8"),
            Self::MissingArchitecture => {
                write!(f, "GGUF metadata is missing required key 'general.architecture'")
            }
            Self::UnreasonableArrayLength(n) => {
                write!(f, "GGUF array of length {n} exceeds the safety cap")
            }
        }
    }
}

impl std::error::Error for MetadataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

/// Per-array safety cap. Vocab/merges arrays for current models top out
/// around ~250k entries; anything an order of magnitude larger than that is
/// either corrupt or hostile.
const MAX_REASONABLE_ARRAY_LEN: u64 = 5_000_000;

pub fn load(path: &Path) -> Result<GgufMetadata, MetadataError> {
    let file = File::open(path).map_err(MetadataError::Io)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    load_from(&mut reader)
}

pub fn load_from<R: Read + Seek>(r: &mut R) -> Result<GgufMetadata, MetadataError> {
    let header = read_header(r)?;
    let mut out = GgufMetadata {
        gguf_version: header.version,
        ..GgufMetadata::default()
    };

    for _ in 0..header.metadata_kv_count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;
        ingest_value(r, &key, value_type, &mut out)?;
    }

    out.architecture = out
        .strings
        .get("general.architecture")
        .cloned()
        .ok_or(MetadataError::MissingArchitecture)?;

    Ok(out)
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

struct RawHeader {
    version: u32,
    metadata_kv_count: u64,
}

fn read_header<R: Read>(r: &mut R) -> Result<RawHeader, MetadataError> {
    let magic = read_u32(r)?;
    if magic != GGUF_MAGIC {
        return Err(MetadataError::NotGguf);
    }
    let version = read_u32(r)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(MetadataError::UnsupportedVersion(version));
    }
    let _tensor_count = read_u64(r)?;
    let metadata_kv_count = read_u64(r)?;
    Ok(RawHeader {
        version,
        metadata_kv_count,
    })
}

// ---------------------------------------------------------------------------
// Per-value ingestion
// ---------------------------------------------------------------------------

fn ingest_value<R: Read + Seek>(
    r: &mut R,
    key: &str,
    tag: u32,
    out: &mut GgufMetadata,
) -> Result<(), MetadataError> {
    match tag {
        0 => {
            out.uints.insert(key.to_string(), read_u8(r)? as u64);
        }
        1 => {
            out.ints.insert(key.to_string(), read_u8(r)? as i8 as i64);
        }
        2 => {
            out.uints.insert(key.to_string(), read_u16(r)? as u64);
        }
        3 => {
            out.ints
                .insert(key.to_string(), read_u16(r)? as i16 as i64);
        }
        4 => {
            out.uints.insert(key.to_string(), read_u32(r)? as u64);
        }
        5 => {
            out.ints
                .insert(key.to_string(), read_u32(r)? as i32 as i64);
        }
        6 => {
            out.floats
                .insert(key.to_string(), f32::from_bits(read_u32(r)?) as f64);
        }
        7 => {
            out.bools.insert(key.to_string(), read_u8(r)? != 0);
        }
        8 => {
            out.strings.insert(key.to_string(), read_string(r)?);
        }
        9 => ingest_array(r, key, out)?,
        10 => {
            out.uints.insert(key.to_string(), read_u64(r)?);
        }
        11 => {
            out.ints.insert(key.to_string(), read_u64(r)? as i64);
        }
        12 => {
            out.floats
                .insert(key.to_string(), f64::from_bits(read_u64(r)?));
        }
        _ => return Err(MetadataError::InvalidValueType(tag)),
    }
    Ok(())
}

fn ingest_array<R: Read + Seek>(
    r: &mut R,
    key: &str,
    out: &mut GgufMetadata,
) -> Result<(), MetadataError> {
    let element_type = read_u32(r)?;
    let len = read_u64(r)?;
    if len > MAX_REASONABLE_ARRAY_LEN {
        return Err(MetadataError::UnreasonableArrayLength(len));
    }
    let len_usize = len as usize;
    match element_type {
        0 => {
            // u8 array — skipped, no consumer cares about it yet
            r.seek(SeekFrom::Current(len as i64)).map_err(MetadataError::Io)?;
        }
        2 => {
            r.seek(SeekFrom::Current((len * 2) as i64)).map_err(MetadataError::Io)?;
        }
        4 => {
            // u32 → store as i64
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(read_u32(r)? as i64);
            }
            out.int_arrays.insert(key.to_string(), v);
        }
        5 => {
            // i32 → store as i64
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(read_u32(r)? as i32 as i64);
            }
            out.int_arrays.insert(key.to_string(), v);
        }
        6 => {
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(f32::from_bits(read_u32(r)?));
            }
            out.float_arrays.insert(key.to_string(), v);
        }
        7 => {
            // bool array — skipped
            r.seek(SeekFrom::Current(len as i64)).map_err(MetadataError::Io)?;
        }
        8 => {
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(read_string(r)?);
            }
            out.string_arrays.insert(key.to_string(), v);
        }
        10 => {
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(read_u64(r)? as i64);
            }
            out.int_arrays.insert(key.to_string(), v);
        }
        11 => {
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(read_u64(r)? as i64);
            }
            out.int_arrays.insert(key.to_string(), v);
        }
        12 => {
            let mut v = Vec::with_capacity(len_usize);
            for _ in 0..len_usize {
                v.push(f64::from_bits(read_u64(r)?) as f32);
            }
            out.float_arrays.insert(key.to_string(), v);
        }
        1 | 3 => {
            // signed 8/16: skip — no use case
            let width = if element_type == 1 { 1 } else { 2 };
            r.seek(SeekFrom::Current((len * width) as i64))
                .map_err(MetadataError::Io)?;
        }
        _ => return Err(MetadataError::InvalidValueType(element_type)),
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Primitive readers (little-endian, matching the GGUF spec)
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> Result<u8, MetadataError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, MetadataError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, MetadataError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, MetadataError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, MetadataError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(map_io)?;
    String::from_utf8(buf).map_err(|_| MetadataError::InvalidUtf8)
}

fn map_io(e: std::io::Error) -> MetadataError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        MetadataError::Truncated
    } else {
        MetadataError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    struct Builder {
        kv_payload: Vec<u8>,
        kv_count: u64,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                kv_payload: Vec::new(),
                kv_count: 0,
            }
        }

        fn string(mut self, key: &str, value: &str) -> Self {
            self.write_string(key);
            self.kv_payload.extend_from_slice(&8u32.to_le_bytes());
            self.write_string(value);
            self.kv_count += 1;
            self
        }

        fn u32(mut self, key: &str, value: u32) -> Self {
            self.write_string(key);
            self.kv_payload.extend_from_slice(&4u32.to_le_bytes());
            self.kv_payload.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn string_array(mut self, key: &str, items: &[&str]) -> Self {
            self.write_string(key);
            self.kv_payload.extend_from_slice(&9u32.to_le_bytes());
            self.kv_payload.extend_from_slice(&8u32.to_le_bytes());
            self.kv_payload.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                self.write_string(it);
            }
            self.kv_count += 1;
            self
        }

        fn i32_array(mut self, key: &str, items: &[i32]) -> Self {
            self.write_string(key);
            self.kv_payload.extend_from_slice(&9u32.to_le_bytes());
            self.kv_payload.extend_from_slice(&5u32.to_le_bytes());
            self.kv_payload.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for v in items {
                self.kv_payload.extend_from_slice(&v.to_le_bytes());
            }
            self.kv_count += 1;
            self
        }

        fn write_string(&mut self, s: &str) {
            self.kv_payload
                .extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.kv_payload.extend_from_slice(s.as_bytes());
        }

        fn finish(self) -> Vec<u8> {
            let mut out = Vec::with_capacity(self.kv_payload.len() + 24);
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kv_payload);
            out
        }
    }

    fn parse(bytes: Vec<u8>) -> Result<GgufMetadata, MetadataError> {
        load_from(&mut Cursor::new(bytes))
    }

    #[test]
    fn captures_arch_and_scalars() {
        let m = parse(
            Builder::new()
                .string("general.architecture", "llama")
                .u32("llama.context_length", 8192)
                .finish(),
        )
        .unwrap();
        assert_eq!(m.architecture, "llama");
        assert_eq!(m.arch_uint("context_length"), Some(8192));
    }

    #[test]
    fn materialises_string_array() {
        let m = parse(
            Builder::new()
                .string("general.architecture", "llama")
                .string_array("tokenizer.ggml.tokens", &["<bos>", "hello", "world"])
                .finish(),
        )
        .unwrap();
        let toks = m.string_arrays.get("tokenizer.ggml.tokens").unwrap();
        assert_eq!(toks, &vec!["<bos>".to_string(), "hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn materialises_int_array_from_i32() {
        let m = parse(
            Builder::new()
                .string("general.architecture", "llama")
                .i32_array("tokenizer.ggml.token_type", &[1, 1, 3, 4])
                .finish(),
        )
        .unwrap();
        assert_eq!(
            m.int_arrays.get("tokenizer.ggml.token_type"),
            Some(&vec![1i64, 1, 3, 4])
        );
    }

    #[test]
    fn rejects_files_without_architecture() {
        let m = parse(Builder::new().finish());
        assert!(matches!(m, Err(MetadataError::MissingArchitecture)));
    }

    #[test]
    fn rejects_unreasonable_array_length() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&1u64.to_le_bytes()); // kv_count
        // key
        let key = "huge";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        // ARRAY of u8, length = u64::MAX
        bytes.extend_from_slice(&9u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, MetadataError::UnreasonableArrayLength(_)));
    }
}
