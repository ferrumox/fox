//! Read the tensor index of a GGUF file (no payload dequantisation yet).
//!
//! [`load_index`] walks the header → metadata → tensor-info sections and
//! returns a [`GgufIndex`] that maps every tensor name to its on-disk
//! location, shape and element type. Subsequent phases of the candle backend
//! will use that index to mmap the file and dequantise tensors lazily.
//!
//! This is intentionally `Read`/`Seek`-based rather than mmap-based: the
//! index is small and the cost of walking it once is negligible. mmap arrives
//! when actual weight access is implemented.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const GGUF_MAGIC: u32 = 0x4655_4747;
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// On-disk location and shape of a single tensor.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    /// Dimensions in row-major order, matching the GGUF on-disk layout.
    pub dims: Vec<u64>,
    /// GGML type ID (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, …). Numeric
    /// rather than enum because the catalogue grows with every llama.cpp
    /// release and the reader does not need to interpret unknown values.
    pub ggml_type: u32,
    /// Byte offset from the start of the tensor data section (not from the
    /// start of the file). Add `tensor_data_offset` to get an absolute file
    /// offset.
    pub offset_in_data: u64,
}

impl TensorEntry {
    /// Total number of scalar elements in this tensor.
    pub fn element_count(&self) -> u64 {
        self.dims.iter().copied().product::<u64>().max(1)
    }
}

/// Result of [`load_index`]: enough information to mmap and read every weight
/// without re-parsing the GGUF metadata.
#[derive(Debug)]
pub struct GgufIndex {
    pub gguf_version: u32,
    pub alignment: u64,
    /// Absolute file offset where the tensor data section begins.
    pub tensor_data_offset: u64,
    /// Tensors keyed by their GGUF name (e.g. `"blk.0.attn_q.weight"`).
    pub tensors: HashMap<String, TensorEntry>,
    /// Order in which the tensors appear in the GGUF tensor-info section.
    /// Useful when the consumer wants to iterate in file order.
    pub order: Vec<String>,
}

impl GgufIndex {
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Errors specific to the tensor-index walker. Distinct from the Phase A
/// `ProbeError` because this reader looks at a strictly larger slice of the
/// file (it must seek past the metadata block to reach tensor info) and so
/// has its own failure modes.
#[derive(Debug)]
pub enum LoaderError {
    Io(std::io::Error),
    NotGguf,
    UnsupportedVersion(u32),
    Truncated,
    InvalidValueType(u32),
    InvalidUtf8,
    InvalidAlignment(u64),
    UnreasonableTensorCount(u64),
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error reading GGUF tensor index: {e}"),
            Self::NotGguf => write!(f, "file does not start with GGUF magic bytes"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported GGUF version {v}"),
            Self::Truncated => write!(f, "GGUF tensor-info section is truncated"),
            Self::InvalidValueType(t) => write!(f, "unknown GGUF metadata value type {t}"),
            Self::InvalidUtf8 => write!(f, "GGUF tensor name is not valid UTF-8"),
            Self::InvalidAlignment(a) => write!(f, "GGUF reports non-power-of-two alignment {a}"),
            Self::UnreasonableTensorCount(n) => {
                write!(f, "GGUF reports {n} tensors — refusing to allocate")
            }
        }
    }
}

impl std::error::Error for LoaderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

/// Sanity cap on tensor count. Real GGUFs have a few hundred to ~1000 tensors;
/// anything above this is almost certainly a corrupt or hostile file and we
/// refuse to allocate a HashMap of that size.
const MAX_REASONABLE_TENSORS: u64 = 100_000;

/// Read the GGUF tensor index from `path`.
pub fn load_index(path: &Path) -> Result<GgufIndex, LoaderError> {
    let file = File::open(path).map_err(LoaderError::Io)?;
    let mut reader = BufReader::with_capacity(64 * 1024, file);
    load_index_from(&mut reader)
}

/// Same as [`load_index`] but operates on any `Read + Seek`. Exposed for
/// tests that build synthetic byte streams.
pub fn load_index_from<R: Read + Seek>(r: &mut R) -> Result<GgufIndex, LoaderError> {
    let header = read_header(r)?;
    if header.tensor_count > MAX_REASONABLE_TENSORS {
        return Err(LoaderError::UnreasonableTensorCount(header.tensor_count));
    }

    let alignment = walk_metadata(r, header.metadata_kv_count)?;

    let entries = read_tensor_infos(r, header.tensor_count)?;

    // Tensor data starts at the next multiple of `alignment` after the end of
    // the tensor-info section.
    let pos = r.stream_position().map_err(LoaderError::Io)?;
    let tensor_data_offset = align_up(pos, alignment);

    let mut tensors = HashMap::with_capacity(entries.len());
    let mut order = Vec::with_capacity(entries.len());
    for entry in entries {
        order.push(entry.name.clone());
        tensors.insert(entry.name.clone(), entry);
    }

    Ok(GgufIndex {
        gguf_version: header.version,
        alignment,
        tensor_data_offset,
        tensors,
        order,
    })
}

// ---------------------------------------------------------------------------
// Header + metadata walking
// ---------------------------------------------------------------------------

struct RawHeader {
    version: u32,
    tensor_count: u64,
    metadata_kv_count: u64,
}

fn read_header<R: Read>(r: &mut R) -> Result<RawHeader, LoaderError> {
    let magic = read_u32(r)?;
    if magic != GGUF_MAGIC {
        return Err(LoaderError::NotGguf);
    }
    let version = read_u32(r)?;
    if !SUPPORTED_VERSIONS.contains(&version) {
        return Err(LoaderError::UnsupportedVersion(version));
    }
    let tensor_count = read_u64(r)?;
    let metadata_kv_count = read_u64(r)?;
    Ok(RawHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Walk the metadata KV section, looking only for `general.alignment`. All
/// other values are skipped without allocation. Returns the alignment to use
/// for the tensor data offset (default: 32 bytes per the GGUF spec).
fn walk_metadata<R: Read + Seek>(r: &mut R, count: u64) -> Result<u64, LoaderError> {
    let mut alignment: u64 = 32;
    for _ in 0..count {
        let key = read_string(r)?;
        let value_type = read_u32(r)?;
        let captured = if key == "general.alignment" {
            Some(read_value_capturing_uint(r, value_type)?)
        } else {
            skip_value(r, value_type)?;
            None
        };
        if let Some(v) = captured {
            if !v.is_power_of_two() {
                return Err(LoaderError::InvalidAlignment(v));
            }
            alignment = v;
        }
    }
    Ok(alignment)
}

/// Read a metadata value when we know the type but want to capture the
/// numeric content (used for `general.alignment`). Falls back to skipping
/// when the value isn't a plain integer.
fn read_value_capturing_uint<R: Read + Seek>(r: &mut R, tag: u32) -> Result<u64, LoaderError> {
    match tag {
        0 => Ok(read_u8(r)? as u64),
        2 => Ok(read_u16(r)? as u64),
        4 => Ok(read_u32(r)? as u64),
        10 => read_u64(r),
        // Any other type for `general.alignment` is malformed; consume the
        // bytes anyway so the walker stays in sync, then fall back to default.
        _ => {
            skip_value(r, tag)?;
            Ok(32)
        }
    }
}

fn skip_value<R: Read + Seek>(r: &mut R, tag: u32) -> Result<(), LoaderError> {
    match tag {
        0 | 1 | 7 => {
            let _ = read_u8(r)?;
        }
        2 | 3 => {
            let _ = read_u16(r)?;
        }
        4..=6 => {
            let _ = read_u32(r)?;
        }
        10..=12 => {
            let _ = read_u64(r)?;
        }
        8 => {
            let len = read_u64(r)?;
            r.seek(SeekFrom::Current(len as i64)).map_err(LoaderError::Io)?;
        }
        9 => {
            let element_type = read_u32(r)?;
            let len = read_u64(r)?;
            skip_array_payload(r, element_type, len)?;
        }
        _ => return Err(LoaderError::InvalidValueType(tag)),
    }
    Ok(())
}

fn skip_array_payload<R: Read + Seek>(
    r: &mut R,
    element_type: u32,
    len: u64,
) -> Result<(), LoaderError> {
    let fixed_width = match element_type {
        0 | 1 | 7 => Some(1u64),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        8 => None,
        _ => return Err(LoaderError::InvalidValueType(element_type)),
    };
    match fixed_width {
        Some(width) => {
            let bytes = width.saturating_mul(len);
            r.seek(SeekFrom::Current(bytes as i64)).map_err(LoaderError::Io)?;
        }
        None => {
            for _ in 0..len {
                let s_len = read_u64(r)?;
                r.seek(SeekFrom::Current(s_len as i64)).map_err(LoaderError::Io)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor-info section
// ---------------------------------------------------------------------------

fn read_tensor_infos<R: Read>(r: &mut R, count: u64) -> Result<Vec<TensorEntry>, LoaderError> {
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        let name = read_string(r)?;
        let n_dims = read_u32(r)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let ggml_type = read_u32(r)?;
        let offset_in_data = read_u64(r)?;
        out.push(TensorEntry {
            name,
            dims,
            ggml_type,
            offset_in_data,
        });
    }
    Ok(out)
}

fn align_up(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment > 0);
    let mask = alignment - 1;
    (value + mask) & !mask
}

// ---------------------------------------------------------------------------
// Primitive readers
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(r: &mut R) -> Result<u8, LoaderError> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(b[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16, LoaderError> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32, LoaderError> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64, LoaderError> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(map_io)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(r: &mut R) -> Result<String, LoaderError> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(map_io)?;
    String::from_utf8(buf).map_err(|_| LoaderError::InvalidUtf8)
}

fn map_io(e: std::io::Error) -> LoaderError {
    if e.kind() == std::io::ErrorKind::UnexpectedEof {
        LoaderError::Truncated
    } else {
        LoaderError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Builder for synthetic GGUF files with a header, a small KV section
    /// and a tensor-info section. No tensor data is written — the reader
    /// stops before that boundary.
    struct Builder {
        kv_payload: Vec<u8>,
        kv_count: u64,
        tensor_payload: Vec<u8>,
        tensor_count: u64,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                kv_payload: Vec::new(),
                kv_count: 0,
                tensor_payload: Vec::new(),
                tensor_count: 0,
            }
        }

        fn kv_alignment(mut self, value: u64) -> Self {
            write_string(&mut self.kv_payload, "general.alignment");
            self.kv_payload.extend_from_slice(&10u32.to_le_bytes()); // UINT64
            self.kv_payload.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn kv_string_array(mut self, key: &str, items: &[&str]) -> Self {
            write_string(&mut self.kv_payload, key);
            self.kv_payload.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
            self.kv_payload.extend_from_slice(&8u32.to_le_bytes()); // STRING elements
            self.kv_payload.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_string(&mut self.kv_payload, item);
            }
            self.kv_count += 1;
            self
        }

        fn tensor(
            mut self,
            name: &str,
            dims: &[u64],
            ggml_type: u32,
            offset_in_data: u64,
        ) -> Self {
            write_string(&mut self.tensor_payload, name);
            self.tensor_payload
                .extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                self.tensor_payload.extend_from_slice(&d.to_le_bytes());
            }
            self.tensor_payload.extend_from_slice(&ggml_type.to_le_bytes());
            self.tensor_payload.extend_from_slice(&offset_in_data.to_le_bytes());
            self.tensor_count += 1;
            self
        }

        fn finish(self) -> Vec<u8> {
            let mut out = Vec::with_capacity(self.kv_payload.len() + self.tensor_payload.len() + 24);
            out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            out.extend_from_slice(&3u32.to_le_bytes()); // version
            out.extend_from_slice(&self.tensor_count.to_le_bytes());
            out.extend_from_slice(&self.kv_count.to_le_bytes());
            out.extend_from_slice(&self.kv_payload);
            out.extend_from_slice(&self.tensor_payload);
            out
        }
    }

    fn write_string(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn parse(bytes: Vec<u8>) -> Result<GgufIndex, LoaderError> {
        let mut cursor = Cursor::new(bytes);
        load_index_from(&mut cursor)
    }

    #[test]
    fn reads_two_tensors_in_order() {
        let bytes = Builder::new()
            .kv_string_array("tokenizer.ggml.tokens", &["a", "b"])
            .tensor("token_embd.weight", &[4096, 32000], 12, 0)
            .tensor("output.weight", &[32000, 4096], 12, 65_536)
            .finish();
        let idx = parse(bytes).unwrap();
        assert_eq!(idx.len(), 2);
        assert_eq!(idx.order, vec!["token_embd.weight", "output.weight"]);
        let t = idx.get("token_embd.weight").unwrap();
        assert_eq!(t.dims, vec![4096, 32000]);
        assert_eq!(t.ggml_type, 12);
        assert_eq!(t.offset_in_data, 0);
        assert_eq!(t.element_count(), 4096 * 32000);
    }

    #[test]
    fn applies_explicit_alignment() {
        let bytes = Builder::new()
            .kv_alignment(64)
            .tensor("a", &[16], 0, 0)
            .finish();
        let idx = parse(bytes).unwrap();
        assert_eq!(idx.alignment, 64);
        // tensor_data_offset must be a multiple of alignment.
        assert_eq!(idx.tensor_data_offset % 64, 0);
    }

    #[test]
    fn defaults_alignment_to_32_when_absent() {
        let bytes = Builder::new().tensor("a", &[16], 0, 0).finish();
        let idx = parse(bytes).unwrap();
        assert_eq!(idx.alignment, 32);
        assert_eq!(idx.tensor_data_offset % 32, 0);
    }

    #[test]
    fn rejects_non_gguf_files() {
        let bytes = b"not a gguf at all".to_vec();
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, LoaderError::NotGguf));
    }

    #[test]
    fn rejects_unsupported_version() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&99u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, LoaderError::UnsupportedVersion(99)));
    }

    #[test]
    fn rejects_unreasonable_tensor_count() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, LoaderError::UnreasonableTensorCount(_)));
    }

    #[test]
    fn rejects_non_power_of_two_alignment() {
        let bytes = Builder::new()
            .kv_alignment(48) // not a power of two
            .tensor("a", &[16], 0, 0)
            .finish();
        let err = parse(bytes).unwrap_err();
        assert!(matches!(err, LoaderError::InvalidAlignment(48)));
    }

    #[test]
    fn skips_unrelated_metadata_array_values() {
        // Mix of metadata kinds: the walker must skip them all without
        // disturbing the tensor section that follows.
        let bytes = Builder::new()
            .kv_string_array("tokenizer.ggml.tokens", &["alpha", "beta", "gamma"])
            .kv_alignment(32)
            .tensor("blk.0.attn_q.weight", &[128, 128], 1, 0)
            .finish();
        let idx = parse(bytes).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(idx.tensors.contains_key("blk.0.attn_q.weight"));
    }
}
