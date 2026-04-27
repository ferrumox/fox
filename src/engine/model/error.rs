//! Typed errors for model loading.
//!
//! `LoadError` exists so the loader cascade and the registry can dispatch on
//! *kind* of failure (OOM-shaped, file IO, missing arch, …) instead of
//! grepping `Display` strings. Today the dominant source of error is the
//! llama.cpp FFI, which only signals failure through a flat `anyhow::Error`
//! whose message text varies between releases — so [`LoadError::classify`]
//! still falls back to substring matching when no structured signal is
//! available, but the matching logic now lives in a single place that other
//! backends and tests can target.

use std::fmt;

/// Cause of a failed `instantiate` call. The enum is intentionally small;
/// add variants only when the loader has a concrete distinction to act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadError {
    /// Out of GPU or system memory. Triggers the loader's degradation
    /// cascade (smaller context length, lower-precision KV cache).
    OutOfMemory,
    /// The requested context length exceeds what the model or the device
    /// can support, but the failure is not memory-pressure shaped.
    ContextTooLarge,
    /// The architecture is recognised but the backend declines to instantiate it.
    ArchUnsupported,
    /// File-system error: missing path, permission denied, truncated file.
    Io,
    /// Anything else — backend-specific failure with no fox-level
    /// classification yet. Carries the original `anyhow` message for logs.
    Backend(String),
}

impl LoadError {
    /// Classify an `anyhow::Error` produced by a backend's `instantiate`.
    ///
    /// The substring patterns are intentionally broad so the cascade catches
    /// CUDA OOM, host-allocator OOM, and the llama.cpp init-failed wrapper —
    /// all symptoms of "the model didn't fit". When none of the known
    /// patterns match the error becomes [`LoadError::Backend`] with the
    /// original message so logs stay actionable.
    pub fn classify(err: &anyhow::Error) -> Self {
        let msg = format!("{err:#}");
        let lower = msg.to_lowercase();

        if lower.contains("out of memory")
            || lower.contains("failed to allocate")
            || lower.contains("cuda error")
            || lower.contains("llama_init_from_model failed")
        {
            return LoadError::OutOfMemory;
        }
        if lower.contains("context") && lower.contains("too") {
            return LoadError::ContextTooLarge;
        }
        if lower.contains("unsupported") || lower.contains("unknown architecture") {
            return LoadError::ArchUnsupported;
        }
        if lower.contains("no such file")
            || lower.contains("permission denied")
            || lower.contains("not a directory")
        {
            return LoadError::Io;
        }
        LoadError::Backend(msg)
    }

    /// True when the loader should retry with a smaller / cheaper config.
    pub fn is_oom(&self) -> bool {
        matches!(self, LoadError::OutOfMemory)
    }
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "model loader ran out of memory"),
            Self::ContextTooLarge => {
                write!(f, "requested context length exceeds the model or device limit")
            }
            Self::ArchUnsupported => {
                write!(f, "the backend declined to instantiate this architecture")
            }
            Self::Io => write!(f, "filesystem error reading the model file"),
            Self::Backend(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::anyhow;

    #[test]
    fn classifies_cuda_oom() {
        let err = anyhow!("CUDA error: out of memory");
        assert_eq!(LoadError::classify(&err), LoadError::OutOfMemory);
        assert!(LoadError::classify(&err).is_oom());
    }

    #[test]
    fn classifies_host_allocator_oom() {
        let err = anyhow!("ggml_new_tensor failed to allocate buffer");
        assert_eq!(LoadError::classify(&err), LoadError::OutOfMemory);
    }

    #[test]
    fn classifies_llama_init_failure_as_oom() {
        let err = anyhow!("llama_init_from_model failed (returned null)");
        assert_eq!(LoadError::classify(&err), LoadError::OutOfMemory);
    }

    #[test]
    fn classifies_io_errors() {
        let err = anyhow!("No such file or directory (os error 2)");
        assert_eq!(LoadError::classify(&err), LoadError::Io);
        assert!(!LoadError::classify(&err).is_oom());
    }

    #[test]
    fn classifies_unknown_arch() {
        let err = anyhow!("unknown architecture: acme-invented-name");
        assert_eq!(LoadError::classify(&err), LoadError::ArchUnsupported);
    }

    #[test]
    fn classifies_context_too_large() {
        let err = anyhow!("context size 131072 too large for this device");
        assert_eq!(LoadError::classify(&err), LoadError::ContextTooLarge);
    }

    #[test]
    fn unrecognised_message_falls_back_to_backend_variant() {
        let err = anyhow!("some weird internal error 0xdeadbeef");
        match LoadError::classify(&err) {
            LoadError::Backend(msg) => assert!(msg.contains("0xdeadbeef")),
            other => panic!("expected Backend variant, got {other:?}"),
        }
    }

    #[test]
    fn case_insensitive_oom_detection() {
        let err = anyhow!("Out Of Memory while allocating");
        assert_eq!(LoadError::classify(&err), LoadError::OutOfMemory);
    }
}
