//! Thin wrapper around the candle-transformers quantised model runners.
//!
//! Despite the file name, this type is polymorphic over the architectures
//! that [`super::arch_runner::CandleRunner`] supports — Llama-family,
//! Qwen 2/3/3.5, Qwen 3 MoE, and Gemma 3. The name is kept for source-
//! compatibility with phases C.3.1 → C.3.3, which only needed Llama; rename
//! is held off until callers can be updated in one pass.
//!
//! Phase C.5.a scope: a single in-flight sequence per `LlamaArch` instance
//! (the inner KV cache lives inside the wrapped `ModelWeights` and grows on
//! every forward call). Multi-batch and KV-cache reset arrive in C.3.4.

use std::fmt;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{DType, Device, Tensor};

use super::arch_runner::{CandleRunner, RunnerError};

pub struct LlamaArch {
    runner: Mutex<CandleRunner>,
    device: Device,
    label: &'static str,
}

#[derive(Debug)]
pub enum ArchError {
    Io(std::io::Error),
    Candle(String),
    /// The architecture string is not handled by the candle backend.
    /// Surfaces unchanged to the router so it can fall back to llama.cpp.
    UnsupportedArch(String),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error opening GGUF file: {e}"),
            Self::Candle(msg) => write!(f, "candle error: {msg}"),
            Self::UnsupportedArch(arch) => write!(
                f,
                "candle backend has no quantised runner for architecture '{arch}'"
            ),
        }
    }
}

impl std::error::Error for ArchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<RunnerError> for ArchError {
    fn from(e: RunnerError) -> Self {
        match e {
            RunnerError::Io(io) => ArchError::Io(io),
            RunnerError::Candle(msg) => ArchError::Candle(msg),
            RunnerError::UnsupportedArch(arch) => ArchError::UnsupportedArch(arch),
        }
    }
}

impl LlamaArch {
    /// Load the appropriate quantised runner for `architecture`.
    ///
    /// `architecture` must be the lowercase string produced by
    /// `inspect::probe` (`general.architecture` from the GGUF). When the
    /// architecture is unknown to the backend the call fails with
    /// [`ArchError::UnsupportedArch`] — caller is expected to route around
    /// the candle backend in that case.
    pub fn from_gguf(
        path: &Path,
        architecture: &str,
        device: Device,
    ) -> Result<Self, ArchError> {
        let runner = CandleRunner::from_gguf(path, architecture, &device)?;
        let label = runner.label();
        Ok(Self {
            runner: Mutex::new(runner),
            device,
            label,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn runner_label(&self) -> &'static str {
        self.label
    }

    /// Run a forward pass on `tokens` starting at absolute KV-cache position
    /// `position`. Returns the logits for the **last** token in the input
    /// (shape: `vocab_size`, in `f32`).
    ///
    /// Calling this on a fresh `LlamaArch`, then again with `position` equal
    /// to the previous call's `tokens.len()`, continues the same conversation.
    /// To start a new conversation, pass `position = 0` — the underlying
    /// runner overwrites its KV cache instead of concatenating.
    pub fn forward(&self, tokens: &[i32], position: usize) -> Result<Vec<f32>, ArchError> {
        let token_u32: Vec<u32> = tokens
            .iter()
            .map(|&t| if t < 0 { 0u32 } else { t as u32 })
            .collect();

        let input = Tensor::new(token_u32.as_slice(), &self.device)
            .map_err(|e| ArchError::Candle(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| ArchError::Candle(e.to_string()))?;

        let mut runner = self
            .runner
            .lock()
            .expect("LlamaArch mutex poisoned — another thread panicked while holding it");
        let logits = runner
            .forward(&input, position)
            .map_err(|e| ArchError::Candle(e.to_string()))?;
        drop(runner);

        let logits = logits
            .squeeze(0)
            .map_err(|e| ArchError::Candle(e.to_string()))?
            .to_dtype(DType::F32)
            .map_err(|e| ArchError::Candle(e.to_string()))?;
        logits
            .to_vec1::<f32>()
            .map_err(|e| ArchError::Candle(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `LlamaArch` is intentionally not `Debug` (its inner runner is not),
    /// so `Result::unwrap_err` will not compile. Extract the error via
    /// `match` instead.
    fn unwrap_err(r: Result<LlamaArch, ArchError>) -> ArchError {
        match r {
            Ok(_) => panic!("expected ArchError, got Ok(LlamaArch)"),
            Err(e) => e,
        }
    }

    #[test]
    fn returns_io_error_for_missing_path() {
        let err = unwrap_err(LlamaArch::from_gguf(
            Path::new("/definitely/does/not/exist.gguf"),
            "llama",
            Device::Cpu,
        ));
        assert!(matches!(err, ArchError::Io(_)));
    }

    #[test]
    fn returns_unsupported_arch_for_unknown_architecture() {
        // No need for a real GGUF file — the architecture check happens
        // before the file is opened.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.gguf");
        std::fs::write(&path, b"unused").unwrap();
        let err = unwrap_err(LlamaArch::from_gguf(
            &path,
            "acme-invented-arch",
            Device::Cpu,
        ));
        assert!(matches!(err, ArchError::UnsupportedArch(_)));
    }

    #[test]
    fn returns_candle_error_for_non_gguf_file_with_known_arch() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.gguf");
        std::fs::write(&path, b"this is not a gguf file").unwrap();
        let err = unwrap_err(LlamaArch::from_gguf(&path, "llama", Device::Cpu));
        assert!(matches!(err, ArchError::Candle(_)));
    }
}
