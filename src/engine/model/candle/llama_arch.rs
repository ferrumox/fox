//! Wrapper around `candle_transformers::quantized_llama::ModelWeights`.
//!
//! Phase C.3 scope: load a quantised GGUF Llama-family model on a chosen
//! device, run forward passes, return logits. The KV cache lives inside the
//! wrapped `ModelWeights` and grows on every forward call — phase C.3 only
//! supports a single in-flight sequence per `LlamaArch` instance. Multi-batch
//! and KV-cache reset are C.3.2 work.

use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{quantized::gguf_file, DType, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

/// Llama-family model loaded into candle. Borrowed pointer + KV cache are
/// hidden behind a Mutex so the type stays `Send + Sync` while still allowing
/// `&mut self` calls into `ModelWeights::forward`.
pub struct LlamaArch {
    weights: Mutex<ModelWeights>,
    device: Device,
}

#[derive(Debug)]
pub enum ArchError {
    Io(std::io::Error),
    Candle(String),
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error opening GGUF file: {e}"),
            Self::Candle(msg) => write!(f, "candle error: {msg}"),
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

impl LlamaArch {
    /// Load weights from `path` onto `device`. Walks the GGUF header and
    /// dequantises each tensor into the quantised candle layout used by
    /// `quantized_llama`. CPU and CUDA devices are both supported.
    pub fn from_gguf(path: &Path, device: Device) -> Result<Self, ArchError> {
        let mut file = File::open(path).map_err(ArchError::Io)?;
        let content =
            gguf_file::Content::read(&mut file).map_err(|e| ArchError::Candle(e.to_string()))?;
        let weights = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| ArchError::Candle(e.to_string()))?;
        Ok(Self {
            weights: Mutex::new(weights),
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run a forward pass on `tokens` starting at absolute KV-cache position
    /// `position`. Returns the logits for the **last** token in the input
    /// (shape: `vocab_size`, in `f32`).
    ///
    /// Calling this on a fresh `LlamaArch`, then again with `position` equal
    /// to the previous call's `tokens.len()`, continues the same conversation.
    /// To start a new conversation the model must be reloaded — see the
    /// module-level note about phase C.3.2.
    pub fn forward(&self, tokens: &[i32], position: usize) -> Result<Vec<f32>, ArchError> {
        let token_u32: Vec<u32> = tokens
            .iter()
            .map(|&t| {
                if t < 0 {
                    0u32
                } else {
                    t as u32
                }
            })
            .collect();

        let input = Tensor::new(token_u32.as_slice(), &self.device)
            .map_err(|e| ArchError::Candle(e.to_string()))?
            .unsqueeze(0)
            .map_err(|e| ArchError::Candle(e.to_string()))?;

        let mut wts = self
            .weights
            .lock()
            .expect("LlamaArch mutex poisoned — another thread panicked while holding it");
        let logits = wts
            .forward(&input, position)
            .map_err(|e| ArchError::Candle(e.to_string()))?;
        drop(wts);

        // `forward` returns `[batch=1, vocab]` for the last token only.
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

    /// `LlamaArch` is intentionally not `Debug` (its inner `ModelWeights` is
    /// not), so `Result::unwrap_err` will not compile. Extract the error via
    /// `match` instead.
    fn unwrap_err(r: Result<LlamaArch, ArchError>) -> ArchError {
        match r {
            Ok(_) => panic!("expected ArchError, got Ok(LlamaArch)"),
            Err(e) => e,
        }
    }

    /// Smoke test exercising the error path only — no fixture model required.
    /// Real-model coverage lives in `tests/candle_llama_smoke.rs`.
    #[test]
    fn returns_io_error_for_missing_path() {
        let err = unwrap_err(LlamaArch::from_gguf(
            Path::new("/definitely/does/not/exist.gguf"),
            Device::Cpu,
        ));
        assert!(matches!(err, ArchError::Io(_)));
    }

    #[test]
    fn returns_candle_error_for_non_gguf_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("garbage.gguf");
        std::fs::write(&path, b"this is not a gguf file").unwrap();
        let err = unwrap_err(LlamaArch::from_gguf(&path, Device::Cpu));
        assert!(matches!(err, ArchError::Candle(_)));
    }
}
