use std::ffi::CString;
use std::os::raw::c_char;

use crate::engine::ffi;
use crate::engine::model::RecommendedSampling;

use super::LlamaCppModel;

impl LlamaCppModel {
    /// Read a single GGUF metadata value by key name.
    /// Returns `None` when the key is absent or cannot be decoded as UTF-8.
    pub(super) fn read_meta_str(&self, key: &str) -> Option<String> {
        let key_c = CString::new(key).ok()?;
        let mut buf = vec![0u8; 512];
        let n = unsafe {
            ffi::llama_model_meta_val_str(
                self._model.as_ptr(),
                key_c.as_ptr(),
                buf.as_mut_ptr() as *mut c_char,
                buf.len(),
            )
        };
        if n < 0 {
            return None;
        }
        Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }

    /// Read a GGUF metadata value as `f32`. Returns `None` when missing or not parseable.
    pub(super) fn read_meta_f32(&self, key: &str) -> Option<f32> {
        self.read_meta_str(key)?.trim().parse::<f32>().ok()
    }

    /// Read a GGUF metadata value as `u32`. Returns `None` when missing or not parseable.
    pub(super) fn read_meta_u32(&self, key: &str) -> Option<u32> {
        self.read_meta_str(key)?.trim().parse::<u32>().ok()
    }

    /// Iterate all GGUF metadata keys/values and look for sampling-related hints.
    /// Logs all keys at TRACE level. Returns a partial `RecommendedSampling`.
    pub(super) fn read_sampling_from_meta(&self) -> RecommendedSampling {
        let count = unsafe { ffi::llama_model_meta_count(self._model.as_ptr()) };
        let mut temperature: Option<f32> = None;
        let mut top_p: Option<f32> = None;
        let mut top_k: Option<u32> = None;

        let mut key_buf = vec![0u8; 256];
        let mut val_buf = vec![0u8; 512];

        for i in 0..count {
            let kn = unsafe {
                ffi::llama_model_meta_key_by_index(
                    self._model.as_ptr(),
                    i,
                    key_buf.as_mut_ptr() as *mut c_char,
                    key_buf.len(),
                )
            };
            let vn = unsafe {
                ffi::llama_model_meta_val_str_by_index(
                    self._model.as_ptr(),
                    i,
                    val_buf.as_mut_ptr() as *mut c_char,
                    val_buf.len(),
                )
            };
            if kn < 0 || vn < 0 {
                continue;
            }
            // Skip values that didn't fit in the buffer (e.g. long chat templates).
            // Sampling parameters are always short numeric strings — no false negatives.
            if vn as usize > val_buf.len() || kn as usize > key_buf.len() {
                continue;
            }
            let key = String::from_utf8_lossy(&key_buf[..kn as usize]).into_owned();
            let val = String::from_utf8_lossy(&val_buf[..vn as usize]).into_owned();
            tracing::trace!(key = %key, value = %val, "GGUF metadata");

            let key_lc = key.to_lowercase();
            if temperature.is_none() && key_lc.contains("temperature") {
                temperature = val.trim().parse::<f32>().ok();
            }
            if top_p.is_none() && key_lc.contains("top_p") {
                top_p = val.trim().parse::<f32>().ok();
            }
            if top_k.is_none() && key_lc.contains("top_k") {
                top_k = val.trim().parse::<u32>().ok();
            }
        }

        RecommendedSampling {
            temperature,
            top_p,
            top_k,
        }
    }
}
