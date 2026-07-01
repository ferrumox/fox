use std::ffi::CString;
use std::os::raw::c_char;

use anyhow::{anyhow, Result};

use crate::engine::ffi;

use super::LlamaCppModel;

impl LlamaCppModel {
    pub(super) fn tokenize_impl(&self, text: &str) -> Result<Vec<i32>> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return Err(anyhow!("vocab is null"));
        }
        // First call with null buffer to get required size
        let n_max = text.len() + 4;
        let mut tokens: Vec<ffi::llama_token> = vec![0; n_max];
        let n = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                tokens.as_mut_ptr(),
                n_max as i32,
                true,  // add_special
                false, // parse_special
            )
        };
        if n < 0 {
            let need = (-n) as usize;
            let mut tokens: Vec<ffi::llama_token> = vec![0; need];
            let n = unsafe {
                ffi::llama_tokenize(
                    vocab,
                    text.as_ptr() as *const c_char,
                    text.len() as i32,
                    tokens.as_mut_ptr(),
                    need as i32,
                    true,
                    false,
                )
            };
            if n < 0 {
                return Err(anyhow!("llama_tokenize failed: {}", n));
            }
            tokens.truncate(n as usize);
            Ok(tokens)
        } else {
            tokens.truncate(n as usize);
            Ok(tokens)
        }
    }

    pub(super) fn token_to_piece_impl(&self, token: i32) -> Result<String> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return Err(anyhow!("vocab is null"));
        }
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0, // lstrip: keep leading spaces so words don't concatenate (e.g. "¡Hola!¿Enquépuedo...")
                    true, // special
                )
            };
            if n < 0 {
                let need = (-n) as usize;
                buf.resize(need, 0);
                let n = unsafe {
                    ffi::llama_token_to_piece(
                        vocab,
                        token,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len() as i32,
                        0, // lstrip: keep leading spaces
                        true,
                    )
                };
                if n < 0 {
                    return Err(anyhow!("llama_token_to_piece failed: {}", n));
                }
                let s = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                return Ok(s);
            }
            if n as usize <= buf.len() {
                let s = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                return Ok(s);
            }
            buf.resize(n as usize, 0);
        }
    }

    /// Raw-bytes variant: same as `token_to_piece_impl` but returns `Vec<u8>`
    /// without any UTF-8 validation so the engine can accumulate partial
    /// multi-byte sequences (e.g. emoji split across BPE tokens) before decoding.
    pub(super) fn token_to_piece_bytes_impl(&self, token: i32) -> Vec<u8> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return vec![];
        }
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe {
                ffi::llama_token_to_piece(
                    vocab,
                    token,
                    buf.as_mut_ptr() as *mut std::ffi::c_char,
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n < 0 {
                let need = (-n) as usize;
                buf.resize(need, 0);
                let n2 = unsafe {
                    ffi::llama_token_to_piece(
                        vocab,
                        token,
                        buf.as_mut_ptr() as *mut std::ffi::c_char,
                        buf.len() as i32,
                        0,
                        true,
                    )
                };
                if n2 < 0 {
                    return vec![];
                }
                return buf[..n2 as usize].to_vec();
            }
            if (n as usize) <= buf.len() {
                return buf[..n as usize].to_vec();
            }
            buf.resize(n as usize, 0);
        }
    }

    pub(super) fn apply_chat_template_impl(&self, messages: &[(String, String)]) -> Result<String> {
        if messages.is_empty() {
            return Ok(String::new());
        }

        let role_cstrings: Vec<CString> = messages
            .iter()
            .map(|(r, _)| {
                // Replace interior null bytes that would truncate the C string
                let sanitized = r.replace('\0', "");
                CString::new(sanitized).unwrap_or_else(|_| CString::new("user").unwrap())
            })
            .collect();
        let content_cstrings: Vec<CString> = messages
            .iter()
            .map(|(_, c)| {
                let sanitized = c.replace('\0', "");
                CString::new(sanitized).unwrap_or_else(|_| CString::new("").unwrap())
            })
            .collect();
        let chat: Vec<ffi::llama_chat_message> = (0..messages.len())
            .map(|i| ffi::llama_chat_message {
                role: role_cstrings[i].as_ptr(),
                content: content_cstrings[i].as_ptr(),
            })
            .collect();

        let estimated_len = messages
            .iter()
            .map(|(r, c)| r.len() + c.len() + 128)
            .sum::<usize>()
            .max(512);

        let model = self._model.as_ptr();

        // Try model's template first, then fallback names
        let templates_to_try: Vec<Option<&str>> = {
            let from_model = unsafe {
                let p = ffi::llama_model_chat_template(model, std::ptr::null());
                if p.is_null() {
                    None
                } else {
                    std::ffi::CStr::from_ptr(p).to_str().ok().map(|s| s as &str)
                }
            };
            if let Some(s) = from_model {
                vec![Some(s)]
            } else {
                vec![None]
            }
        };

        // Built-in template names - llama_chat_apply_template looks them up by name
        let fallback_names = [
            "chatml",     // Qwen, Phi, OpenHermes, many models
            "llama3",     // Meta Llama 3
            "phi3",       // Microsoft Phi-3
            "llama2",     // Meta Llama 2, Mistral
            "mistral-v1", // Mistral
            "gemma",      // Google Gemma
        ];

        let mut template_strings: Vec<String> = templates_to_try
            .into_iter()
            .filter_map(|o| o.map(String::from))
            .collect();
        for name in fallback_names {
            if !template_strings.iter().any(|s| s == name) {
                template_strings.push(name.to_string());
            }
        }

        for tmpl_str in &template_strings {
            let mut buf = vec![0u8; estimated_len];
            let tmpl_c = match CString::new(tmpl_str.as_str()) {
                Ok(c) => c,
                Err(_) => continue,
            };
            let n = unsafe {
                ffi::llama_chat_apply_template(
                    tmpl_c.as_ptr(),
                    chat.as_ptr(),
                    messages.len(),
                    true, // add_ass
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                )
            };
            if n >= 0 && (n as usize) <= buf.len() {
                // Buffer was large enough — return the complete result.
                let result = String::from_utf8_lossy(&buf[..n as usize]).into_owned();
                if !result.is_empty() {
                    tracing::debug!(template = tmpl_str, "applied chat template");
                    return Ok(result);
                }
            } else if n > 0 {
                // Buffer was too small — resize to the exact size needed and retry.
                let need = n as usize;
                buf.resize(need, 0);
                let n2 = unsafe {
                    ffi::llama_chat_apply_template(
                        tmpl_c.as_ptr(),
                        chat.as_ptr(),
                        messages.len(),
                        true,
                        buf.as_mut_ptr() as *mut c_char,
                        buf.len() as i32,
                    )
                };
                if n2 >= 0 && (n2 as usize) <= buf.len() {
                    let result = String::from_utf8_lossy(&buf[..n2 as usize]).into_owned();
                    if !result.is_empty() {
                        tracing::debug!(template = tmpl_str, "applied chat template (resized)");
                        return Ok(result);
                    }
                }
            }
        }

        Err(anyhow!("no chat template could be applied"))
    }
}
