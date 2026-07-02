use std::ffi::CString;
use std::os::raw::c_char;

use anyhow::{anyhow, Result};
use minijinja::{context, Environment};

use crate::engine::ffi;

use super::LlamaCppModel;

impl LlamaCppModel {
    /// Tokenize arbitrary user text / raw strings: prepend BOS (add_special) and
    /// treat any `<|...|>`-looking text as literal, NOT as special tokens.
    pub(super) fn tokenize_impl(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenize_with(text, true, false)
    }

    /// Tokenize an already chat-templated prompt: the template already supplies
    /// BOS (add_special = false, avoids a double BOS) and its control markers must
    /// become the real special tokens (parse_special = true), not literal text.
    pub(super) fn tokenize_prompt_impl(&self, text: &str) -> Result<Vec<i32>> {
        self.tokenize_with(text, false, true)
    }

    fn tokenize_with(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<i32>> {
        let vocab = self.vocab;
        if vocab.is_null() {
            return Err(anyhow!("vocab is null"));
        }
        // First call sizes the buffer; retry once if it was too small.
        let n_max = text.len() + 4;
        let mut tokens: Vec<ffi::llama_token> = vec![0; n_max];
        let n = unsafe {
            ffi::llama_tokenize(
                vocab,
                text.as_ptr() as *const c_char,
                text.len() as i32,
                tokens.as_mut_ptr(),
                n_max as i32,
                add_special,
                parse_special,
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
                    add_special,
                    parse_special,
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

    /// Render the model's real Jinja chat template (from the GGUF) with `minijinja`,
    /// threading `enable_thinking`. Returns `None` when the model has no embedded
    /// template or it fails to render — the caller then falls back to the built-in
    /// llama.cpp format.
    ///
    /// TODO(perf): this parses the template on every call; cache the compiled
    /// template on the model once the interface settles.
    fn render_chat_jinja(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
    ) -> Option<String> {
        // Full template string straight from the model (NOT read_meta_str, whose
        // 512-byte buffer would truncate a multi-KB chat template).
        let template = unsafe {
            let p = ffi::llama_model_chat_template(self._model.as_ptr(), std::ptr::null());
            if p.is_null() {
                return None;
            }
            std::ffi::CStr::from_ptr(p).to_str().ok()?.to_owned()
        };

        let bos = {
            let id = unsafe { ffi::llama_vocab_bos(self.vocab) };
            self.token_to_piece_impl(id).unwrap_or_default()
        };
        let eos = self.token_to_piece_impl(self.eos_token).unwrap_or_default();

        let msgs: Vec<minijinja::Value> = messages
            .iter()
            .map(|(role, content)| context! { role => role, content => content })
            .collect();

        let mut env = Environment::new();
        // Chat templates lean on Python string methods (.strip(), .split(), …).
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template_owned("chat", template).ok()?;
        let tmpl = env.get_template("chat").ok()?;

        tmpl.render(context! {
            messages => msgs,
            add_generation_prompt => true,
            enable_thinking => enable_thinking,
            bos_token => bos,
            eos_token => eos,
        })
        .ok()
    }

    /// Build the final prompt token ids for a chat request. Prefers the model's
    /// real Jinja template; falls back to llama.cpp's built-in format. Each path
    /// tokenizes with the flags appropriate to how it produced the string.
    pub(super) fn build_prompt_tokens_impl(
        &self,
        messages: &[(String, String)],
        enable_thinking: bool,
    ) -> Result<Vec<i32>> {
        if let Some(rendered) = self.render_chat_jinja(messages, enable_thinking) {
            tracing::debug!(
                chars = rendered.len(),
                enable_thinking,
                "chat prompt: rendered via model Jinja template"
            );
            return self.tokenize_prompt_impl(&rendered);
        }
        tracing::debug!("chat prompt: fell back to llama.cpp built-in template");
        // Fallback: built-in format + manual <think> prefill, tokenized the legacy way.
        let mut prompt = self.apply_chat_template_impl(messages)?;
        if enable_thinking {
            prompt.push_str("<think>\n");
        }
        self.tokenize_impl(&prompt)
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
