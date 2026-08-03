use serde::{Deserialize, Serialize};

use super::embeddings::EmbeddingInput;
use super::shared::deserialize_stop;
use super::tools::Tool;

// --- Ollama Management ---

#[derive(Debug, Serialize)]
pub struct OllamaDetails {
    pub format: String,
    pub family: String,
    pub parameter_size: String,
    pub quantization_level: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaModel {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaDetails,
    pub modified_at: String,
}

#[derive(Debug, Serialize)]
pub struct TagsResponse {
    pub models: Vec<OllamaModel>,
}

#[derive(Debug, Serialize)]
pub struct PsEntry {
    pub name: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaDetails,
    pub expires_at: String,
    pub size_vram: u64,
}

#[derive(Debug, Serialize)]
pub struct PsResponse {
    pub models: Vec<PsEntry>,
}

#[derive(Debug, Deserialize)]
pub struct ShowRequest {
    pub name: String,
    #[serde(default)]
    pub verbose: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ShowResponse {
    pub modelfile: String,
    pub parameters: String,
    pub template: String,
    pub details: OllamaDetails,
    pub model_info: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct DeleteRequest {
    pub name: String,
}

/// POST /api/copy — copy a model to a new name.
#[derive(Debug, Deserialize)]
pub struct CopyRequest {
    pub source: String,
    pub destination: String,
}

/// POST /api/create — create a model from a Modelfile.
#[derive(Debug, Deserialize)]
pub struct CreateRequest {
    pub model: String,
    #[serde(default)]
    pub modelfile: Option<String>,
    #[serde(default)]
    pub stream: Option<bool>,
}

// --- Ollama Embeddings ---

#[derive(Debug, Deserialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
}

// --- Ollama Tool Calls ---

/// A tool call emitted by the assistant in an Ollama response.
/// NOTE: `arguments` is a JSON *object* (not a JSON string like OpenAI).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaToolCall {
    pub function: OllamaToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaToolCallFunction {
    pub name: String,
    pub arguments: serde_json::Value,
}

// --- Ollama Generate (POST /api/generate) ---

/// Sampling options shared by /api/generate and /api/chat.
#[derive(Debug, Deserialize, Default)]
pub struct OllamaOptions {
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Min-P sampling threshold (mirrors upstream Ollama).
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    /// How far back `repeat_penalty` (and fox's frequency/presence penalties) look,
    /// in generated tokens: `-1` = whole history, `0` = disabled, `n` = last `n`.
    /// Mirrors upstream Ollama. Falls back to the server's `--repeat-last-n`.
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    /// Keep only tokens within N standard deviations of the top logit (0 = disabled).
    /// llama.cpp's `top_n_sigma`; upstream Ollama has no equivalent.
    #[serde(default)]
    pub top_n_sigma: Option<f32>,
    /// Floor on candidates left by any truncation step.
    #[serde(default)]
    pub min_keep: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Maximum tokens to generate (equivalent to max_tokens).
    #[serde(default)]
    pub num_predict: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,

    // ---------------------------------------------------------------------
    // Accepted for compatibility, but NOT honoured. Declared rather than left
    // to serde's silent drop so `unsupported_options()` can name them in a
    // warning: a caller who sets `mirostat` and sees no effect has no other
    // way to find out. Rejecting them outright would break clients that send
    // Ollama's defaults on every request.
    // ---------------------------------------------------------------------
    /// Per-request context size. fox sizes the context when the model is loaded,
    /// so honouring this would mean reloading mid-flight. Use `--max-context-len`.
    #[serde(default)]
    pub num_ctx: Option<u32>,
    /// Tokens preserved at the front on a context roll. fox has the knob, but as
    /// the server-wide `--context-keep`; it is not yet per-request.
    #[serde(default)]
    pub num_keep: Option<i32>,
    /// Locally typical sampling. Not implemented — see
    /// `docs/design/llama-server-gap-analysis.md` §3 for why it is not a
    /// one-liner against fox's adaptive candidate pool.
    #[serde(default)]
    pub typical_p: Option<f32>,
    /// Mirostat mode. Not implemented (needs per-request feedback state).
    #[serde(default)]
    pub mirostat: Option<i32>,
    #[serde(default)]
    pub mirostat_tau: Option<f32>,
    #[serde(default)]
    pub mirostat_eta: Option<f32>,
    /// Whether to apply the repeat penalty to newlines. Not implemented.
    #[serde(default)]
    pub penalize_newline: Option<bool>,
}

impl OllamaOptions {
    /// Names of the options this request set that fox will not act on, so the
    /// handler can say so once per request instead of silently ignoring them.
    pub fn unsupported_options(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.num_ctx.is_some() {
            out.push("num_ctx");
        }
        if self.num_keep.is_some() {
            out.push("num_keep");
        }
        if self.typical_p.is_some() {
            out.push("typical_p");
        }
        if self.mirostat.is_some() {
            out.push("mirostat");
        }
        if self.mirostat_tau.is_some() {
            out.push("mirostat_tau");
        }
        if self.mirostat_eta.is_some() {
            out.push("mirostat_eta");
        }
        if self.penalize_newline.is_some() {
            out.push("penalize_newline");
        }
        out
    }
}

/// Parse Ollama's `keep_alive`: a duration string (`"5m"`, `"30s"`, `"1h"`, `"500ms"`),
/// a bare number of seconds (string or JSON number), or a negative value meaning
/// "never evict".
///
/// Returns `None` when the value is absent or unparseable — an unparseable value falls
/// back to the server default rather than erroring, matching how Ollama treats it.
pub fn parse_keep_alive(v: Option<&serde_json::Value>) -> Option<KeepAlive> {
    let v = v?;
    if let Some(n) = v.as_f64() {
        return Some(KeepAlive::from_secs_f64(n));
    }
    let s = v.as_str()?.trim();
    if s.is_empty() {
        return None;
    }
    // Split the trailing unit off the number.
    let split = s
        .find(|c: char| !c.is_ascii_digit() && c != '-' && c != '+' && c != '.')
        .unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let n: f64 = num.parse().ok()?;
    let secs = match unit.trim() {
        "" | "s" => n,
        "ms" => n / 1000.0,
        "m" => n * 60.0,
        "h" => n * 3600.0,
        _ => return None,
    };
    Some(KeepAlive::from_secs_f64(secs))
}

/// How long to keep a model resident after this request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepAlive {
    /// Unload as soon as the request finishes (Ollama's `keep_alive: 0`).
    Immediate,
    /// Never evict on a timer (Ollama's negative values).
    Forever,
    /// Evict after this many idle seconds.
    Secs(u64),
}

impl KeepAlive {
    fn from_secs_f64(n: f64) -> Self {
        if n < 0.0 {
            KeepAlive::Forever
        } else if n == 0.0 {
            KeepAlive::Immediate
        } else {
            KeepAlive::Secs(n.round().max(1.0) as u64)
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub system: Option<String>,
    /// true = stream tokens as NDJSON (default), false = wait for full response.
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    /// "json" (string) or a JSON Schema object for structured output.
    #[serde(default)]
    pub format: Option<serde_json::Value>,
    /// How long to keep the model resident after this request: a duration string
    /// (`"5m"`), a number of seconds, `0` to unload immediately, or a negative value
    /// to pin it. Ollama sends both strings and numbers here, so this is a raw
    /// `Value` parsed by [`parse_keep_alive`].
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
    /// Base64-encoded images (no data URI prefix, per Ollama's wire format) —
    /// encoded via mtmd when the loaded model has a paired mmproj
    /// (`entry.engine.supports_vision()`), otherwise dropped with a warning.
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

/// A single token event in the /api/generate stream.
#[derive(Debug, Serialize)]
pub struct OllamaGenerateChunk {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

// --- Ollama Chat (POST /api/chat) ---

/// Ollama chat message (used in both request and response).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    /// Empty string (not null) when tool_calls is present — Ollama convention.
    #[serde(default)]
    pub content: String,
    /// Reasoning content from thinking models (Qwen3, DeepSeek-R1…).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Tool calls made by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
    /// For role=="tool" messages: id matching the assistant's prior tool_call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Base64-encoded images (no data URI prefix, per Ollama's wire format) —
    /// encoded via mtmd when the loaded model has a paired mmproj
    /// (`entry.engine.supports_vision()`), otherwise dropped with a warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
    /// Tool definitions (same structure as OpenAI).
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// "json" (string) or a JSON Schema object for structured output.
    #[serde(default)]
    pub format: Option<serde_json::Value>,
    /// How long to keep the model resident after this request — see
    /// [`parse_keep_alive`]. Accepts a duration string or a number.
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
    /// Enable thinking/reasoning. Can be bool or string ("high", "medium", "low").
    #[serde(default)]
    pub think: Option<serde_json::Value>,
}

/// A single message event in the /api/chat stream.
#[derive(Debug, Serialize)]
pub struct OllamaChatChunk {
    pub model: String,
    pub created_at: String,
    pub message: OllamaChatMessage,
    pub done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keep_alive_parses_durations_and_numbers() {
        let ka = |v: serde_json::Value| parse_keep_alive(Some(&v));
        assert_eq!(ka(json!("5m")), Some(KeepAlive::Secs(300)));
        assert_eq!(ka(json!("30s")), Some(KeepAlive::Secs(30)));
        assert_eq!(ka(json!("1h")), Some(KeepAlive::Secs(3600)));
        assert_eq!(
            ka(json!("500ms")),
            Some(KeepAlive::Secs(1)),
            "sub-second rounds up to 1s, never to 0"
        );
        // Ollama sends bare numbers as well as strings.
        assert_eq!(ka(json!(120)), Some(KeepAlive::Secs(120)));
        assert_eq!(ka(json!("120")), Some(KeepAlive::Secs(120)));
    }

    #[test]
    fn keep_alive_zero_and_negative_are_distinct() {
        let ka = |v: serde_json::Value| parse_keep_alive(Some(&v));
        assert_eq!(ka(json!(0)), Some(KeepAlive::Immediate));
        assert_eq!(ka(json!("0")), Some(KeepAlive::Immediate));
        assert_eq!(ka(json!(-1)), Some(KeepAlive::Forever));
        assert_eq!(ka(json!("-1m")), Some(KeepAlive::Forever));
    }

    #[test]
    fn keep_alive_absent_or_junk_falls_back_to_the_server_default() {
        assert_eq!(parse_keep_alive(None), None);
        assert_eq!(parse_keep_alive(Some(&json!(""))), None);
        assert_eq!(parse_keep_alive(Some(&json!("later"))), None);
        assert_eq!(parse_keep_alive(Some(&json!("5 fortnights"))), None);
    }

    #[test]
    fn unsupported_options_names_exactly_what_is_ignored() {
        let opts = OllamaOptions {
            temperature: Some(0.5),
            num_ctx: Some(4096),
            mirostat: Some(2),
            ..Default::default()
        };
        assert_eq!(opts.unsupported_options(), vec!["num_ctx", "mirostat"]);
        // A request using only supported options must not produce a warning.
        let clean = OllamaOptions {
            temperature: Some(0.5),
            top_k: Some(40),
            ..Default::default()
        };
        assert!(clean.unsupported_options().is_empty());
    }
}
