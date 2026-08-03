use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::shared::{default_max_tokens, deserialize_stop, Usage};
use super::tools::{ResponseFormat, StreamOptions, Tool, ToolCall, ToolCallDelta};

// ---------------------------------------------------------------------------
// Message content (string or array of content blocks — OpenAI multimodal spec)
// ---------------------------------------------------------------------------

/// OpenAI `image_url` content block payload.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ImageUrl {
    pub url: String,
}

/// A single content block inside a message content array.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentBlock {
    /// "text" | "image_url" | "input_audio" etc.
    #[serde(rename = "type")]
    pub block_type: String,
    /// Present when type == "text".
    #[serde(default)]
    pub text: Option<String>,
    /// Present when type == "image_url". Only `data:` base64 URIs are supported —
    /// fox has no outbound HTTP fetch path for remote image URLs (avoids an SSRF
    /// surface and matches fox's local-first posture).
    #[serde(default)]
    pub image_url: Option<ImageUrl>,
    // input_audio and other block types are accepted but not processed.
}

/// Decode an OpenAI `image_url.url` data URI (`data:image/png;base64,...`) into raw
/// image bytes. Returns a human-readable error for anything else (remote
/// `http(s)://` URLs, malformed/non-base64 data URIs) so the caller can reject the
/// request with a clear `400` instead of silently dropping the image.
fn decode_data_uri_image(url: &str) -> Result<Vec<u8>, String> {
    let rest = url.strip_prefix("data:").ok_or_else(|| {
        "image_url must be a base64 data URI (data:<mime>;base64,<data>) — fox does not fetch \
         remote image URLs"
            .to_string()
    })?;
    let (meta, b64) = rest
        .split_once(',')
        .ok_or_else(|| "malformed data URI: missing ','".to_string())?;
    if !meta.contains("base64") {
        return Err("data URI must be base64-encoded (;base64,...)".to_string());
    }
    STANDARD
        .decode(b64)
        .map_err(|e| format!("invalid base64 image data: {e}"))
}

/// OpenAI-compatible message content: either a plain string or an array of blocks.
#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    Array(Vec<ContentBlock>),
}

impl MessageContent {
    /// Number of non-text, non-image blocks (`input_audio`, …) that `as_text`
    /// drops. fox has no audio support, so callers can warn on these instead of
    /// dropping them silently. `image_url` is excluded — whether it's dropped
    /// depends on the model's vision support, checked separately via
    /// [`Self::has_images`].
    pub fn non_text_blocks(&self) -> usize {
        match self {
            MessageContent::Text(_) => 0,
            MessageContent::Array(blocks) => blocks
                .iter()
                .filter(|b| b.block_type != "text" && b.block_type != "image_url")
                .count(),
        }
    }

    /// Whether this content carries any `image_url` block.
    pub fn has_images(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Array(blocks) => blocks.iter().any(|b| b.block_type == "image_url"),
        }
    }

    /// Extract all text from the content, joining text blocks with a newline.
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Array(blocks) => blocks
                .iter()
                .filter_map(|b| {
                    if b.block_type == "text" {
                        b.text.as_deref()
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Like `as_text`, but keeps image content: each `image_url` block is replaced
    /// with `marker` (in place, preserving block order) and its decoded bytes are
    /// returned alongside. Mirrors llama.cpp server's own approach of flattening
    /// content parts into one marker-laced string *before* the chat template runs,
    /// rather than teaching every Jinja template about images.
    ///
    /// Errors if an `image_url` block can't be decoded (remote URL, malformed data
    /// URI) — the caller should reject the request rather than silently drop it.
    pub fn as_text_with_media_marker(
        &self,
        marker: &str,
    ) -> Result<(String, Vec<Vec<u8>>), String> {
        match self {
            MessageContent::Text(s) => Ok((s.clone(), Vec::new())),
            MessageContent::Array(blocks) => {
                let mut text = String::new();
                let mut images = Vec::new();
                for b in blocks {
                    match b.block_type.as_str() {
                        "text" => {
                            if let Some(t) = b.text.as_deref() {
                                text.push_str(t);
                            }
                        }
                        "image_url" => {
                            let url = b.image_url.as_ref().ok_or_else(|| {
                                "image_url block missing image_url field".to_string()
                            })?;
                            images.push(decode_data_uri_image(&url.url)?);
                            text.push_str(marker);
                        }
                        _ => {} // input_audio and other block types: still dropped
                    }
                }
                Ok((text, images))
            }
        }
    }
}

impl<'de> Deserialize<'de> for MessageContent {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Text(String),
            Array(Vec<ContentBlock>),
        }
        match Raw::deserialize(d)? {
            Raw::Text(s) => Ok(MessageContent::Text(s)),
            Raw::Array(v) => Ok(MessageContent::Array(v)),
        }
    }
}

impl Serialize for MessageContent {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            MessageContent::Text(t) => s.serialize_str(t),
            MessageContent::Array(v) => v.serialize(s),
        }
    }
}

// ---------------------------------------------------------------------------
// Chat Completions
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
    /// Alias for max_tokens (newer OpenAI API name). Takes effect when max_tokens is absent.
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Top-K filter: keep only the K most likely tokens before nucleus sampling (0 = disabled).
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Penalize tokens that already appear in the output (1.0 = disabled).
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// fox extension mirroring llama-server's `grammar`: a raw GBNF grammar
    /// constraining generation. More expressive than `response_format`, which can
    /// only describe JSON. Mutually exclusive with it.
    #[serde(default)]
    pub grammar: Option<String>,
    /// fox extension: keep only tokens whose logit is within N standard deviations
    /// of the top logit (0 = disabled). Unlike `top_p`, the cutoff is on the logit
    /// scale, so it does not shift as `temperature` changes.
    #[serde(default)]
    pub top_n_sigma: Option<f32>,
    /// fox extension: floor on how few candidates any truncation step (`min_p`,
    /// `top_p`, `top_n_sigma`) may leave.
    #[serde(default)]
    pub min_keep: Option<usize>,
    /// fox extension (OpenAI has no equivalent): how far back the repetition,
    /// frequency and presence penalties look, in generated tokens.
    /// `-1` = whole history, `0` = disabled, `n` = last `n`. Falls back to the
    /// server's `--repeat-last-n`.
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    /// Fixed RNG seed for reproducible outputs.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stop generation when output ends with any of these strings (OpenAI-compatible).
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI function/tool definitions.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// "auto" | "none" | "required" | specific tool selector.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// When false, limit tool calls in a single turn to at most one.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// Structured output format.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
    /// Return per-token log-probabilities when true.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of most-likely alternatives to return per token (0–20). Only meaningful
    /// with `logprobs: true`.
    #[serde(default)]
    pub top_logprobs: Option<u8>,
    /// Additive per-token logit bias, keyed by token id (as a string, OpenAI style).
    /// Values around ±100 effectively force or ban a token.
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// Min-P sampling: drop tokens below `min_p × max_prob` (fox extension).
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Suppress end-of-generation until at least this many tokens are produced (fox
    /// extension).
    #[serde(default)]
    pub min_tokens: Option<usize>,
    /// fox extension: opt in to the model's native reasoning/thinking when it
    /// supports it. Default off — thinking is NOT enabled unless requested.
    #[serde(default)]
    pub think: Option<bool>,
    /// Options for the streaming response (include_usage etc.).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Frequency penalty — accepted, not applied (no llama.cpp equivalent).
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty — accepted, not applied.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Caller identifier — accepted for API compatibility, not used.
    #[serde(default)]
    pub user: Option<String>,
    /// Number of completion choices to return (1–8). Each choice is an
    /// independent generation over the same prompt, not a shared-prefill fork
    /// — see `docs/design/n-best-of-support.md`.
    #[serde(default)]
    pub n: Option<u32>,
    /// Generate this many candidates server-side and return only the `n` with
    /// the highest total log-likelihood. Must be >= `n`; incompatible with
    /// `stream: true`.
    #[serde(default)]
    pub best_of: Option<u32>,
}

impl ChatCompletionRequest {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(t) = self.temperature {
            if t < 0.0 {
                return Err(format!("temperature must be >= 0, got {t}"));
            }
        }
        if let Some(p) = self.top_p {
            if !(0.0..=1.0).contains(&p) {
                return Err(format!("top_p must be in [0, 1], got {p}"));
            }
        }
        if let Some(r) = self.repetition_penalty {
            if r < 0.0 {
                return Err(format!("repetition_penalty must be >= 0, got {r}"));
            }
        }
        if let Some(n) = self.n {
            if !(1..=crate::api::shared::sampling_defaults::openai::MAX_N).contains(&n) {
                return Err(format!(
                    "n must be in [1, {}], got {n}",
                    crate::api::shared::sampling_defaults::openai::MAX_N
                ));
            }
        }
        if let Some(best_of) = self.best_of {
            if !(1..=crate::api::shared::sampling_defaults::openai::MAX_N).contains(&best_of) {
                return Err(format!(
                    "best_of must be in [1, {}], got {best_of}",
                    crate::api::shared::sampling_defaults::openai::MAX_N
                ));
            }
            let n = self.n.unwrap_or(1);
            if best_of < n {
                return Err(format!("best_of ({best_of}) must be >= n ({n})"));
            }
            if best_of > n && self.stream {
                return Err("best_of > n is not supported with stream: true".to_string());
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// May be absent/null when role=="assistant" and tool_calls is present.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// For role=="tool": identifies which tool_call this result responds to.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// For role=="assistant" in multi-turn history: prior tool calls.
    #[serde(default)]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional sender name.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<Usage>,
    /// Always null — included for client compatibility.
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessageResponse,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// OpenAI `logprobs` block: one entry per generated token.
#[derive(Debug, Serialize)]
pub struct ChatLogprobs {
    pub content: Vec<ChatLogprobEntry>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatLogprobEntry {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<ChatTopLogprob>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ChatTopLogprob {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

impl From<crate::scheduler::TokenLogprob> for ChatLogprobEntry {
    fn from(l: crate::scheduler::TokenLogprob) -> Self {
        ChatLogprobEntry {
            token: l.token,
            logprob: l.logprob,
            bytes: l.bytes,
            top_logprobs: l
                .top
                .into_iter()
                .map(|t| ChatTopLogprob {
                    token: t.token,
                    logprob: t.logprob,
                    bytes: t.bytes,
                })
                .collect(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    /// Null when finish_reason=="tool_calls" (OpenAI spec).
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// --- Streaming ---

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// Token usage — only present in the final chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Always null — included for client compatibility.
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

// --- Completions (legacy) ---

/// OpenAI's legacy `/v1/completions` request.
///
/// Every sampling field here is threaded through to the shared generation path —
/// they were previously declared nowhere and hard-coded to `None` in the handler,
/// so `top_p`, `stop`, `seed`, `logit_bias` and the penalties silently did nothing
/// on this endpoint.
#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub repeat_last_n: Option<i32>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub top_n_sigma: Option<f32>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub min_keep: Option<usize>,
    /// fox extension, as on `/v1/chat/completions`.
    #[serde(default)]
    pub min_p: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// fox extension: raw GBNF grammar, as on `/v1/chat/completions`.
    #[serde(default)]
    pub grammar: Option<String>,
    /// Legacy semantics: an *integer* count of alternatives per token, unlike chat's
    /// boolean `logprobs` + `top_logprobs` pair. Mapped onto the latter.
    #[serde(default)]
    pub logprobs: Option<u8>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub best_of: Option<u32>,
    /// Accepted so the handler can reject it explicitly — fox cannot echo the prompt,
    /// and silently ignoring it would be undetectable by the caller.
    /// and silently ignoring it would be undetectable by the caller.
    #[serde(default)]
    pub echo: Option<bool>,
    /// Same: accepted only to be rejected explicitly.
    #[serde(default)]
    pub suffix: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub index: u32,
    pub text: String,
    pub finish_reason: Option<String>,
}

// --- Models ---

#[derive(Debug, Serialize)]
pub struct ModelsResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat_req(extra: &str) -> ChatCompletionRequest {
        let json =
            format!(r#"{{"model":"m","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#);
        serde_json::from_str(&json).unwrap()
    }

    #[test]
    fn validate_n_within_bounds_ok() {
        assert!(chat_req(r#","n":8"#).validate().is_ok());
        assert!(chat_req(r#","n":1"#).validate().is_ok());
    }

    #[test]
    fn validate_n_zero_rejected() {
        assert!(chat_req(r#","n":0"#).validate().is_err());
    }

    #[test]
    fn validate_n_over_max_rejected() {
        assert!(chat_req(r#","n":9"#).validate().is_err());
    }

    #[test]
    fn validate_best_of_within_bounds_ok() {
        assert!(chat_req(r#","n":2,"best_of":4"#).validate().is_ok());
    }

    #[test]
    fn validate_best_of_less_than_n_rejected() {
        let err = chat_req(r#","n":4,"best_of":2"#).validate().unwrap_err();
        assert!(err.contains("best_of"));
    }

    #[test]
    fn validate_best_of_over_max_rejected() {
        assert!(chat_req(r#","best_of":9"#).validate().is_err());
    }

    #[test]
    fn validate_best_of_greater_than_n_with_stream_rejected() {
        let err = chat_req(r#","n":1,"best_of":3,"stream":true"#)
            .validate()
            .unwrap_err();
        assert!(err.contains("stream"));
    }

    #[test]
    fn validate_best_of_greater_than_n_without_stream_ok() {
        assert!(chat_req(r#","n":1,"best_of":3"#).validate().is_ok());
    }

    #[test]
    fn validate_best_of_equal_to_n_with_stream_ok() {
        assert!(chat_req(r#","n":2,"best_of":2,"stream":true"#)
            .validate()
            .is_ok());
    }

    fn text_block(s: &str) -> ContentBlock {
        ContentBlock {
            block_type: "text".to_string(),
            text: Some(s.to_string()),
            image_url: None,
        }
    }

    fn image_block(data_uri: &str) -> ContentBlock {
        ContentBlock {
            block_type: "image_url".to_string(),
            text: None,
            image_url: Some(ImageUrl {
                url: data_uri.to_string(),
            }),
        }
    }

    #[test]
    fn has_images_detects_image_url_blocks() {
        let none = MessageContent::Array(vec![text_block("hi")]);
        assert!(!none.has_images());
        let one = MessageContent::Array(vec![
            text_block("hi"),
            image_block("data:image/png;base64,AAAA"),
        ]);
        assert!(one.has_images());
        assert!(!MessageContent::Text("hi".to_string()).has_images());
    }

    #[test]
    fn non_text_blocks_excludes_images_but_counts_audio() {
        let content = MessageContent::Array(vec![
            text_block("hi"),
            image_block("data:image/png;base64,AAAA"),
            ContentBlock {
                block_type: "input_audio".to_string(),
                text: None,
                image_url: None,
            },
        ]);
        // image_url is not "dropped" the way audio is — has_images() is the
        // signal callers should use for it instead.
        assert_eq!(content.non_text_blocks(), 1);
    }

    #[test]
    fn as_text_with_media_marker_preserves_order_and_collects_images() {
        // 1x1 fully-transparent PNG, a real (tiny) base64 payload — decode must succeed.
        let png_b64 =
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
        let content = MessageContent::Array(vec![
            text_block("Look at this: "),
            image_block(&format!("data:image/png;base64,{png_b64}")),
            text_block(" what is it?"),
        ]);
        let (text, images) = content.as_text_with_media_marker("<MARK>").unwrap();
        assert_eq!(text, "Look at this: <MARK> what is it?");
        assert_eq!(images.len(), 1);
        assert!(!images[0].is_empty());
    }

    #[test]
    fn as_text_with_media_marker_rejects_remote_url() {
        let content = MessageContent::Array(vec![image_block("https://example.com/cat.png")]);
        let err = content
            .as_text_with_media_marker("<MARK>")
            .expect_err("remote URLs must be rejected, not silently dropped");
        assert!(err.contains("data URI"), "unexpected error message: {err}");
    }

    #[test]
    fn as_text_with_media_marker_rejects_malformed_data_uri() {
        let content = MessageContent::Array(vec![image_block("data:image/png;base64")]);
        assert!(content.as_text_with_media_marker("<MARK>").is_err());
    }

    #[test]
    fn as_text_with_media_marker_plain_text_has_no_images() {
        let content = MessageContent::Text("just text".to_string());
        let (text, images) = content.as_text_with_media_marker("<MARK>").unwrap();
        assert_eq!(text, "just text");
        assert!(images.is_empty());
    }
}
