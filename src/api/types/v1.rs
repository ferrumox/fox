use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::shared::{default_max_tokens, deserialize_stop, Usage};
use super::tools::{ResponseFormat, StreamOptions, Tool, ToolCall, ToolCallDelta};

// ---------------------------------------------------------------------------
// Message content (string or array of content blocks — OpenAI multimodal spec)
// ---------------------------------------------------------------------------

/// A single content block inside a message content array.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ContentBlock {
    /// "text" | "image_url" | "input_audio" etc.
    #[serde(rename = "type")]
    pub block_type: String,
    /// Present when type == "text".
    #[serde(default)]
    pub text: Option<String>,
    // image_url/audio blocks are accepted but not processed (no vision support).
}

/// OpenAI-compatible message content: either a plain string or an array of blocks.
#[derive(Debug, Clone)]
pub enum MessageContent {
    Text(String),
    Array(Vec<ContentBlock>),
}

impl MessageContent {
    /// Number of non-text blocks (`image_url`, `input_audio`, …) that `as_text`
    /// drops. fox has no vision/audio support, so callers can warn on these
    /// instead of dropping them silently.
    pub fn non_text_blocks(&self) -> usize {
        match self {
            MessageContent::Text(_) => 0,
            MessageContent::Array(blocks) => {
                blocks.iter().filter(|b| b.block_type != "text").count()
            }
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

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub stream: bool,
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
