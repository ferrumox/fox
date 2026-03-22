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
    #[serde(default)]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Maximum tokens to generate (equivalent to max_tokens).
    #[serde(default)]
    pub num_predict: Option<u32>,
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
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
    /// How long to keep the model loaded (e.g. "5m", "0" to unload immediately).
    #[serde(default)]
    pub keep_alive: Option<String>,
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
    /// How long to keep the model loaded.
    #[serde(default)]
    pub keep_alive: Option<String>,
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
