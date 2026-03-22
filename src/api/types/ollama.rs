use serde::{Deserialize, Serialize};

use super::embeddings::EmbeddingInput;
use super::shared::deserialize_stop;

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
    pub eval_count: Option<u32>,
}

// --- Ollama Chat (POST /api/chat) ---

/// Ollama chat message (used in both request and response).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
    /// Reasoning content from thinking models (Qwen3, DeepSeek-R1…).
    /// Matches the field name used by Ollama ≥0.7 for client compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub options: Option<OllamaOptions>,
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
    pub eval_count: Option<u32>,
}
