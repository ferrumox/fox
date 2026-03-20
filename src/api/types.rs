// OpenAI-compatible request/response types.

use serde::{Deserialize, Deserializer, Serialize};

/// Deserialize the OpenAI `stop` field which can be either a string or an array of strings.
fn deserialize_stop<'de, D>(deserializer: D) -> Result<Option<Vec<String>>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StopField {
        Single(String),
        Multiple(Vec<String>),
    }

    let opt = Option::<StopField>::deserialize(deserializer)?;
    Ok(opt.map(|v| match v {
        StopField::Single(s) => vec![s],
        StopField::Multiple(v) => v,
    }))
}

// --- Chat Completions ---

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<u32>,
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
    /// Accepts a single string or an array of strings.
    #[serde(default, deserialize_with = "deserialize_stop")]
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI function/tool definitions.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// "auto" | "none" | specific tool selector.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// Structured output format.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,
}

pub const DEFAULT_MAX_TOKENS: u32 = 256;

fn default_max_tokens() -> Option<u32> {
    Some(DEFAULT_MAX_TOKENS)
}

impl ChatCompletionRequest {
    /// Validate numeric fields. Called by request handlers before processing.
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

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessageResponse,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageResponse {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// --- Streaming ---

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
    /// Token usage — only present in the final chunk (when finish_reason is set).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

// --- Completions ---

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
}

// --- Ollama Compatibility ---

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

// --- Embeddings (OpenAI-compatible) ---

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

// --- Embeddings (Ollama-compatible) ---

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

// --- Pull (Ollama-compatible) ---

#[derive(Debug, Deserialize)]
pub struct PullRequest {
    pub name: String,
}

#[derive(Debug, Serialize)]
pub struct PullStatus {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

// --- Health ---

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub kv_cache_usage: f32,
    pub queue_depth: usize,
    pub active_requests: usize,
    pub model_name: String,
    pub started_at: u64,
}

// --- Version ---

#[derive(Debug, Serialize)]
pub struct VersionResponse {
    pub version: String,
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

// --- Function Calling (OpenAI tool use) ---

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments serialized as a JSON string (OpenAI spec).
    pub arguments: String,
}

// --- Structured Output ---

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseFormat {
    /// "json_object" | "text"
    #[serde(rename = "type")]
    pub format_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_deserialize() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
    }

    #[test]
    fn test_chat_completion_request_deserialize() {
        let json = r#"{"model":"llama","messages":[{"role":"user","content":"Hi"}],"stream":true}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama");
        assert!(req.stream);
    }

    #[test]
    fn test_chat_completion_request_with_tools() {
        let json = r#"{
            "model": "llama",
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather"}}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert!(req.tools.is_some());
        assert_eq!(req.tools.unwrap()[0].function.name, "get_weather");
    }

    #[test]
    fn test_ollama_generate_request_deserialize() {
        let json = r#"{"model":"llama3.2","prompt":"Why is the sky blue?","stream":false}"#;
        let req: OllamaGenerateRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3.2");
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn test_ollama_chat_request_deserialize() {
        let json = r#"{"model":"llama3.2","messages":[{"role":"user","content":"Hi"}]}"#;
        let req: OllamaChatRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "llama3.2");
        assert_eq!(req.messages[0].role, "user");
    }
}
