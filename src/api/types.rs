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
}

fn default_max_tokens() -> Option<u32> {
    Some(256)
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
}
