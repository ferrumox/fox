mod embeddings;
mod ollama;
mod pull;
mod shared;
mod tools;
mod v1;

pub use embeddings::*;
pub use ollama::*;
pub use pull::*;
pub use shared::{HealthResponse, Usage, VersionResponse, DEFAULT_MAX_TOKENS};
pub use tools::*;
pub use v1::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_message_deserialize() {
        let json = r#"{"role":"user","content":"Hello"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "user");
        assert_eq!(
            msg.content.as_ref().map(|c| c.as_text()).as_deref(),
            Some("Hello")
        );
    }

    #[test]
    fn test_chat_message_null_content() {
        let json = r#"{"role":"assistant","content":null}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
    }

    #[test]
    fn test_chat_message_tool_result() {
        let json = r#"{"role":"tool","tool_call_id":"call_abc","content":"Sunny, 22°C"}"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "tool");
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_abc"));
        assert_eq!(
            msg.content.as_ref().map(|c| c.as_text()).as_deref(),
            Some("Sunny, 22°C")
        );
    }

    #[test]
    fn test_chat_message_assistant_with_tool_calls() {
        let json = r#"{
            "role": "assistant",
            "content": null,
            "tool_calls": [{"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":"{}"}}]
        }"#;
        let msg: ChatMessage = serde_json::from_str(json).unwrap();
        assert_eq!(msg.role, "assistant");
        assert!(msg.content.is_none());
        let calls = msg.tool_calls.expect("should have tool_calls");
        assert_eq!(calls[0].function.name, "get_weather");
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
