use serde::{Deserialize, Serialize};

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

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: ToolCallFunction,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCallFunction {
    pub name: String,
    /// Arguments serialized as a JSON string (OpenAI spec).
    pub arguments: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseFormat {
    /// "json_object" | "json_schema" | "text"
    #[serde(rename = "type")]
    pub format_type: String,
    /// Only present when type == "json_schema".
    #[serde(default)]
    pub json_schema: Option<JsonSchemaFormat>,
}

/// Schema definition for `response_format: { "type": "json_schema" }`.
#[derive(Debug, Deserialize, Clone)]
pub struct JsonSchemaFormat {
    pub name: String,
    #[serde(default)]
    pub strict: Option<bool>,
    #[serde(default)]
    pub schema: Option<serde_json::Value>,
}

/// `stream_options` field in chat completion requests.
#[derive(Debug, Deserialize, Clone)]
pub struct StreamOptions {
    /// When true, include token usage in the final streaming chunk.
    /// Fox always includes usage in the final chunk so this is accepted but has no extra effect.
    #[serde(default)]
    pub include_usage: Option<bool>,
}

/// Tool call delta used in streaming SSE responses.
#[derive(Debug, Serialize, Clone)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ToolCallFunctionDelta,
}

#[derive(Debug, Serialize, Clone)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}
