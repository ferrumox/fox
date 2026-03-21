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

#[derive(Debug, Deserialize, Clone)]
pub struct ResponseFormat {
    /// "json_object" | "text"
    #[serde(rename = "type")]
    pub format_type: String,
}
