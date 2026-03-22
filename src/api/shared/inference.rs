// Prompt preparation and inference parameter helpers.

use uuid::Uuid;

use crate::model_registry::EngineEntry;
use crate::scheduler::SamplingParams;

use crate::api::types::{OllamaOptions, ResponseFormat, Tool, ToolCall, ToolCallFunction};

// ---------------------------------------------------------------------------
// Message representation for template rendering
// ---------------------------------------------------------------------------

/// A chat message carrying all fields needed for prompt building.
///
/// This is an internal type — not part of the public API. Callers convert their
/// own message types (OpenAI `ChatMessage`, Ollama `OllamaChatMessage`) into
/// `MessageForTemplate` before calling [`prepare_prompt`].
pub struct MessageForTemplate {
    pub role: String,
    pub content: Option<String>,
    /// Tool calls made by the assistant (present in multi-turn assistant messages).
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `role == "tool"` messages: id matching the prior assistant tool_call.
    pub tool_call_id: Option<String>,
}

/// Flatten a [`MessageForTemplate`] into a `(role, content)` tuple suitable
/// for `apply_chat_template`, which speaks the llama.cpp FFI `(role, content)` format.
///
/// * `role == "tool"` → content is the tool's return value (passed through).
/// * `role == "assistant"` with tool_calls → tool calls are serialised into content.
/// * All other roles → content is used as-is (empty string when `None`).
fn flatten_message_for_template(msg: &MessageForTemplate) -> (String, String) {
    match msg.role.as_str() {
        "assistant" if msg.tool_calls.is_some() => {
            let calls = msg.tool_calls.as_ref().unwrap();
            let serialized = calls
                .iter()
                .map(|tc| format!("[tool_call: {}({})]", tc.function.name, tc.function.arguments))
                .collect::<Vec<_>>()
                .join(" ");
            let content = match &msg.content {
                Some(c) if !c.is_empty() => format!("{c}\n{serialized}"),
                _ => serialized,
            };
            (msg.role.clone(), content)
        }
        _ => {
            let content = msg.content.as_deref().unwrap_or("").to_string();
            (msg.role.clone(), content)
        }
    }
}

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

/// Build `SamplingParams` + `max_tokens` from Ollama-style options.
///
/// `show_thinking` should be set to `engine.supports_thinking()` by the caller
/// so that thinking tokens are forwarded only when the model actually supports them.
pub fn sampling_from_ollama(
    opts: Option<&OllamaOptions>,
    show_thinking: bool,
) -> (SamplingParams, usize) {
    let (temp, top_p, top_k, rep, seed, max_tokens, stop) = match opts {
        Some(o) => (
            o.temperature.unwrap_or(0.8),
            o.top_p.unwrap_or(0.9),
            o.top_k.unwrap_or(40),
            o.repeat_penalty.unwrap_or(1.1),
            o.seed,
            o.num_predict.unwrap_or(128) as usize,
            o.stop.clone(),
        ),
        None => (0.8, 0.9, 40, 1.1, None, 128, None),
    };
    (
        SamplingParams {
            temperature: temp,
            top_p,
            top_k,
            repetition_penalty: rep,
            seed,
            stop,
            show_thinking,
            initial_in_thinking: show_thinking,
            max_thinking_chars: 8192,
        },
        max_tokens,
    )
}

// ---------------------------------------------------------------------------
// Thinking extraction
// ---------------------------------------------------------------------------

/// Split a model response that may contain `<think>…</think>` into
/// `(thinking_content, visible_content)`.
///
/// Used by Ollama-compatible endpoints to populate the separate `thinking`
/// field that Ollama ≥0.7 exposes for reasoning models.
pub fn extract_thinking(text: &str) -> (Option<String>, String) {
    let end_tag = "</think>";
    if let Some(end) = text.find(end_tag) {
        let think_start = text.find("<think>").map(|i| i + "<think>".len()).unwrap_or(0);
        let thinking = text[think_start..end].trim().to_string();
        let content = text[end + end_tag.len()..].trim().to_string();
        let thinking = if thinking.is_empty() { None } else { Some(thinking) };
        return (thinking, content);
    }
    (None, text.to_string())
}

// ---------------------------------------------------------------------------
// Tool call helpers
// ---------------------------------------------------------------------------

/// Build a system message that describes the available tools to the model.
pub fn tools_system_message(tools: &[Tool]) -> String {
    let json = serde_json::to_string_pretty(tools).unwrap_or_default();
    format!(
        "You have access to the following tools:\n{json}\n\n\
         When you want to call a tool, respond ONLY with a JSON object:\n\
         {{\"name\": \"<tool_name>\", \"arguments\": {{<key>: <value>}}}}\n\n\
         If you don't need a tool, respond normally."
    )
}

/// Try to parse `response` text as a JSON tool call.
/// Returns `(content, tool_calls)` — when a tool call is detected, `content` is empty.
pub fn try_parse_tool_call(response: &str) -> (String, Option<Vec<ToolCall>>) {
    let trimmed = response.trim();
    let value: serde_json::Value = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(_) => return (response.to_string(), None),
    };

    // Pattern: {"name": "...", "arguments": {...}}
    if let (Some(name), Some(args)) = (
        value.get("name").and_then(|n| n.as_str()),
        value.get("arguments"),
    ) {
        let call = ToolCall {
            id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
            call_type: "function".to_string(),
            function: ToolCallFunction {
                name: name.to_string(),
                arguments: args.to_string(),
            },
        };
        return (String::new(), Some(vec![call]));
    }

    // Pattern: {"tool_calls": [...]}
    if let Some(calls) = value.get("tool_calls").and_then(|tc| tc.as_array()) {
        let tool_calls: Vec<ToolCall> = calls
            .iter()
            .filter_map(|c| {
                let name = c.get("name")?.as_str()?.to_string();
                let args = c.get("arguments")?.to_string();
                Some(ToolCall {
                    id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
                    call_type: "function".to_string(),
                    function: ToolCallFunction {
                        name,
                        arguments: args,
                    },
                })
            })
            .collect();
        if !tool_calls.is_empty() {
            return (String::new(), Some(tool_calls));
        }
    }

    (response.to_string(), None)
}

// ---------------------------------------------------------------------------
// Prompt preparation
// ---------------------------------------------------------------------------

/// Build the JSON mode system instruction string.
fn json_mode_instruction(response_format: Option<&ResponseFormat>) -> Option<String> {
    let rf = response_format?;
    match rf.format_type.as_str() {
        "json_object" => Some(
            "Respond ONLY with valid JSON. Do not include any explanation or markdown.".to_string(),
        ),
        "json_schema" => {
            let schema_hint = rf
                .json_schema
                .as_ref()
                .and_then(|js| js.schema.as_ref())
                .and_then(|s| serde_json::to_string(s).ok())
                .unwrap_or_default();
            if schema_hint.is_empty() {
                Some(
                    "Respond ONLY with valid JSON. Do not include any explanation or markdown."
                        .to_string(),
                )
            } else {
                Some(format!(
                    "Respond ONLY with valid JSON matching this schema: {schema_hint}. \
                     Do not include any explanation or markdown."
                ))
            }
        }
        _ => None,
    }
}

/// Inject system messages (system prompt, tools, JSON mode), apply the chat
/// template, and tokenise the result.
///
/// When `show_thinking` is true the opening `<think>\n` tag is appended to the
/// rendered prompt so the model enters reasoning mode immediately.
///
/// Returns `(tokens, token_count)`.
pub fn prepare_prompt(
    entry: &EngineEntry,
    mut messages: Vec<MessageForTemplate>,
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
    response_format: Option<&ResponseFormat>,
    show_thinking: bool,
) -> (Vec<i32>, usize) {
    // Inject server-level system prompt when none is already present.
    if let Some(sp) = system_prompt {
        if messages.first().map(|m| m.role.as_str()) != Some("system") {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(sp.to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
    }

    // Append tool descriptions to the system message (or create one).
    if let Some(tools) = tools {
        let tool_msg = tools_system_message(tools);
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys_content = messages[0].content.get_or_insert_with(String::new);
            sys_content.push_str(&format!("\n\n{tool_msg}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(tool_msg),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
    }

    // Inject JSON-mode instruction.
    if let Some(instr) = json_mode_instruction(response_format) {
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys_content = messages[0].content.get_or_insert_with(String::new);
            sys_content.push_str(&format!("\n\n{instr}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(instr),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }
    }

    // Flatten to (role, content) pairs for apply_chat_template (FFI boundary).
    let flat: Vec<(String, String)> = messages.iter().map(flatten_message_for_template).collect();

    let mut prompt = entry
        .engine
        .apply_chat_template(&flat)
        .unwrap_or_else(|_| {
            flat.iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        });

    // For reasoning models (Qwen3, DeepSeek-R1…) append the opening <think> tag
    // so the model enters reasoning mode. Mirrors the behaviour of `fox run --show-thinking`.
    if show_thinking {
        prompt.push_str("<think>\n");
    }

    let tokens: Vec<i32> = entry.engine.tokenize(&prompt).unwrap_or_else(|_| {
        if prompt.is_empty() {
            vec![0]
        } else {
            prompt.bytes().map(|b| b as i32).take(4096).collect()
        }
    });

    let len = tokens.len();
    (tokens, len)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_from_ollama_defaults() {
        let (params, max_tokens) = sampling_from_ollama(None, false);
        assert_eq!(max_tokens, 128);
        assert!((params.temperature - 0.8).abs() < f32::EPSILON);
        assert!((params.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(params.top_k, 40);
    }

    #[test]
    fn test_sampling_from_ollama_custom() {
        use crate::api::types::OllamaOptions;
        let opts = OllamaOptions {
            temperature: Some(0.3),
            top_p: Some(0.5),
            top_k: Some(10),
            repeat_penalty: Some(1.2),
            seed: Some(42),
            num_predict: Some(64),
            stop: None,
        };
        let (params, max_tokens) = sampling_from_ollama(Some(&opts), false);
        assert_eq!(max_tokens, 64);
        assert!((params.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(params.seed, Some(42));
        assert_eq!(params.top_k, 10);
    }

    #[test]
    fn test_try_parse_tool_call_valid_single() {
        let response = r#"{"name":"get_weather","arguments":{"city":"Madrid"}}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
        assert_eq!(calls[0].call_type, "function");
        assert!(calls[0].id.starts_with("call_"));
    }

    #[test]
    fn test_try_parse_tool_call_plain_text() {
        let response = "The sky is blue because of Rayleigh scattering.";
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_invalid_json() {
        let response = "not { json at all }";
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_json_no_name() {
        let response = r#"{"answer": "42"}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_tool_calls_array() {
        let response = r#"{"tool_calls":[{"name":"search","arguments":{"query":"rust"}}]}"#;
        let (content, calls) = try_parse_tool_call(response);
        assert!(content.is_empty());
        let calls = calls.expect("should detect tool_calls array");
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_try_parse_tool_call_whitespace_trimmed() {
        let response = "  \n{\"name\":\"foo\",\"arguments\":{}}\n  ";
        let (_content, calls) = try_parse_tool_call(response);
        assert!(calls.is_some());
    }

    #[test]
    fn test_tools_system_message_contains_tool_name() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: Some("Get current weather".to_string()),
                parameters: None,
            },
        }];
        let msg = tools_system_message(&tools);
        assert!(msg.contains("get_weather"));
        assert!(msg.contains("tool"));
        assert!(msg.contains("JSON"));
    }

    #[test]
    fn test_tools_system_message_empty_tools() {
        let msg = tools_system_message(&[]);
        assert!(msg.contains("JSON"));
    }

    #[test]
    fn test_flatten_message_assistant_with_tool_calls() {
        use crate::api::types::{ToolCall, ToolCallFunction};
        let msg = MessageForTemplate {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_abc".to_string(),
                call_type: "function".to_string(),
                function: ToolCallFunction {
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Madrid"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        };
        let (role, content) = flatten_message_for_template(&msg);
        assert_eq!(role, "assistant");
        assert!(content.contains("get_weather"));
        assert!(content.contains("Madrid"));
    }

    #[test]
    fn test_flatten_message_tool_result() {
        let msg = MessageForTemplate {
            role: "tool".to_string(),
            content: Some("Sunny, 22°C".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_abc".to_string()),
        };
        let (role, content) = flatten_message_for_template(&msg);
        assert_eq!(role, "tool");
        assert_eq!(content, "Sunny, 22°C");
    }
}
