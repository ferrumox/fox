// Prompt preparation and inference parameter helpers.

use uuid::Uuid;

use crate::model_registry::EngineEntry;
use crate::scheduler::SamplingParams;

use crate::api::types::{OllamaOptions, ResponseFormat, Tool, ToolCall, ToolCallFunction};

// ---------------------------------------------------------------------------
// Message representation for template rendering
// ---------------------------------------------------------------------------

/// A chat message carrying all fields needed for prompt building.
/// Content is already extracted to plain text (callers handle MessageContent → String).
#[derive(Clone)]
pub struct MessageForTemplate {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
    /// Raw image bytes for vision processing (decoded from base64/URL by the handler).
    pub image_data: Option<Vec<u8>>,
}

/// Flatten a [`MessageForTemplate`] into a `(role, content)` tuple for
/// `apply_chat_template` (llama.cpp FFI boundary accepts only string pairs).
fn flatten_message_for_template(msg: &MessageForTemplate) -> (String, String) {
    match msg.role.as_str() {
        "assistant" if msg.tool_calls.is_some() => {
            let calls = msg.tool_calls.as_ref().unwrap();
            let serialized = calls
                .iter()
                .map(|tc| {
                    format!(
                        "[tool_call: {}({})]",
                        tc.function.name, tc.function.arguments
                    )
                })
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
// Tool choice resolution
// ---------------------------------------------------------------------------

/// Resolved tool choice from the request's `tool_choice` field.
pub struct ToolChoiceResult {
    /// Effective tools to inject (None when tool_choice == "none").
    pub tools: Option<Vec<Tool>>,
    /// Whether the model is required to call a tool (tool_choice == "required" or specific).
    pub required: bool,
    /// Specific tool name the model must call (tool_choice == {"function": {"name": "X"}}).
    pub specific: Option<String>,
}

/// Resolve `tools` + `tool_choice` into an effective configuration.
///
/// * `"none"` → no tools, model responds normally.
/// * `"auto"` / absent → tools available, model decides.
/// * `"required"` → tools available, model must call one.
/// * `{"type":"function","function":{"name":"X"}}` → only tool X, model must call it.
pub fn resolve_tool_choice(
    tools: Option<&[Tool]>,
    tool_choice: Option<&serde_json::Value>,
) -> ToolChoiceResult {
    let tools = match tools {
        None | Some([]) => {
            return ToolChoiceResult {
                tools: None,
                required: false,
                specific: None,
            }
        }
        Some(t) => t,
    };

    match tool_choice {
        // "none" — no tool use this turn
        Some(v) if v.as_str() == Some("none") => ToolChoiceResult {
            tools: None,
            required: false,
            specific: None,
        },
        // "required" — model must call a tool
        Some(v) if v.as_str() == Some("required") => ToolChoiceResult {
            tools: Some(tools.to_vec()),
            required: true,
            specific: None,
        },
        // specific function: {"type":"function","function":{"name":"X"}}
        Some(v) if v.is_object() => {
            let name = v
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(String::from);
            let filtered: Vec<Tool> = tools
                .iter()
                .filter(|t| {
                    name.as_deref()
                        .map(|n| t.function.name == n)
                        .unwrap_or(true)
                })
                .cloned()
                .collect();
            ToolChoiceResult {
                tools: Some(if filtered.is_empty() {
                    tools.to_vec()
                } else {
                    filtered
                }),
                required: true,
                specific: name,
            }
        }
        // "auto" / absent / unknown string — default behaviour
        _ => ToolChoiceResult {
            tools: Some(tools.to_vec()),
            required: false,
            specific: None,
        },
    }
}

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

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
            o.num_predict
                .unwrap_or(if show_thinking { 2048 } else { 512 }) as usize,
            o.stop.clone(),
        ),
        None => (
            0.8,
            0.9,
            40,
            1.1,
            None,
            if show_thinking { 2048 } else { 512 },
            None,
        ),
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

pub fn extract_thinking(text: &str) -> (Option<String>, String) {
    let start_tag = "<think>";
    let end_tag = "</think>";
    if let Some(end) = text.find(end_tag) {
        // Well-formed <think>...</think> block.
        let think_start = text
            .find(start_tag)
            .map(|i| i + start_tag.len())
            .unwrap_or(0);
        let thinking = text[think_start..end].trim().to_string();
        let content = text[end + end_tag.len()..].trim().to_string();
        let thinking = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };
        return (thinking, content);
    }
    if let Some(start) = text.find(start_tag) {
        // Unclosed <think> block — generation was cut off before </think>.
        // Treat everything after <think> as thinking; visible content is empty.
        let thinking = text[start + start_tag.len()..].trim().to_string();
        let thinking = if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        };
        return (thinking, String::new());
    }
    (None, text.to_string())
}

// ---------------------------------------------------------------------------
// Tool call helpers
// ---------------------------------------------------------------------------

/// Build a system message describing available tools.
/// `required` and `specific` come from [`ToolChoiceResult`].
pub fn tools_system_message(tools: &[Tool], required: bool, specific: Option<&str>) -> String {
    let json = serde_json::to_string_pretty(tools).unwrap_or_default();
    let usage = "When you want to call a tool, respond ONLY with a JSON object:\n\
                 {\"name\": \"<tool_name>\", \"arguments\": {<key>: <value>}}";
    let constraint = if let Some(name) = specific {
        format!("You MUST call the tool '{name}'. Do not respond without calling it.")
    } else if required {
        "You MUST call one of the available tools. Do not respond without calling a tool."
            .to_string()
    } else {
        "If you don't need a tool, respond normally.".to_string()
    };
    format!("You have access to the following tools:\n{json}\n\n{usage}\n\n{constraint}")
}

/// Try to parse `response` text as a JSON tool call.
/// Returns `(content, tool_calls)`.
pub fn try_parse_tool_call(
    response: &str,
    known_tools: Option<&[Tool]>,
) -> (String, Option<Vec<ToolCall>>) {
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
        // Validate name against known tools when provided.
        if let Some(tools) = known_tools {
            if !tools.iter().any(|t| t.function.name == name) {
                return (response.to_string(), None);
            }
        }
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
                // Validate name if known_tools provided.
                if let Some(tools) = known_tools {
                    if !tools.iter().any(|t| t.function.name == name) {
                        return None;
                    }
                }
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

/// Inject system messages, apply the chat template, and tokenise the result.
///
/// * `tools` — already filtered by [`resolve_tool_choice`] (None = no tools).
/// * `tool_required` / `specific_tool` — from [`ToolChoiceResult`].
#[allow(clippy::too_many_arguments)]
pub fn prepare_prompt(
    entry: &EngineEntry,
    mut messages: Vec<MessageForTemplate>,
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
    tool_required: bool,
    specific_tool: Option<&str>,
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
                    image_data: None,
                },
            );
        }
    }

    // Append tool descriptions.
    if let Some(tools) = tools {
        let tool_msg = tools_system_message(tools, tool_required, specific_tool);
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys = messages[0].content.get_or_insert_with(String::new);
            sys.push_str(&format!("\n\n{tool_msg}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(tool_msg),
                    tool_calls: None,
                    tool_call_id: None,
                    image_data: None,
                },
            );
        }
    }

    // Inject JSON-mode instruction.
    if let Some(instr) = json_mode_instruction(response_format) {
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys = messages[0].content.get_or_insert_with(String::new);
            sys.push_str(&format!("\n\n{instr}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(instr),
                    tool_calls: None,
                    tool_call_id: None,
                    image_data: None,
                },
            );
        }
    }

    let flat: Vec<(String, String)> = messages.iter().map(flatten_message_for_template).collect();

    let mut prompt = entry.engine.apply_chat_template(&flat).unwrap_or_else(|_| {
        flat.iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    });

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

/// Media marker used by mtmd to denote where an image should be inserted.
const MEDIA_MARKER: &str = "<__media__>";

/// Build the prompt text for a vision request, inserting the media marker
/// where the image should appear. Returns the raw prompt string (NOT tokenized —
/// mtmd_tokenize handles tokenization + image interleaving).
#[allow(clippy::too_many_arguments)]
pub fn prepare_vision_prompt(
    entry: &EngineEntry,
    mut messages: Vec<MessageForTemplate>,
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
    tool_required: bool,
    specific_tool: Option<&str>,
    response_format: Option<&ResponseFormat>,
    show_thinking: bool,
) -> String {
    // For the message with image_data, insert the media marker before the text content.
    for msg in &mut messages {
        if msg.image_data.is_some() {
            let text = msg.content.as_deref().unwrap_or("");
            msg.content = Some(format!("{MEDIA_MARKER}\n{text}"));
        }
    }

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
                    image_data: None,
                },
            );
        }
    }

    // Append tool descriptions.
    if let Some(tools) = tools {
        let tool_msg = tools_system_message(tools, tool_required, specific_tool);
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys = messages[0].content.get_or_insert_with(String::new);
            sys.push_str(&format!("\n\n{tool_msg}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(tool_msg),
                    tool_calls: None,
                    tool_call_id: None,
                    image_data: None,
                },
            );
        }
    }

    // Inject JSON-mode instruction.
    if let Some(instr) = json_mode_instruction(response_format) {
        if messages.first().map(|m| m.role.as_str()) == Some("system") {
            let sys = messages[0].content.get_or_insert_with(String::new);
            sys.push_str(&format!("\n\n{instr}"));
        } else {
            messages.insert(
                0,
                MessageForTemplate {
                    role: "system".to_string(),
                    content: Some(instr),
                    tool_calls: None,
                    tool_call_id: None,
                    image_data: None,
                },
            );
        }
    }

    let flat: Vec<(String, String)> = messages.iter().map(flatten_message_for_template).collect();

    let mut prompt = entry.engine.apply_chat_template(&flat).unwrap_or_else(|_| {
        flat.iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    });

    if show_thinking {
        prompt.push_str("<think>\n");
    }

    prompt
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
        assert_eq!(max_tokens, 512);
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
        let (content, calls) = try_parse_tool_call(response, None);
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
        let (content, calls) = try_parse_tool_call(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_invalid_json() {
        let response = "not { json at all }";
        let (content, calls) = try_parse_tool_call(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_json_no_name() {
        let response = r#"{"answer": "42"}"#;
        let (content, calls) = try_parse_tool_call(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_try_parse_tool_call_tool_calls_array() {
        let response = r#"{"tool_calls":[{"name":"search","arguments":{"query":"rust"}}]}"#;
        let (content, calls) = try_parse_tool_call(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should detect tool_calls array");
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_try_parse_tool_call_whitespace_trimmed() {
        let response = "  \n{\"name\":\"foo\",\"arguments\":{}}\n  ";
        let (_content, calls) = try_parse_tool_call(response, None);
        assert!(calls.is_some());
    }

    #[test]
    fn test_try_parse_tool_call_validates_against_known_tools() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];
        // Known tool name → detected
        let (_, calls) =
            try_parse_tool_call(r#"{"name":"get_weather","arguments":{}}"#, Some(&tools));
        assert!(calls.is_some());

        // Unknown tool name → rejected
        let (content, calls) =
            try_parse_tool_call(r#"{"name":"unknown_tool","arguments":{}}"#, Some(&tools));
        assert!(calls.is_none());
        assert!(!content.is_empty());
    }

    #[test]
    fn test_extract_thinking_well_formed() {
        let (thinking, content) = extract_thinking("<think>\nsome thought\n</think>\nthe answer");
        assert_eq!(thinking.as_deref(), Some("some thought"));
        assert_eq!(content, "the answer");
    }

    #[test]
    fn test_extract_thinking_unclosed_block() {
        // Generation cut off before </think> — thinking leaks into content without this fix.
        let (thinking, content) =
            extract_thinking("<think>\nI was still thinking when tokens ran out");
        assert_eq!(
            thinking.as_deref(),
            Some("I was still thinking when tokens ran out")
        );
        assert_eq!(content, "");
    }

    #[test]
    fn test_extract_thinking_no_think_tag() {
        let (thinking, content) = extract_thinking("plain response");
        assert!(thinking.is_none());
        assert_eq!(content, "plain response");
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
        let msg = tools_system_message(&tools, false, None);
        assert!(msg.contains("get_weather"));
        assert!(msg.contains("tool"));
        assert!(msg.contains("JSON"));
    }

    #[test]
    fn test_tools_system_message_required() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let msg = tools_system_message(&tools, true, None);
        assert!(msg.contains("MUST call"));
    }

    #[test]
    fn test_tools_system_message_specific_tool() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "search".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let msg = tools_system_message(&tools, true, Some("search"));
        assert!(msg.contains("'search'"));
    }

    #[test]
    fn test_resolve_tool_choice_none_string() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "f".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let v = serde_json::json!("none");
        let r = resolve_tool_choice(Some(&tools), Some(&v));
        assert!(r.tools.is_none());
        assert!(!r.required);
    }

    #[test]
    fn test_resolve_tool_choice_required() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "f".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let v = serde_json::json!("required");
        let r = resolve_tool_choice(Some(&tools), Some(&v));
        assert!(r.tools.is_some());
        assert!(r.required);
        assert!(r.specific.is_none());
    }

    #[test]
    fn test_resolve_tool_choice_specific() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![
            Tool {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: "a".to_string(),
                    description: None,
                    parameters: None,
                },
            },
            Tool {
                tool_type: "function".to_string(),
                function: ToolFunction {
                    name: "b".to_string(),
                    description: None,
                    parameters: None,
                },
            },
        ];
        let v = serde_json::json!({"type": "function", "function": {"name": "a"}});
        let r = resolve_tool_choice(Some(&tools), Some(&v));
        let eff = r.tools.expect("should have tools");
        assert_eq!(eff.len(), 1);
        assert_eq!(eff[0].function.name, "a");
        assert!(r.required);
        assert_eq!(r.specific.as_deref(), Some("a"));
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
            image_data: None,
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
            image_data: None,
        };
        let (role, content) = flatten_message_for_template(&msg);
        assert_eq!(role, "tool");
        assert_eq!(content, "Sunny, 22°C");
    }
}
