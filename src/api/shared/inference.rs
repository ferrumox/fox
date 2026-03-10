// Prompt preparation and inference parameter helpers.

use uuid::Uuid;

use crate::model_registry::EngineEntry;
use crate::scheduler::SamplingParams;

use crate::api::types::{OllamaOptions, Tool, ToolCall, ToolCallFunction};

// ---------------------------------------------------------------------------
// Sampling parameters
// ---------------------------------------------------------------------------

/// Build `SamplingParams` + `max_tokens` from Ollama-style options.
pub fn sampling_from_ollama(opts: Option<&OllamaOptions>) -> (SamplingParams, usize) {
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
            show_thinking: false,
        },
        max_tokens,
    )
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
                    function: ToolCallFunction { name, arguments: args },
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

/// Inject system messages (system prompt, tools, JSON mode), apply the chat
/// template, and tokenise the result.
///
/// Returns `(tokens, token_count)`.
pub fn prepare_prompt(
    entry: &EngineEntry,
    mut messages: Vec<(String, String)>,
    system_prompt: Option<&str>,
    tools: Option<&[Tool]>,
    json_mode: bool,
) -> (Vec<i32>, usize) {
    // Inject system prompt when configured and none is already present.
    if let Some(sp) = system_prompt {
        if messages.first().map(|(r, _)| r.as_str()) != Some("system") {
            messages.insert(0, ("system".to_string(), sp.to_string()));
        }
    }

    // Append tool descriptions to the system message (or create one).
    if let Some(tools) = tools {
        let tool_msg = tools_system_message(tools);
        if messages.first().map(|(r, _)| r.as_str()) == Some("system") {
            messages[0].1.push_str(&format!("\n\n{tool_msg}"));
        } else {
            messages.insert(0, ("system".to_string(), tool_msg));
        }
    }

    // Inject JSON-mode instruction.
    if json_mode {
        let json_instr =
            "Respond ONLY with valid JSON. Do not include any explanation or markdown.";
        if messages.first().map(|(r, _)| r.as_str()) == Some("system") {
            messages[0].1.push_str(&format!("\n\n{json_instr}"));
        } else {
            messages.insert(0, ("system".to_string(), json_instr.to_string()));
        }
    }

    let prompt = entry.engine.apply_chat_template(&messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{r}: {c}"))
            .collect::<Vec<_>>()
            .join("\n")
    });

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
        let (params, max_tokens) = sampling_from_ollama(None);
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
        let (params, max_tokens) = sampling_from_ollama(Some(&opts));
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
}
