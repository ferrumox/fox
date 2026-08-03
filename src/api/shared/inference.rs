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
pub struct MessageForTemplate {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
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
    use super::sampling_defaults as defaults;
    let default_max_tokens = if show_thinking {
        defaults::ollama::MAX_TOKENS_THINKING
    } else {
        defaults::ollama::MAX_TOKENS
    };
    let (temp, top_p, top_k, rep, seed, max_tokens, stop) = match opts {
        Some(o) => (
            o.temperature.unwrap_or(defaults::TEMPERATURE),
            o.top_p.unwrap_or(defaults::TOP_P),
            o.top_k.unwrap_or(defaults::ollama::TOP_K),
            o.repeat_penalty.unwrap_or(defaults::ollama::REPEAT_PENALTY),
            o.seed,
            o.num_predict
                .map(|n| n as usize)
                .unwrap_or(default_max_tokens),
            o.stop.clone(),
        ),
        None => (
            defaults::TEMPERATURE,
            defaults::TOP_P,
            defaults::ollama::TOP_K,
            defaults::ollama::REPEAT_PENALTY,
            None,
            default_max_tokens,
            None,
        ),
    };
    (
        SamplingParams {
            temperature: temp,
            top_p,
            top_k,
            repetition_penalty: rep,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed,
            stop,
            show_thinking,
            initial_in_thinking: show_thinking,
            max_thinking_chars: defaults::MAX_THINKING_CHARS,
            grammar: None,
            logprobs: None,
            min_p: opts.and_then(|o| o.min_p).unwrap_or(0.0).clamp(0.0, 1.0),
            min_tokens: 0,
            logit_bias: None,
        },
        max_tokens,
    )
}

// ---------------------------------------------------------------------------
// Thinking extraction
// ---------------------------------------------------------------------------

pub fn extract_thinking(text: &str, start_tag: &str, end_tag: &str) -> (Option<String>, String) {
    if let Some(end) = text.find(end_tag) {
        // Well-formed <open>...<close> block.
        let think_start = text
            .find(start_tag)
            .map(|i| i + start_tag.len())
            .unwrap_or(0)
            .min(end); // guard against a malformed open-after-close ordering
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

/// Build a validated `ToolCall` from a name + arguments value, rejecting names not
/// present in `known_tools` (when provided). Shared by every tool-call output format
/// (generic JSON, Hermes tags, …) so the whitelist/id-generation logic lives once.
fn build_tool_call(
    name: &str,
    args: &serde_json::Value,
    known_tools: Option<&[Tool]>,
) -> Option<ToolCall> {
    if let Some(tools) = known_tools {
        if !tools.iter().any(|t| t.function.name == name) {
            return None;
        }
    }
    Some(ToolCall {
        id: format!("call_{}", &Uuid::new_v4().to_string()[..8]),
        call_type: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: args.to_string(),
        },
    })
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
        return match build_tool_call(name, args, known_tools) {
            Some(call) => (String::new(), Some(vec![call])),
            None => (response.to_string(), None),
        };
    }

    // Pattern: {"tool_calls": [...]}
    if let Some(calls) = value.get("tool_calls").and_then(|tc| tc.as_array()) {
        let tool_calls: Vec<ToolCall> = calls
            .iter()
            .filter_map(|c| {
                let name = c.get("name")?.as_str()?;
                let args = c.get("arguments")?;
                build_tool_call(name, args, known_tools)
            })
            .collect();
        if !tool_calls.is_empty() {
            return (String::new(), Some(tool_calls));
        }
    }

    (response.to_string(), None)
}

/// One or more `<tool_call>{"name":..,"arguments":..}</tool_call>` blocks — the
/// output format Hermes/Qwen tool-use chat templates instruct the model to emit.
/// Parallel tool calls are native to the format (one block per call). Text outside
/// the blocks (leading/trailing prose) becomes the returned content. A block with
/// invalid JSON, or naming a tool not in `known_tools`, is silently dropped — same
/// whitelist behavior as the generic parser.
fn parse_hermes_tool_calls(
    response: &str,
    known_tools: Option<&[Tool]>,
) -> (String, Option<Vec<ToolCall>>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = response;

    while let Some(start) = rest.find(OPEN) {
        content.push_str(&rest[..start]);
        let after_open = &rest[start + OPEN.len()..];
        let Some(end) = after_open.find(CLOSE) else {
            // Unterminated block (generation cut off) — keep it out of content and stop.
            rest = "";
            break;
        };
        let block = after_open[..end].trim();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(block) {
            if let (Some(name), Some(args)) = (
                value.get("name").and_then(|n| n.as_str()),
                value.get("arguments"),
            ) {
                if let Some(call) = build_tool_call(name, args, known_tools) {
                    calls.push(call);
                }
            }
        }
        rest = &after_open[end + CLOSE.len()..];
    }
    content.push_str(rest);

    if calls.is_empty() {
        (response.to_string(), None)
    } else {
        (content.trim().to_string(), Some(calls))
    }
}

/// One or more Mistral-style `[TOOL_CALLS]`-marked calls. Two real wire formats
/// exist depending on the model/template version, so both are attempted:
///
/// 1. Classic (documented at docs.mistral.ai, vLLM's `--tool-call-parser mistral`):
///    a single `[TOOL_CALLS]` marker followed by a JSON array —
///    `[TOOL_CALLS] [{"name": "...", "arguments": {...}}]`.
/// 2. Newer (the currently-vendored llama.cpp's own PEG chat parser): one marker
///    per call, each followed by `<name>[ARGS]<json-object>` —
///    `[TOOL_CALLS]get_weather[ARGS]{"city": "Madrid"}`, repeated for parallel calls.
///
/// Text before the first marker becomes the returned content. No marker, or neither
/// shape parses, falls through to passthrough (same behavior as the other parsers).
fn parse_mistral_tool_calls(
    response: &str,
    known_tools: Option<&[Tool]>,
) -> (String, Option<Vec<ToolCall>>) {
    const MARKER: &str = "[TOOL_CALLS]";
    const ARGS: &str = "[ARGS]";

    let Some(start) = response.find(MARKER) else {
        return (response.to_string(), None);
    };
    let before = response[..start].trim().to_string();
    let after_first = response[start + MARKER.len()..].trim();

    // Attempt 1: classic JSON-array format directly after the (single) marker.
    if let Ok(serde_json::Value::Array(items)) =
        serde_json::from_str::<serde_json::Value>(after_first)
    {
        let calls: Vec<ToolCall> = items
            .iter()
            .filter_map(|item| {
                let name = item.get("name")?.as_str()?;
                let args = item.get("arguments")?;
                build_tool_call(name, args, known_tools)
            })
            .collect();
        if !calls.is_empty() {
            return (before, Some(calls));
        }
    }

    // Attempt 2: repeated `[TOOL_CALLS]name[ARGS]{...}` per call.
    let mut calls = Vec::new();
    for segment in response[start..].split(MARKER).skip(1) {
        let Some(args_start) = segment.find(ARGS) else {
            continue;
        };
        let name = segment[..args_start].trim();
        let rest = segment[args_start + ARGS.len()..].trim();
        if let Ok(args) = serde_json::from_str::<serde_json::Value>(rest) {
            if let Some(call) = build_tool_call(name, &args, known_tools) {
                calls.push(call);
            }
        }
    }
    if calls.is_empty() {
        (response.to_string(), None)
    } else {
        (before, Some(calls))
    }
}

/// Which format to parse tool calls out of the model's raw output text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallParserKind {
    /// `{"name":..,"arguments":..}` or `{"tool_calls":[...]}` as the whole response
    /// (fox's original prompt-engineered format — the default/fallback).
    Generic,
    /// One or more `<tool_call>{...}</tool_call>` blocks (Hermes/Qwen tool-use
    /// template format).
    Hermes,
    /// `[TOOL_CALLS]`-marked calls (Mistral tool-use template format) — see
    /// [`parse_mistral_tool_calls`] for the two wire-format variants handled.
    Mistral,
    /// A single JSON object, optionally prefixed by `<|python_tag|>`, whose args key
    /// is `arguments` or `parameters` (Llama3's custom-tool JSON convention).
    /// Explicit-opt-in only — see [`crate::engine::model::NativeToolFormat`].
    Llama3,
}

/// One JSON object (optionally `<|python_tag|>`-prefixed), args key `arguments` or
/// `parameters` — Llama3's custom-tool-calling convention. Single call only: the
/// format has no parallel-call shape.
fn parse_llama3_tool_calls(
    response: &str,
    known_tools: Option<&[Tool]>,
) -> (String, Option<Vec<ToolCall>>) {
    const PY_TAG: &str = "<|python_tag|>";
    let trimmed = response.trim();
    let json_part = trimmed.strip_prefix(PY_TAG).unwrap_or(trimmed).trim();

    let Ok(value) = serde_json::from_str::<serde_json::Value>(json_part) else {
        return (response.to_string(), None);
    };
    let name = value.get("name").and_then(|n| n.as_str());
    let args = value.get("arguments").or_else(|| value.get("parameters"));

    if let (Some(name), Some(args)) = (name, args) {
        return match build_tool_call(name, args, known_tools) {
            Some(call) => (String::new(), Some(vec![call])),
            None => (response.to_string(), None),
        };
    }
    (response.to_string(), None)
}

/// Resolve the effective parser. An explicit `configured` override (`"generic"` /
/// `"hermes"` / `"mistral"` / `"llama3"`) wins; anything else (including `"auto"`,
/// the default) picks Hermes/Mistral when the loaded model's own chat template
/// natively formats tool calls that way, generic otherwise — see
/// `InferenceEngine::native_tool_call_format`. Llama3 is never auto-selected (see
/// [`crate::engine::model::NativeToolFormat`]).
pub fn resolve_tool_call_parser(
    configured: &str,
    native: Option<crate::engine::model::NativeToolFormat>,
) -> ToolCallParserKind {
    use crate::engine::model::NativeToolFormat;
    match configured {
        "generic" => ToolCallParserKind::Generic,
        "hermes" => ToolCallParserKind::Hermes,
        "mistral" => ToolCallParserKind::Mistral,
        "llama3" => ToolCallParserKind::Llama3,
        _ => match native {
            Some(NativeToolFormat::Hermes) => ToolCallParserKind::Hermes,
            Some(NativeToolFormat::Mistral) => ToolCallParserKind::Mistral,
            None => ToolCallParserKind::Generic,
        },
    }
}

/// Parse `response` for tool calls using the given parser. Returns `(content, tool_calls)`.
pub fn parse_tool_call(
    response: &str,
    known_tools: Option<&[Tool]>,
    parser: ToolCallParserKind,
) -> (String, Option<Vec<ToolCall>>) {
    match parser {
        ToolCallParserKind::Generic => try_parse_tool_call(response, known_tools),
        ToolCallParserKind::Hermes => parse_hermes_tool_calls(response, known_tools),
        ToolCallParserKind::Mistral => parse_mistral_tool_calls(response, known_tools),
        ToolCallParserKind::Llama3 => parse_llama3_tool_calls(response, known_tools),
    }
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
                },
            );
        }
    }

    // Whether the model's own chat template natively formats tool calls (Hermes/Qwen
    // tool-use templates). When it does, `tools` is threaded straight into the Jinja
    // render below and the template renders its own tool listing — injecting fox's
    // generic system-message listing too would duplicate it in the prompt.
    let model_native_tools = tools.is_some() && entry.engine.native_tool_call_format().is_some();

    // Append tool descriptions — only for models without a native tool-formatting
    // template; see `model_native_tools` above.
    if let Some(tools) = tools {
        if !model_native_tools {
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
                    },
                );
            }
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
                },
            );
        }
    }

    let flat: Vec<(String, String)> = messages.iter().map(flatten_message_for_template).collect();

    // Serialize tools once for the Jinja context (OpenAI-shaped `Tool` already
    // matches what native tool-use templates expect — no transform needed).
    let tools_json = tools.map(|t| serde_json::to_value(t).unwrap_or_default());

    // Build the prompt tokens via the model's chat template (real Jinja when the
    // model has one, threading `enable_thinking` and `tools`; built-in format
    // otherwise).
    let tokens: Vec<i32> = entry
        .engine
        .build_prompt_tokens(&flat, show_thinking, tools_json.as_ref())
        .unwrap_or_else(|_| {
            // Last-ditch fallback: plain "role: content" text, byte-tokenized.
            let prompt = flat
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n");
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
            min_p: Some(0.05),
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
    fn test_parse_hermes_tool_call_single() {
        let response = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Madrid"}}
</tool_call>"#;
        let (content, calls) = parse_hermes_tool_calls(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
    }

    #[test]
    fn test_parse_hermes_tool_call_parallel() {
        let response = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Madrid"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Lisbon"}}
</tool_call>"#;
        let (content, calls) = parse_hermes_tool_calls(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should have parallel tool calls");
        assert_eq!(calls.len(), 2);
        assert!(calls[0].function.arguments.contains("Madrid"));
        assert!(calls[1].function.arguments.contains("Lisbon"));
    }

    #[test]
    fn test_parse_hermes_tool_call_with_surrounding_prose() {
        let response = r#"Sure, let me check that for you.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Madrid"}}
</tool_call>
Let me know if you need anything else."#;
        let (content, calls) = parse_hermes_tool_calls(response, None);
        assert!(calls.is_some());
        assert!(content.contains("Sure, let me check"));
        assert!(content.contains("Let me know"));
        assert!(!content.contains("tool_call"));
    }

    #[test]
    fn test_parse_hermes_tool_call_unknown_tool_rejected() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let response = r#"<tool_call>
{"name": "unknown_tool", "arguments": {}}
</tool_call>"#;
        let (content, calls) = parse_hermes_tool_calls(response, Some(&tools));
        assert!(calls.is_none());
        assert_eq!(content, response);
    }

    #[test]
    fn test_parse_hermes_no_tool_call_tags_is_plain_text() {
        let response = "The sky is blue because of Rayleigh scattering.";
        let (content, calls) = parse_hermes_tool_calls(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_resolve_tool_call_parser_explicit_overrides_win() {
        use crate::engine::model::NativeToolFormat;
        assert_eq!(
            resolve_tool_call_parser("generic", Some(NativeToolFormat::Hermes)),
            ToolCallParserKind::Generic
        );
        assert_eq!(
            resolve_tool_call_parser("hermes", None),
            ToolCallParserKind::Hermes
        );
        assert_eq!(
            resolve_tool_call_parser("mistral", None),
            ToolCallParserKind::Mistral
        );
        assert_eq!(
            resolve_tool_call_parser("llama3", None),
            ToolCallParserKind::Llama3
        );
        // Llama3 is never reached via "auto", even with an explicit override
        // present for the OTHER formats — only its own literal string selects it.
        assert_eq!(
            resolve_tool_call_parser("llama3", Some(NativeToolFormat::Hermes)),
            ToolCallParserKind::Llama3
        );
    }

    #[test]
    fn test_resolve_tool_call_parser_auto_follows_model_native() {
        use crate::engine::model::NativeToolFormat;
        assert_eq!(
            resolve_tool_call_parser("auto", Some(NativeToolFormat::Hermes)),
            ToolCallParserKind::Hermes
        );
        assert_eq!(
            resolve_tool_call_parser("auto", Some(NativeToolFormat::Mistral)),
            ToolCallParserKind::Mistral
        );
        assert_eq!(
            resolve_tool_call_parser("auto", None),
            ToolCallParserKind::Generic
        );
    }

    #[test]
    fn test_parse_tool_call_dispatches_by_parser_kind() {
        let generic_response = r#"{"name":"f","arguments":{}}"#;
        let (_, calls) = parse_tool_call(generic_response, None, ToolCallParserKind::Generic);
        assert!(calls.is_some());

        let hermes_response = "<tool_call>\n{\"name\":\"f\",\"arguments\":{}}\n</tool_call>";
        let (_, calls) = parse_tool_call(hermes_response, None, ToolCallParserKind::Hermes);
        assert!(calls.is_some());

        // Cross-format text must NOT parse under the wrong parser.
        let (_, calls) = parse_tool_call(hermes_response, None, ToolCallParserKind::Generic);
        assert!(calls.is_none());

        let mistral_response = r#"[TOOL_CALLS] [{"name":"f","arguments":{}}]"#;
        let (_, calls) = parse_tool_call(mistral_response, None, ToolCallParserKind::Mistral);
        assert!(calls.is_some());

        let llama3_response = r#"{"name":"f","parameters":{}}"#;
        let (_, calls) = parse_tool_call(llama3_response, None, ToolCallParserKind::Llama3);
        assert!(calls.is_some());
    }

    #[test]
    fn test_parse_mistral_tool_call_classic_array_single() {
        let response = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Madrid"}}]"#;
        let (content, calls) = parse_mistral_tool_calls(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
    }

    #[test]
    fn test_parse_mistral_tool_call_classic_array_parallel() {
        let response = r#"[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Madrid"}}, {"name": "get_weather", "arguments": {"city": "Lisbon"}}]"#;
        let (_, calls) = parse_mistral_tool_calls(response, None);
        let calls = calls.expect("should have parallel tool calls");
        assert_eq!(calls.len(), 2);
        assert!(calls[0].function.arguments.contains("Madrid"));
        assert!(calls[1].function.arguments.contains("Lisbon"));
    }

    #[test]
    fn test_parse_mistral_tool_call_new_format_single() {
        let response = r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Madrid"}"#;
        let (content, calls) = parse_mistral_tool_calls(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
    }

    #[test]
    fn test_parse_mistral_tool_call_new_format_parallel() {
        let response = r#"[TOOL_CALLS]get_weather[ARGS]{"city": "Madrid"}[TOOL_CALLS]get_weather[ARGS]{"city": "Lisbon"}"#;
        let (_, calls) = parse_mistral_tool_calls(response, None);
        let calls = calls.expect("should have parallel tool calls");
        assert_eq!(calls.len(), 2);
        assert!(calls[0].function.arguments.contains("Madrid"));
        assert!(calls[1].function.arguments.contains("Lisbon"));
    }

    #[test]
    fn test_parse_mistral_tool_call_with_surrounding_prose() {
        let response = r#"Let me check that.[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Madrid"}}]"#;
        let (content, calls) = parse_mistral_tool_calls(response, None);
        assert!(calls.is_some());
        assert_eq!(content, "Let me check that.");
    }

    #[test]
    fn test_parse_mistral_no_marker_is_plain_text() {
        let response = "The sky is blue because of Rayleigh scattering.";
        let (content, calls) = parse_mistral_tool_calls(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_mistral_invalid_json_is_plain_text() {
        let response = "[TOOL_CALLS] not json at all";
        let (content, calls) = parse_mistral_tool_calls(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_mistral_unknown_tool_rejected() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let response = r#"[TOOL_CALLS] [{"name": "unknown_tool", "arguments": {}}]"#;
        let (_, calls) = parse_mistral_tool_calls(response, Some(&tools));
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_llama3_tool_call_arguments_key() {
        let response = r#"{"name": "get_weather", "arguments": {"city": "Madrid"}}"#;
        let (content, calls) = parse_llama3_tool_calls(response, None);
        assert!(content.is_empty());
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
    }

    #[test]
    fn test_parse_llama3_tool_call_parameters_key() {
        let response = r#"{"name": "get_weather", "parameters": {"city": "Madrid"}}"#;
        let (_, calls) = parse_llama3_tool_calls(response, None);
        let calls = calls.expect("should have tool calls");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Madrid"));
    }

    #[test]
    fn test_parse_llama3_tool_call_python_tag_prefix_stripped() {
        let response = r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "Madrid"}}"#;
        let (_, calls) = parse_llama3_tool_calls(response, None);
        assert!(calls.is_some());
    }

    #[test]
    fn test_parse_llama3_invalid_json_is_plain_text() {
        let response = "The sky is blue because of Rayleigh scattering.";
        let (content, calls) = parse_llama3_tool_calls(response, None);
        assert_eq!(content, response);
        assert!(calls.is_none());
    }

    #[test]
    fn test_parse_llama3_unknown_tool_rejected() {
        use crate::api::types::{Tool, ToolFunction};
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "get_weather".to_string(),
                description: None,
                parameters: None,
            },
        }];
        let response = r#"{"name": "unknown_tool", "parameters": {}}"#;
        let (_, calls) = parse_llama3_tool_calls(response, Some(&tools));
        assert!(calls.is_none());
    }

    #[test]
    fn test_extract_thinking_well_formed() {
        let (thinking, content) = extract_thinking(
            "<think>\nsome thought\n</think>\nthe answer",
            "<think>",
            "</think>",
        );
        assert_eq!(thinking.as_deref(), Some("some thought"));
        assert_eq!(content, "the answer");
    }

    #[test]
    fn test_extract_thinking_unclosed_block() {
        // Generation cut off before </think> — thinking leaks into content without this fix.
        let (thinking, content) = extract_thinking(
            "<think>\nI was still thinking when tokens ran out",
            "<think>",
            "</think>",
        );
        assert_eq!(
            thinking.as_deref(),
            Some("I was still thinking when tokens ran out")
        );
        assert_eq!(content, "");
    }

    #[test]
    fn test_extract_thinking_no_think_tag() {
        let (thinking, content) = extract_thinking("plain response", "<think>", "</think>");
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
