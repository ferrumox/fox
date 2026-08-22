// Jinja chat-template rendering, kept free of any llama.cpp dependency so it
// builds and is unit-tested in stub builds (`FOX_SKIP_LLAMA=1`, what CI runs).
//
// The model side (`llama_cpp::vocab`) only supplies the template source, the
// BOS/EOS pieces and the messages; everything about *how* a template is compiled
// and rendered lives here.

use minijinja::{context, Environment};

/// Compile a model's chat template into a minijinja environment.
///
/// Returns `None` — with a warning — when the template is not usable, so the
/// caller can fall back to llama.cpp's built-in format.
// Only `llama_cpp::vocab` calls this, and that module is absent from stub builds.
#[cfg_attr(fox_stub, allow(dead_code))]
pub(crate) fn build_env(template: String) -> Option<Environment<'static>> {
    // Some GGUF conversions store a legacy template NAME (e.g. "vicuna", "chatml")
    // in `tokenizer.chat_template` instead of real Jinja source — a pre-Jinja
    // convention meant for llama.cpp's own name-based classifier. Trusting a bare
    // name as Jinja doesn't error — minijinja renders any string with no `{{`/`{%`
    // tags as literal text — so the entire prompt silently becomes that one word.
    // Require actual template syntax before committing to the Jinja path.
    if !template.contains("{{") && !template.contains("{%") {
        return None;
    }

    let mut env = Environment::new();
    // Chat templates lean on Python string methods (.strip(), .split(), …).
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    if let Err(e) = env.add_template_owned("chat", template) {
        tracing::warn!(
            error = %e,
            "chat template failed to parse; falling back to llama.cpp's built-in format, \
             which cannot render tool definitions"
        );
        return None;
    }
    Some(env)
}

/// Render the chat prompt. `tools` are OpenAI-shaped tool definitions, passed
/// straight through: a native tool-use template (Qwen, Hermes) unwraps
/// `tool.function.{name,description,parameters}` itself.
#[cfg_attr(fox_stub, allow(dead_code))]
pub(crate) fn render(
    env: &Environment<'static>,
    messages: &[(String, String)],
    enable_thinking: bool,
    bos_token: &str,
    eos_token: &str,
    tools: Option<&serde_json::Value>,
) -> Option<String> {
    let tmpl = env.get_template("chat").ok()?;

    let msgs: Vec<minijinja::Value> = messages
        .iter()
        .map(|(role, content)| context! { role => role, content => content })
        .collect();

    // A template that doesn't reference `tools` (the vast majority of models)
    // simply ignores this context key — no behavior change for them.
    let tools_value = tools
        .map(minijinja::Value::from_serialize)
        .unwrap_or(minijinja::Value::UNDEFINED);

    tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => true,
        enable_thinking => enable_thinking,
        bos_token => bos_token,
        eos_token => eos_token,
        tools => tools_value,
    })
    .inspect_err(|e| {
        // Never fail silently here: the fallback quietly drops `tools`, so a render
        // error surfaces to users as an agent whose tools "just don't work".
        tracing::warn!(
            error = %e,
            has_tools = tools.is_some(),
            "chat template render failed; falling back to llama.cpp's built-in format, \
             which cannot render tool definitions"
        );
    })
    .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tool-listing block every native tool-use template (Qwen, Hermes,
    /// Mistral) uses. `tojson` lives behind minijinja's non-default `json`
    /// feature: without it this render fails, the caller falls back to the
    /// built-in format, and the model never learns the tools exist.
    const TOOL_TEMPLATE: &str = r#"
{%- if tools %}
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>
{%- endif %}
{%- for message in messages %}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}"#;

    fn tools_json() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "run_sql",
                "description": "Run a read-only SQL query",
                "parameters": {
                    "type": "object",
                    "properties": {"sql": {"type": "string"}},
                    "required": ["sql"],
                },
            },
        }])
    }

    fn messages() -> Vec<(String, String)> {
        vec![("user".to_string(), "hi".to_string())]
    }

    #[test]
    fn renders_tool_definitions_into_the_prompt() {
        let env = build_env(TOOL_TEMPLATE.to_string()).expect("template should compile");
        let out = render(
            &env,
            &messages(),
            false,
            "",
            "<|im_end|>",
            Some(&tools_json()),
        )
        .expect("render with tools must succeed");

        // The whole schema has to reach the model, not just the name: a model that
        // cannot see the arguments cannot fill them in.
        assert!(
            out.contains("run_sql"),
            "tool name missing from prompt:\n{out}"
        );
        assert!(
            out.contains("Run a read-only SQL query"),
            "description missing:\n{out}"
        );
        assert!(out.contains("\"sql\""), "parameter schema missing:\n{out}");
    }

    #[test]
    fn renders_without_tools_when_none_are_offered() {
        let env = build_env(TOOL_TEMPLATE.to_string()).unwrap();
        let out = render(&env, &messages(), false, "", "<|im_end|>", None).unwrap();
        assert!(
            !out.contains("<tools>"),
            "tool block leaked without tools:\n{out}"
        );
        assert!(out.contains("hi"));
    }

    #[test]
    fn rejects_a_legacy_template_name() {
        // GGUFs that store "chatml" instead of Jinja source must not be rendered
        // as a literal one-word prompt.
        assert!(build_env("chatml".to_string()).is_none());
    }

    #[test]
    fn rejects_a_template_that_does_not_parse() {
        assert!(build_env("{% for x in %}".to_string()).is_none());
    }

    #[test]
    fn render_failure_returns_none_instead_of_panicking() {
        let env = build_env("{{ nope() }}".to_string()).unwrap();
        assert!(render(&env, &messages(), false, "", "", None).is_none());
    }
}
