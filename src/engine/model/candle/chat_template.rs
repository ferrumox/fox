//! Apply a GGUF-embedded Jinja chat template to a list of `(role, content)`
//! messages, producing the prompt string that goes into the tokenizer.
//!
//! Modern instruct-tuned models ship their conversation format inside the
//! GGUF as a Jinja2 template (`tokenizer.chat_template`). Rendering that
//! template gives a prompt that's byte-for-byte compatible with what
//! llama.cpp / transformers produce for the same input — meaning the model
//! sees the chat structure it was trained on.

use minijinja::{context, value::Value, Environment};
use std::fmt;

use super::tokenizer::Vocab;

#[derive(Debug)]
pub enum TemplateError {
    Missing,
    Render(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing => write!(
                f,
                "vocab does not carry a chat template (tokenizer.chat_template absent)"
            ),
            Self::Render(e) => write!(f, "chat template rendering failed: {e}"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Render `vocab.chat_template` over `messages`. When `add_generation_prompt`
/// is true, the template is asked to leave the prompt open at the assistant
/// turn so generation continues from there.
pub fn apply_chat_template(
    vocab: &Vocab,
    messages: &[(String, String)],
    add_generation_prompt: bool,
) -> Result<String, TemplateError> {
    let template_str = vocab
        .chat_template
        .as_deref()
        .ok_or(TemplateError::Missing)?;

    let mut env = Environment::new();
    // Jinja in HF templates uses `raise_exception` to abort with a custom
    // message when the conversation shape is invalid. minijinja does not
    // ship that filter — provide a small shim so common templates render
    // without surprises.
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat", template_str)
        .map_err(|e| TemplateError::Render(e.to_string()))?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TemplateError::Render(e.to_string()))?;

    let messages_value: Vec<Value> = messages
        .iter()
        .map(|(role, content)| {
            context! {
                role => role.as_str(),
                content => content.as_str(),
            }
        })
        .collect();

    let bos_token = vocab
        .specials
        .bos
        .and_then(|id| vocab.token(id))
        .unwrap_or("")
        .to_string();
    let eos_token = vocab
        .specials
        .eos
        .and_then(|id| vocab.token(id))
        .unwrap_or("")
        .to_string();

    let ctx = context! {
        messages => messages_value,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => add_generation_prompt,
    };

    tmpl.render(ctx)
        .map_err(|e| TemplateError::Render(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::candle::tokenizer::{SpecialTokens, Vocab};

    fn vocab_with_template(template: &str) -> Vocab {
        Vocab {
            tokens: vec!["<bos>".into(), "<eos>".into()],
            token_types: vec![],
            merges: vec![],
            specials: SpecialTokens {
                bos: Some(0),
                eos: Some(1),
                ..Default::default()
            },
            model_kind: "gpt2".into(),
            chat_template: Some(template.into()),
        }
    }

    #[test]
    fn renders_a_minimal_chatml_like_template() {
        let template = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}<|/{{ m.role }}|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}";
        let vocab = vocab_with_template(template);
        let messages = vec![
            ("system".to_string(), "you are helpful".to_string()),
            ("user".to_string(), "hi".to_string()),
        ];
        let out = apply_chat_template(&vocab, &messages, true).unwrap();
        assert_eq!(
            out,
            "<|system|>you are helpful<|/system|><|user|>hi<|/user|><|assistant|>"
        );
    }

    #[test]
    fn injects_bos_and_eos_tokens() {
        let template =
            "{{ bos_token }}{% for m in messages %}{{ m.content }}{{ eos_token }}{% endfor %}";
        let vocab = vocab_with_template(template);
        let messages = vec![("user".to_string(), "hi".to_string())];
        let out = apply_chat_template(&vocab, &messages, false).unwrap();
        assert_eq!(out, "<bos>hi<eos>");
    }

    #[test]
    fn raise_exception_surfaces_as_render_error() {
        let template =
            "{% if messages|length == 0 %}{{ raise_exception('no messages') }}{% endif %}";
        let vocab = vocab_with_template(template);
        let err = apply_chat_template(&vocab, &[], false).unwrap_err();
        match err {
            TemplateError::Render(msg) => assert!(msg.contains("no messages")),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn missing_template_returns_missing_error() {
        let mut vocab = vocab_with_template("ignored");
        vocab.chat_template = None;
        let err = apply_chat_template(&vocab, &[], false).unwrap_err();
        assert!(matches!(err, TemplateError::Missing));
    }

    #[test]
    fn add_generation_prompt_flag_reaches_template() {
        let template = "{{ add_generation_prompt }}";
        let vocab = vocab_with_template(template);
        let on = apply_chat_template(&vocab, &[], true).unwrap();
        let off = apply_chat_template(&vocab, &[], false).unwrap();
        assert_eq!(on, "true");
        assert_eq!(off, "false");
    }
}
