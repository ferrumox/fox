// JSON Schema → GBNF grammar (0.14, S2).
//
// Guided decoding constrains generation to a GBNF grammar (see the engine's
// `sample_constrained`). llama.cpp ships a JSON-schema→grammar converter, but it lives
// in `common/`, which fox does not build, so the conversion is ours.
//
// This covers the pragmatic subset that matters for structured output:
//   - `type`: object / array / string / integer / number / boolean / null
//   - `properties` + `required` on objects, including genuinely optional properties
//   - `items` on arrays
//   - `enum` (any JSON literal)
//   - arbitrary nesting
//   - untyped / empty schema → any JSON value
//
// Two documented divergences, both deliberate:
//
//   1. **Optional properties appear in declaration order**, not in an arbitrary
//      permutation — modelling every ordering is exponential in the property count.
//      llama.cpp's own converter has the same limitation. (Until 0.19 optional
//      properties were *dropped* instead, which was a bug rather than a
//      simplification: the grammar actively forbade a field the schema declared.)
//   2. **An absent `required` means "all properties required"**, where JSON Schema
//      says it means "none are". Structured-output callers essentially always want
//      the full declared shape, and OpenAI's strict `json_schema` mode requires
//      `required` to list every property anyway. An explicitly empty `"required": []`
//      is honoured literally, so an all-optional object is still expressible.

use std::sync::Arc;

use serde_json::Value;

use crate::api::types::ResponseFormat;

/// Shared primitive rules appended to every generated grammar. `value`/`object`/`array`
/// back untyped nodes and `additionalProperties`-style "any JSON" positions.
const PREAMBLE: &str = r#"ws ::= [ \t\n]*
value ::= object | array | string | number | boolean | null
object ::= "{" ws ( string ws ":" ws value ws ( "," ws string ws ":" ws value ws )* )? "}"
array ::= "[" ws ( value ws ( "," ws value ws )* )? "]"
string ::= "\"" ( [^"\\] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] ) )* "\""
integer ::= "-"? ( "0" | [1-9] [0-9]* )
number ::= "-"? ( "0" | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?
boolean ::= "true" | "false"
null ::= "null"
"#;

/// A permissive grammar matching any JSON value — backs OpenAI `response_format:
/// { "type": "json_object" }` and Ollama `format: "json"`.
pub fn any_json_gbnf() -> String {
    format!("root ::= ws value ws\n{PREAMBLE}")
}

/// Convert a JSON Schema value into a GBNF grammar whose `root` matches a conforming
/// JSON value. Returns `Err` with a human-readable reason for unsupported constructs.
pub fn schema_to_gbnf(schema: &Value) -> Result<String, String> {
    let mut b = Builder::default();
    let root = b.rule_for(schema)?;
    let mut out = format!("root ::= ws {root} ws\n");
    for (name, def) in &b.rules {
        out.push_str(&format!("{name} ::= {def}\n"));
    }
    out.push_str(PREAMBLE);
    Ok(out)
}

/// Grammar for an OpenAI `response_format`, if it requests constrained output.
/// `text` → `None` (unconstrained); `json_object` → any JSON; `json_schema` → the
/// schema (or any JSON when no schema is supplied). `Err` (→ HTTP 400) on a schema that
/// can't be converted.
pub fn grammar_from_response_format(rf: &ResponseFormat) -> Result<Option<Arc<str>>, String> {
    match rf.format_type.as_str() {
        "text" => Ok(None),
        "json_object" => Ok(Some(Arc::from(any_json_gbnf()))),
        "json_schema" => match rf.json_schema.as_ref().and_then(|s| s.schema.as_ref()) {
            Some(schema) => Ok(Some(Arc::from(schema_to_gbnf(schema)?))),
            None => Ok(Some(Arc::from(any_json_gbnf()))),
        },
        other => Err(format!("unsupported response_format type {other:?}")),
    }
}

/// Grammar for an Ollama `format` field: the string `"json"` → any JSON, a JSON schema
/// object → that schema, anything else → `None`. `Err` (→ HTTP 400) on a bad schema.
pub fn grammar_from_ollama_format(format: Option<&Value>) -> Result<Option<Arc<str>>, String> {
    match format {
        Some(Value::String(s)) if s == "json" => Ok(Some(Arc::from(any_json_gbnf()))),
        Some(v) if v.is_object() => Ok(Some(Arc::from(schema_to_gbnf(v)?))),
        _ => Ok(None),
    }
}

#[derive(Default)]
struct Builder {
    rules: Vec<(String, String)>,
    counter: usize,
}

impl Builder {
    /// Register a new named rule and return its name.
    fn add(&mut self, prefix: &str, def: String) -> String {
        let name = format!("{prefix}-{}", self.counter);
        self.counter += 1;
        self.rules.push((name.clone(), def));
        name
    }

    /// Return a GBNF expression (a rule name) matching a value conforming to `schema`.
    fn rule_for(&mut self, schema: &Value) -> Result<String, String> {
        // A boolean schema (`true`/`false`) or non-object is treated as "any JSON".
        let Some(obj) = schema.as_object() else {
            return Ok("value".to_string());
        };

        // `enum` takes precedence: an alternation of the exact JSON literals.
        if let Some(en) = obj.get("enum").and_then(|v| v.as_array()) {
            if en.is_empty() {
                return Err("enum must list at least one value".to_string());
            }
            let alts: Vec<String> = en.iter().map(json_literal).collect();
            return Ok(self.add("enum", alts.join(" | ")));
        }

        match obj.get("type").and_then(|v| v.as_str()) {
            None => Ok("value".to_string()), // untyped → any JSON value
            Some("string") => Ok("string".to_string()),
            Some("integer") => Ok("integer".to_string()),
            Some("number") => Ok("number".to_string()),
            Some("boolean") => Ok("boolean".to_string()),
            Some("null") => Ok("null".to_string()),
            Some("array") => {
                let item = match obj.get("items") {
                    Some(items) => self.rule_for(items)?,
                    None => "value".to_string(),
                };
                let def = format!(
                    "\"[\" ws ( {item} ws ( \",\" ws {item} ws )* )? \"]\"",
                    item = item
                );
                Ok(self.add("arr", def))
            }
            Some("object") => self.object_rule(obj),
            Some(other) => Err(format!("unsupported schema type: {other:?}")),
        }
    }

    fn object_rule(&mut self, obj: &serde_json::Map<String, Value>) -> Result<String, String> {
        let props = match obj.get("properties").and_then(|v| v.as_object()) {
            Some(p) if !p.is_empty() => p,
            // Object with no declared properties → any JSON object.
            _ => return Ok("object".to_string()),
        };
        // An *absent* `required` and an explicitly *empty* one mean different things
        // and must not be collapsed: `"required": []` says nothing is required, and
        // honouring that is what makes an all-optional object expressible at all.
        let required_field = obj.get("required").and_then(|v| v.as_array());
        let required: Vec<&str> = required_field
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();

        // With `required` absent entirely, every declared property is treated as
        // required. That is a deliberate divergence from JSON Schema (where absent
        // `required` requires nothing): for structured output the caller almost always
        // wants the whole declared shape, and OpenAI's own strict `json_schema` mode
        // demands `required` list every property anyway.
        if required_field.is_none() {
            let keys: Vec<&String> = props.keys().collect();
            return self.emit_object(props, &keys, &[]);
        }

        // Required keys in the caller's order (serde_json's Map is alphabetically
        // sorted, so declaration order is not recoverable from `props` alone).
        let req_keys: Vec<&String> = required
            .iter()
            .filter_map(|r| props.get_key_value(*r).map(|(k, _)| k))
            .collect();
        // Everything else is optional — and must be *allowed*, not omitted.
        let opt_keys: Vec<&String> = props
            .keys()
            .filter(|k| !required.contains(&k.as_str()))
            .collect();

        if req_keys.is_empty() && opt_keys.is_empty() {
            return Ok("object".to_string());
        }
        self.emit_object(props, &req_keys, &opt_keys)
    }

    /// Emit an object rule: `req` keys always present in order, then `opt` keys each
    /// independently present-or-absent, also in order.
    ///
    /// Optional properties used to be dropped from the grammar entirely, which did not
    /// merely make the grammar stricter — it made it **wrong**: the model was forbidden
    /// from ever emitting a declared optional field, so a schema like
    /// `{properties:{a,b}, required:[a]}` could never produce `b`.
    ///
    /// Known limitation (shared with llama.cpp's own converter): optional properties may
    /// only appear in declaration order, not in an arbitrary permutation. Modelling every
    /// permutation is exponential in the property count; declaration order covers what
    /// models actually emit, since they follow the schema's own ordering.
    fn emit_object(
        &mut self,
        props: &serde_json::Map<String, Value>,
        req: &[&String],
        opt: &[&String],
    ) -> Result<String, String> {
        // One `"key" ws ":" ws <rule> ws` member, without any separator.
        let member = |b: &mut Self, key: &String| -> Result<String, String> {
            let sub = b.rule_for(&props[key])?;
            Ok(format!(
                "{} ws {} ws {sub} ws",
                json_literal(&Value::String(key.clone())),
                str_literal(":")
            ))
        };

        let comma = str_literal(",");
        let mut body = String::new();

        if req.is_empty() {
            // No required keys: the first *present* optional carries no leading comma,
            // so a flat list of independently-optional groups would emit `{, "b": 1}`.
            // Build a chain instead, where `opt-i` means "at least one of properties
            // i.. is present, in order":
            //     opt-i ::= member-i ( "," ws opt-{i+1} )? | opt-{i+1}
            //     opt-last ::= member-last
            // and the whole chain is itself optional, so `{}` stays valid.
            let mut next: Option<String> = None;
            for key in opt.iter().rev() {
                let m = member(self, key)?;
                let def = match &next {
                    Some(n) => format!("{m} ( {comma} ws {n} )? | {n}"),
                    None => m,
                };
                next = Some(self.add("optchain", def));
            }
            if let Some(chain) = next {
                body = format!("( {chain} )?");
            }
        } else {
            for (i, key) in req.iter().enumerate() {
                let m = member(self, key)?;
                if i > 0 {
                    body.push_str(&format!("{comma} ws "));
                }
                body.push_str(&m);
                body.push(' ');
            }
            // A required member always precedes these, so each optional can carry its
            // own leading comma and stay independent of the others.
            for key in opt {
                let m = member(self, key)?;
                body.push_str(&format!("( {comma} ws {m} )? "));
            }
        }

        let def = format!("{} ws {body}{}", str_literal("{"), str_literal("}"));
        Ok(self.add("obj", def))
    }
}

/// GBNF double-quoted literal matching the exact text `s` (escaping GBNF metacharacters).
fn str_literal(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// GBNF literal matching the exact JSON serialization of `v` (e.g. a string enum value
/// becomes a literal *including* its surrounding quotes).
fn json_literal(v: &Value) -> String {
    // serde_json::to_string is infallible for plain Values.
    let json = serde_json::to_string(v).unwrap_or_default();
    str_literal(&json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn any_json_has_root_and_primitives() {
        let g = any_json_gbnf();
        assert!(g.starts_with("root ::= ws value ws"));
        assert!(g.contains("object ::="));
        assert!(g.contains("string ::="));
    }

    #[test]
    fn primitive_types_map_to_preamble_rules() {
        assert!(schema_to_gbnf(&json!({"type": "string"}))
            .unwrap()
            .starts_with("root ::= ws string ws"));
        assert!(schema_to_gbnf(&json!({"type": "integer"}))
            .unwrap()
            .starts_with("root ::= ws integer ws"));
        assert!(schema_to_gbnf(&json!({"type": "boolean"}))
            .unwrap()
            .starts_with("root ::= ws boolean ws"));
    }

    #[test]
    fn object_emits_keys_in_order_with_literals() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });
        let g = schema_to_gbnf(&schema).unwrap();
        // The object rule quotes each key literal and separates pairs with a comma.
        assert!(
            g.contains(r#""\"name\"" ws ":" ws string ws "," ws "\"age\"" ws ":" ws integer"#),
            "object rule malformed:\n{g}"
        );
        assert!(g.contains("\"{\" ws") && g.contains("\"}\""));
    }

    #[test]
    fn optional_properties_are_allowed_not_forbidden() {
        // Regression: optional properties used to be omitted from the grammar, which
        // meant the model was *forbidden* from emitting a field the schema declares.
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a"]
        });
        let g = schema_to_gbnf(&schema).unwrap();
        assert!(g.contains(r#""\"a\"""#), "required key a must appear:\n{g}");
        assert!(
            g.contains(r#""\"b\"""#),
            "optional key b must be allowed, not dropped:\n{g}"
        );
        // ...and it must be optional, not mandatory.
        assert!(
            g.contains(r#"( "," ws "\"b\"""#),
            "optional key b must sit in an optional group:\n{g}"
        );
    }

    #[test]
    fn all_optional_object_never_emits_a_leading_comma() {
        // With no required keys, a flat list of optional groups would allow
        // `{, "b": 1}`. The chain construction exists to prevent exactly that.
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": []
        });
        let g = schema_to_gbnf(&schema).unwrap();
        assert!(
            g.contains("optchain"),
            "all-optional object should use the chain construction:\n{g}"
        );
        assert!(
            !g.contains(r#""{" ws ( "," "#),
            "grammar must not allow a leading comma:\n{g}"
        );
        // Both keys reachable, and the whole body optional so `{}` stays valid.
        assert!(g.contains(r#""\"a\"""#) && g.contains(r#""\"b\"""#), "{g}");
    }

    #[test]
    fn absent_required_still_means_every_property() {
        // Deliberate divergence from JSON Schema, documented in object_rule.
        let schema = json!({
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}}
        });
        let g = schema_to_gbnf(&schema).unwrap();
        assert!(
            !g.contains("optchain"),
            "no property should be optional:\n{g}"
        );
        assert!(g.contains(r#""\"a\"""#) && g.contains(r#""\"b\"""#), "{g}");
    }

    #[test]
    fn array_of_items_nests() {
        let schema = json!({"type": "array", "items": {"type": "number"}});
        let g = schema_to_gbnf(&schema).unwrap();
        assert!(
            g.contains("\"[\" ws ( number ws"),
            "array rule malformed:\n{g}"
        );
    }

    #[test]
    fn enum_becomes_literal_alternation() {
        let schema = json!({"enum": ["red", "green", 3]});
        let g = schema_to_gbnf(&schema).unwrap();
        // string enum values keep their quotes; numbers do not.
        assert!(
            g.contains(r#""\"red\"" | "\"green\"" | "3""#),
            "enum rule malformed:\n{g}"
        );
    }

    #[test]
    fn nested_object_generates_distinct_rules() {
        let schema = json!({
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"]
                }
            },
            "required": ["user"]
        });
        let g = schema_to_gbnf(&schema).unwrap();
        // Two distinct object rules (outer + nested) plus the preamble.
        assert!(
            g.matches("obj-").count() >= 3,
            "expected outer+nested obj rules:\n{g}"
        );
    }

    #[test]
    fn untyped_schema_is_any_value() {
        assert!(schema_to_gbnf(&json!({}))
            .unwrap()
            .starts_with("root ::= ws value ws"));
    }

    #[test]
    fn unsupported_type_errors() {
        let err = schema_to_gbnf(&json!({"type": "widget"})).unwrap_err();
        assert!(err.contains("unsupported"), "got: {err}");
    }

    use crate::api::types::{JsonSchemaFormat, ResponseFormat};

    fn rf(t: &str, schema: Option<Value>) -> ResponseFormat {
        ResponseFormat {
            format_type: t.to_string(),
            json_schema: schema.map(|s| JsonSchemaFormat {
                name: "x".to_string(),
                strict: None,
                schema: Some(s),
            }),
        }
    }

    #[test]
    fn response_format_text_is_unconstrained() {
        assert!(grammar_from_response_format(&rf("text", None))
            .unwrap()
            .is_none());
    }

    #[test]
    fn response_format_json_object_is_any_json() {
        let g = grammar_from_response_format(&rf("json_object", None))
            .unwrap()
            .unwrap();
        assert!(g.starts_with("root ::= ws value ws"));
    }

    #[test]
    fn response_format_json_schema_uses_schema() {
        let g = grammar_from_response_format(&rf("json_schema", Some(json!({"type": "string"}))))
            .unwrap()
            .unwrap();
        assert!(g.starts_with("root ::= ws string ws"));
    }

    #[test]
    fn response_format_json_schema_without_schema_falls_back_to_any() {
        let g = grammar_from_response_format(&rf("json_schema", None))
            .unwrap()
            .unwrap();
        assert!(g.starts_with("root ::= ws value ws"));
    }

    #[test]
    fn ollama_format_json_string_and_schema_object() {
        assert!(grammar_from_ollama_format(Some(&json!("json")))
            .unwrap()
            .unwrap()
            .starts_with("root ::= ws value ws"));
        assert!(
            grammar_from_ollama_format(Some(&json!({"type": "integer"})))
                .unwrap()
                .unwrap()
                .starts_with("root ::= ws integer ws")
        );
        assert!(grammar_from_ollama_format(None).unwrap().is_none());
    }
}
