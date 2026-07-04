// JSON Schema → GBNF grammar (0.14, S2).
//
// Guided decoding constrains generation to a GBNF grammar (see the engine's
// `sample_constrained`). llama.cpp ships a JSON-schema→grammar converter, but it lives
// in `common/`, which fox does not build, so the conversion is ours.
//
// This covers the pragmatic subset that matters for structured output:
//   - `type`: object / array / string / integer / number / boolean / null
//   - `properties` + `required` on objects
//   - `items` on arrays
//   - `enum` (any JSON literal)
//   - arbitrary nesting
//   - untyped / empty schema → any JSON value
//
// Deliberate simplification: object properties are emitted in declaration order and the
// grammar requires exactly the `required` set (or every declared property when
// `required` is absent). Optional (non-required) properties are omitted from the
// grammar rather than modelled as an order-independent optional set — the constrained
// output is always a schema-valid object, just never *more* permissive than the
// required core. Full optional-property flexibility can come later.

use serde_json::Value;

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
        let required: Vec<&str> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            .unwrap_or_default();

        // Keys to emit and their order. With `required` present, emit exactly those keys
        // in the caller's order (serde_json's Map is alphabetically sorted, so we can't
        // rely on declaration order); otherwise emit every declared property.
        let keys: Vec<&String> = if required.is_empty() {
            props.keys().collect()
        } else {
            required
                .iter()
                .filter_map(|r| props.get_key_value(*r).map(|(k, _)| k))
                .collect()
        };
        if keys.is_empty() {
            return Ok("object".to_string());
        }

        let mut tokens: Vec<String> = vec![str_literal("{"), "ws".to_string()];
        for (i, key) in keys.iter().enumerate() {
            if i > 0 {
                tokens.push(str_literal(","));
                tokens.push("ws".to_string());
            }
            let sub = self.rule_for(&props[*key])?;
            tokens.push(json_literal(&Value::String((*key).clone()))); // "\"key\""
            tokens.push("ws".to_string());
            tokens.push(str_literal(":"));
            tokens.push("ws".to_string());
            tokens.push(sub);
            tokens.push("ws".to_string());
        }
        tokens.push(str_literal("}"));
        Ok(self.add("obj", tokens.join(" ")))
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
    fn required_subset_drops_optional_properties() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"}
            },
            "required": ["a"]
        });
        let g = schema_to_gbnf(&schema).unwrap();
        assert!(g.contains(r#""\"a\"""#), "required key a must appear");
        assert!(!g.contains(r#""\"b\"""#), "optional key b must be dropped");
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
}
