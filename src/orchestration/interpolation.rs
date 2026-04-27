//! Resolve `{{stage_id.path.to.field}}` placeholders inside a JSON value
//! using the outputs of previously-executed stages.
//!
//! Syntax:
//! * `{{foo}}` — substitute the entire output of stage `foo`.
//! * `{{foo.bar}}` — JSON Pointer-style lookup; equivalent to
//!   `outputs["foo"].pointer("/bar")`.
//! * `{{foo.bar.0.baz}}` — array indices and nested keys are written with
//!   dots; the parser converts them to a `/`-separated JSON Pointer.
//!
//! Substitution is recursive over the input JSON: every string leaf is
//! scanned for placeholders, and the substitution result keeps the JSON
//! type of the looked-up value when the entire string is exactly one
//! placeholder. Mixed strings (template + literal) coerce non-string
//! values to their JSON encoding.

use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq)]
pub enum InterpolationError {
    UnterminatedPlaceholder,
    UnknownStage(String),
    UnresolvedPath { stage: String, path: String },
}

impl std::fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedPlaceholder => write!(f, "unterminated `{{{{` placeholder"),
            Self::UnknownStage(s) => write!(f, "stage '{s}' has no output to interpolate"),
            Self::UnresolvedPath { stage, path } => write!(
                f,
                "stage '{stage}' output does not contain path '{path}'"
            ),
        }
    }
}

impl std::error::Error for InterpolationError {}

/// Walk every string leaf of `value` and substitute placeholders.
pub fn substitute(
    value: &serde_json::Value,
    outputs: &HashMap<String, serde_json::Value>,
) -> Result<serde_json::Value, InterpolationError> {
    match value {
        serde_json::Value::String(s) => substitute_string(s, outputs),
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(substitute(item, outputs)?);
            }
            Ok(serde_json::Value::Array(out))
        }
        serde_json::Value::Object(map) => {
            let mut out = serde_json::Map::with_capacity(map.len());
            for (k, v) in map {
                out.insert(k.clone(), substitute(v, outputs)?);
            }
            Ok(serde_json::Value::Object(out))
        }
        // Numbers, booleans and null pass through untouched.
        other => Ok(other.clone()),
    }
}

/// Collect the set of `stage_id`s referenced anywhere inside `value`.
/// Used by the graph builder to derive implicit `depends_on` edges so a
/// pipeline author doesn't have to declare them by hand.
pub fn referenced_stages(value: &serde_json::Value) -> Result<HashSet<String>, InterpolationError> {
    let mut acc = HashSet::new();
    walk(value, &mut acc)?;
    Ok(acc)
}

fn walk(
    value: &serde_json::Value,
    acc: &mut HashSet<String>,
) -> Result<(), InterpolationError> {
    match value {
        serde_json::Value::String(s) => {
            for placeholder in scan(s)? {
                acc.insert(placeholder.stage);
            }
            Ok(())
        }
        serde_json::Value::Array(items) => {
            for item in items {
                walk(item, acc)?;
            }
            Ok(())
        }
        serde_json::Value::Object(map) => {
            for v in map.values() {
                walk(v, acc)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn substitute_string(
    s: &str,
    outputs: &HashMap<String, serde_json::Value>,
) -> Result<serde_json::Value, InterpolationError> {
    let placeholders = scan(s)?;
    if placeholders.is_empty() {
        return Ok(serde_json::Value::String(s.to_string()));
    }

    // Special case: the entire string is exactly one placeholder. Keep the
    // looked-up value's JSON type so `args: "{{foo}}"` flows the original
    // shape (object, array, number) into the next stage rather than
    // collapsing to a string.
    if placeholders.len() == 1 {
        let p = &placeholders[0];
        if p.start == 0 && p.end == s.len() {
            return resolve(p, outputs);
        }
    }

    // Mixed: rebuild the string, coercing each lookup with `to_string()` for
    // strings or the compact JSON encoding for everything else.
    let mut out = String::with_capacity(s.len());
    let mut cursor = 0;
    for p in &placeholders {
        out.push_str(&s[cursor..p.start]);
        let value = resolve(p, outputs)?;
        match &value {
            serde_json::Value::String(sub) => out.push_str(sub),
            other => out.push_str(&other.to_string()),
        }
        cursor = p.end;
    }
    out.push_str(&s[cursor..]);
    Ok(serde_json::Value::String(out))
}

fn resolve(
    p: &Placeholder,
    outputs: &HashMap<String, serde_json::Value>,
) -> Result<serde_json::Value, InterpolationError> {
    let stage_value = outputs
        .get(&p.stage)
        .ok_or_else(|| InterpolationError::UnknownStage(p.stage.clone()))?;
    if p.segments.is_empty() {
        return Ok(stage_value.clone());
    }
    let pointer = format!("/{}", p.segments.join("/"));
    stage_value
        .pointer(&pointer)
        .cloned()
        .ok_or_else(|| InterpolationError::UnresolvedPath {
            stage: p.stage.clone(),
            path: pointer,
        })
}

/// Every `{{…}}` occurrence in a string. Returned in source order.
struct Placeholder {
    /// Byte offset of the opening `{{`.
    start: usize,
    /// Byte offset just past the closing `}}`.
    end: usize,
    /// First identifier — the stage id.
    stage: String,
    /// Subsequent dotted segments — the JSON Pointer path inside that stage.
    segments: Vec<String>,
}

fn scan(s: &str) -> Result<Vec<Placeholder>, InterpolationError> {
    let bytes = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] == b'{' && bytes[i + 1] == b'{' {
            let body_start = i + 2;
            let close = find_close(bytes, body_start)
                .ok_or(InterpolationError::UnterminatedPlaceholder)?;
            let raw = &s[body_start..close];
            let trimmed = raw.trim();
            let mut parts = trimmed.split('.');
            let stage = parts
                .next()
                .map(str::trim)
                .map(str::to_string)
                .unwrap_or_default();
            let segments: Vec<String> = parts.map(|p| p.trim().to_string()).collect();
            out.push(Placeholder {
                start: i,
                end: close + 2,
                stage,
                segments,
            });
            i = close + 2;
        } else {
            i += 1;
        }
    }
    Ok(out)
}

fn find_close(bytes: &[u8], from: usize) -> Option<usize> {
    let mut i = from;
    while i + 1 < bytes.len() {
        if bytes[i] == b'}' && bytes[i + 1] == b'}' {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn outputs() -> HashMap<String, serde_json::Value> {
        let mut o = HashMap::new();
        o.insert(
            "fetch".to_string(),
            json!({"status": 200, "body": "hello", "users": [{"name": "alice"}, {"name": "bob"}]}),
        );
        o.insert("count".to_string(), json!(42));
        o
    }

    #[test]
    fn whole_string_placeholder_preserves_value_type() {
        let v = substitute(&json!("{{count}}"), &outputs()).unwrap();
        assert_eq!(v, json!(42));
    }

    #[test]
    fn whole_string_placeholder_returns_object_unchanged() {
        let v = substitute(&json!("{{fetch}}"), &outputs()).unwrap();
        assert_eq!(v["status"], json!(200));
    }

    #[test]
    fn dotted_path_resolves_through_objects_and_arrays() {
        let v = substitute(&json!("{{fetch.users.0.name}}"), &outputs()).unwrap();
        assert_eq!(v, json!("alice"));
    }

    #[test]
    fn mixed_string_concatenates_with_coercion() {
        let v = substitute(
            &json!("status was {{fetch.status}}, body: {{fetch.body}}"),
            &outputs(),
        )
        .unwrap();
        assert_eq!(v, json!("status was 200, body: hello"));
    }

    #[test]
    fn substitution_recurses_into_objects_and_arrays() {
        let v = substitute(
            &json!({
                "summary": "user {{fetch.users.0.name}} count={{count}}",
                "raw": "{{fetch}}"
            }),
            &outputs(),
        )
        .unwrap();
        assert_eq!(v["summary"], json!("user alice count=42"));
        assert_eq!(v["raw"]["body"], json!("hello"));
    }

    #[test]
    fn unknown_stage_is_reported() {
        let err = substitute(&json!("{{ghost}}"), &outputs()).unwrap_err();
        assert!(matches!(err, InterpolationError::UnknownStage(s) if s == "ghost"));
    }

    #[test]
    fn unresolved_path_is_reported() {
        let err = substitute(&json!("{{fetch.nope}}"), &outputs()).unwrap_err();
        match err {
            InterpolationError::UnresolvedPath { stage, path } => {
                assert_eq!(stage, "fetch");
                assert_eq!(path, "/nope");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn unterminated_placeholder_is_reported() {
        let err = substitute(&json!("hello {{fetch"), &outputs()).unwrap_err();
        assert_eq!(err, InterpolationError::UnterminatedPlaceholder);
    }

    #[test]
    fn referenced_stages_collects_from_all_leaves() {
        let v = json!({
            "a": "{{x}}",
            "b": ["{{y.body}}", "no placeholder", "{{x}}"]
        });
        let refs = referenced_stages(&v).unwrap();
        assert!(refs.contains("x"));
        assert!(refs.contains("y"));
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn whitespace_inside_placeholder_is_tolerated() {
        let v = substitute(&json!("{{  count  }}"), &outputs()).unwrap();
        assert_eq!(v, json!(42));
    }

    #[test]
    fn plain_string_passes_through_untouched() {
        let v = substitute(&json!("nothing to substitute"), &outputs()).unwrap();
        assert_eq!(v, json!("nothing to substitute"));
    }
}
