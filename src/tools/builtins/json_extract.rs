//! `json_extract` builtin — pull a value out of a JSON document by path.
//!
//! Uses RFC 6901 JSON Pointer (`/foo/0/bar`) which is what `serde_json`
//! understands natively. Pure function: no I/O, no hidden state.

use async_trait::async_trait;

use crate::tools::{ToolCtx, ToolDescriptor, ToolError, ToolHandler};

pub struct JsonExtract;

#[async_trait]
impl ToolHandler for JsonExtract {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "json_extract".into(),
            description:
                "Extract a value from a JSON document using an RFC 6901 JSON Pointer path \
                 (e.g. '/results/0/title')."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "data": {
                        "description": "JSON document to extract from",
                    },
                    "path": {
                        "type": "string",
                        "description": "JSON Pointer path. Empty string returns the whole document.",
                    }
                },
                "required": ["data", "path"]
            }),
        }
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        _ctx: &ToolCtx,
    ) -> Result<serde_json::Value, ToolError> {
        let data = args
            .get("data")
            .ok_or_else(|| ToolError::InvalidArgs("missing 'data' field".into()))?;
        let path = args
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("missing 'path' string field".into()))?;

        match data.pointer(path) {
            Some(v) => Ok(v.clone()),
            None => Err(ToolError::Execution(format!(
                "JSON Pointer '{path}' did not resolve to any value"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn extracts_nested_value() {
        let tool = JsonExtract;
        let args = json!({
            "data": {"users": [{"name": "alice"}, {"name": "bob"}]},
            "path": "/users/1/name"
        });
        let v = tool.invoke(args, &ToolCtx::default()).await.unwrap();
        assert_eq!(v, json!("bob"));
    }

    #[tokio::test]
    async fn empty_path_returns_whole_document() {
        let tool = JsonExtract;
        let args = json!({"data": {"k": 1}, "path": ""});
        let v = tool.invoke(args, &ToolCtx::default()).await.unwrap();
        assert_eq!(v, json!({"k": 1}));
    }

    #[tokio::test]
    async fn missing_data_field_is_invalid_args() {
        let tool = JsonExtract;
        let err = tool
            .invoke(json!({"path": ""}), &ToolCtx::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs(_)));
    }

    #[tokio::test]
    async fn unresolvable_pointer_is_execution_error() {
        let tool = JsonExtract;
        let args = json!({"data": {"a": 1}, "path": "/b"});
        let err = tool.invoke(args, &ToolCtx::default()).await.unwrap_err();
        assert!(matches!(err, ToolError::Execution(_)));
    }

    #[tokio::test]
    async fn descriptor_is_well_formed() {
        let d = JsonExtract.descriptor();
        assert_eq!(d.name, "json_extract");
        assert!(d.parameters["properties"]["data"].is_object());
        assert!(d.parameters["properties"]["path"]["type"] == "string");
    }
}
