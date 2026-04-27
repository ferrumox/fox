//! `http_fetch` builtin — bounded GET against an http or https URL.
//!
//! Honours [`ToolCtx::max_response_bytes`] so a runaway response cannot exhaust
//! server memory. Returns a `{ "status": <code>, "headers": {...}, "body": "..." }`
//! shape so a follow-up `json_extract` can pluck out the relevant bits in a
//! pipeline without re-parsing.

use async_trait::async_trait;

use crate::tools::{ToolCtx, ToolDescriptor, ToolError, ToolHandler};

pub struct HttpFetch;

#[async_trait]
impl ToolHandler for HttpFetch {
    fn descriptor(&self) -> ToolDescriptor {
        ToolDescriptor {
            name: "http_fetch".into(),
            description:
                "GET an http(s) URL and return its status code and body. \
                 The body is decoded as UTF-8 (lossy when the response is binary). \
                 Bounded by the caller-supplied `max_response_bytes` budget."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute http or https URL"
                    }
                },
                "required": ["url"]
            }),
        }
    }

    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &ToolCtx,
    ) -> Result<serde_json::Value, ToolError> {
        let url = args
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::InvalidArgs("missing 'url' string field".into()))?;
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(ToolError::InvalidArgs(
                "url must start with http:// or https://".into(),
            ));
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(15))
            .build()
            .map_err(|e| ToolError::Execution(format!("client build failed: {e}")))?;

        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| ToolError::Execution(format!("request failed: {e}")))?;

        let status = resp.status().as_u16();
        // Capture headers before consuming the body so the response shape
        // includes useful metadata even when the body is empty.
        let headers: serde_json::Map<String, serde_json::Value> = resp
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|s| (k.to_string(), serde_json::Value::String(s.to_string())))
            })
            .collect();

        let bytes = resp
            .bytes()
            .await
            .map_err(|e| ToolError::Execution(format!("read body failed: {e}")))?;
        if bytes.len() > ctx.max_response_bytes {
            return Err(ToolError::Execution(format!(
                "response body is {} bytes — exceeds the {}-byte budget",
                bytes.len(),
                ctx.max_response_bytes
            )));
        }
        let body = String::from_utf8_lossy(&bytes).into_owned();

        Ok(serde_json::json!({
            "status": status,
            "headers": serde_json::Value::Object(headers),
            "body": body,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn rejects_missing_url() {
        let err = HttpFetch
            .invoke(json!({}), &ToolCtx::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArgs(_)));
    }

    #[tokio::test]
    async fn rejects_non_http_scheme() {
        let err = HttpFetch
            .invoke(json!({"url": "file:///etc/passwd"}), &ToolCtx::default())
            .await
            .unwrap_err();
        match err {
            ToolError::InvalidArgs(msg) => assert!(msg.contains("http")),
            other => panic!("expected InvalidArgs, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn descriptor_advertises_url_parameter() {
        let d = HttpFetch.descriptor();
        assert_eq!(d.name, "http_fetch");
        assert!(d.parameters["properties"]["url"]["type"] == "string");
        assert!(d.parameters["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v == "url"));
    }
}
