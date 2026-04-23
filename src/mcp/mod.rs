// MCP (Model Context Protocol) server — JSON-RPC 2.0 over stdio transport.
// Reads Content-Length framed messages from stdin, writes responses to stdout.
// All diagnostic logging goes to stderr (stdout is reserved for the protocol).

use std::io::{BufRead, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::model_registry::ModelRegistry;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub(crate) struct JsonRpcRequest {
    #[allow(dead_code)]
    pub jsonrpc: String,
    pub id: Option<serde_json::Value>,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub(crate) struct JsonRpcResponse {
    pub jsonrpc: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
pub(crate) struct JsonRpcError {
    pub code: i64,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

impl JsonRpcResponse {
    fn success(id: Option<serde_json::Value>, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Option<serde_json::Value>, code: i64, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// JSON-RPC error codes
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
const INTERNAL_ERROR: i64 = -32603;

// ---------------------------------------------------------------------------
// Content-Length framing (LSP-style)
// ---------------------------------------------------------------------------

/// Read one Content-Length framed JSON message from the reader.
/// Returns `None` on EOF.
pub(crate) fn read_message(reader: &mut impl BufRead) -> std::io::Result<Option<String>> {
    let mut content_length: Option<usize> = None;
    let mut header_line = String::new();

    loop {
        header_line.clear();
        let n = reader.read_line(&mut header_line)?;
        if n == 0 {
            return Ok(None); // EOF
        }
        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            break; // end of headers
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            if let Ok(len) = val.trim().parse::<usize>() {
                content_length = Some(len);
            }
        }
    }

    let len = match content_length {
        Some(l) => l,
        None => return Ok(None),
    };

    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    Ok(Some(String::from_utf8_lossy(&body).into_owned()))
}

/// Write one Content-Length framed JSON message to the writer.
pub(crate) fn write_message(writer: &mut impl Write, body: &str) -> std::io::Result<()> {
    write!(writer, "Content-Length: {}\r\n\r\n{}", body.len(), body)?;
    writer.flush()
}

// ---------------------------------------------------------------------------
// MCP tool/resource definitions
// ---------------------------------------------------------------------------

fn tool_definitions() -> serde_json::Value {
    serde_json::json!([
        {
            "name": "generate",
            "description": "Generate text completion from a prompt.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Model name or path" },
                    "prompt": { "type": "string", "description": "Text prompt" },
                    "max_tokens": { "type": "integer", "description": "Maximum tokens to generate" },
                    "temperature": { "type": "number", "description": "Sampling temperature" }
                },
                "required": ["model", "prompt"]
            }
        },
        {
            "name": "chat",
            "description": "Chat completion with message history.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Model name or path" },
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": { "type": "string" },
                                "content": { "type": "string" }
                            },
                            "required": ["role", "content"]
                        },
                        "description": "Chat messages"
                    },
                    "max_tokens": { "type": "integer", "description": "Maximum tokens to generate" },
                    "temperature": { "type": "number", "description": "Sampling temperature" }
                },
                "required": ["model", "messages"]
            }
        },
        {
            "name": "embed",
            "description": "Generate embeddings for text input.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model": { "type": "string", "description": "Model name or path" },
                    "input": {
                        "description": "Text or array of texts to embed",
                        "oneOf": [
                            { "type": "string" },
                            { "type": "array", "items": { "type": "string" } }
                        ]
                    }
                },
                "required": ["model", "input"]
            }
        }
    ])
}

fn resource_definitions() -> serde_json::Value {
    serde_json::json!([
        {
            "uri": "fox://models",
            "name": "models",
            "description": "List available models.",
            "mimeType": "application/json"
        }
    ])
}

// ---------------------------------------------------------------------------
// MCP server
// ---------------------------------------------------------------------------

pub(crate) struct McpServer {
    registry: Arc<ModelRegistry>,
    models_dir: std::path::PathBuf,
}

impl McpServer {
    pub fn new(registry: Arc<ModelRegistry>, models_dir: std::path::PathBuf) -> Self {
        Self {
            registry,
            models_dir,
        }
    }

    /// Run the stdio event loop. Blocks until stdin is closed.
    pub async fn run(self) -> anyhow::Result<()> {
        let server = Arc::new(self);

        // Spawn a blocking reader thread since stdin is synchronous.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        std::thread::spawn(move || {
            let stdin = std::io::stdin();
            let mut reader = stdin.lock();
            loop {
                match read_message(&mut reader) {
                    Ok(Some(msg)) => {
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                    Ok(None) => break, // EOF
                    Err(e) => {
                        eprintln!("mcp: stdin read error: {e}");
                        break;
                    }
                }
            }
        });

        while let Some(msg) = rx.recv().await {
            let server = server.clone();
            let response = server.handle_message(&msg).await;
            if let Some(resp) = response {
                let body = serde_json::to_string(&resp).unwrap_or_default();
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                if let Err(e) = write_message(&mut out, &body) {
                    eprintln!("mcp: stdout write error: {e}");
                    break;
                }
            }
        }

        Ok(())
    }

    async fn handle_message(&self, raw: &str) -> Option<JsonRpcResponse> {
        let req: JsonRpcRequest = match serde_json::from_str(raw) {
            Ok(r) => r,
            Err(e) => {
                return Some(JsonRpcResponse::error(
                    None,
                    -32700, // Parse error
                    format!("invalid JSON: {e}"),
                ));
            }
        };

        // Notifications (no id) do not get responses.
        req.id.as_ref()?;

        let result = match req.method.as_str() {
            "initialize" => self.handle_initialize(),
            "initialized" => return None, // notification acknowledgment
            "tools/list" => self.handle_tools_list(),
            "tools/call" => self.handle_tools_call(&req.params).await,
            "resources/list" => self.handle_resources_list(),
            "resources/read" => self.handle_resources_read(&req.params),
            _ => Err((METHOD_NOT_FOUND, format!("unknown method: {}", req.method))),
        };

        Some(match result {
            Ok(val) => JsonRpcResponse::success(req.id, val),
            Err((code, msg)) => JsonRpcResponse::error(req.id, code, msg),
        })
    }

    fn handle_initialize(&self) -> Result<serde_json::Value, (i64, String)> {
        Ok(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {}
            },
            "serverInfo": {
                "name": "fox",
                "version": env!("CARGO_PKG_VERSION")
            }
        }))
    }

    fn handle_tools_list(&self) -> Result<serde_json::Value, (i64, String)> {
        Ok(serde_json::json!({ "tools": tool_definitions() }))
    }

    fn handle_resources_list(&self) -> Result<serde_json::Value, (i64, String)> {
        Ok(serde_json::json!({ "resources": resource_definitions() }))
    }

    fn handle_resources_read(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, (i64, String)> {
        let uri = params
            .get("uri")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing uri".to_string()))?;

        match uri {
            "fox://models" => {
                let models = crate::cli::list_models(&self.models_dir).unwrap_or_default();
                let list: Vec<serde_json::Value> = models
                    .iter()
                    .map(|(path, meta)| {
                        let name = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("unknown");
                        serde_json::json!({
                            "name": name,
                            "size": meta.len(),
                        })
                    })
                    .collect();

                Ok(serde_json::json!({
                    "contents": [{
                        "uri": "fox://models",
                        "mimeType": "application/json",
                        "text": serde_json::to_string(&list).unwrap_or_else(|_| "[]".to_string())
                    }]
                }))
            }
            _ => Err((INVALID_PARAMS, format!("unknown resource uri: {uri}"))),
        }
    }

    async fn handle_tools_call(
        &self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, (i64, String)> {
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing tool name".to_string()))?;
        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

        match name {
            "generate" => self.tool_generate(&arguments).await,
            "chat" => self.tool_chat(&arguments).await,
            "embed" => self.tool_embed(&arguments).await,
            _ => Err((INVALID_PARAMS, format!("unknown tool: {name}"))),
        }
    }

    // -----------------------------------------------------------------------
    // Tool implementations
    // -----------------------------------------------------------------------

    async fn tool_generate(
        &self,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, (i64, String)> {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing model".to_string()))?;
        let prompt = args
            .get("prompt")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing prompt".to_string()))?;
        let max_tokens = args
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let temperature = args
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8) as f32;

        let entry = self
            .registry
            .get_or_load(model)
            .await
            .map_err(|e| (INTERNAL_ERROR, format!("failed to load model: {e}")))?;
        let engine = &entry.engine;

        let messages = vec![("user".to_string(), prompt.to_string())];
        let prompt_text = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
            messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        });

        let tokens = engine
            .tokenize(&prompt_text)
            .map_err(|e| (INTERNAL_ERROR, format!("tokenize failed: {e}")))?;

        let text = run_inference(engine, tokens, max_tokens, temperature).await?;

        Ok(mcp_text_result(&text))
    }

    async fn tool_chat(
        &self,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, (i64, String)> {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing model".to_string()))?;
        let messages_val = args
            .get("messages")
            .and_then(|v| v.as_array())
            .ok_or_else(|| (INVALID_PARAMS, "missing messages array".to_string()))?;
        let max_tokens = args
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(512) as usize;
        let temperature = args
            .get("temperature")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.8) as f32;

        let messages: Vec<(String, String)> = messages_val
            .iter()
            .filter_map(|m| {
                let role = m.get("role")?.as_str()?.to_string();
                let content = m.get("content")?.as_str()?.to_string();
                Some((role, content))
            })
            .collect();

        if messages.is_empty() {
            return Err((INVALID_PARAMS, "messages array is empty".to_string()));
        }

        let entry = self
            .registry
            .get_or_load(model)
            .await
            .map_err(|e| (INTERNAL_ERROR, format!("failed to load model: {e}")))?;
        let engine = &entry.engine;

        let prompt_text = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
            messages
                .iter()
                .map(|(r, c)| format!("{r}: {c}"))
                .collect::<Vec<_>>()
                .join("\n")
        });

        let tokens = engine
            .tokenize(&prompt_text)
            .map_err(|e| (INTERNAL_ERROR, format!("tokenize failed: {e}")))?;

        let text = run_inference(engine, tokens, max_tokens, temperature).await?;

        Ok(mcp_text_result(&text))
    }

    async fn tool_embed(
        &self,
        args: &serde_json::Value,
    ) -> Result<serde_json::Value, (i64, String)> {
        let model = args
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or_else(|| (INVALID_PARAMS, "missing model".to_string()))?;
        let input = args
            .get("input")
            .ok_or_else(|| (INVALID_PARAMS, "missing input".to_string()))?;

        let texts: Vec<String> = if let Some(s) = input.as_str() {
            vec![s.to_string()]
        } else if let Some(arr) = input.as_array() {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        } else {
            return Err((
                INVALID_PARAMS,
                "input must be a string or array".to_string(),
            ));
        };

        let entry = self
            .registry
            .get_or_load(model)
            .await
            .map_err(|e| (INTERNAL_ERROR, format!("failed to load model: {e}")))?;
        let engine = &entry.engine;

        let mut embeddings = Vec::with_capacity(texts.len());
        for text in &texts {
            let emb = engine
                .embed(text)
                .await
                .map_err(|e| (INTERNAL_ERROR, format!("embedding failed: {e}")))?;
            embeddings.push(emb);
        }

        let result = serde_json::to_string(&embeddings).unwrap_or_else(|_| "[]".to_string());

        Ok(mcp_text_result(&result))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn mcp_text_result(text: &str) -> serde_json::Value {
    serde_json::json!({
        "content": [{
            "type": "text",
            "text": text
        }]
    })
}

async fn run_inference(
    engine: &crate::engine::InferenceEngine,
    tokens: Vec<i32>,
    max_tokens: usize,
    temperature: f32,
) -> Result<String, (i64, String)> {
    use crate::scheduler::{InferenceRequest, SamplingParams};

    let sampling = SamplingParams {
        temperature,
        top_p: 0.9,
        top_k: 0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: None,
        stop: None,
        show_thinking: false,
        initial_in_thinking: false,
        max_thinking_chars: 8192,
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, tokens, max_tokens, sampling, tx);
    engine.submit_request(req);

    let mut result = String::new();
    while let Some(token) = rx.recv().await {
        if !token.text.is_empty() {
            result.push_str(&token.text);
        }
        if token.stop_reason.is_some() {
            break;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_message_basic() {
        let input = b"Content-Length: 17\r\n\r\n{\"hello\":\"world\"}";
        let mut cursor = std::io::Cursor::new(input.as_slice());
        let msg = read_message(&mut cursor).unwrap();
        assert_eq!(msg, Some("{\"hello\":\"world\"}".to_string()));
    }

    #[test]
    fn test_read_message_eof() {
        let input = b"";
        let mut cursor = std::io::Cursor::new(input.as_slice());
        let msg = read_message(&mut cursor).unwrap();
        assert!(msg.is_none());
    }

    #[test]
    fn test_write_message() {
        let mut buf = Vec::new();
        write_message(&mut buf, "{\"ok\":true}").unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.starts_with("Content-Length: 11\r\n\r\n"));
        assert!(output.ends_with("{\"ok\":true}"));
    }

    #[test]
    fn test_roundtrip_message() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#;
        let mut buf = Vec::new();
        write_message(&mut buf, body).unwrap();

        let mut cursor = std::io::Cursor::new(buf.as_slice());
        let read_back = read_message(&mut cursor).unwrap();
        assert_eq!(read_back, Some(body.to_string()));
    }

    #[test]
    fn test_json_rpc_response_success_serialization() {
        let resp = JsonRpcResponse::success(Some(serde_json::json!(1)), serde_json::json!("ok"));
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 1);
        assert_eq!(json["result"], "ok");
        assert!(json.get("error").is_none());
    }

    #[test]
    fn test_json_rpc_response_error_serialization() {
        let resp = JsonRpcResponse::error(Some(serde_json::json!(2)), -32600, "invalid request");
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["jsonrpc"], "2.0");
        assert_eq!(json["id"], 2);
        assert_eq!(json["error"]["code"], -32600);
        assert_eq!(json["error"]["message"], "invalid request");
        assert!(json.get("result").is_none());
    }

    #[test]
    fn test_json_rpc_request_deserialization() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    #[test]
    fn test_json_rpc_request_with_params() {
        let raw = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"generate","arguments":{"model":"llama","prompt":"hi"}}}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.method, "tools/call");
        assert_eq!(req.params["name"], "generate");
        assert_eq!(req.params["arguments"]["prompt"], "hi");
    }

    #[test]
    fn test_tool_definitions_has_all_tools() {
        let tools = tool_definitions();
        let arr = tools.as_array().unwrap();
        let names: Vec<&str> = arr.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"generate"));
        assert!(names.contains(&"chat"));
        assert!(names.contains(&"embed"));
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_resource_definitions_has_models() {
        let resources = resource_definitions();
        let arr = resources.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["uri"], "fox://models");
    }

    #[test]
    fn test_mcp_text_result_format() {
        let result = mcp_text_result("hello world");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "hello world");
    }

    #[tokio::test]
    async fn test_handle_initialize() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "fox");
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
    }

    #[tokio::test]
    async fn test_handle_tools_list() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_none());
        let tools = resp.result.unwrap()["tools"].as_array().unwrap().clone();
        assert_eq!(tools.len(), 3);
    }

    #[tokio::test]
    async fn test_handle_resources_list() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":3,"method":"resources/list"}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_none());
        let resources = resp.result.unwrap()["resources"]
            .as_array()
            .unwrap()
            .clone();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0]["uri"], "fox://models");
    }

    #[tokio::test]
    async fn test_handle_resources_read_models() {
        let dir = tempfile::tempdir().unwrap();
        // Create a fake model file
        std::fs::write(dir.path().join("test-model.gguf"), b"fake").unwrap();

        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw =
            r#"{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"fox://models"}}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_none());
        let result = resp.result.unwrap();
        let contents = result["contents"].as_array().unwrap();
        assert_eq!(contents[0]["uri"], "fox://models");
        let text = contents[0]["text"].as_str().unwrap();
        let models: Vec<serde_json::Value> = serde_json::from_str(text).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["name"], "test-model");
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":5,"method":"nonexistent/method"}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn test_handle_invalid_json() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let resp = server.handle_message("not valid json").await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32700);
    }

    #[tokio::test]
    async fn test_handle_notification_no_response() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","method":"initialized"}"#;
        let resp = server.handle_message(raw).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn test_handle_resources_read_unknown_uri() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":6,"method":"resources/read","params":{"uri":"fox://unknown"}}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
    }

    #[tokio::test]
    async fn test_handle_tools_call_unknown_tool() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::model_registry::RegistryConfig {
            models_dir: dir.path().to_path_buf(),
            max_models: 1,
            max_batch_size: 4,
            max_context_len: Some(512),
            block_size: 16,
            gpu_memory_bytes: 4 * 1024 * 1024,
            gpu_memory_fraction: 0.9,
            metrics: None,
            keep_alive_secs: 0,
            type_k: 1,
            type_v: 1,
            main_gpu: 0,
            split_mode: 1,
            tensor_split: vec![],
            moe_offload_cpu: false,
            mmproj_path: None,
            vision_contexts: 1,
            discovered_models: vec![],
            flash_attn: true,
        };
        let registry = Arc::new(ModelRegistry::new(cfg, std::collections::HashMap::new()));
        let server = McpServer::new(registry, dir.path().to_path_buf());

        let raw = r#"{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"nonexistent","arguments":{}}}"#;
        let resp = server.handle_message(raw).await.unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
    }
}
