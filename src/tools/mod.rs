//! Pluggable tool handlers and the registry that holds them.
//!
//! A "tool" is anything the model can call: a JSON pointer extractor, an HTTP
//! fetcher, an in-house data lookup, etc. Handlers implement [`ToolHandler`]
//! and register themselves on the shared [`ToolBoard`] at startup. The board
//! is `Send + Sync + Clone` so it can sit inside `AppState` and be reused by
//! every request without locking.
//!
//! Two HTTP entry points expose the board: `GET /v1/tools` returns the list
//! of registered descriptors (matches the OpenAI tools schema), and
//! `POST /v1/tools/execute` runs a tool by name. Function-calling responses
//! emitted during chat completion can also resolve `tool_calls.name` against
//! the board to execute server-side tools transparently — that wireup
//! arrives in E.3.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use serde::{Deserialize, Serialize};

pub mod builtins;

/// JSON Schema descriptor — same shape as `function` blocks in the OpenAI
/// tool API so a registered tool can be advertised to the model verbatim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    /// JSON Schema for the tool's `args` payload (OpenAI uses the same shape
    /// under `function.parameters`).
    pub parameters: serde_json::Value,
}

/// Errors a [`ToolHandler::invoke`] may return.
#[derive(Debug)]
pub enum ToolError {
    /// The supplied `args` failed validation against the tool's schema.
    InvalidArgs(String),
    /// The tool ran but produced an error.
    Execution(String),
    /// The named tool is not registered on the board.
    NotFound(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgs(m) => write!(f, "invalid arguments: {m}"),
            Self::Execution(m) => write!(f, "tool execution failed: {m}"),
            Self::NotFound(name) => write!(f, "tool '{name}' is not registered"),
        }
    }
}

impl std::error::Error for ToolError {}

/// Per-invocation context. Today carries a single byte budget that handlers
/// can honour to bound their output. Future fields: caller identity for
/// auditing, request id for tracing, deadline for timeout enforcement.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    pub max_response_bytes: usize,
}

impl Default for ToolCtx {
    fn default() -> Self {
        Self {
            // 1 MiB is generous for JSON-shaped responses while still
            // protecting the server from being asked to ferry a GiB.
            max_response_bytes: 1024 * 1024,
        }
    }
}

#[async_trait]
pub trait ToolHandler: Send + Sync {
    fn descriptor(&self) -> ToolDescriptor;
    async fn invoke(
        &self,
        args: serde_json::Value,
        ctx: &ToolCtx,
    ) -> Result<serde_json::Value, ToolError>;
}

/// Shared registry. `Clone` is cheap — internally an `Arc<DashMap<…>>`.
#[derive(Default, Clone)]
pub struct ToolBoard {
    handlers: Arc<DashMap<String, Arc<dyn ToolHandler>>>,
}

impl ToolBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `handler` under the name carried in its descriptor. Returns
    /// `true` when the slot was previously empty; existing registrations
    /// are overwritten so a startup script can re-register at will.
    pub fn register(&self, handler: Arc<dyn ToolHandler>) -> bool {
        let name = handler.descriptor().name;
        self.handlers.insert(name, handler).is_none()
    }

    pub fn deregister(&self, name: &str) -> bool {
        self.handlers.remove(name).is_some()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    pub fn list(&self) -> Vec<ToolDescriptor> {
        let mut out: Vec<ToolDescriptor> = self
            .handlers
            .iter()
            .map(|e| e.value().descriptor())
            .collect();
        // Stable order so `GET /v1/tools` is deterministic for clients.
        out.sort_by(|a, b| a.name.cmp(&b.name));
        out
    }

    /// Resolve `name` and forward the call. The board itself is read-only
    /// during invoke, so multiple concurrent requests can hit the same
    /// handler without contention.
    pub async fn invoke(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolCtx,
    ) -> Result<serde_json::Value, ToolError> {
        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| ToolError::NotFound(name.to_string()))?
            .clone();
        handler.invoke(args, ctx).await
    }
}

/// Build a board pre-populated with the default builtins (JSON extractor and
/// HTTP fetcher). Callers that want a different set should construct
/// `ToolBoard::new()` and register their own handlers.
pub fn default_board() -> ToolBoard {
    let board = ToolBoard::new();
    board.register(Arc::new(builtins::json_extract::JsonExtract));
    board.register(Arc::new(builtins::http_fetch::HttpFetch));
    board
}

/// Convenience: turn a board snapshot into a name-keyed map. Used by the
/// future pipelines runtime when validating a stage's `tool` reference.
#[allow(dead_code)]
pub fn descriptor_index(board: &ToolBoard) -> HashMap<String, ToolDescriptor> {
    board
        .list()
        .into_iter()
        .map(|d| (d.name.clone(), d))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    #[async_trait]
    impl ToolHandler for Echo {
        fn descriptor(&self) -> ToolDescriptor {
            ToolDescriptor {
                name: "echo".into(),
                description: "Echo the args back".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn invoke(
            &self,
            args: serde_json::Value,
            _ctx: &ToolCtx,
        ) -> Result<serde_json::Value, ToolError> {
            Ok(args)
        }
    }

    #[tokio::test]
    async fn register_and_invoke_round_trip() {
        let board = ToolBoard::new();
        let was_new = board.register(Arc::new(Echo));
        assert!(was_new);
        assert!(board.contains("echo"));

        let result = board
            .invoke("echo", serde_json::json!({"hi": 1}), &ToolCtx::default())
            .await
            .unwrap();
        assert_eq!(result, serde_json::json!({"hi": 1}));
    }

    #[tokio::test]
    async fn invoke_returns_not_found_for_missing_tool() {
        let board = ToolBoard::new();
        let err = board
            .invoke("nope", serde_json::Value::Null, &ToolCtx::default())
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::NotFound(_)));
    }

    #[tokio::test]
    async fn list_returns_descriptors_in_stable_order() {
        let board = ToolBoard::new();
        board.register(Arc::new(Echo));
        let list = board.list();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "echo");
    }

    #[tokio::test]
    async fn deregister_removes_the_handler() {
        let board = ToolBoard::new();
        board.register(Arc::new(Echo));
        assert!(board.deregister("echo"));
        assert!(!board.contains("echo"));
        // Idempotent.
        assert!(!board.deregister("echo"));
    }

    #[tokio::test]
    async fn default_board_registers_builtins() {
        let board = default_board();
        assert!(board.contains("json_extract"));
        assert!(board.contains("http_fetch"));
    }
}
