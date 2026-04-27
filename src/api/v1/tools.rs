//! HTTP entry points for the [`crate::tools::ToolBoard`].
//!
//! * `GET  /v1/tools` — list every registered tool's OpenAI-shaped descriptor.
//! * `POST /v1/tools/execute` — invoke a tool by name with a JSON args payload.

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;

use crate::api::router::AppState;
use crate::tools::{ToolCtx, ToolDescriptor, ToolError};

#[derive(Debug, Deserialize)]
pub struct ExecuteRequest {
    pub name: String,
    /// Arbitrary JSON payload — validated by the tool itself against its
    /// declared schema.
    #[serde(default)]
    pub args: serde_json::Value,
    /// Optional per-call output budget. Falls back to `ToolCtx::default`.
    #[serde(default)]
    pub max_response_bytes: Option<usize>,
}

pub async fn list_tools(State(state): State<AppState>) -> Json<Vec<ToolDescriptor>> {
    Json(state.tool_board.list())
}

pub async fn execute_tool(
    State(state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let ctx = match req.max_response_bytes {
        Some(n) => ToolCtx {
            max_response_bytes: n,
        },
        None => ToolCtx::default(),
    };
    state
        .tool_board
        .invoke(&req.name, req.args, &ctx)
        .await
        .map(Json)
        .map_err(error_to_status)
}

fn error_to_status(e: ToolError) -> (StatusCode, String) {
    match &e {
        ToolError::NotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
        ToolError::InvalidArgs(_) => (StatusCode::BAD_REQUEST, e.to_string()),
        ToolError::Execution(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
