//! `POST /v1/pipelines/run` — execute a `Pipeline` synchronously and return
//! the per-stage outputs.

use axum::{extract::State, http::StatusCode, Json};
use std::sync::Arc;

use crate::api::router::AppState;
use crate::orchestration::{Pipeline, PipelineError, PipelineRun, PipelineRunner};

pub async fn run_pipeline(
    State(state): State<AppState>,
    Json(pipeline): Json<Pipeline>,
) -> Result<Json<PipelineRun>, (StatusCode, String)> {
    let runner = PipelineRunner::new(Arc::new(state.tool_board.clone()));
    runner
        .run(&pipeline)
        .await
        .map(Json)
        .map_err(error_to_status)
}

fn error_to_status(e: PipelineError) -> (StatusCode, String) {
    use crate::orchestration::graph::GraphError;
    use crate::tools::ToolError;
    match &e {
        PipelineError::Graph(g) => match g {
            GraphError::DuplicateStageId(_) | GraphError::UnknownDependency { .. } => {
                (StatusCode::BAD_REQUEST, e.to_string())
            }
            GraphError::Cycle(_) => (StatusCode::BAD_REQUEST, e.to_string()),
        },
        PipelineError::Interpolation { .. } => (StatusCode::BAD_REQUEST, e.to_string()),
        PipelineError::Tool { source, .. } => match source {
            ToolError::NotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
            ToolError::InvalidArgs(_) => (StatusCode::BAD_REQUEST, e.to_string()),
            ToolError::Execution(_) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        },
    }
}
