//! Pipelines: small DAG runtime over the [`crate::tools::ToolBoard`].
//!
//! A pipeline is a list of stages with explicit dependencies plus the
//! implicit ones derived from `{{stage_id.path}}` placeholders inside their
//! arguments. The runner topologically sorts the stages, resolves the
//! placeholders against the running outputs map and dispatches each stage
//! to its handler.
//!
//! Phase E.2 ships two stage types — `Tool` (invoke a registered tool) and
//! `Transform` (substitute a JSON expression). The `Llm` stage that lets a
//! pipeline ask a model for an intermediate completion is E.2.b work; it
//! needs the chat-completion path refactored into a reusable function and
//! is intentionally held back here.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::tools::{ToolBoard, ToolCtx, ToolError};

pub mod graph;
pub mod interpolation;

/// User-supplied pipeline. Deserialised straight from
/// `POST /v1/pipelines/run`.
#[derive(Debug, Clone, Deserialize)]
pub struct Pipeline {
    pub id: String,
    pub stages: Vec<Stage>,
}

/// One node of the DAG. The `type` tag in JSON discriminates the variant —
/// `tool` or `transform`. New stage types add a variant here, a handler in
/// [`PipelineRunner::run_stage`] and one branch each in `id`/`depends_on`/
/// `args`.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Stage {
    Tool(ToolStage),
    Transform(TransformStage),
}

/// Invoke a registered tool with `args` (after `{{…}}` substitution).
#[derive(Debug, Clone, Deserialize)]
pub struct ToolStage {
    pub id: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub tool: String,
    #[serde(default)]
    pub args: serde_json::Value,
    /// Optional per-stage byte budget. Falls back to `ToolCtx::default`.
    #[serde(default)]
    pub max_response_bytes: Option<usize>,
}

/// Substitute placeholders inside `value` and emit the result as the stage's
/// output. Useful for renaming, restructuring, or pulling a sub-tree out of
/// a previous stage's response.
#[derive(Debug, Clone, Deserialize)]
pub struct TransformStage {
    pub id: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    pub value: serde_json::Value,
}

impl Stage {
    pub fn id(&self) -> &str {
        match self {
            Self::Tool(s) => &s.id,
            Self::Transform(s) => &s.id,
        }
    }

    pub fn declared_deps(&self) -> &[String] {
        match self {
            Self::Tool(s) => &s.depends_on,
            Self::Transform(s) => &s.depends_on,
        }
    }

    /// Source JSON used both as the stage's runtime input and as the place
    /// to mine implicit `depends_on` edges from.
    pub fn args_value(&self) -> &serde_json::Value {
        match self {
            Self::Tool(s) => &s.args,
            Self::Transform(s) => &s.value,
        }
    }
}

/// Successful pipeline run. `outputs[stage_id]` is the value the stage
/// produced; `order` is the execution order so callers can replay it for
/// debugging or step-through visualisation.
#[derive(Debug, Serialize)]
pub struct PipelineRun {
    pub pipeline_id: String,
    pub order: Vec<String>,
    pub outputs: HashMap<String, serde_json::Value>,
}

#[derive(Debug)]
pub enum PipelineError {
    Graph(graph::GraphError),
    Interpolation {
        stage: String,
        source: interpolation::InterpolationError,
    },
    Tool {
        stage: String,
        source: ToolError,
    },
}

impl fmt::Display for PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Graph(e) => write!(f, "pipeline graph error: {e}"),
            Self::Interpolation { stage, source } => {
                write!(f, "stage '{stage}' interpolation failed: {source}")
            }
            Self::Tool { stage, source } => {
                write!(f, "stage '{stage}' tool execution failed: {source}")
            }
        }
    }
}

impl std::error::Error for PipelineError {}

#[derive(Clone)]
pub struct PipelineRunner {
    board: Arc<ToolBoard>,
}

impl PipelineRunner {
    pub fn new(board: Arc<ToolBoard>) -> Self {
        Self { board }
    }

    /// Validate a pipeline without running anything: builds the graph and
    /// the topological order, returning either the order or an error.
    pub fn plan(&self, pipeline: &Pipeline) -> Result<Vec<String>, PipelineError> {
        let nodes = build_dependency_graph(pipeline)?;
        graph::topological_order(&nodes).map_err(PipelineError::Graph)
    }

    /// Execute every stage in topological order, threading outputs through
    /// the running `outputs` map.
    pub async fn run(&self, pipeline: &Pipeline) -> Result<PipelineRun, PipelineError> {
        let order = self.plan(pipeline)?;
        let by_id: HashMap<&str, &Stage> = pipeline
            .stages
            .iter()
            .map(|s| (s.id(), s))
            .collect();

        let mut outputs: HashMap<String, serde_json::Value> = HashMap::new();
        for stage_id in &order {
            let stage = by_id
                .get(stage_id.as_str())
                .copied()
                .expect("topological order references only declared stages");
            let value = self.run_stage(stage, &outputs).await?;
            outputs.insert(stage_id.clone(), value);
        }
        Ok(PipelineRun {
            pipeline_id: pipeline.id.clone(),
            order,
            outputs,
        })
    }

    async fn run_stage(
        &self,
        stage: &Stage,
        outputs: &HashMap<String, serde_json::Value>,
    ) -> Result<serde_json::Value, PipelineError> {
        match stage {
            Stage::Transform(t) => {
                interpolation::substitute(&t.value, outputs).map_err(|source| {
                    PipelineError::Interpolation {
                        stage: t.id.clone(),
                        source,
                    }
                })
            }
            Stage::Tool(t) => {
                let resolved_args = interpolation::substitute(&t.args, outputs).map_err(
                    |source| PipelineError::Interpolation {
                        stage: t.id.clone(),
                        source,
                    },
                )?;
                let ctx = match t.max_response_bytes {
                    Some(n) => ToolCtx {
                        max_response_bytes: n,
                    },
                    None => ToolCtx::default(),
                };
                self.board
                    .invoke(&t.tool, resolved_args, &ctx)
                    .await
                    .map_err(|source| PipelineError::Tool {
                        stage: t.id.clone(),
                        source,
                    })
            }
        }
    }
}

/// Build the `(stage_id, deps)` table the toposort consumes. Combines the
/// explicit `depends_on` list with the implicit edges discovered by scanning
/// the stage's `args` for `{{…}}` placeholders.
fn build_dependency_graph(
    pipeline: &Pipeline,
) -> Result<Vec<(String, HashSet<String>)>, PipelineError> {
    let mut nodes = Vec::with_capacity(pipeline.stages.len());
    for stage in &pipeline.stages {
        let mut deps: HashSet<String> = stage
            .declared_deps()
            .iter()
            .cloned()
            .collect();
        let implicit = interpolation::referenced_stages(stage.args_value()).map_err(|source| {
            PipelineError::Interpolation {
                stage: stage.id().to_string(),
                source,
            }
        })?;
        deps.extend(implicit);
        // A stage cannot depend on itself, even by accident.
        deps.remove(stage.id());
        nodes.push((stage.id().to_string(), deps));
    }
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::default_board;
    use serde_json::json;

    fn pipeline_from(json_str: &str) -> Pipeline {
        serde_json::from_str(json_str).expect("pipeline parses")
    }

    #[tokio::test]
    async fn transform_only_pipeline_passes_value_through() {
        let p = pipeline_from(
            r#"{
                "id": "demo",
                "stages": [
                    {"type": "transform", "id": "hello", "value": {"greeting": "hi"}}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let run = runner.run(&p).await.unwrap();
        assert_eq!(run.order, vec!["hello"]);
        assert_eq!(run.outputs["hello"], json!({"greeting": "hi"}));
    }

    #[tokio::test]
    async fn implicit_dependency_via_placeholder_orders_stages_correctly() {
        let p = pipeline_from(
            r#"{
                "id": "demo",
                "stages": [
                    {"type": "transform", "id": "second", "value": "{{first.k}}"},
                    {"type": "transform", "id": "first",  "value": {"k": "v"}}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let run = runner.run(&p).await.unwrap();
        assert_eq!(run.order, vec!["first", "second"]);
        assert_eq!(run.outputs["second"], json!("v"));
    }

    #[tokio::test]
    async fn tool_stage_invokes_json_extract_through_the_board() {
        let p = pipeline_from(
            r#"{
                "id": "demo",
                "stages": [
                    {
                        "type": "transform",
                        "id": "data",
                        "value": {"users": [{"name": "alice"}, {"name": "bob"}]}
                    },
                    {
                        "type": "tool",
                        "id": "pick",
                        "tool": "json_extract",
                        "args": {"data": "{{data}}", "path": "/users/1/name"}
                    }
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let run = runner.run(&p).await.unwrap();
        assert_eq!(run.outputs["pick"], json!("bob"));
    }

    #[tokio::test]
    async fn cycle_is_reported_during_plan() {
        let p = pipeline_from(
            r#"{
                "id": "loop",
                "stages": [
                    {"type": "transform", "id": "a", "value": "{{b}}"},
                    {"type": "transform", "id": "b", "value": "{{a}}"}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let err = runner.run(&p).await.unwrap_err();
        assert!(matches!(err, PipelineError::Graph(graph::GraphError::Cycle(_))));
    }

    #[tokio::test]
    async fn duplicate_stage_id_is_rejected() {
        let p = pipeline_from(
            r#"{
                "id": "dup",
                "stages": [
                    {"type": "transform", "id": "x", "value": 1},
                    {"type": "transform", "id": "x", "value": 2}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let err = runner.run(&p).await.unwrap_err();
        assert!(matches!(
            err,
            PipelineError::Graph(graph::GraphError::DuplicateStageId(s)) if s == "x"
        ));
    }

    #[tokio::test]
    async fn unknown_tool_surfaces_as_tool_error() {
        let p = pipeline_from(
            r#"{
                "id": "bad",
                "stages": [
                    {"type": "tool", "id": "ghost", "tool": "no_such_tool", "args": {}}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let err = runner.run(&p).await.unwrap_err();
        match err {
            PipelineError::Tool { stage, source } => {
                assert_eq!(stage, "ghost");
                assert!(matches!(source, ToolError::NotFound(_)));
            }
            other => panic!("expected tool error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn explicit_depends_on_is_honoured_even_without_placeholders() {
        let p = pipeline_from(
            r#"{
                "id": "explicit",
                "stages": [
                    {"type": "transform", "id": "second", "depends_on": ["first"], "value": "after"},
                    {"type": "transform", "id": "first",  "value": "before"}
                ]
            }"#,
        );
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let run = runner.run(&p).await.unwrap();
        assert_eq!(run.order, vec!["first", "second"]);
    }

    #[tokio::test]
    async fn plan_returns_order_without_executing() {
        let p = pipeline_from(
            r#"{
                "id": "demo",
                "stages": [
                    {"type": "tool", "id": "a", "tool": "no_such_tool", "args": {}}
                ]
            }"#,
        );
        // Plan succeeds even though the tool doesn't exist — we only walk
        // the graph, no invocation happens.
        let runner = PipelineRunner::new(Arc::new(default_board()));
        let order = runner.plan(&p).unwrap();
        assert_eq!(order, vec!["a"]);
    }
}
