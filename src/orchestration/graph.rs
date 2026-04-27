//! Topological ordering and cycle detection over a pipeline's stages.
//!
//! Each stage exposes its dependencies through the [`StageDeps`] trait —
//! today implemented by the four built-in stage types, plus the union of
//! `depends_on` plus the implicit edges discovered from `{{…}}` placeholders
//! inside `args`. The function returns the stage ids in execution order or
//! reports a cycle by listing the participating ids.

use std::collections::{HashMap, HashSet};

#[derive(Debug, PartialEq, Eq)]
pub enum GraphError {
    DuplicateStageId(String),
    UnknownDependency { stage: String, missing: String },
    Cycle(Vec<String>),
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateStageId(id) => write!(f, "duplicate stage id '{id}'"),
            Self::UnknownDependency { stage, missing } => write!(
                f,
                "stage '{stage}' depends on '{missing}', which is not defined"
            ),
            Self::Cycle(cycle) => write!(f, "pipeline has a cycle: {}", cycle.join(" → ")),
        }
    }
}

impl std::error::Error for GraphError {}

/// Compute an execution order for `(id, deps)` pairs.
///
/// Dependencies must reference declared stage ids. The returned `Vec` lists
/// every stage id exactly once such that every stage appears after all of
/// its dependencies. Stable on the input order for siblings.
pub fn topological_order<S: AsRef<str>>(
    stages: &[(S, HashSet<String>)],
) -> Result<Vec<String>, GraphError> {
    // Validate: no duplicate ids, every dep refers to a declared id.
    let mut declared: HashMap<String, &HashSet<String>> = HashMap::with_capacity(stages.len());
    for (id, deps) in stages {
        let id = id.as_ref().to_string();
        if declared.insert(id.clone(), deps).is_some() {
            return Err(GraphError::DuplicateStageId(id));
        }
    }
    for (id, deps) in stages {
        for dep in deps.iter() {
            if !declared.contains_key(dep) {
                return Err(GraphError::UnknownDependency {
                    stage: id.as_ref().to_string(),
                    missing: dep.clone(),
                });
            }
        }
    }

    // DFS with three colours so cycles produce the actual offending path.
    let mut color: HashMap<String, GraphColor> = declared
        .keys()
        .map(|k| (k.clone(), GraphColor::White))
        .collect();
    let mut order: Vec<String> = Vec::with_capacity(stages.len());

    for (id, _) in stages {
        let id = id.as_ref();
        if color.get(id).copied() == Some(GraphColor::Black) {
            continue;
        }
        let mut path: Vec<String> = Vec::new();
        visit(id, &declared, &mut color, &mut path, &mut order)?;
    }
    Ok(order)
}

fn visit(
    id: &str,
    declared: &HashMap<String, &HashSet<String>>,
    color: &mut HashMap<String, GraphColor>,
    path: &mut Vec<String>,
    order: &mut Vec<String>,
) -> Result<(), GraphError> {
    match color.get(id).copied().unwrap_or(GraphColor::White) {
        GraphColor::Black => return Ok(()),
        GraphColor::Gray => {
            // We're back at a node currently on the recursion stack — cycle.
            let mut cycle = path.clone();
            cycle.push(id.to_string());
            // Trim the prefix that comes before the first occurrence of `id`
            // so the reported cycle is the minimal loop.
            if let Some(start) = cycle.iter().position(|n| n == id) {
                cycle.drain(..start);
            }
            return Err(GraphError::Cycle(cycle));
        }
        GraphColor::White => {}
    }
    color.insert(id.to_string(), GraphColor::Gray);
    path.push(id.to_string());

    if let Some(deps) = declared.get(id).copied() {
        // Sort dependencies for deterministic order across runs.
        let mut sorted: Vec<&String> = deps.iter().collect();
        sorted.sort();
        for dep in sorted {
            visit(dep, declared, color, path, order)?;
        }
    }

    path.pop();
    color.insert(id.to_string(), GraphColor::Black);
    order.push(id.to_string());
    Ok(())
}

/// Three-colour DFS marker. Local copy because `visit` recurses and the
/// `Color` enum from `topological_order` would not be reachable.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GraphColor {
    White,
    Gray,
    Black,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deps(items: &[&str]) -> HashSet<String> {
        items.iter().map(|s| (*s).to_string()).collect()
    }

    #[test]
    fn linear_chain_orders_in_dependency_order() {
        let stages = vec![
            ("c", deps(&["b"])),
            ("a", deps(&[])),
            ("b", deps(&["a"])),
        ];
        let order = topological_order(&stages).unwrap();
        let pos = |id: &str| order.iter().position(|s| s == id).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("b") < pos("c"));
    }

    #[test]
    fn diamond_dependency_is_resolved() {
        // a → b, a → c, both → d
        let stages = vec![
            ("a", deps(&[])),
            ("b", deps(&["a"])),
            ("c", deps(&["a"])),
            ("d", deps(&["b", "c"])),
        ];
        let order = topological_order(&stages).unwrap();
        let pos = |id: &str| order.iter().position(|s| s == id).unwrap();
        assert!(pos("a") < pos("b"));
        assert!(pos("a") < pos("c"));
        assert!(pos("b") < pos("d"));
        assert!(pos("c") < pos("d"));
    }

    #[test]
    fn duplicate_stage_id_is_rejected() {
        let stages = vec![("a", deps(&[])), ("a", deps(&[]))];
        let err = topological_order(&stages).unwrap_err();
        assert!(matches!(err, GraphError::DuplicateStageId(s) if s == "a"));
    }

    #[test]
    fn unknown_dependency_is_rejected() {
        let stages = vec![("a", deps(&["ghost"]))];
        let err = topological_order(&stages).unwrap_err();
        assert!(matches!(
            err,
            GraphError::UnknownDependency { stage, missing }
                if stage == "a" && missing == "ghost"
        ));
    }

    #[test]
    fn self_loop_is_a_cycle() {
        let stages = vec![("a", deps(&["a"]))];
        let err = topological_order(&stages).unwrap_err();
        match err {
            GraphError::Cycle(cycle) => {
                assert!(cycle.contains(&"a".to_string()));
            }
            other => panic!("expected cycle, got {other:?}"),
        }
    }

    #[test]
    fn three_node_cycle_is_reported() {
        let stages = vec![
            ("a", deps(&["c"])),
            ("b", deps(&["a"])),
            ("c", deps(&["b"])),
        ];
        let err = topological_order(&stages).unwrap_err();
        assert!(matches!(err, GraphError::Cycle(_)));
    }
}
