// `fox inspect` — inspect a downloaded GGUF model and print a backend
// recommendation derived from its metadata (no GPU work performed).

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use crate::model_registry::{probe, recommend_backend, render_diagnostic, ModelDiagnostic};

use super::{list_models, models_dir};

#[derive(Parser, Debug)]
pub struct InspectArgs {
    /// Model name (stem), filename, or absolute path to a GGUF file.
    pub model: String,

    /// Directory to search for models (defaults to ~/.cache/ferrumox/models).
    #[arg(long)]
    pub path: Option<PathBuf>,
}

pub async fn run_inspect(args: InspectArgs) -> Result<()> {
    let path = resolve_path(&args)?;

    let profile =
        probe(&path).with_context(|| format!("failed to inspect '{}'", path.display()))?;
    let (hint, notes) = recommend_backend(&profile);
    let diagnostic = ModelDiagnostic {
        profile,
        hint,
        notes,
    };

    println!("File: {}", path.display());
    println!();
    print!("{}", render_diagnostic(&diagnostic));

    Ok(())
}

fn resolve_path(args: &InspectArgs) -> Result<PathBuf> {
    let direct = PathBuf::from(&args.model);
    if direct.is_file() {
        return Ok(direct);
    }

    let dir = args.path.clone().unwrap_or_else(models_dir);
    let entries = list_models(&dir)?;

    for (path, _) in &entries {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if stem == args.model || name == args.model {
            return Ok(path.clone());
        }
    }

    let lower = args.model.to_lowercase();
    for (path, _) in &entries {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.to_lowercase().contains(&lower) {
                return Ok(path.clone());
            }
        }
    }

    anyhow::bail!("model '{}' not found in {}", args.model, dir.display());
}
