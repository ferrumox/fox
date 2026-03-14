// `fox rm` — remove a downloaded GGUF model.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use super::{list_models, models_dir};

#[derive(Parser, Debug)]
pub struct RmArgs {
    /// Model name (stem) or filename to remove
    pub model: String,

    /// Skip confirmation prompt
    #[arg(long, short = 'y')]
    pub yes: bool,

    /// Directory to search for models (defaults to ~/.cache/ferrumox/models)
    #[arg(long)]
    pub path: Option<PathBuf>,
}

pub async fn run_rm(args: RmArgs) -> Result<()> {
    let dir = args.path.unwrap_or_else(models_dir);
    let models = list_models(&dir)?;

    let target: Option<PathBuf> = models
        .iter()
        .find(|(path, _)| {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
            stem == args.model || name == args.model
        })
        .map(|(p, _)| p.clone());

    let path = match target {
        Some(p) => p,
        None => {
            eprintln!("Model '{}' not found.", args.model);
            if models.is_empty() {
                eprintln!("No models found in {}.", dir.display());
            } else {
                eprintln!("Available models:");
                for (p, _) in &models {
                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                    eprintln!("  {}", stem);
                }
            }
            anyhow::bail!("model not found");
        }
    };

    if !args.yes {
        let confirmed = dialoguer::Confirm::new()
            .with_prompt(format!("Remove '{}'?", args.model))
            .default(false)
            .interact()?;
        if !confirmed {
            eprintln!("Aborted.");
            return Ok(());
        }
    }

    std::fs::remove_file(&path)?;
    println!("Removed {}", args.model);
    Ok(())
}
