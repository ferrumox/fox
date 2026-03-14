// `fox list` — list downloaded GGUF models.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use crossterm::style::Color;

use super::theme;
use super::{format_age, format_size, list_models, models_dir};

#[derive(Parser, Debug)]
pub struct ListArgs {
    /// Directory to search for models (defaults to ~/.cache/ferrumox/models)
    #[arg(long)]
    pub path: Option<PathBuf>,
}

pub async fn run_list(args: ListArgs) -> Result<()> {
    let dir = args.path.unwrap_or_else(models_dir);
    let models = list_models(&dir)?;

    if models.is_empty() {
        eprintln!(
            "No models found in {}. Run `fox pull <model-id>` to download one.",
            dir.display()
        );
        return Ok(());
    }

    let max_name_len = models
        .iter()
        .filter_map(|(p, _)| p.file_stem().and_then(|s| s.to_str()).map(|s| s.len()))
        .max()
        .unwrap_or(4)
        .max(4);
    let name_col = max_name_len + 2;

    theme::print_table_header(&[("NAME", name_col), ("SIZE", 10), ("MODIFIED", 20)]);
    theme::print_separator(name_col + 31);

    for (path, meta) in &models {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
        let size = format_size(meta.len());
        let age = meta
            .modified()
            .map(format_age)
            .unwrap_or_else(|_| "unknown".to_string());

        print!("{:<width$} ", name, width = name_col);
        theme::print_styled(Some(Color::Blue), false, false, &format!("{:<10} ", size));
        theme::print_styled(None, false, true, &age);
        println!();
    }

    Ok(())
}
