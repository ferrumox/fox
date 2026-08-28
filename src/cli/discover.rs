// `fox discover` — scan well-known directories for GGUF models.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use crossterm::style::Color;

use super::theme;
use crate::cli::format_size;
use crate::model_discovery::{discover_models, parse_model_dirs};

#[derive(Parser, Debug)]
pub struct DiscoverArgs {
    /// Semicolon-separated list of additional directories to scan for GGUF models
    #[arg(long, env = "FOX_MODEL_DIRS")]
    pub model_dirs: Option<String>,
}

pub async fn run_discover(args: DiscoverArgs) -> Result<()> {
    let extra: Vec<PathBuf> = args
        .model_dirs
        .as_deref()
        .map(parse_model_dirs)
        .unwrap_or_default();

    let models = discover_models(&extra);

    if models.is_empty() {
        eprintln!("No models found. Run `fox pull <model-id>` to download one.");
        return Ok(());
    }

    println!("Found {} models:\n", models.len());

    let max_name = models
        .iter()
        .map(|m| m.name.len())
        .max()
        .unwrap_or(4)
        .max(4);
    let name_col = max_name + 2;

    theme::print_table_header(&[
        ("NAME", name_col),
        ("SIZE", 10),
        ("SOURCE", 14),
        ("PATH", 20),
    ]);
    theme::print_separator(name_col + 46);

    for m in &models {
        print!("{:<width$} ", m.name, width = name_col);
        theme::print_styled(
            Some(Color::Blue),
            false,
            false,
            &format!("{:<10} ", format_size(m.size_bytes)),
        );
        theme::print_styled(
            Some(Color::Magenta),
            false,
            false,
            &format!("{:<14} ", m.source),
        );

        let display_path = shorten_home(&m.path);
        theme::print_styled(None, false, true, &display_path);
        println!();
    }

    Ok(())
}

fn shorten_home(path: &Path) -> String {
    if let Some(home) = dirs::home_dir() {
        if let Ok(rest) = path.strip_prefix(&home) {
            return format!("~/{}", rest.display());
        }
    }
    path.display().to_string()
}
