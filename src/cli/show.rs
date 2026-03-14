// `fox show` — show details about a downloaded GGUF model.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use super::theme;
use super::{format_age, format_size, list_models, models_dir};

#[derive(Parser, Debug)]
pub struct ShowArgs {
    /// Model name (stem) or filename to inspect
    pub model: String,

    /// Directory to search for models (defaults to ~/.cache/ferrumox/models)
    #[arg(long)]
    pub path: Option<PathBuf>,
}

static QUANT_TAGS: &[&str] = &[
    "IQ1_S", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_S", "IQ4_NL", "IQ4_XS", "Q2_K",
    "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_1", "Q5_K_S",
    "Q5_K_M", "Q6_K", "Q8_0", "F16", "F32",
];

static ARCH_TAGS: &[&str] = &[
    "codellama",
    "deepseek",
    "wizardlm",
    "internlm",
    "stablelm",
    "baichuan",
    "mistral",
    "vicuna",
    "alpaca",
    "gemma",
    "llama",
    "qwen",
    "falcon",
    "bloom",
    "orca",
    "phi",
    "mpt",
    "gpt",
    "yi",
];

fn parse_quantization(stem: &str) -> Option<&'static str> {
    let upper = stem.to_uppercase();
    // Longer tags first to avoid prefix shadowing (e.g. Q4_K_M before Q4_K_S)
    let mut tags = QUANT_TAGS.to_vec();
    tags.sort_by_key(|t| std::cmp::Reverse(t.len()));
    tags.into_iter()
        .find(|&tag| upper.contains(&tag.to_uppercase()))
        .map(|v| v as _)
}

fn parse_architecture(stem: &str) -> Option<&'static str> {
    let lower = stem.to_lowercase();
    // Longer names first to avoid prefix shadowing (e.g. "codellama" before "llama")
    ARCH_TAGS
        .iter()
        .find(|&arch| lower.contains(*arch))
        .map(|v| v as _)
}

pub async fn run_show(args: ShowArgs) -> Result<()> {
    let dir = args.path.unwrap_or_else(models_dir);
    let models = list_models(&dir)?;

    let target: Option<(PathBuf, std::fs::Metadata)> = models.into_iter().find(|(path, _)| {
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        stem == args.model || name == args.model
    });

    let (path, meta) = match target {
        Some(m) => m,
        None => {
            eprintln!("Model '{}' not found.", args.model);
            let available = list_models(&dir)?;
            if !available.is_empty() {
                eprintln!("Available models:");
                for (p, _) in &available {
                    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
                    eprintln!("  {}", stem);
                }
            }
            anyhow::bail!("model not found");
        }
    };

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");
    let arch = parse_architecture(stem).unwrap_or("unknown");
    let quant = parse_quantization(stem).unwrap_or("unknown");
    let size = format_size(meta.len());
    let age = meta
        .modified()
        .map(format_age)
        .unwrap_or_else(|_| "unknown".to_string());

    theme::print_kv_pair("Name", stem);
    theme::print_kv_pair("Architecture", arch);
    theme::print_kv_pair("Quantization", quant);
    theme::print_kv_pair("Size", &size);
    theme::print_kv_pair("Modified", &age);
    theme::print_kv_pair("Path", &path.display().to_string());

    Ok(())
}
