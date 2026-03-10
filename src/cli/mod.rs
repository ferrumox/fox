// CLI entry point — dispatch to subcommands.

pub mod list;
pub mod ps;
pub mod pull;
pub mod rm;
pub mod run;
pub mod serve;
pub mod show;
pub mod theme;

use std::fs::Metadata;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "fox")]
#[command(about = "High-performance LLM inference engine")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the OpenAI-compatible HTTP inference server
    Serve(serve::ServeArgs),
    /// Run single-shot inference and stream output to stdout
    Run(run::RunArgs),
    /// Download a GGUF model from HuggingFace Hub
    Pull(pull::PullArgs),
    /// List downloaded models
    List(list::ListArgs),
    /// Remove a downloaded model
    Rm(rm::RmArgs),
    /// Show details about a downloaded model
    Show(show::ShowArgs),
    /// Show running model servers
    Ps(ps::PsArgs),
}

pub async fn run() -> anyhow::Result<()> {
    // Load config file before clap parses CLI args so env-var-backed flags
    // pick up config values as their effective defaults.
    crate::config::load_config_into_env();

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve::run_serve(args).await,
        Command::Run(args) => run::run_run(args).await,
        Command::Pull(args) => pull::run_pull(args).await,
        Command::List(args) => list::run_list(args).await,
        Command::Rm(args) => rm::run_rm(args).await,
        Command::Show(args) => show::run_show(args).await,
        Command::Ps(args) => ps::run_ps(args).await,
    }
}

/// Query total GPU memory via nvidia-smi. Falls back to 8 GiB on CPU-only builds.
pub(crate) fn get_gpu_memory_bytes() -> usize {
    #[cfg(feature = "cuda")]
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(mib) = s.trim().parse::<usize>() {
                    return mib * 1024 * 1024;
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024
}

/// Expand a leading `~` to the user's home directory.
pub(crate) fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Ok(home) = std::env::var("HOME") {
            let rest = s.strip_prefix("~").unwrap_or("");
            return PathBuf::from(home).join(rest.trim_start_matches('/'));
        }
    }
    path.to_path_buf()
}

/// Default models directory: `~/.cache/ferrumox/models`.
pub(crate) fn models_dir() -> PathBuf {
    expand_tilde(Path::new("~/.cache/ferrumox/models"))
}

/// List all `.gguf` files in `dir`, sorted by filename.
/// Returns an empty vec if `dir` does not exist.
pub(crate) fn list_models(dir: &Path) -> anyhow::Result<Vec<(PathBuf, Metadata)>> {
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut entries: Vec<(PathBuf, Metadata)> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("gguf"))
                .unwrap_or(false)
        })
        .filter_map(|e| {
            let path = e.path();
            e.metadata().ok().map(|m| (path, m))
        })
        .collect();
    entries.sort_by(|a, b| a.0.file_name().cmp(&b.0.file_name()));
    Ok(entries)
}

/// Format byte count as human-readable string (GB / MB / B).
pub(crate) fn format_size(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format a `SystemTime` as a human-readable relative age string.
pub(crate) fn format_age(modified: SystemTime) -> String {
    let elapsed = SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default();
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{} seconds ago", secs)
    } else if secs < 3600 {
        let m = secs / 60;
        if m == 1 {
            "1 minute ago".to_string()
        } else {
            format!("{} minutes ago", m)
        }
    } else if secs < 86400 {
        let h = secs / 3600;
        if h == 1 {
            "1 hour ago".to_string()
        } else {
            format!("{} hours ago", h)
        }
    } else {
        let d = secs / 86400;
        if d == 1 {
            "1 day ago".to_string()
        } else {
            format!("{} days ago", d)
        }
    }
}
