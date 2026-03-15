// CLI entry point — dispatch to subcommands.

pub mod alias;
pub mod bench;
pub mod list;
pub mod models;
pub mod ps;
pub mod pull;
pub mod rm;
pub mod run;
pub mod search;
pub mod serve;
pub mod show;
pub mod theme;

use std::collections::HashMap;
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
    /// Benchmark model load time and inference throughput
    Bench(bench::BenchArgs),
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
    /// List curated models available to pull
    Models(models::ModelsArgs),
    /// Search HuggingFace Hub for GGUF models
    Search(search::SearchArgs),
    /// Manage model name aliases
    Alias(alias::AliasArgs),
}

/// Known subcommand names — anything else is treated as `fox run <arg>`.
const SUBCOMMANDS: &[&str] = &[
    "serve", "run", "bench", "pull", "list", "rm", "show", "ps", "models", "search", "alias",
    "help",
];

pub async fn run() -> anyhow::Result<()> {
    // Load config file before clap parses CLI args so env-var-backed flags
    // pick up config values as their effective defaults.
    crate::config::load_config_into_env();

    // If the first non-flag argument is not a known subcommand, inject "run"
    // so that `fox llama "Hello"` works as `fox run llama "Hello"`.
    let raw: Vec<String> = std::env::args().collect();
    let effective: Vec<String> = match raw.get(1).map(String::as_str) {
        Some(first)
            if !first.starts_with('-') && !SUBCOMMANDS.contains(&first) =>
        {
            let mut v = vec![raw[0].clone(), "run".to_string()];
            v.extend(raw[1..].iter().cloned());
            v
        }
        _ => raw,
    };

    let cli = Cli::parse_from(effective);
    match cli.command {
        Command::Serve(args) => serve::run_serve(args).await,
        Command::Run(args) => run::run_run(args).await,
        Command::Bench(args) => bench::run_bench(args).await,
        Command::Pull(args) => pull::run_pull(args).await,
        Command::List(args) => list::run_list(args).await,
        Command::Rm(args) => rm::run_rm(args).await,
        Command::Show(args) => show::run_show(args).await,
        Command::Ps(args) => ps::run_ps(args).await,
        Command::Models(args) => models::run_models(args).await,
        Command::Search(args) => search::run_search(args).await,
        Command::Alias(args) => alias::run_alias(args).await,
    }
}

/// Query total GPU memory via nvidia-smi. Falls back to 8 GiB if no GPU is found.
pub(crate) fn get_gpu_memory_bytes() -> usize {
    let nvidia_smi = if cfg!(target_os = "windows") { "nvidia-smi.exe" } else { "nvidia-smi" };
    if let Ok(out) = std::process::Command::new(nvidia_smi)
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

/// Expand a leading `~` to the user's home directory (cross-platform).
pub(crate) fn expand_tilde(path: &Path) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Some(home) = dirs::home_dir() {
            let rest = s.strip_prefix("~").unwrap_or("");
            return home.join(rest.trim_start_matches('/'));
        }
    }
    path.to_path_buf()
}

/// Default models directory (platform-appropriate).
///
/// - Linux:   `~/.cache/ferrumox/models`
/// - macOS:   `~/Library/Caches/ferrumox/models`
/// - Windows: `%LOCALAPPDATA%\ferrumox\models`
pub(crate) fn models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ferrumox")
        .join("models")
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

/// Load model aliases from a TOML file.
/// Defaults to `~/.config/ferrumox/aliases.toml` if `path` is `None`.
pub(crate) fn load_aliases(path: Option<PathBuf>) -> HashMap<String, String> {
    let path = path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(".config/ferrumox/aliases.toml")
    });

    if !path.exists() {
        return HashMap::new();
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };

    #[derive(serde::Deserialize)]
    struct AliasesFile {
        #[serde(default)]
        aliases: HashMap<String, String>,
    }

    match toml::from_str::<AliasesFile>(&content) {
        Ok(f) => f.aliases,
        Err(_) => HashMap::new(),
    }
}

/// Resolve a user-supplied model name (or path) to `(stem, PathBuf)`.
///
/// Resolution order:
/// 1. If `name` points to an existing file on disk → use it directly.
/// 2. Alias lookup from `alias_file` (defaults to `~/.config/ferrumox/aliases.toml`).
/// 3. Exact case-insensitive stem match inside `models_dir()`.
/// 4. Starts-with match.
/// 5. Contains match.
///
/// On failure prints available models and returns an error.
pub(crate) fn resolve_model_path(
    name: &str,
    alias_file: Option<&Path>,
) -> anyhow::Result<(String, PathBuf)> {
    // 1. Direct path on disk
    let as_path = PathBuf::from(name);
    if as_path.exists() && as_path.is_file() {
        let stem = as_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(name)
            .to_string();
        return Ok((stem, as_path));
    }

    // 2. Alias lookup
    let aliases = load_aliases(alias_file.map(|p| p.to_path_buf()));
    let resolved = aliases
        .get(name)
        .map(String::as_str)
        .unwrap_or(name);

    let dir = models_dir();
    let entries = list_models(&dir).unwrap_or_default();
    let lower = resolved.to_lowercase();

    // Exact match
    for (path, _) in &entries {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.eq_ignore_ascii_case(resolved) {
                return Ok((stem.to_string(), path.clone()));
            }
        }
    }

    // Starts-with
    for (path, _) in &entries {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.to_lowercase().starts_with(&lower) {
                return Ok((stem.to_string(), path.clone()));
            }
        }
    }

    // Contains
    for (path, _) in &entries {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.to_lowercase().contains(&lower) {
                return Ok((stem.to_string(), path.clone()));
            }
        }
    }

    // Nothing found — show available models
    let available: Vec<String> = entries
        .iter()
        .filter_map(|(p, _)| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
        .collect();

    if available.is_empty() {
        anyhow::bail!(
            "model '{}' not found and no models are available in {}.\n\
             Run `fox pull <model>` to download one.",
            name,
            dir.display()
        );
    } else {
        anyhow::bail!(
            "model '{}' not found in {}.\nAvailable models:\n  {}",
            name,
            dir.display(),
            available.join("\n  ")
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    // --- format_size ---

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(999), "999 B");
        assert_eq!(format_size(999_999), "999999 B");
    }

    #[test]
    fn test_format_size_mb() {
        assert_eq!(format_size(1_000_000), "1.0 MB");
        assert_eq!(format_size(5_500_000), "5.5 MB");
        assert_eq!(format_size(999_999_999), "1000.0 MB");
    }

    #[test]
    fn test_format_size_gb() {
        assert_eq!(format_size(1_000_000_000), "1.0 GB");
        assert_eq!(format_size(7_300_000_000), "7.3 GB");
    }

    // --- format_age ---

    #[test]
    fn test_format_age_seconds() {
        let t = SystemTime::now() - Duration::from_secs(30);
        assert_eq!(format_age(t), "30 seconds ago");
    }

    #[test]
    fn test_format_age_one_minute() {
        let t = SystemTime::now() - Duration::from_secs(60);
        assert_eq!(format_age(t), "1 minute ago");
    }

    #[test]
    fn test_format_age_minutes() {
        let t = SystemTime::now() - Duration::from_secs(5 * 60);
        assert_eq!(format_age(t), "5 minutes ago");
    }

    #[test]
    fn test_format_age_one_hour() {
        let t = SystemTime::now() - Duration::from_secs(3600);
        assert_eq!(format_age(t), "1 hour ago");
    }

    #[test]
    fn test_format_age_hours() {
        let t = SystemTime::now() - Duration::from_secs(3 * 3600);
        assert_eq!(format_age(t), "3 hours ago");
    }

    #[test]
    fn test_format_age_one_day() {
        let t = SystemTime::now() - Duration::from_secs(86400);
        assert_eq!(format_age(t), "1 day ago");
    }

    #[test]
    fn test_format_age_days() {
        let t = SystemTime::now() - Duration::from_secs(3 * 86400);
        assert_eq!(format_age(t), "3 days ago");
    }

    // --- models_dir ---

    #[test]
    fn test_models_dir_ends_with_ferrumox_models() {
        let dir = models_dir();
        let s = dir.to_string_lossy();
        assert!(
            s.ends_with("ferrumox/models") || s.ends_with("ferrumox\\models"),
            "models_dir should end with ferrumox/models, got: {s}"
        );
    }

    // --- expand_tilde ---

    #[test]
    fn test_expand_tilde_absolute_path_unchanged() {
        let path = PathBuf::from("/absolute/path");
        assert_eq!(expand_tilde(&path), path);
    }

    #[test]
    fn test_expand_tilde_relative_path_unchanged() {
        let path = PathBuf::from("relative/path");
        assert_eq!(expand_tilde(&path), path);
    }

    #[test]
    fn test_expand_tilde_expands_home() {
        if let Some(home) = dirs::home_dir() {
            let path = PathBuf::from("~/mydir");
            let expanded = expand_tilde(&path);
            assert!(
                expanded.starts_with(&home),
                "expanded path should start with home dir"
            );
            assert!(
                expanded.to_string_lossy().contains("mydir"),
                "expanded path should contain the original subdirectory"
            );
        }
    }

    #[test]
    fn test_expand_tilde_only_tilde() {
        if let Some(home) = dirs::home_dir() {
            let path = PathBuf::from("~");
            let expanded = expand_tilde(&path);
            assert_eq!(expanded, home);
        }
    }
}
