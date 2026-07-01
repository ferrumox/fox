use std::collections::HashMap;
use std::fs::Metadata;
use std::path::{Path, PathBuf};

/// Expand a leading `~` to the user's home directory (cross-platform).
pub fn expand_tilde(path: &Path) -> PathBuf {
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
pub fn models_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ferrumox")
        .join("models")
}

/// List all `.gguf` files in `dir`, sorted by filename.
/// Returns an empty vec if `dir` does not exist.
pub fn list_models(dir: &Path) -> anyhow::Result<Vec<(PathBuf, Metadata)>> {
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

/// Load model aliases from a TOML file.
/// Defaults to `~/.config/ferrumox/aliases.toml` if `path` is `None`.
pub fn load_aliases(path: Option<PathBuf>) -> HashMap<String, String> {
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
pub fn resolve_model_path(
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
    let resolved = aliases.get(name).map(String::as_str).unwrap_or(name);

    let dir = models_dir();
    let entries = list_models(&dir).unwrap_or_default();
    // Normalize colon-notation (e.g. "qwen3.5:2b" → "qwen3.5-2b") so fuzzy
    // matching works against filenames that use dashes as separators.
    let normalized = resolved.replace(':', "-");
    let lower = normalized.to_lowercase();

    // Exact match
    for (path, _) in &entries {
        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if stem.eq_ignore_ascii_case(&normalized) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_models_dir_ends_with_ferrumox_models() {
        let dir = models_dir();
        let s = dir.to_string_lossy();
        assert!(
            s.ends_with("ferrumox/models") || s.ends_with("ferrumox\\models"),
            "models_dir should end with ferrumox/models, got: {s}"
        );
    }

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
