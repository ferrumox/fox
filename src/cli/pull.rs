// `fox pull` — download a GGUF model from HuggingFace Hub.
//
// Usage:
//   fox pull gemma3                                      (top HF result)
//   fox pull gemma3:12b                                  (specific size)
//   fox pull gemma3:12b-q4                               (size + quant prefix)
//   fox pull bartowski/gemma-3-12b-it-GGUF               (raw HF repo)
//   fox pull bartowski/gemma-3-12b-it-GGUF:q4            (raw HF repo + quant)

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

use super::theme;

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_CDN_BASE: &str = "https://huggingface.co";

#[derive(Parser, Debug)]
pub struct PullArgs {
    /// Model to download. Formats:
    ///   name              — e.g. `gemma3`
    ///   name:size         — e.g. `gemma3:12b`
    ///   name:size-quant   — e.g. `gemma3:12b-q4`
    ///   owner/repo        — raw HuggingFace repo
    ///   owner/repo:quant  — raw HuggingFace repo + quant prefix
    pub model_id: String,

    /// Specific GGUF filename to download (overrides auto-selection).
    #[arg(long, short)]
    pub filename: Option<String>,

    /// Directory where the model file will be saved.
    /// Defaults to the platform cache directory (e.g. ~/.cache/ferrumox/models).
    #[arg(long)]
    pub output_dir: Option<PathBuf>,

    /// HuggingFace API token for private or gated models
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,
}

/// Parsed model spec from user input.
struct ModelSpec {
    /// HF repo if input was `owner/repo`, otherwise None (will be searched).
    raw_repo: Option<String>,
    /// Search query to find the repo (e.g. "gemma3 12b").
    search_query: String,
    /// Quantization prefix to filter files (e.g. "Q4").
    quant: Option<String>,
}

/// Parse user input into a ModelSpec.
///
/// Raw HF repo (contains `/`):
///   `bartowski/gemma-3-12b-it-GGUF`      → raw_repo=Some(...), quant=None
///   `bartowski/gemma-3-12b-it-GGUF:q4`   → raw_repo=Some(...), quant=Some("Q4")
///
/// Friendly name:
///   `gemma3`           → search="gemma3",      quant=None
///   `gemma3:12b`       → search="gemma3 12b",  quant=None
///   `gemma3:12b-q4`    → search="gemma3 12b",  quant=Some("Q4")
fn parse_model_spec(input: &str) -> ModelSpec {
    if input.contains('/') {
        // Raw HF repo — optionally with :quant suffix
        let (repo, quant) = match input.split_once(':') {
            Some((r, q)) => (r.to_string(), Some(q.to_uppercase())),
            None => (input.to_string(), None),
        };
        return ModelSpec {
            raw_repo: Some(repo.clone()),
            search_query: repo,
            quant,
        };
    }

    // Friendly name: split on ':' to get name and optional size-quant tag
    let (name, tag) = match input.split_once(':') {
        Some((n, t)) => (n, Some(t)),
        None => (input, None),
    };

    match tag {
        None => ModelSpec {
            raw_repo: None,
            search_query: name.to_string(),
            quant: None,
        },
        Some(tag) => {
            // Tag may be "12b", "12b-q4", or just "q4"
            // Split on '-' from the right: last segment is quant if it starts with q/iq/f
            let parts: Vec<&str> = tag.splitn(2, '-').collect();
            match parts.as_slice() {
                [size, quant] => ModelSpec {
                    raw_repo: None,
                    search_query: format!("{} {}", name, size),
                    quant: Some(quant.to_uppercase()),
                },
                [only] => {
                    let up = only.to_uppercase();
                    if up.starts_with('Q') || up.starts_with("IQ") || up.starts_with('F') {
                        // It's a quant with no size: "gemma3:q4"
                        ModelSpec {
                            raw_repo: None,
                            search_query: name.to_string(),
                            quant: Some(up),
                        }
                    } else {
                        // It's a size with no quant: "gemma3:12b"
                        ModelSpec {
                            raw_repo: None,
                            search_query: format!("{} {}", name, only),
                            quant: None,
                        }
                    }
                }
                _ => ModelSpec {
                    raw_repo: None,
                    search_query: name.to_string(),
                    quant: None,
                },
            }
        }
    }
}

#[derive(Deserialize)]
struct HfSearchResult {
    #[serde(rename = "modelId")]
    model_id: String,
}

/// Search HF for the most downloaded GGUF repo matching `query`.
async fn search_top_repo(query: &str, client: &reqwest::Client) -> Result<String> {
    let encoded = query.replace(' ', "+");
    let url =
        format!("{HF_API_BASE}?search={encoded}&filter=gguf&sort=downloads&direction=-1&limit=1");
    let results: Vec<HfSearchResult> = client
        .get(&url)
        .send()
        .await
        .context("searching HuggingFace")?
        .json()
        .await
        .context("parsing HuggingFace search response")?;

    results
        .into_iter()
        .next()
        .map(|r| r.model_id)
        .ok_or_else(|| anyhow::anyhow!("No GGUF model found for \"{}\" on HuggingFace", query))
}

pub async fn run_pull(args: PullArgs) -> Result<()> {
    let output_dir = match args.output_dir {
        Some(ref d) => super::expand_tilde(d),
        None => super::models_dir(),
    };
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output dir {:?}", output_dir))?;

    let client = build_client(args.hf_token.as_deref())?;
    let spec = parse_model_spec(&args.model_id);

    // Resolve HF repo: registry → raw input → HF search.
    let (hf_repo, registry_filename) = resolve_repo(&spec, &client).await?;

    // Fetch file list from HF Hub API.
    let url = format!("{HF_API_BASE}/{hf_repo}");
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("fetching metadata for {}", hf_repo))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "HuggingFace API returned {} for `{}`. \
             Check the repo name and ensure HF_TOKEN is set for private models.",
            resp.status(),
            hf_repo
        );
    }

    let meta: serde_json::Value = resp.json().await.context("parsing HF API response")?;
    let siblings = meta["siblings"]
        .as_array()
        .context("unexpected HF API response: missing `siblings`")?;

    let gguf_files: Vec<String> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|name| name.to_lowercase().ends_with(".gguf"))
        .map(String::from)
        .collect();

    if gguf_files.is_empty() {
        anyhow::bail!(
            "No .gguf files found in `{}`. \
             This repository may not contain GGUF quantizations.",
            hf_repo
        );
    }

    // Select file: --filename > registry recommended > quant prefix > pick balanced.
    let filename = if let Some(f) = args.filename {
        if !gguf_files.contains(&f) {
            anyhow::bail!(
                "File `{}` not found in `{}`.\nAvailable files:\n{}",
                f,
                hf_repo,
                gguf_files
                    .iter()
                    .map(|s| format!("  - {}", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        f
    } else if let Some(ref rec) = registry_filename {
        if gguf_files.contains(rec) {
            rec.clone()
        } else {
            let all: Vec<&String> = gguf_files.iter().collect();
            pick_balanced(&all).to_string()
        }
    } else if let Some(ref q) = spec.quant {
        let matches: Vec<&String> = gguf_files
            .iter()
            .filter(|name| name.to_uppercase().contains(q.as_str()))
            .collect();
        if matches.is_empty() {
            anyhow::bail!(
                "No GGUF file with quantization `{}` found in `{}`.\nAvailable files:\n{}",
                q,
                hf_repo,
                gguf_files
                    .iter()
                    .map(|s| format!("  - {}", s))
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
        pick_balanced(&matches).to_string()
    } else {
        let all: Vec<&String> = gguf_files.iter().collect();
        pick_balanced(&all).to_string()
    };

    eprintln!("Selected: {}", filename);

    // Download with progress bar.
    let dest = output_dir.join(&filename);
    if dest.exists() {
        eprintln!("{} already exists, skipping download.", dest.display());
        return Ok(());
    }

    let download_url = format!("{HF_CDN_BASE}/{hf_repo}/resolve/main/{filename}");
    eprintln!("Downloading {} …", filename);

    let resp = client
        .get(&download_url)
        .send()
        .await
        .with_context(|| format!("downloading {}", download_url))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "download failed with status {} for {}",
            resp.status(),
            download_url
        );
    }

    let total_bytes = resp.content_length();
    let pb = match total_bytes {
        Some(n) => {
            let pb = ProgressBar::new(n);
            pb.set_style(
                ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:50.cyan/blue}] \
                     {bytes}/{total_bytes} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
            );
            pb
        }
        None => {
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::with_template("{spinner:.green} {bytes} downloaded ({elapsed})")
                    .unwrap(),
            );
            pb
        }
    };

    let tmp_dest = dest.with_extension("gguf.part");
    let mut file =
        std::fs::File::create(&tmp_dest).with_context(|| format!("creating {:?}", tmp_dest))?;

    let mut stream = resp;
    while let Some(chunk) = stream
        .chunk()
        .await
        .context("error reading download stream")?
    {
        file.write_all(&chunk).context("error writing to file")?;
        pb.inc(chunk.len() as u64);
    }
    pb.finish_with_message("download complete");

    std::fs::rename(&tmp_dest, &dest)
        .with_context(|| format!("renaming {:?} to {:?}", tmp_dest, dest))?;

    let stem = dest
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&filename);

    eprintln!();
    theme::print_success(&format!("Saved to {}", dest.display()));
    theme::eprint_styled(
        None,
        false,
        true,
        &format!("     Run:   fox run {}\n", stem),
    );
    theme::eprint_styled(None, false, true, "     Serve: fox serve\n");

    Ok(())
}

/// Resolve a model spec to an HF repo, checking the curated registry first.
/// Returns `(repo, Option<recommended_filename>)`.
async fn resolve_repo(
    spec: &ModelSpec,
    client: &reqwest::Client,
) -> Result<(String, Option<String>)> {
    // If the user gave a raw HF repo path, use it directly.
    if let Some(ref repo) = spec.raw_repo {
        return Ok((repo.clone(), None));
    }

    // Check the curated registry before searching HF.
    let registry = crate::registry::Registry::load();
    if let Some((_canonical, model)) = registry.resolve(&spec.search_query) {
        eprintln!("Using curated registry: {}", model.repo);
        return Ok((model.repo, Some(model.recommended)));
    }

    // Also try the original input (before parse_model_spec split it).
    // e.g. "llama3" won't match search_query "llama3" if it was mapped to
    // an alias, but the split format "gemma3 12b" won't be in the registry.

    eprintln!("Searching HuggingFace for \"{}\"…", spec.search_query);
    let repo = search_top_repo(&spec.search_query, client).await?;
    eprintln!("Found: {}", repo);
    Ok((repo, None))
}

fn build_client(token: Option<&str>) -> Result<reqwest::Client> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(tok) = token {
        let auth = format!("Bearer {}", tok);
        headers.insert(
            reqwest::header::AUTHORIZATION,
            auth.parse().context("invalid HF token")?,
        );
    }
    reqwest::Client::builder()
        .default_headers(headers)
        .user_agent("ferrumox/1.0.0")
        .build()
        .context("building HTTP client")
}

/// From a list of GGUF files, pick the most balanced quantization.
/// Priority: Q4_K_M > Q4_K_S > Q5_K_M > Q4_0 > Q8_0 > first available.
fn pick_balanced<'a>(files: &[&'a String]) -> &'a String {
    let priority = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "Q8_0"];
    for variant in &priority {
        if let Some(f) = files.iter().find(|f| f.to_uppercase().contains(variant)) {
            return f;
        }
    }
    files[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_spec_friendly_name() {
        let spec = parse_model_spec("gemma3");
        assert!(spec.raw_repo.is_none());
        assert_eq!(spec.search_query, "gemma3");
        assert!(spec.quant.is_none());
    }

    #[test]
    fn test_parse_model_spec_with_size() {
        let spec = parse_model_spec("gemma3:12b");
        assert!(spec.raw_repo.is_none());
        assert_eq!(spec.search_query, "gemma3 12b");
        assert!(spec.quant.is_none());
    }

    #[test]
    fn test_parse_model_spec_with_size_and_quant() {
        let spec = parse_model_spec("gemma3:12b-q4");
        assert!(spec.raw_repo.is_none());
        assert_eq!(spec.search_query, "gemma3 12b");
        assert_eq!(spec.quant.as_deref(), Some("Q4"));
    }

    #[test]
    fn test_parse_model_spec_raw_repo() {
        let spec = parse_model_spec("bartowski/gemma-3-12b-it-GGUF");
        assert_eq!(
            spec.raw_repo.as_deref(),
            Some("bartowski/gemma-3-12b-it-GGUF")
        );
        assert!(spec.quant.is_none());
    }

    #[test]
    fn test_parse_model_spec_raw_repo_with_quant() {
        let spec = parse_model_spec("bartowski/gemma-3-12b-it-GGUF:q8");
        assert_eq!(
            spec.raw_repo.as_deref(),
            Some("bartowski/gemma-3-12b-it-GGUF")
        );
        assert_eq!(spec.quant.as_deref(), Some("Q8"));
    }

    #[tokio::test]
    async fn test_resolve_repo_from_registry() {
        let spec = ModelSpec {
            raw_repo: None,
            search_query: "llama3".to_string(),
            quant: None,
        };
        let client = reqwest::Client::new();
        let (repo, recommended) = resolve_repo(&spec, &client).await.unwrap();
        assert_eq!(repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert_eq!(
            recommended.as_deref(),
            Some("Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        );
    }

    #[tokio::test]
    async fn test_resolve_repo_from_registry_exact_key() {
        let spec = ModelSpec {
            raw_repo: None,
            search_query: "nomic-embed".to_string(),
            quant: None,
        };
        let client = reqwest::Client::new();
        let (repo, recommended) = resolve_repo(&spec, &client).await.unwrap();
        assert!(repo.contains("nomic"));
        assert!(recommended.is_some());
    }

    #[tokio::test]
    async fn test_resolve_repo_raw_repo_bypasses_registry() {
        let spec = ModelSpec {
            raw_repo: Some("someone/custom-repo".to_string()),
            search_query: "someone/custom-repo".to_string(),
            quant: None,
        };
        let client = reqwest::Client::new();
        let (repo, recommended) = resolve_repo(&spec, &client).await.unwrap();
        assert_eq!(repo, "someone/custom-repo");
        assert!(recommended.is_none());
    }

    #[test]
    fn test_pick_balanced_prefers_q4_k_m() {
        let files = [
            "model-Q5_K_M.gguf".to_string(),
            "model-Q4_K_M.gguf".to_string(),
            "model-Q8_0.gguf".to_string(),
        ];
        let refs: Vec<&String> = files.iter().collect();
        assert_eq!(pick_balanced(&refs), "model-Q4_K_M.gguf");
    }

    #[test]
    fn test_pick_balanced_falls_back() {
        let files = ["model-IQ2_M.gguf".to_string()];
        let refs: Vec<&String> = files.iter().collect();
        assert_eq!(pick_balanced(&refs), "model-IQ2_M.gguf");
    }
}
