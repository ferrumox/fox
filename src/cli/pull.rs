// `fox pull` — download a GGUF model from HuggingFace Hub.
//
// Usage:
//   fox pull bartowski/Llama-3.2-1B-Instruct-GGUF
//   fox pull bartowski/Llama-3.2-1B-Instruct-GGUF --filename Llama-3.2-1B-Instruct-Q4_K_M.gguf
//   fox pull bartowski/Llama-3.2-1B-Instruct-GGUF --output-dir ./models

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use super::theme;

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_CDN_BASE: &str = "https://huggingface.co";

#[derive(Parser, Debug)]
pub struct PullArgs {
    /// HuggingFace model ID in the form `owner/repo`
    /// (e.g. `bartowski/Llama-3.2-1B-Instruct-GGUF`)
    pub model_id: String,

    /// Specific GGUF filename to download.
    /// If omitted and multiple files are found, an interactive list is shown.
    #[arg(long, short)]
    pub filename: Option<String>,

    /// Directory where the model file will be saved.
    /// Defaults to `~/.cache/ferrumox/models/`
    #[arg(long, default_value = "~/.cache/ferrumox/models")]
    pub output_dir: PathBuf,

    /// HuggingFace API token for private or gated models
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,
}

pub async fn run_pull(args: PullArgs) -> Result<()> {
    let output_dir = super::expand_tilde(&args.output_dir);
    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("creating output dir {:?}", output_dir))?;

    // 1. Fetch model metadata from HF Hub API.
    eprintln!("Fetching model info for {}…", args.model_id);
    let url = format!("{}/{}", HF_API_BASE, args.model_id);
    let client = build_client(args.hf_token.as_deref())?;
    let resp = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("fetching {}", url))?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "HuggingFace API returned {} for model `{}`. \
             Check the model ID and ensure HF_TOKEN is set for private models.",
            resp.status(),
            args.model_id
        );
    }

    let meta: serde_json::Value = resp.json().await.context("parsing HF API response")?;

    // 2. Extract .gguf file list from `siblings`.
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
            args.model_id
        );
    }

    // 3. Resolve which file to download.
    let filename = match args.filename {
        Some(f) => {
            if !gguf_files.contains(&f) {
                anyhow::bail!(
                    "File `{}` not found in `{}`. Available GGUF files:\n{}",
                    f,
                    args.model_id,
                    gguf_files
                        .iter()
                        .map(|s| format!("  - {}", s))
                        .collect::<Vec<_>>()
                        .join("\n")
                );
            }
            f
        }
        None if gguf_files.len() == 1 => {
            eprintln!("Found 1 GGUF file: {}", gguf_files[0]);
            gguf_files[0].clone()
        }
        None => select_file_interactive(&gguf_files)?,
    };

    // 4. Download with progress bar.
    let dest = output_dir.join(&filename);

    if dest.exists() {
        eprintln!("{} already exists, skipping download.", dest.display());
        eprintln!("Path: {}", dest.display());
        return Ok(());
    }

    let download_url = format!(
        "{}/{}/resolve/main/{}",
        HF_CDN_BASE, args.model_id, filename
    );
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

    // Write to a temp file first, rename on success.
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

    eprintln!();
    theme::print_success(&format!("Saved to {}", dest.display()));
    theme::eprint_styled(
        None,
        false,
        true,
        &format!("     Run:   fox run --model-path \"{}\"\n", dest.display()),
    );
    theme::eprint_styled(
        None,
        false,
        true,
        &format!(
            "     Serve: fox serve --model-path \"{}\"\n",
            dest.display()
        ),
    );

    Ok(())
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
        .user_agent("ferrumox/0.6.0")
        .build()
        .context("building HTTP client")
}

/// Ask the user to choose a file from a numbered list.
fn select_file_interactive(files: &[String]) -> Result<String> {
    let selection = dialoguer::Select::new()
        .with_prompt("Multiple GGUF files found — which one do you want to download?")
        .items(files)
        .default(0)
        .interact()
        .context("interactive selection")?;
    Ok(files[selection].clone())
}
