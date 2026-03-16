// `fox search` — search HuggingFace Hub for GGUF models.
//
// Usage:
//   fox search gemma
//   fox search llama 3b
//   fox search "reasoning model" --limit 5

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;

use super::theme;
use super::{list_models, models_dir};

const HF_API_BASE: &str = "https://huggingface.co/api/models";

#[derive(Parser, Debug)]
pub struct SearchArgs {
    /// Search query (model name, architecture, task, etc.)
    #[arg(num_args = 1.., value_delimiter = ' ')]
    pub query: Vec<String>,

    /// Maximum number of results to show (default: 15)
    #[arg(long, short, default_value = "15")]
    pub limit: usize,

    /// Sort results by: downloads (default) or likes
    #[arg(long, default_value = "downloads")]
    pub sort: String,

    /// HuggingFace API token (for accessing private/gated models in results)
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,
}

#[derive(Deserialize, Debug)]
struct HfModel {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u32,
}

pub async fn run_search(args: SearchArgs) -> Result<()> {
    let query = args.query.join(" ");
    if query.is_empty() {
        anyhow::bail!("provide a search query, e.g. `fox search gemma`");
    }

    eprintln!(
        "Searching HuggingFace for GGUF models matching \"{}\"…",
        query
    );

    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(ref tok) = args.hf_token {
        let auth = format!("Bearer {tok}");
        if let Ok(val) = auth.parse() {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .user_agent("ferrumox/1.0.0")
        .build()
        .context("building HTTP client")?;

    let sort = match args.sort.as_str() {
        "likes" => "likes",
        _ => "downloads",
    };
    let encoded_query = query.replace(' ', "+");
    let url = format!(
        "{HF_API_BASE}?search={}&filter=gguf&sort={}&direction=-1&limit={}",
        encoded_query, sort, args.limit,
    );

    let resp = client
        .get(&url)
        .send()
        .await
        .context("contacting HuggingFace API")?;

    if !resp.status().is_success() {
        anyhow::bail!("HuggingFace API returned {}", resp.status());
    }

    let models: Vec<HfModel> = resp
        .json()
        .await
        .context("parsing HuggingFace API response")?;

    if models.is_empty() {
        eprintln!("No GGUF models found for \"{}\".", query);
        return Ok(());
    }

    // Check which repos we already have locally (rough check by stem substring).
    let local_dir = models_dir();
    let local_files: Vec<String> = list_models(&local_dir)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(p, _)| p.file_name().and_then(|n| n.to_str()).map(String::from))
        .collect();

    let repo_w = 52usize;
    let dl_w = 11usize;
    let likes_w = 6usize;

    // Header
    theme::print_table_header(&[("REPO", repo_w), ("DOWNLOADS", dl_w), ("LIKES", likes_w)]);
    theme::print_separator(repo_w + dl_w + likes_w + 6);

    for m in &models {
        let repo_name = &m.model_id;

        // Mark as downloaded if any local file contains the repo's model name fragment.
        let repo_stem = repo_name.split('/').next_back().unwrap_or(repo_name);
        let downloaded = local_files.iter().any(|f| {
            let f_lower = f.to_lowercase();
            let stem_lower = repo_stem.to_lowercase();
            // Match on a meaningful prefix (first two dash-segments) to reduce false positives.
            let key: String = stem_lower
                .splitn(3, '-')
                .take(2)
                .collect::<Vec<_>>()
                .join("-");
            !key.is_empty() && f_lower.contains(&key)
        });

        let dl_str = format_downloads(m.downloads);
        let mark = if downloaded { "✓ " } else { "  " };

        // Truncate long repo names.
        let display_repo = if repo_name.len() + 2 > repo_w {
            format!("{}…", &repo_name[..repo_w.saturating_sub(3)])
        } else {
            repo_name.clone()
        };

        if downloaded {
            theme::print_styled(
                Some(crossterm::style::Color::Green),
                false,
                false,
                &format!(
                    "{}{:<repo_w$} {:>dl_w$} {:>likes_w$}\n",
                    mark, display_repo, dl_str, m.likes
                ),
            );
        } else {
            println!(
                "{}{:<repo_w$} {:>dl_w$} {:>likes_w$}",
                mark, display_repo, dl_str, m.likes
            );
        }
    }

    println!();
    println!("Pull a model:   fox pull <REPO>");
    println!(
        "Example:        fox pull {}",
        models
            .first()
            .map(|m| m.model_id.as_str())
            .unwrap_or("bartowski/gemma-3-4b-it-GGUF")
    );

    Ok(())
}

fn format_downloads(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}
