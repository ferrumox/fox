// `fox pull` — download a GGUF model from HuggingFace Hub.
//
// Usage:
//   fox pull llama3.2                                    (curated registry entry)
//   fox pull gemma3:12b                                  (registry entry, or specific size)
//   fox pull gemma3:12b-q4                                (size + quant prefix)
//   fox pull bartowski/gemma-3-12b-it-GGUF               (raw HF repo)
//   fox pull bartowski/gemma-3-12b-it-GGUF:q4            (raw HF repo + quant)
//
// A friendly name is checked against the curated `registry.json` catalog first
// (exact key, alias, or `<key>-<quant>` — the same names `fox models` lists);
// only when nothing matches does it fall back to a live HuggingFace search by
// name (the historical, catalog-unaware behavior, unchanged for anything not
// in the registry).

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

use super::theme;
use crate::registry::{Registry, RegistryModel};

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
/// `("Kimi-K3-Q4_K_M", "00005")` from `Kimi-K3-Q4_K_M-00002-of-00005.gguf`.
fn split_shard_name(filename: &str) -> Option<(&str, &str)> {
    let stem = filename.strip_suffix(".gguf")?;
    let (rest, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = rest.rsplit_once('-')?;
    let five_digits = |s: &str| s.len() == 5 && s.chars().all(|c| c.is_ascii_digit());
    (five_digits(index) && five_digits(total)).then_some((prefix, total))
}

/// Every file making up the same sharded GGUF as `filename`, in order.
///
/// Large models are published split across `name-00001-of-00005.gguf` … and llama.cpp
/// loads the whole set when handed the first part. Downloading one part therefore
/// leaves an unusable file, which is why several of the most-downloaded models on
/// HuggingFace — Kimi K3, DeepSeek V4, GLM 5.2, MiniMax M3 — were unreachable through
/// `fox pull` before this existed.
///
/// Returns just `filename` when it is not part of a set, so unsharded pulls are
/// unchanged.
fn shard_set(filename: &str, all: &[String]) -> Vec<String> {
    let Some((prefix, total)) = split_shard_name(filename) else {
        return vec![filename.to_string()];
    };
    let mut set: Vec<String> = all
        .iter()
        .filter(|f| split_shard_name(f).is_some_and(|(p, t)| p == prefix && t == total))
        .cloned()
        .collect();
    // Sorting is what puts part 1 first; the download order does not matter but the
    // path reported back to the user does, since that is what llama.cpp must be given.
    set.sort();
    if set.is_empty() {
        vec![filename.to_string()]
    } else {
        set
    }
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

/// Try to resolve `input` against the curated registry (exact key, alias, or
/// `<key>-<quant>` where the quant suffix is stripped for a second lookup —
/// e.g. `gemma3:12b-q4` matches registry key `gemma3:12b` with quant `Q4`,
/// mirroring `parse_model_spec`'s own size-quant tag splitting). Only tried
/// for friendly names — raw `owner/repo` input never reaches this.
fn resolve_from_registry(input: &str) -> Option<(RegistryModel, Option<String>)> {
    let registry = Registry::load();
    if let Some((_, model)) = registry.resolve(input) {
        return Some((model, None));
    }
    let (base, quant) = input.rsplit_once('-')?;
    let up = quant.to_uppercase();
    if up.starts_with('Q') || up.starts_with("IQ") || up.starts_with('F') {
        let (_, model) = registry.resolve(base)?;
        Some((model, Some(up)))
    } else {
        None
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

    // A friendly name checks the curated registry (exact key, alias, or
    // `<key>-<quant>`) before ever hitting the network — `fox models` already
    // advertises "fox pull <name>" for these names, so a registry entry must
    // actually resolve to the repo/file it names, not coincidentally rely on a
    // live search happening to surface the same repo.
    let registry_hit = if args.model_id.contains('/') {
        None
    } else {
        resolve_from_registry(&args.model_id)
    };

    let (hf_repo, recommended_filename, mmproj_hint, registry_quant) = match registry_hit {
        Some((model, quant)) => {
            eprintln!(
                "Resolved \"{}\" → {} (curated registry)",
                args.model_id, model.repo
            );
            (model.repo, Some(model.recommended), model.mmproj, quant)
        }
        None => {
            let spec = parse_model_spec(&args.model_id);
            let repo = match spec.raw_repo {
                Some(repo) => repo,
                None => {
                    eprintln!("Searching HuggingFace for \"{}\"…", spec.search_query);
                    let repo = search_top_repo(&spec.search_query, &client).await?;
                    eprintln!("Found: {}", repo);
                    repo
                }
            };
            (repo, None, None, spec.quant)
        }
    };

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

    // Select file: --filename > quant prefix > curated "recommended" > pick
    // balanced from all files.
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
    } else if let Some(ref q) = registry_quant {
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
    } else if let Some(rec) = recommended_filename.filter(|f| gguf_files.contains(f)) {
        rec
    } else {
        let all: Vec<&String> = gguf_files.iter().collect();
        pick_balanced(&all).to_string()
    };

    let all_files: Vec<String> = gguf_files.to_vec();
    let files = shard_set(&filename, &all_files);
    if files.len() > 1 {
        eprintln!("Selected: {} ({} shards)", filename, files.len());
    } else {
        eprintln!("Selected: {}", filename);
    }

    // The first shard is the handle for the whole set — it is what gets reported back
    // and what llama.cpp is pointed at.
    let dest = output_dir.join(&files[0]);
    let mut fetched = 0usize;
    for (i, name) in files.iter().enumerate() {
        let part_dest = output_dir.join(name);
        if part_dest.exists() {
            eprintln!("{} already exists, skipping.", part_dest.display());
            continue;
        }
        let label = if files.len() > 1 {
            format!("{name} ({}/{})", i + 1, files.len())
        } else {
            name.clone()
        };
        download_file(&client, &hf_repo, name, &part_dest, &label).await?;
        fetched += 1;
    }
    if fetched == 0 && files.len() == 1 {
        eprintln!("Nothing to download.");
        return Ok(());
    }

    let stem = dest
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&files[0]);

    eprintln!();
    theme::print_success(&format!("Saved to {}", dest.display()));
    theme::eprint_styled(
        None,
        false,
        true,
        &format!("     Run:   fox run {}\n", stem),
    );
    theme::eprint_styled(None, false, true, "     Serve: fox serve\n");
    if let Some(mmproj) = mmproj_hint {
        theme::eprint_styled(
            None,
            false,
            true,
            &format!(
                "     Vision: this model has a paired mmproj — also run \
                 `fox pull {hf_repo} --filename {mmproj}`, then serve with `--mmproj {}`\n",
                mmproj.trim_end_matches(".gguf")
            ),
        );
    }

    Ok(())
}

/// Fetch one file to `dest`, showing progress under `label`.
///
/// Writes to a `.part` file and renames on completion, so an interrupted download
/// cannot leave behind something that looks like a finished model — which matters more
/// with shards, where one truncated part poisons the whole set.
async fn download_file(
    client: &reqwest::Client,
    hf_repo: &str,
    filename: &str,
    dest: &std::path::Path,
    label: &str,
) -> Result<()> {
    let download_url = format!("{HF_CDN_BASE}/{hf_repo}/resolve/main/{filename}");
    eprintln!("Downloading {label} …");

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

    let pb = match resp.content_length() {
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

    // Sharded repos publish parts under a per-quant subdirectory
    // (`UD-IQ1_S/Model-UD-IQ1_S-00001-of-00014.gguf`), so the destination's parent may
    // not exist yet. Verified against the real file lists rather than assumed: every
    // sharded repo checked (Kimi K3, DeepSeek V4, GLM 5.2, MiniMax M3) nests this way.
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("creating {:?}", parent))?;
    }
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

    std::fs::rename(&tmp_dest, dest)
        .with_context(|| format!("renaming {:?} to {:?}", tmp_dest, dest))?;
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
    fn shard_set_groups_every_part_in_order() {
        let all: Vec<String> = vec![
            "m-00003-of-00003.gguf",
            "m-00001-of-00003.gguf",
            "m-00002-of-00003.gguf",
            "other-00001-of-00002.gguf",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        // Handed any part, the whole set comes back — a pull that started from part 2
        // must still fetch 1 and 3, or llama.cpp gets an incomplete model.
        let got = shard_set("m-00002-of-00003.gguf", &all);
        assert_eq!(
            got,
            vec![
                "m-00001-of-00003.gguf",
                "m-00002-of-00003.gguf",
                "m-00003-of-00003.gguf"
            ]
        );
    }

    #[test]
    fn shard_set_handles_parts_nested_in_a_quant_directory() {
        // Real layout: every sharded repo checked on HuggingFace nests parts under a
        // per-quant directory, so the grouping key has to survive a path separator.
        let all: Vec<String> = vec![
            "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00002-of-00003.gguf",
            "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00003.gguf",
            "UD-IQ1_S/Kimi-K3-UD-IQ1_S-00003-of-00003.gguf",
            "UD-IQ1_M/Kimi-K3-UD-IQ1_M-00001-of-00002.gguf",
        ]
        .into_iter()
        .map(String::from)
        .collect();
        let got = shard_set("UD-IQ1_S/Kimi-K3-UD-IQ1_S-00001-of-00003.gguf", &all);
        assert_eq!(
            got.len(),
            3,
            "must not pull in the other quant's parts: {got:?}"
        );
        assert!(
            got[0].ends_with("00001-of-00003.gguf"),
            "part 1 must sort first"
        );
    }

    #[test]
    fn shard_set_leaves_unsharded_files_alone() {
        let all: Vec<String> = vec!["a.gguf".to_string(), "b-00001-of-00002.gguf".to_string()];
        assert_eq!(shard_set("a.gguf", &all), vec!["a.gguf"]);
    }

    #[test]
    fn shard_set_does_not_mix_different_totals() {
        // Same prefix, different split counts: a repo that was re-sharded keeps both
        // sets. Mixing them yields a model that cannot load.
        let all: Vec<String> = vec![
            "m-00001-of-00002.gguf".to_string(),
            "m-00001-of-00003.gguf".to_string(),
        ];
        assert_eq!(
            shard_set("m-00001-of-00002.gguf", &all),
            vec!["m-00001-of-00002.gguf"]
        );
    }

    #[test]
    fn shard_like_names_that_are_not_shards_are_not_split() {
        // Quant names carry digits and dashes too; only the five-digit NNNNN-of-NNNNN
        // form is a shard marker.
        let all: Vec<String> = vec!["Qwen3-Coder-30B-A3B-Q4_K_M.gguf".to_string()];
        assert_eq!(split_shard_name("Qwen3-Coder-30B-A3B-Q4_K_M.gguf"), None);
        assert_eq!(shard_set(&all[0], &all), vec![all[0].clone()]);
    }

    #[test]
    fn resolve_from_registry_exact_key() {
        let (model, quant) = resolve_from_registry("llama3.2").expect("known registry key");
        assert_eq!(model.repo, "bartowski/Llama-3.2-3B-Instruct-GGUF");
        assert!(quant.is_none());
    }

    #[test]
    fn resolve_from_registry_alias() {
        let (model, quant) = resolve_from_registry("moondream").expect("known alias");
        assert_eq!(model.repo, "ggml-org/moondream2-20250414-GGUF");
        assert!(quant.is_none());
    }

    #[test]
    fn resolve_from_registry_key_with_colon() {
        // Registry keys can themselves contain ':' (e.g. "gemma3:12b") — must
        // match verbatim before any quant-suffix stripping is attempted.
        let (model, quant) = resolve_from_registry("gemma3:12b").expect("compound key");
        assert!(quant.is_none());
        assert!(model.repo.contains("gemma-3-12b"));
    }

    #[test]
    fn resolve_from_registry_strips_trailing_quant() {
        let (model, quant) =
            resolve_from_registry("gemma3:12b-q4").expect("key with trailing quant");
        assert_eq!(quant.as_deref(), Some("Q4"));
        assert!(model.repo.contains("gemma-3-12b"));
    }

    #[test]
    fn resolve_from_registry_unknown_name_returns_none() {
        assert!(resolve_from_registry("totally-not-a-registry-name-xyz").is_none());
    }

    #[test]
    fn resolve_from_registry_trailing_segment_not_a_quant_returns_none() {
        // "foo-bar" isn't a registry key, and "bar" doesn't look like a quant
        // (doesn't start with q/iq/f), so no quant-stripping retry should fire.
        assert!(resolve_from_registry("gemma3:12b-bar").is_none());
    }
}
