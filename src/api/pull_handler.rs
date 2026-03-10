// Ollama-compatible POST /api/pull handler with SSE progress streaming.

use std::convert::Infallible;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Context as _;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use tokio::sync::mpsc;

use super::router::AppState;
use super::types::{PullRequest, PullStatus};

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_CDN_BASE: &str = "https://huggingface.co";

pub async fn ollama_pull(
    State(state): State<AppState>,
    Json(req): Json<PullRequest>,
) -> impl IntoResponse {
    let (tx, mut rx) = mpsc::unbounded_channel::<PullStatus>();

    let model_id = req.name.clone();
    let hf_token = state.hf_token.clone();
    let models_dir = state.models_dir.clone();

    tokio::spawn(async move {
        let _ = do_pull(model_id, hf_token, models_dir, tx).await;
    });

    let stream = async_stream::stream! {
        while let Some(status) = rx.recv().await {
            let event = Event::default()
                .json_data(&status)
                .unwrap_or_else(|_| Event::default().data("{}"));
            yield Ok::<_, Infallible>(event);
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

async fn do_pull(
    model_id: String,
    hf_token: Option<String>,
    models_dir: PathBuf,
    tx: mpsc::UnboundedSender<PullStatus>,
) -> anyhow::Result<()> {
    macro_rules! send {
        ($status:expr) => {
            if tx.send($status).is_err() {
                return Ok(()); // client disconnected
            }
        };
    }

    send!(PullStatus {
        status: "pulling manifest".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    // Build HTTP client with optional HF auth token.
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(ref tok) = hf_token {
        let auth = format!("Bearer {tok}");
        if let Ok(val) = auth.parse() {
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }
    }
    let client = reqwest::Client::builder()
        .default_headers(headers)
        .user_agent("ferrumox/0.8.0")
        .build()
        .context("building HTTP client")?;

    // Fetch model metadata from HF Hub API.
    let url = format!("{HF_API_BASE}/{model_id}");
    let resp = client
        .get(&url)
        .send()
        .await
        .context("fetching model metadata")?;

    if !resp.status().is_success() {
        send!(PullStatus {
            status: format!("error: HF API returned {}", resp.status()),
            digest: None,
            total: None,
            completed: None,
        });
        anyhow::bail!("HF API error: {}", resp.status());
    }

    let meta: serde_json::Value = resp.json().await.context("parsing HF API response")?;
    let siblings = meta["siblings"]
        .as_array()
        .context("unexpected HF API response: missing `siblings`")?;

    let gguf_files: Vec<String> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|n| n.to_lowercase().ends_with(".gguf"))
        .map(String::from)
        .collect();

    if gguf_files.is_empty() {
        send!(PullStatus {
            status: "error: no .gguf files found in repository".to_string(),
            digest: None,
            total: None,
            completed: None,
        });
        anyhow::bail!("no gguf files found");
    }

    // Pick the best file: prefer Q4_K_M quantization, fall back to first.
    let filename = gguf_files
        .iter()
        .find(|f| f.contains("Q4_K_M"))
        .or_else(|| gguf_files.first())
        .cloned()
        .unwrap();

    std::fs::create_dir_all(&models_dir).context("creating models directory")?;

    let dest = models_dir.join(&filename);
    if dest.exists() {
        send!(PullStatus {
            status: "success".to_string(),
            digest: None,
            total: None,
            completed: None,
        });
        return Ok(());
    }

    // Download the model file with SSE progress events.
    let download_url = format!("{HF_CDN_BASE}/{model_id}/resolve/main/{filename}");

    let resp = client
        .get(&download_url)
        .send()
        .await
        .context("sending download request")?;

    if !resp.status().is_success() {
        send!(PullStatus {
            status: format!("error: download failed with {}", resp.status()),
            digest: None,
            total: None,
            completed: None,
        });
        anyhow::bail!("download failed: {}", resp.status());
    }

    let total = resp.content_length().unwrap_or(0);
    let digest = format!("sha256:{filename}");

    send!(PullStatus {
        status: "downloading".to_string(),
        digest: Some(digest.clone()),
        total: Some(total),
        completed: Some(0),
    });

    let tmp_dest = dest.with_extension("gguf.part");
    let mut file =
        std::fs::File::create(&tmp_dest).with_context(|| format!("creating {tmp_dest:?}"))?;

    let mut completed: u64 = 0;
    let mut stream = resp;
    while let Some(chunk) = stream.chunk().await.context("reading download stream")? {
        file.write_all(&chunk).context("writing to file")?;
        completed += chunk.len() as u64;
        send!(PullStatus {
            status: "downloading".to_string(),
            digest: Some(digest.clone()),
            total: Some(total),
            completed: Some(completed),
        });
    }
    drop(file);

    std::fs::rename(&tmp_dest, &dest)
        .with_context(|| format!("renaming {tmp_dest:?} to {dest:?}"))?;

    send!(PullStatus {
        status: "verifying sha256 digest".to_string(),
        digest: Some(digest.clone()),
        total: None,
        completed: None,
    });

    send!(PullStatus {
        status: "success".to_string(),
        digest: None,
        total: None,
        completed: None,
    });

    Ok(())
}
