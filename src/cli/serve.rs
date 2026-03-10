// `fox serve` — start the OpenAI-compatible HTTP inference server.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::api::router;
use crate::metrics::Metrics;
use crate::model_registry::{ModelRegistry, RegistryConfig};

use super::get_gpu_memory_bytes;
use super::models_dir as default_models_dir;
use super::theme;

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Path to the GGUF model file
    #[arg(long, env = "FOX_MODEL_PATH")]
    pub model_path: PathBuf,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long, default_value = "0.85", env = "FOX_GPU_MEMORY_FRACTION")]
    pub gpu_memory_fraction: f32,

    /// Maximum batch size for inference
    #[arg(long, default_value = "32", env = "FOX_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Tokens per KV block
    #[arg(long, default_value = "16", env = "FOX_BLOCK_SIZE")]
    pub block_size: usize,

    /// Host to bind the server to
    #[arg(long, default_value = "0.0.0.0", env = "FOX_HOST")]
    pub host: String,

    /// Port to bind the server to
    #[arg(long, default_value = "8080", env = "FOX_PORT")]
    pub port: u16,

    /// Maximum context length (tokens)
    #[arg(long, default_value = "4096", env = "FOX_MAX_CONTEXT_LEN")]
    pub max_context_len: u32,

    /// Default system prompt injected when no system message is present.
    /// Pass an empty string to disable injection.
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        env = "FOX_SYSTEM_PROMPT"
    )]
    pub system_prompt: String,

    /// Fraction of GPU memory reserved for CPU↔GPU KV-cache swap space (0.0-1.0).
    ///
    /// When GPU memory is exhausted, preempted requests will have their KV blocks
    /// swapped to CPU RAM instead of being discarded (re-prefill not required on
    /// resume).  Set to 0 to disable swapping (default).
    ///
    /// Note: swap transfer is currently a placeholder pending llama.cpp tensor-access
    /// API availability.  The flag is accepted to avoid breaking future config files.
    #[arg(long, default_value = "0.0", env = "FOX_SWAP_FRACTION")]
    pub swap_fraction: f32,

    /// Use JSON log format (for production)
    #[arg(long, env = "FOX_JSON_LOGS")]
    pub json_logs: bool,

    /// HuggingFace API token for authenticated model pulls via POST /api/pull.
    /// Can also be set with the HF_TOKEN environment variable.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Maximum number of models to keep in memory simultaneously (LRU eviction).
    #[arg(long, default_value = "1", env = "FOX_MAX_MODELS")]
    pub max_models: usize,

    /// Path to aliases TOML file. Default: ~/.config/ferrumox/aliases.toml
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,
}

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    if args.gpu_memory_fraction <= 0.0 || args.gpu_memory_fraction > 1.0 {
        anyhow::bail!(
            "gpu_memory_fraction must be in range (0, 1], got {}",
            args.gpu_memory_fraction
        );
    }
    if args.max_context_len == 0 {
        anyhow::bail!("max_context_len must be greater than 0");
    }

    if args.json_logs {
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(EnvFilter::from_default_env())
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }

    let model_name = args
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default")
        .to_string();

    let gpu_memory_bytes = get_gpu_memory_bytes();

    let metrics = match Metrics::new() {
        Ok(m) => {
            tracing::info!("Prometheus metrics registered; scrape at /metrics");
            Some(std::sync::Arc::new(m))
        }
        Err(e) => {
            tracing::warn!("failed to register Prometheus metrics: {}", e);
            None
        }
    };

    // Load model aliases from TOML file (optional).
    let aliases = load_aliases(args.alias_file.clone());

    let models_dir = default_models_dir();

    let registry_cfg = RegistryConfig {
        models_dir: models_dir.clone(),
        max_models: args.max_models.max(1),
        max_batch_size: args.max_batch_size,
        max_context_len: args.max_context_len,
        block_size: args.block_size,
        gpu_memory_bytes,
        gpu_memory_fraction: args.gpu_memory_fraction,
        metrics,
    };

    let registry = std::sync::Arc::new(ModelRegistry::new(registry_cfg, aliases));

    // Pre-load the initial model so the first request is instant.
    tracing::info!("loading model from {:?}", args.model_path);
    registry.get_or_load(&model_name).await?;

    let addr: std::net::SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{}:{}': {}", args.host, args.port, e))?;

    let system_prompt = if args.system_prompt.is_empty() {
        None
    } else {
        Some(args.system_prompt)
    };

    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let app = router(
        registry,
        model_name.clone(),
        system_prompt,
        started_at,
        models_dir,
        args.hf_token,
    )
    .layer(tower_http::cors::CorsLayer::permissive());

    tracing::info!("listening on {}", addr);
    theme::print_serve_ready(&model_name, &addr.to_string());

    let server = axum::serve(tokio::net::TcpListener::bind(addr).await?, app);

    tokio::select! {
        r = server => { r?; }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received, exiting");
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm =
            signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = sigterm.recv() => {}
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}

/// Load aliases from a TOML file.
///
/// The file format is:
/// ```toml
/// [aliases]
/// "llama3" = "Llama-3.2-3B-Instruct-f16"
/// ```
///
/// If the file does not exist or cannot be parsed, returns an empty map.
fn load_aliases(path: Option<PathBuf>) -> HashMap<String, String> {
    let path = path.unwrap_or_else(|| {
        let home = std::env::var("HOME").unwrap_or_default();
        PathBuf::from(home).join(".config/ferrumox/aliases.toml")
    });

    if !path.exists() {
        return HashMap::new();
    }

    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("failed to read alias file {:?}: {}", path, e);
            return HashMap::new();
        }
    };

    #[derive(serde::Deserialize)]
    struct AliasesFile {
        #[serde(default)]
        aliases: HashMap<String, String>,
    }

    match toml::from_str::<AliasesFile>(&content) {
        Ok(f) => {
            tracing::info!("loaded {} alias(es) from {:?}", f.aliases.len(), path);
            f.aliases
        }
        Err(e) => {
            tracing::warn!("failed to parse alias file {:?}: {}", path, e);
            HashMap::new()
        }
    }
}
