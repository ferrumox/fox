// `fox serve` — start the OpenAI-compatible HTTP inference server.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// Default values for ServeArgs — centralised so grep/docs find them in one place.
const DEFAULT_GPU_MEMORY_FRACTION: &str = "0.85";
const DEFAULT_MAX_BATCH_SIZE: &str = "32";
const DEFAULT_BLOCK_SIZE: &str = "16";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8080";
const DEFAULT_MAX_CONTEXT_LEN: &str = "4096";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const DEFAULT_SWAP_FRACTION: &str = "0.0";
const DEFAULT_MAX_MODELS: &str = "1";
const DEFAULT_KEEP_ALIVE_SECS: &str = "300";
const DEFAULT_TYPE_KV: &str = "f16";

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::api::router;
use crate::metrics::Metrics;
use crate::model_registry::{ModelRegistry, RegistryConfig};

use super::get_gpu_memory_bytes;
use super::list_models;
use super::load_aliases;
use super::models_dir as default_models_dir;
use super::theme;

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Path to the GGUF model file (optional; if omitted models are loaded on demand)
    #[arg(long, env = "FOX_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long, default_value = DEFAULT_GPU_MEMORY_FRACTION, env = "FOX_GPU_MEMORY_FRACTION")]
    pub gpu_memory_fraction: f32,

    /// Maximum batch size for inference
    #[arg(long, default_value = DEFAULT_MAX_BATCH_SIZE, env = "FOX_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Tokens per KV block
    #[arg(long, default_value = DEFAULT_BLOCK_SIZE, env = "FOX_BLOCK_SIZE")]
    pub block_size: usize,

    /// Host to bind the server to
    #[arg(long, default_value = DEFAULT_HOST, env = "FOX_HOST")]
    pub host: String,

    /// Port to bind the server to
    #[arg(long, default_value = DEFAULT_PORT, env = "FOX_PORT")]
    pub port: u16,

    /// Maximum context length (tokens)
    #[arg(long, default_value = DEFAULT_MAX_CONTEXT_LEN, env = "FOX_MAX_CONTEXT_LEN")]
    pub max_context_len: u32,

    /// Default system prompt injected when no system message is present.
    /// Pass an empty string to disable injection.
    #[arg(
        long,
        default_value = DEFAULT_SYSTEM_PROMPT,
        env = "FOX_SYSTEM_PROMPT"
    )]
    pub system_prompt: String,

    /// Fraction of GPU memory reserved for CPU↔GPU KV-cache swap space (0.0-1.0).
    #[arg(long, default_value = DEFAULT_SWAP_FRACTION, env = "FOX_SWAP_FRACTION")]
    pub swap_fraction: f32,

    /// Use JSON log format (for production)
    #[arg(long, env = "FOX_JSON_LOGS")]
    pub json_logs: bool,

    /// HuggingFace API token for authenticated model pulls via POST /api/pull.
    /// Can also be set with the HF_TOKEN environment variable.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Maximum number of models to keep in memory simultaneously (LRU eviction).
    #[arg(long, default_value = DEFAULT_MAX_MODELS, env = "FOX_MAX_MODELS")]
    pub max_models: usize,

    /// Path to aliases TOML file. Default: ~/.config/ferrumox/aliases.toml
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Seconds a model stays in memory after its last request (0 = never evict by time).
    #[arg(long, default_value = DEFAULT_KEEP_ALIVE_SECS, env = "FOX_KEEP_ALIVE_SECS")]
    pub keep_alive_secs: u64,

    /// KV cache quantization: f16 (default), q8_0, q4_0
    #[arg(long, default_value = DEFAULT_TYPE_KV, env = "FOX_TYPE_KV")]
    pub type_kv: String,

    /// Require `Authorization: Bearer <key>` on every API request.
    /// Omit to run without authentication.
    #[arg(long, env = "FOX_API_KEY")]
    pub api_key: Option<String>,
}

fn parse_type_kv(s: &str) -> u32 {
    match s {
        "q8_0" => 8,
        "q4_0" => 2,
        _ => 1, // f16
    }
}

impl ServeArgs {
    fn validate(&self) -> Result<()> {
        if self.gpu_memory_fraction <= 0.0 || self.gpu_memory_fraction > 1.0 {
            anyhow::bail!(
                "gpu_memory_fraction must be in range (0, 1], got {}",
                self.gpu_memory_fraction
            );
        }
        if self.max_context_len == 0 {
            anyhow::bail!("max_context_len must be greater than 0");
        }
        if self.max_batch_size == 0 {
            anyhow::bail!("max_batch_size must be greater than 0");
        }
        if self.block_size == 0 {
            anyhow::bail!("block_size must be greater than 0");
        }
        Ok(())
    }
}

fn setup_logging(json: bool) {
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("ferrumox=info,warn"));

    if json {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().pretty())
            .init();
    }
}

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    args.validate()?;
    setup_logging(args.json_logs);

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
        keep_alive_secs: args.keep_alive_secs,
        type_kv: parse_type_kv(&args.type_kv),
    };

    let registry = std::sync::Arc::new(ModelRegistry::new(registry_cfg, aliases));

    // Pre-load the initial model if specified; otherwise use lazy loading.
    let primary_model = match &args.model_path {
        Some(path) => {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("default")
                .to_string();
            tracing::info!("pre-loading model from {:?}", path);
            registry.get_or_load(&name).await?;
            name
        }
        None => {
            // Lazy mode: discover the first model in models_dir as the primary.
            let first = match list_models(&models_dir) {
                Ok(models) => models
                    .into_iter()
                    .next()
                    .and_then(|(p, _)| p.file_stem().and_then(|s| s.to_str()).map(str::to_string)),
                Err(e) => {
                    tracing::error!("failed to list models in {}: {}", models_dir.display(), e);
                    None
                }
            };
            if let Some(ref name) = first {
                tracing::info!(
                    "lazy mode: primary model set to '{}' (not pre-loaded)",
                    name
                );
            } else {
                tracing::warn!(
                    "no model specified and no .gguf files found in {}; \
                     requests will fail until a model is available",
                    models_dir.display()
                );
            }
            first.unwrap_or_default()
        }
    };

    // Start background keep-alive eviction task.
    std::sync::Arc::clone(&registry).start_eviction_task();

    let addr: std::net::SocketAddr =
        format!("{}:{}", args.host, args.port)
            .parse()
            .map_err(|e| {
                anyhow::anyhow!("invalid bind address '{}:{}': {}", args.host, args.port, e)
            })?;

    let system_prompt = if args.system_prompt.is_empty() {
        None
    } else {
        Some(args.system_prompt)
    };

    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if args.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

    let app = router(
        registry,
        primary_model.clone(),
        system_prompt,
        started_at,
        models_dir,
        args.hf_token,
        args.api_key,
    )
    .layer(tower_http::cors::CorsLayer::permissive());

    tracing::info!("listening on {}", addr);
    let display_name = if primary_model.is_empty() {
        "none (lazy)"
    } else {
        &primary_model
    };
    theme::print_serve_ready(display_name, &addr.to_string());

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
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!("failed to listen for Ctrl-C: {}", e);
            // Block forever — server stays up if signal setup fails.
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = sigterm.recv() => {}
                }
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {}", e);
                ctrl_c.await;
            }
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}
