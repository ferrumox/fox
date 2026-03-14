// `fox serve` — start the OpenAI-compatible HTTP inference server.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::api::router;
use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::metrics::Metrics;
use crate::scheduler::Scheduler;

use super::get_gpu_memory_bytes;
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

    tracing::info!("loading model from {:?}", args.model_path);
    let model = LlamaCppModel::load(&args.model_path, args.max_batch_size, args.max_context_len)?;
    let model_config = model.model_config();

    let model_name = args
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default")
        .to_string();

    let gpu_memory_bytes = get_gpu_memory_bytes();
    let kv_cache = std::sync::Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        args.gpu_memory_fraction,
        args.block_size,
    ));

    let scheduler = std::sync::Arc::new(Scheduler::new(kv_cache.clone(), args.max_batch_size));

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

    let model = std::sync::Arc::new(model);
    let engine = std::sync::Arc::new(InferenceEngine::new(
        model.clone(),
        scheduler.clone(),
        kv_cache.clone(),
        model_name,
        metrics,
    ));

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

    let app = router(engine.clone(), system_prompt, started_at)
        .layer(tower_http::cors::CorsLayer::permissive());

    tracing::info!("listening on {}", addr);
    theme::print_serve_ready(
        args.model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model"),
        &addr.to_string(),
    );

    let server = axum::serve(tokio::net::TcpListener::bind(addr).await?, app);

    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.run_loop().await {
                tracing::error!("engine loop error: {}", e);
            }
        })
    };

    tokio::select! {
        r = server => { r?; }
        r = engine_loop => { r?; }
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
