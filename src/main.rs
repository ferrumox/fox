// Entry point: parse config, init tracing, load model, start server + engine.

use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use ferrum_engine::api::router;
use ferrum_engine::config::Config;
use ferrum_engine::engine::model::{LlamaCppModel, Model};
use ferrum_engine::engine::InferenceEngine;
use ferrum_engine::kv_cache::KVCacheManager;
use ferrum_engine::scheduler::Scheduler;

#[tokio::main]
async fn main() -> Result<()> {
    let config = Config::from_args();

    // Validate config ranges early so the user gets a clear error message.
    if config.gpu_memory_fraction <= 0.0 || config.gpu_memory_fraction > 1.0 {
        anyhow::bail!(
            "gpu_memory_fraction must be in range (0, 1], got {}",
            config.gpu_memory_fraction
        );
    }
    if config.max_context_len == 0 {
        anyhow::bail!("max_context_len must be greater than 0");
    }

    if config.json_logs {
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

    tracing::info!("loading model from {:?}", config.model_path);
    let model = LlamaCppModel::load(&config.model_path, config.max_batch_size, config.max_context_len)?;
    let model_config = model.model_config();

    // Derive a human-readable model name from the file stem (e.g. "qwen2-7b-instruct-q4")
    let model_name = config
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default")
        .to_string();

    let gpu_memory_bytes = get_gpu_memory_bytes();
    let kv_cache = KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        config.gpu_memory_fraction,
        config.block_size,
    );
    let kv_cache = std::sync::Arc::new(kv_cache);

    let scheduler = Scheduler::new(kv_cache.clone());
    let scheduler = std::sync::Arc::new(scheduler);

    let model = std::sync::Arc::new(model);
    let engine = InferenceEngine::new(model.clone(), scheduler.clone(), kv_cache.clone(), model_name);
    let engine = std::sync::Arc::new(engine);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid bind address '{}:{}': {}", config.host, config.port, e))?;
    let app = router(engine.clone()).layer(tower_http::cors::CorsLayer::permissive());

    tracing::info!("listening on {}", addr);

    let server = axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app,
    );

    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = engine.run_loop().await {
                tracing::error!("engine loop error: {}", e);
            }
        })
    };

    tokio::select! {
        r = server => {
            r?;
        }
        r = engine_loop => {
            r?;
        }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received, exiting");
        }
    }

    Ok(())
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };

    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = ctrl_c => {}
            _ = sigterm.recv() => {}
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}

fn get_gpu_memory_bytes() -> usize {
    #[cfg(feature = "cuda")]
    {
        if let Ok(cu) = cudarc::driver::CudaDevice::new(0) {
            if let Ok((_free, total)) = cu.get_mem_info() {
                return total;
            }
        }
    }
    // Fallback: 8 GB
    8 * 1024 * 1024 * 1024
}
