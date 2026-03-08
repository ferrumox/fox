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
    let model = LlamaCppModel::load(&config.model_path, config.max_batch_size)?;
    let model_config = model.model_config();

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
    let engine = InferenceEngine::new(model.clone(), scheduler.clone(), kv_cache.clone());
    let engine = std::sync::Arc::new(engine);

    let addr: std::net::SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .expect("valid address");
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
    }

    Ok(())
}

fn get_gpu_memory_bytes() -> usize {
    #[cfg(feature = "cuda")]
    {
        if let Ok(cu) = cudarc::driver::CudaDevice::new(0) {
            if let Ok((free, total)) = cu.get_mem_info() {
                return total;
            }
        }
    }
    // Fallback: 8 GB
    8 * 1024 * 1024 * 1024
}
