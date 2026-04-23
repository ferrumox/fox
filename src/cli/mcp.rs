// `fox mcp` — start an MCP (Model Context Protocol) server over stdio.
// Designed for IDE integration (Cursor, VS Code, Claude Code).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;

use crate::mcp::McpServer;
use crate::model_registry::{ModelRegistry, RegistryConfig};

use super::get_gpu_memory_bytes;
use super::load_aliases;
use super::models_dir as default_models_dir;

#[derive(Parser, Debug)]
pub struct McpArgs {
    /// Path to GGUF model file (optional; models are loaded on demand)
    #[arg(long, env = "FOX_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Host for model registry
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Port (not used for stdio transport, but needed for model registry)
    #[arg(long, default_value_t = 8080)]
    pub port: u16,

    /// Path to aliases TOML file. Default: ~/.config/ferrumox/aliases.toml
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,
}

pub async fn run_mcp(args: McpArgs) -> Result<()> {
    // Log to stderr only — stdout is the MCP transport.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("ferrumox=info,warn")),
        )
        .init();

    let gpu_memory_bytes = get_gpu_memory_bytes();
    let aliases = load_aliases(args.alias_file);
    let models_dir = default_models_dir();

    let registry_cfg = RegistryConfig {
        models_dir: models_dir.clone(),
        max_models: 1,
        max_batch_size: 4,
        max_context_len: None,
        block_size: 16,
        gpu_memory_bytes,
        gpu_memory_fraction: 0.85,
        metrics: None,
        keep_alive_secs: 300,
        type_k: 1, // F16
        type_v: 1, // F16
        main_gpu: 0,
        split_mode: 1, // layer
        tensor_split: vec![],
        moe_offload_cpu: false,
    };

    let registry = Arc::new(ModelRegistry::new(registry_cfg, aliases));

    if let Some(path) = &args.model_path {
        eprintln!("mcp: pre-loading model from {:?}", path);
        registry
            .get_or_load(path.to_string_lossy().as_ref())
            .await?;
    }

    Arc::clone(&registry).start_eviction_task();

    eprintln!("mcp: server ready on stdio");
    let server = McpServer::new(registry, models_dir);
    server.run().await
}
