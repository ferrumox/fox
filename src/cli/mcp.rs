// `fox mcp` — start an MCP (Model Context Protocol) server over stdio.
// Designed for IDE integration (Cursor, VS Code, and other MCP clients).

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

    let registry_cfg = RegistryConfig::embedded(models_dir.clone(), gpu_memory_bytes);
    let registry = Arc::new(ModelRegistry::new(registry_cfg, aliases));

    if let Some(path) = &args.model_path {
        eprintln!("mcp: pre-loading model from {:?}", path);
        registry
            .get_or_load(path.to_string_lossy().as_ref())
            .await?;
    }

    eprintln!("fox mcp server listening on stdio");
    McpServer::new(registry, models_dir).run().await
}
