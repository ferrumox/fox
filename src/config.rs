use std::path::PathBuf;

use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "ferrum-engine")]
#[command(about = "High-performance LLM inference engine")]
pub struct Config {
    /// Path to the GGUF model file
    #[arg(long, env = "FERRUM_MODEL_PATH")]
    pub model_path: PathBuf,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long, default_value = "0.85", env = "FERRUM_GPU_MEMORY_FRACTION")]
    pub gpu_memory_fraction: f32,

    /// Maximum batch size for inference
    #[arg(long, default_value = "32", env = "FERRUM_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Tokens per KV block
    #[arg(long, default_value = "16", env = "FERRUM_BLOCK_SIZE")]
    pub block_size: usize,

    /// Host to bind the server to
    #[arg(long, default_value = "0.0.0.0", env = "FERRUM_HOST")]
    pub host: String,

    /// Port to bind the server to
    #[arg(long, default_value = "8080", env = "FERRUM_PORT")]
    pub port: u16,

    /// Maximum context length (tokens) for the model
    #[arg(long, default_value = "4096", env = "FERRUM_MAX_CONTEXT_LEN")]
    pub max_context_len: u32,

    /// Use JSON log format (for production)
    #[arg(long, env = "FERRUM_JSON_LOGS")]
    pub json_logs: bool,
}

impl Config {
    pub fn from_args() -> Self {
        Self::parse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config {
            model_path: PathBuf::from("/tmp/model.gguf"),
            gpu_memory_fraction: 0.85,
            max_batch_size: 32,
            block_size: 16,
            max_context_len: 4096,
            host: "0.0.0.0".to_string(),
            port: 8080,
            json_logs: false,
        };
        assert_eq!(config.gpu_memory_fraction, 0.85);
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.block_size, 16);
    }
}
