// CLI entry point — dispatch to serve / run / pull subcommands.

pub mod pull;
pub mod run;
pub mod serve;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "ferrum")]
#[command(about = "High-performance LLM inference engine")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Start the OpenAI-compatible HTTP inference server
    Serve(serve::ServeArgs),
    /// Run single-shot inference and stream output to stdout
    Run(run::RunArgs),
    /// Download a GGUF model from HuggingFace Hub
    Pull(pull::PullArgs),
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => serve::run_serve(args).await,
        Command::Run(args) => run::run_run(args).await,
        Command::Pull(args) => pull::run_pull(args).await,
    }
}

/// Query total GPU memory via nvidia-smi. Falls back to 8 GiB on CPU-only builds.
pub(crate) fn get_gpu_memory_bytes() -> usize {
    #[cfg(feature = "cuda")]
    if let Ok(out) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                if let Ok(mib) = s.trim().parse::<usize>() {
                    return mib * 1024 * 1024;
                }
            }
        }
    }
    8 * 1024 * 1024 * 1024
}
