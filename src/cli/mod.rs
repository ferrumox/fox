// CLI entry point — dispatch to subcommands.

pub mod alias;
pub mod bench;
pub mod bench_kv;
pub mod discover;
pub mod list;
pub mod models;
pub mod ps;
pub mod pull;
pub mod rm;
pub mod run;
pub mod search;
pub mod serve;
pub mod show;
pub mod theme;
pub mod utils;

pub use utils::{
    expand_tilde, format_age, format_size, get_all_gpu_memory_bytes, get_gpu_info,
    get_gpu_memory_bytes, get_ram_info, get_total_gpu_memory_bytes, list_models, load_aliases,
    models_dir, resolve_model_path, GpuInfo, RamInfo,
};

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "fox")]
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
    /// Benchmark model load time and inference throughput
    Bench(bench::BenchArgs),
    /// Compare KV cache quantization types (F16, Q8_0, TurboQuant) side-by-side
    BenchKv(bench_kv::BenchKvArgs),
    /// Download a GGUF model from HuggingFace Hub
    Pull(pull::PullArgs),
    /// List downloaded models
    List(list::ListArgs),
    /// Remove a downloaded model
    Rm(rm::RmArgs),
    /// Show details about a downloaded model
    Show(show::ShowArgs),
    /// Show running model servers
    Ps(ps::PsArgs),
    /// List curated models available to pull
    Models(models::ModelsArgs),
    /// Search HuggingFace Hub for GGUF models
    Search(search::SearchArgs),
    /// Manage model name aliases
    Alias(alias::AliasArgs),
    /// Discover GGUF models across well-known directories
    Discover(discover::DiscoverArgs),
}

/// Known subcommand names — anything else is treated as `fox run <arg>`.
const SUBCOMMANDS: &[&str] = &[
    "serve", "run", "bench", "bench-kv", "pull", "list", "rm", "show", "ps", "models", "search",
    "alias", "discover", "help",
];

pub async fn run() -> anyhow::Result<()> {
    // Load config file before clap parses CLI args so env-var-backed flags
    // pick up config values as their effective defaults.
    crate::config::load_config_into_env();

    // If the first non-flag argument is not a known subcommand, inject "run"
    // so that `fox llama "Hello"` works as `fox run llama "Hello"`.
    let raw: Vec<String> = std::env::args().collect();
    let effective: Vec<String> = match raw.get(1).map(String::as_str) {
        Some(first) if !first.starts_with('-') && !SUBCOMMANDS.contains(&first) => {
            let mut v = vec![raw[0].clone(), "run".to_string()];
            v.extend(raw[1..].iter().cloned());
            v
        }
        _ => raw,
    };

    let cli = Cli::parse_from(effective);
    match cli.command {
        Command::Serve(args) => serve::run_serve(args).await,
        Command::Run(args) => run::run_run(args).await,
        Command::Bench(args) => bench::run_bench(args).await,
        Command::BenchKv(args) => bench_kv::run_bench_kv(args).await,
        Command::Pull(args) => pull::run_pull(args).await,
        Command::List(args) => list::run_list(args).await,
        Command::Rm(args) => rm::run_rm(args).await,
        Command::Show(args) => show::run_show(args).await,
        Command::Ps(args) => ps::run_ps(args).await,
        Command::Models(args) => models::run_models(args).await,
        Command::Search(args) => search::run_search(args).await,
        Command::Alias(args) => alias::run_alias(args).await,
        Command::Discover(args) => discover::run_discover(args).await,
    }
}
