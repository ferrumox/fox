// `ferrum run` — single-shot inference, streaming output to stdout.
// Reuses the full engine stack (Scheduler + InferenceEngine) without an HTTP server.

use std::io::Write as _;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::{InferenceRequest, SamplingParams};

use super::get_gpu_memory_bytes;

#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Path to the GGUF model file
    #[arg(long, env = "FERRUM_MODEL_PATH")]
    pub model_path: PathBuf,

    /// The prompt to send to the model
    pub prompt: String,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "512")]
    pub max_new_tokens: usize,

    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value = "1.0")]
    pub temperature: f32,

    /// Top-p nucleus sampling threshold
    #[arg(long, default_value = "1.0")]
    pub top_p: f32,

    /// Top-K filter (0 = disabled)
    #[arg(long, default_value = "0")]
    pub top_k: u32,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value = "1.0")]
    pub repetition_penalty: f32,

    /// RNG seed for reproducible output
    #[arg(long)]
    pub seed: Option<u64>,

    /// System prompt prepended to the conversation
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        env = "FERRUM_SYSTEM_PROMPT"
    )]
    pub system_prompt: String,

    /// Disable system prompt injection entirely
    #[arg(long)]
    pub no_system_prompt: bool,

    /// Maximum context length (tokens)
    #[arg(long, default_value = "4096")]
    pub max_context_len: u32,

    /// Fraction of GPU/RAM to use for KV cache
    #[arg(long, default_value = "0.85")]
    pub gpu_memory_fraction: f32,

    /// Tokens per KV block
    #[arg(long, default_value = "16")]
    pub block_size: usize,

    /// Show engine logs (hidden by default for cleaner output)
    #[arg(long)]
    pub verbose: bool,
}

pub async fn run_run(args: RunArgs) -> Result<()> {
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }

    eprint!("Loading model… ");
    let _ = std::io::stderr().flush();

    let model = LlamaCppModel::load(&args.model_path, 1, args.max_context_len)?;
    let model_config = model.model_config();
    eprintln!("done.");

    let gpu_memory_bytes = get_gpu_memory_bytes();
    let kv_cache = std::sync::Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        args.gpu_memory_fraction,
        args.block_size,
    ));
    let scheduler = std::sync::Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));

    let model_name = args
        .model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("default")
        .to_string();

    let model = std::sync::Arc::new(model);
    let engine = std::sync::Arc::new(InferenceEngine::new(
        model.clone(),
        scheduler.clone(),
        kv_cache,
        model_name,
        None,
    ));

    // Build message list and apply chat template.
    let mut messages: Vec<(String, String)> = Vec::new();
    if !args.no_system_prompt && !args.system_prompt.is_empty() {
        messages.push(("system".to_string(), args.system_prompt.clone()));
    }
    messages.push(("user".to_string(), args.prompt.clone()));

    let prompt = engine.apply_chat_template(&messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens = engine.tokenize(&prompt).unwrap_or_else(|_| {
        prompt.bytes().map(|b| b as i32).take(4096).collect()
    });

    let sampling = SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        stop: None,
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine.submit_request(req);

    // Drive the engine loop in the background.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };

    // Stream tokens to stdout.
    let stdout = std::io::stdout();
    while let Some(token) = rx.recv().await {
        if !token.text.is_empty() {
            print!("{}", token.text);
            let _ = stdout.lock().flush();
        }
        if token.stop_reason.is_some() {
            break;
        }
    }
    println!(); // trailing newline

    engine_loop.abort();
    Ok(())
}
