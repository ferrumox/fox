// `fox run` — single-shot inference or interactive REPL, streaming output to stdout.
// Reuses the full engine stack (Scheduler + InferenceEngine) without an HTTP server.
//
// fox run llama "Hello"   → one-shot (resolved from models_dir)
// fox run llama           → opens interactive REPL
// fox run /abs/path/to/model.gguf "Hello"  → direct path

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::scheduler::{InferenceRequest, SamplingParams};

use super::get_gpu_memory_bytes;
use super::resolve_model_path;
use super::theme;

#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Model name, alias, or path to a GGUF file.
    /// Resolved against ~/.cache/ferrumox/models with alias → exact → prefix → contains fallback.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// The prompt to send to the model.
    /// If omitted, an interactive chat session is started.
    pub prompt: Option<String>,

    /// Path to aliases TOML file (default: ~/.config/ferrumox/aliases.toml)
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Maximum number of tokens to generate per turn
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
        env = "FOX_SYSTEM_PROMPT"
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

    /// Fraction of GPU memory reserved for CPU↔GPU KV-cache swap space (0.0-1.0).
    /// Set to 0 to disable (default). Currently a placeholder — see `fox serve --help`.
    #[arg(long, default_value = "0.0")]
    pub swap_fraction: f32,

    /// Show the model's internal <think>…</think> reasoning block in the output.
    /// By default reasoning is suppressed; only the final answer is printed.
    #[arg(long)]
    pub show_thinking: bool,

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

    // Resolve model — auto-pull from HuggingFace if not found locally.
    let (model_name, model_path) =
        match resolve_model_path(&args.model, args.alias_file.as_deref()) {
            Ok(r) => r,
            Err(_) => {
                eprintln!(
                    "Model '{}' not found locally. Pulling from HuggingFace…",
                    args.model
                );
                super::pull::run_pull(super::pull::PullArgs {
                    model_id: args.model.clone(),
                    filename: None,
                    output_dir: None,
                    hf_token: std::env::var("HF_TOKEN").ok(),
                })
                .await?;
                resolve_model_path(&args.model, args.alias_file.as_deref())?
            }
        };

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .expect("valid template")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("Loading model…");
    spinner.enable_steady_tick(Duration::from_millis(80));

    let gpu_memory_bytes_load = get_gpu_memory_bytes();
    let model = LlamaCppModel::load(&model_path, 1, args.max_context_len, gpu_memory_bytes_load, args.gpu_memory_fraction, 1)?;
    let model_config = model.model_config();

    spinner.finish_and_clear();
    theme::print_success("Model loaded.");

    let gpu_memory_bytes = get_gpu_memory_bytes();
    let kv_cache = std::sync::Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        args.gpu_memory_fraction,
        args.block_size,
        1,
    ));
    let scheduler = std::sync::Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));

    let model = std::sync::Arc::new(model);
    let engine = std::sync::Arc::new(InferenceEngine::new(
        model.clone(),
        scheduler.clone(),
        kv_cache,
        model_name,
        None,
    ));

    match args.prompt.clone() {
        Some(prompt) => run_oneshot(&args, &engine, prompt).await,
        None => run_repl(&args, &engine).await,
    }
}

/// One-shot mode: send a single prompt and stream the response to stdout.
async fn run_oneshot(args: &RunArgs, engine: &Arc<InferenceEngine>, prompt: String) -> Result<()> {
    let mut messages: Vec<(String, String)> = Vec::new();
    if !args.no_system_prompt && !args.system_prompt.is_empty() {
        messages.push(("system".to_string(), args.system_prompt.clone()));
    }
    messages.push(("user".to_string(), prompt));

    stream_turn(args, engine, &messages).await?;
    println!();
    Ok(())
}

/// Interactive REPL mode: maintain conversation history across multiple turns.
async fn run_repl(args: &RunArgs, engine: &Arc<InferenceEngine>) -> Result<()> {
    let model_name = engine.model_name();

    theme::print_banner(model_name, args.max_context_len);

    // Keep the engine loop running for the lifetime of the session.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };

    let mut show_thinking = args.show_thinking;
    let mut messages: Vec<(String, String)> = Vec::new();
    if !args.no_system_prompt && !args.system_prompt.is_empty() {
        messages.push(("system".to_string(), args.system_prompt.clone()));
    }

    loop {
        theme::print_prompt_glyph();

        // Read input via spawn_blocking to avoid blocking the tokio runtime thread,
        // which would starve the engine loop task running concurrently.
        // Typing `"""` on its own line enters multiline mode; a second `"""` submits.
        let result = tokio::task::spawn_blocking(|| {
            let mut first_line = String::new();
            let n = std::io::stdin().read_line(&mut first_line)?;
            if n == 0 {
                return Ok::<(String, usize), std::io::Error>((first_line, 0));
            }
            if first_line.trim() == "\"\"\"" {
                let mut buf = String::new();
                let mut total = n;
                loop {
                    eprint!("  · ");
                    let _ = std::io::stderr().flush();
                    let mut line = String::new();
                    let m = std::io::stdin().read_line(&mut line)?;
                    if m == 0 {
                        break;
                    }
                    total += m;
                    if line.trim() == "\"\"\"" {
                        break;
                    }
                    buf.push_str(&line);
                }
                Ok((buf, total))
            } else {
                Ok((first_line, n))
            }
        })
        .await
        .expect("spawn_blocking panicked");

        let (line, n) = match result {
            Ok(v) => v,
            Err(e) => {
                eprintln!("\nError reading input: {}", e);
                break;
            }
        };

        eprintln!();

        if n == 0 {
            // EOF (Ctrl+D)
            break;
        }

        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/bye" || input == "/exit" || input == "exit" || input == "quit" {
            break;
        }

        if input == "/think" {
            show_thinking = !show_thinking;
            let status = if show_thinking { "activado" } else { "desactivado" };
            theme::eprint_styled(None, false, true, &format!("  Razonamiento {status}\n\n"));
            continue;
        }

        messages.push(("user".to_string(), input));

        // Thinking spinner
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("  {spinner:.dim} {msg:.dim}")
                .expect("valid template")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message("Thinking…");
        spinner.enable_steady_tick(Duration::from_millis(80));

        let start = Instant::now();
        let (response, token_count) =
            stream_turn_collecting(args, engine, &messages, spinner, show_thinking).await?;
        let elapsed = start.elapsed();

        println!();
        let secs = elapsed.as_secs_f64();
        let toks_per_sec = if secs > 0.0 { token_count as f64 / secs } else { 0.0 };
        theme::eprint_styled(
            None,
            false,
            true,
            &format!(
                "  {} tokens · {:.1}s · {:.1} tok/s\n\n",
                token_count,
                secs,
                toks_per_sec
            ),
        );

        if response.is_empty() {
            eprintln!("(Context window full — clearing conversation history.)");
            eprintln!();
            messages.truncate(if args.no_system_prompt { 0 } else { 1 });
        } else {
            messages.push(("assistant".to_string(), response));
        }
    }

    engine_loop.abort();
    Ok(())
}

/// Run one inference turn, stream tokens to stdout, and return `(response_text, token_count)`.
async fn stream_turn_collecting(
    args: &RunArgs,
    engine: &Arc<InferenceEngine>,
    messages: &[(String, String)],
    spinner: ProgressBar,
    show_thinking: bool,
) -> Result<(String, usize)> {
    let prompt = engine.apply_chat_template(messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens = engine
        .tokenize(&prompt)
        .unwrap_or_else(|_| prompt.bytes().map(|b| b as i32).take(4096).collect());

    let mut sampling = build_sampling_params(args);
    sampling.show_thinking = show_thinking;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine.submit_request(req);

    let stdout = std::io::stdout();
    let mut response = String::new();
    let mut token_count: usize = 0;
    let mut first_token = true;

    while let Some(token) = rx.recv().await {
        if !token.text.is_empty() {
            if first_token {
                spinner.finish_and_clear();
                eprintln!();
                theme::print_fox_label();
                let _ = std::io::stderr().flush();
                first_token = false;
            }
            print!("{}", token.text);
            let _ = stdout.lock().flush();
            response.push_str(&token.text);
            token_count += 1;
        }
        if token.stop_reason.is_some() {
            if first_token {
                spinner.finish_and_clear();
            }
            break;
        }
    }

    Ok((response, token_count))
}

/// Run one inference turn streaming to stdout (no response collection — for one-shot mode).
async fn stream_turn(
    args: &RunArgs,
    engine: &Arc<InferenceEngine>,
    messages: &[(String, String)],
) -> Result<()> {
    let prompt = engine.apply_chat_template(messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    let prompt_tokens = engine
        .tokenize(&prompt)
        .unwrap_or_else(|_| prompt.bytes().map(|b| b as i32).take(4096).collect());

    let sampling = build_sampling_params(args);

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine.submit_request(req);

    // Drive the engine loop in the background for this single request.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };

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

    engine_loop.abort();
    Ok(())
}

fn build_sampling_params(args: &RunArgs) -> SamplingParams {
    SamplingParams {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        stop: None,
        show_thinking: args.show_thinking,
    }
}
