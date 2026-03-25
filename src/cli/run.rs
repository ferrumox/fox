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

use super::resolve_model_path;
use super::theme;
use super::{get_gpu_info, get_gpu_memory_bytes, get_ram_info, get_total_gpu_memory_bytes};

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
    #[arg(long, default_value = "4096")]
    pub max_new_tokens: usize,

    /// Sampling temperature (0 = greedy). Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.8.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling threshold. Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.9.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Top-K filter (0 = disabled). Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.
    #[arg(long)]
    pub top_k: Option<u32>,

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

    /// Maximum context length per sequence (tokens).
    /// If omitted, fox auto-detects the model's trained context length.
    #[arg(long)]
    pub max_context_len: Option<u32>,

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

    /// Primary GPU index (0-based). Used when split_mode=none, or as main GPU for splits.
    #[arg(long, default_value = "0", env = "FOX_MAIN_GPU")]
    pub main_gpu: i32,

    /// How to split the model across multiple GPUs: none, layer (default), row.
    #[arg(long, default_value = "layer", env = "FOX_SPLIT_MODE")]
    pub split_mode: String,

    /// Comma-separated VRAM proportions for tensor splitting (e.g. "3,1" for 75%/25%).
    #[arg(long, env = "FOX_TENSOR_SPLIT")]
    pub tensor_split: Option<String>,

    /// Offload MoE expert tensors to CPU RAM instead of VRAM.
    #[arg(long, env = "FOX_MOE_CPU")]
    pub moe_cpu: bool,

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
    let (model_name, model_path) = match resolve_model_path(&args.model, args.alias_file.as_deref())
    {
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

    let split_mode = match args.split_mode.as_str() {
        "row" => 2u32,
        "none" => 0u32,
        _ => 1u32, // layer
    };
    let tensor_split_parsed: Vec<f32> = args
        .tensor_split
        .as_deref()
        .map(|s| {
            let raw: Vec<f32> = s
                .split(',')
                .filter_map(|p| p.trim().parse::<f32>().ok())
                .collect();
            let sum: f32 = raw.iter().sum();
            if sum > 0.0 {
                raw.iter().map(|&v| v / sum).collect()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();
    let gpu_memory_bytes_load = if split_mode != 0 {
        get_total_gpu_memory_bytes()
    } else {
        get_gpu_memory_bytes()
    };
    let model = LlamaCppModel::load(
        &model_path,
        1,
        args.max_context_len,
        gpu_memory_bytes_load,
        args.gpu_memory_fraction,
        1,
        args.main_gpu,
        split_mode,
        &tensor_split_parsed,
        args.moe_cpu,
    )?;
    let model_config = model.model_config();

    spinner.finish_and_clear();
    theme::print_success("Model loaded.");

    let gpu_memory_bytes = gpu_memory_bytes_load;
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

    let effective_ctx = engine.context_len();
    let supports_thinking = engine.supports_thinking();
    let mut show_thinking = supports_thinking;

    theme::print_banner(model_name, effective_ctx, supports_thinking);
    let startup_gpu = get_gpu_info();
    let startup_ram = get_ram_info();
    theme::print_system_info(
        startup_gpu.as_ref(),
        &startup_ram,
        effective_ctx,
        supports_thinking,
        show_thinking,
    );

    // Keep the engine loop running for the lifetime of the session.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };
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
            if !supports_thinking {
                theme::eprint_styled(
                    Some(crossterm::style::Color::Yellow),
                    false,
                    false,
                    "  This model has no native reasoning support (<think> token not found)\n\n",
                );
            } else {
                show_thinking = !show_thinking;
                let status = if show_thinking { "on" } else { "off" };
                theme::eprint_styled(None, false, true, &format!("  think · {status}\n\n"));
            }
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
        let toks_per_sec = if secs > 0.0 {
            token_count as f64 / secs
        } else {
            0.0
        };

        if response.is_empty() {
            eprintln!("(Context window full — clearing conversation history.)");
            eprintln!();
            messages.truncate(if args.no_system_prompt { 0 } else { 1 });
        } else {
            messages.push(("assistant".to_string(), response));
        }

        let ctx_tokens = engine
            .apply_chat_template(&messages)
            .and_then(|p| engine.tokenize(&p).map(|t| t.len()))
            .unwrap_or(0);
        let gpu_info = get_gpu_info();
        let ram_info = get_ram_info();
        theme::print_status_line(
            ctx_tokens,
            engine.context_len(),
            gpu_info.as_ref(),
            &ram_info,
            toks_per_sec,
        );
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
    let mut prompt = engine.apply_chat_template(messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    // For models that require an explicit thinking activation (e.g. Qwen3-Instruct),
    // append the opening <think> tag to the prompt so the model starts in reasoning
    // mode.  The engine state is also initialised with in_thinking=true so the output
    // filter knows the first generated tokens are reasoning content, not regular output.
    if show_thinking {
        prompt.push_str("<think>\n");
    }

    let prompt_tokens = engine
        .tokenize(&prompt)
        .unwrap_or_else(|_| prompt.bytes().map(|b| b as i32).take(4096).collect());

    let recommended = engine.recommended_sampling();
    let mut sampling = build_sampling_params(args, recommended.as_ref());
    sampling.show_thinking = show_thinking;
    sampling.initial_in_thinking = show_thinking;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine.submit_request(req);

    let stdout = std::io::stdout();
    let mut response = String::new();
    let mut token_count: usize = 0;
    let mut first_token = true;
    // Track whether we are currently inside a <think>…</think> block so we
    // can apply ANSI dim styling to the reasoning section.
    let mut in_thinking_display = show_thinking;

    while let Some(token) = rx.recv().await {
        if !token.text.is_empty() {
            if first_token {
                spinner.finish_and_clear();
                eprintln!();
                theme::print_fox_label();
                let _ = std::io::stderr().flush();
                // The <think> tag was injected into the prompt; emit it
                // synthetically with dim styling so the user sees it.
                if show_thinking {
                    println!("\x1b[2m<think>");
                    let _ = stdout.lock().flush();
                }
                first_token = false;
            }

            if in_thinking_display {
                if let Some(idx) = token.text.find("</think>") {
                    // Print everything up to and including </think> in dim,
                    // then reset and print any text that follows normally.
                    let end = idx + "</think>".len();
                    print!("{}\x1b[0m{}", &token.text[..end], &token.text[end..]);
                    in_thinking_display = false;
                } else {
                    // Still inside thinking block — dim mode stays active.
                    print!("{}", token.text);
                }
            } else {
                print!("{}", token.text);
            }
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
    let mut prompt = engine.apply_chat_template(messages).unwrap_or_else(|_| {
        messages
            .iter()
            .map(|(r, c)| format!("{}: {}", r, c))
            .collect::<Vec<_>>()
            .join("\n")
    });

    if args.show_thinking {
        prompt.push_str("<think>\n");
    }

    let prompt_tokens = engine
        .tokenize(&prompt)
        .unwrap_or_else(|_| prompt.bytes().map(|b| b as i32).take(4096).collect());

    let recommended = engine.recommended_sampling();
    let mut sampling = build_sampling_params(args, recommended.as_ref());
    sampling.initial_in_thinking = args.show_thinking;

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
    if args.show_thinking {
        println!("<think>");
        let _ = stdout.lock().flush();
    }
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

fn build_sampling_params(
    args: &RunArgs,
    recommended: Option<&crate::engine::model::RecommendedSampling>,
) -> SamplingParams {
    // Priority: user flag > model metadata > hardcoded default.
    let temperature = args
        .temperature
        .or_else(|| recommended.and_then(|r| r.temperature))
        .unwrap_or(0.8);
    let top_p = args
        .top_p
        .or_else(|| recommended.and_then(|r| r.top_p))
        .unwrap_or(0.9);
    let top_k = args
        .top_k
        .or_else(|| recommended.and_then(|r| r.top_k))
        .unwrap_or(0);

    SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
        stop: None,
        show_thinking: args.show_thinking,
        initial_in_thinking: false, // set by callers that force thinking mode
        max_thinking_chars: 8192,
    }
}
