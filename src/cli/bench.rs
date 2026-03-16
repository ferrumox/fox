// `fox bench` — measure model load time and inference throughput.
//
// fox bench phi                    → benchmark with default prompt
// fox bench phi --prompt "Hello"   → custom prompt
// fox bench phi --runs 3           → average over 3 runs

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

use super::{get_gpu_memory_bytes, resolve_model_path, theme};

const DEFAULT_PROMPT: &str = "Explain what a large language model is in two sentences.";

#[derive(Parser, Debug)]
pub struct BenchArgs {
    /// Model name, alias, or path to a GGUF file.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// Prompt to use for benchmarking
    #[arg(long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,

    /// Path to aliases TOML file (default: ~/.config/ferrumox/aliases.toml)
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Maximum tokens to generate per run
    #[arg(long, default_value = "200")]
    pub max_new_tokens: usize,

    /// Maximum context length (tokens)
    #[arg(long, default_value = "4096")]
    pub max_context_len: u32,

    /// Number of runs to average results over
    #[arg(long, default_value = "1")]
    pub runs: usize,
}

pub async fn run_bench(args: BenchArgs) -> Result<()> {
    let (model_name, model_path) = resolve_model_path(&args.model, args.alias_file.as_deref())?;

    // ── Load model (timed) ───────────────────────────────────────────────────
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .expect("valid template")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("Loading model…");
    spinner.enable_steady_tick(Duration::from_millis(80));

    let load_start = Instant::now();
    let gpu_memory_bytes_load = get_gpu_memory_bytes();
    let model = LlamaCppModel::load(
        &model_path,
        1,
        args.max_context_len,
        gpu_memory_bytes_load,
        0.85,
        1,
    )?;
    let model_config = model.model_config();
    let load_elapsed = load_start.elapsed();

    spinner.finish_and_clear();

    // ── Engine setup ─────────────────────────────────────────────────────────
    let gpu_memory_bytes = get_gpu_memory_bytes();
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        0.85,
        16,
        1,
    ));
    let scheduler = Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));
    let model = Arc::new(model);
    let engine = Arc::new(InferenceEngine::new(
        model.clone(),
        scheduler.clone(),
        kv_cache,
        model_name.clone(),
        None,
    ));

    // ── Tokenize prompt ──────────────────────────────────────────────────────
    let messages: Vec<(String, String)> = vec![
        (
            "system".to_string(),
            "You are a helpful assistant.".to_string(),
        ),
        ("user".to_string(), args.prompt.clone()),
    ];
    let prompt_text = engine
        .apply_chat_template(&messages)
        .unwrap_or_else(|_| format!("user: {}", args.prompt));
    let prompt_tokens = engine
        .tokenize(&prompt_text)
        .unwrap_or_else(|_| prompt_text.bytes().map(|b| b as i32).take(4096).collect());
    let n_prompt_tokens = prompt_tokens.len();

    // ── Engine loop ──────────────────────────────────────────────────────────
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };

    // ── Benchmark runs ───────────────────────────────────────────────────────
    let runs = args.runs.max(1);
    let mut ttft_samples: Vec<f64> = Vec::new();
    let mut gen_tok_samples: Vec<usize> = Vec::new();
    let mut gen_sec_samples: Vec<f64> = Vec::new();

    for run_idx in 0..runs {
        if runs > 1 {
            eprint!("  Run {}/{}…\r", run_idx + 1, runs);
            let _ = std::io::stderr().flush();
        }

        let sampling = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: Some(42),
            stop: None,
            show_thinking: false,
            initial_in_thinking: false,
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let req_id = engine.next_request_id();
        let req = InferenceRequest::new(
            req_id,
            prompt_tokens.clone(),
            args.max_new_tokens,
            sampling,
            tx,
        );

        let submit_time = Instant::now();
        engine.submit_request(req);

        let mut first_token_elapsed: Option<f64> = None;
        let mut gen_tokens: usize = 0;
        let mut gen_start: Option<Instant> = None;

        while let Some(token) = rx.recv().await {
            if !token.text.is_empty() {
                if first_token_elapsed.is_none() {
                    first_token_elapsed = Some(submit_time.elapsed().as_secs_f64());
                    gen_start = Some(Instant::now());
                } else {
                    gen_tokens += 1;
                }
            }
            if token.stop_reason.is_some() {
                break;
            }
        }

        if let Some(ttft) = first_token_elapsed {
            ttft_samples.push(ttft);
        }
        gen_tok_samples.push(gen_tokens);
        if let Some(gs) = gen_start {
            gen_sec_samples.push(gs.elapsed().as_secs_f64());
        }
    }

    engine_loop.abort();

    // Clear run counter line
    if runs > 1 {
        eprint!("                      \r");
    }

    // ── Compute averages ─────────────────────────────────────────────────────
    let avg_ttft = if ttft_samples.is_empty() {
        0.0
    } else {
        ttft_samples.iter().sum::<f64>() / ttft_samples.len() as f64
    };
    let total_gen_toks: usize = gen_tok_samples.iter().sum();
    let total_gen_secs: f64 = gen_sec_samples.iter().sum();
    let avg_gen_toks = total_gen_toks as f64 / runs as f64;
    let avg_gen_secs = total_gen_secs / runs as f64;
    let gen_speed = if avg_gen_secs > 0.0 {
        avg_gen_toks / avg_gen_secs
    } else {
        0.0
    };

    // ── Print results ─────────────────────────────────────────────────────────
    eprintln!();
    theme::eprint_styled(None, false, false, "  🦊  ");
    theme::eprint_styled(
        Some(crossterm::style::Color::White),
        true,
        false,
        &model_name,
    );
    eprintln!();
    theme::eprint_styled(None, false, true, &format!("  {}\n\n", "─".repeat(44)));

    theme::eprint_kv_pair("Prompt", &format!("{} tokens", n_prompt_tokens));
    theme::eprint_kv_pair("Load", &format!("{:.2} s", load_elapsed.as_secs_f64()));
    theme::eprint_kv_pair("TTFT", &format!("{:.3} s", avg_ttft));
    theme::eprint_kv_pair(
        "Generation",
        &format!(
            "{:.1} tok/s  ({} tokens · {:.2}s)",
            gen_speed,
            total_gen_toks / runs,
            avg_gen_secs
        ),
    );
    if runs > 1 {
        theme::eprint_kv_pair("Runs", &format!("{}", runs));
    }
    eprintln!();

    Ok(())
}
