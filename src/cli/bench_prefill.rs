// `fox bench-prefill` — prove chunked prefill keeps a short request responsive
// while a long prompt is being prefilled.
//
// The validation for 0.13's chunked prefill (design `serving-robustness.md` §1, S3):
// submit a LONG prompt (whose prefill would otherwise head-of-line-block the engine)
// and, concurrently, a SHORT request that generates tokens. We measure the short
// request's *worst stall* — the largest gap between consecutive tokens, including the
// time-to-first-token. With chunking on, that stall is bounded by one prefill chunk;
// with chunking off (single-shot), it balloons to the full long-prompt prefill.
//
//   fox bench-prefill phi
//   fox bench-prefill phi --long-prompt-tokens 4096 --chunks 512,0
//   fox bench-prefill phi --short-new-tokens 48 --max-context-len 8192

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::{get_gpu_memory_bytes, get_total_gpu_memory_bytes, resolve_model_path, theme};
use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::InferenceEngine;
use crate::kv_cache::KVCacheManager;
use crate::model_registry::kv_type;
use crate::scheduler::{InferenceRequest, SamplingParams};

const BLOCK_SIZE: usize = 16;
const GPU_FRACTION: f32 = 0.85;
// Enough concurrent sequences for the long + short pair with headroom.
const MAX_BATCH: usize = 4;
// The long request only needs to finish its prefill to exercise the stall; a few
// decode tokens are plenty.
const LONG_NEW_TOKENS: usize = 4;
// Repeated to synthesize a long prompt of an exact token length with valid token ids.
const FILLER: &str = "The quick brown fox jumps over the lazy dog. ";
const SHORT_PROMPT: &str = "In one word, what colour is the sky?";

#[derive(Parser, Debug)]
pub struct BenchPrefillArgs {
    /// Model name, alias, or path to a GGUF file.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// Length (in tokens) of the long prompt that competes for the engine.
    #[arg(long, default_value = "2048")]
    pub long_prompt_tokens: usize,

    /// Tokens the short request generates while the long prompt prefills.
    #[arg(long, default_value = "24")]
    pub short_new_tokens: usize,

    /// Comma-separated `--max-prefill-chunk` values to compare. `0` = single-shot
    /// (chunking off). Default compares chunked (512) against single-shot.
    #[arg(long, default_value = "512,0")]
    pub chunks: String,

    /// Maximum context length. Defaults to the model's trained context.
    #[arg(long)]
    pub max_context_len: Option<u32>,

    /// Path to aliases TOML file.
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Primary GPU index (0-based).
    #[arg(long, default_value = "0", env = "FOX_MAIN_GPU")]
    pub main_gpu: i32,

    /// How to split across GPUs: none, layer (default), row.
    #[arg(long, default_value = "layer", env = "FOX_SPLIT_MODE")]
    pub split_mode: String,

    /// Comma-separated VRAM proportions for tensor splitting.
    #[arg(long, env = "FOX_TENSOR_SPLIT")]
    pub tensor_split: Option<String>,

    /// Offload MoE expert tensors to CPU RAM.
    #[arg(long, env = "FOX_MOE_CPU")]
    pub moe_cpu: bool,
}

struct ScenarioResult {
    chunk: usize,
    short_ttft_secs: f64,
    short_worst_stall_secs: f64,
    short_gen_toks_per_sec: f64,
    long_prefill_secs: f64,
}

fn sampling() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        seed: Some(42),
        stop: None,
        show_thinking: false,
        initial_in_thinking: false,
        max_thinking_chars: 8192,
        grammar: None,
        logprobs: None,
        min_p: 0.0,
        min_tokens: 0,
        logit_bias: None,
    }
}

/// Build a prompt of exactly `target` tokens by repeating a filler sentence.
fn synth_long_prompt(engine: &InferenceEngine, target: usize) -> Vec<i32> {
    let base = engine
        .tokenize(FILLER)
        .unwrap_or_else(|_| FILLER.bytes().map(|b| b as i32).collect());
    let base = if base.is_empty() { vec![1] } else { base };
    let mut tokens = Vec::with_capacity(target);
    while tokens.len() < target {
        tokens.extend_from_slice(&base);
    }
    tokens.truncate(target);
    tokens
}

async fn run_scenario(
    base_model: &LlamaCppModel,
    model_name: &str,
    chunk: usize,
    args: &BenchPrefillArgs,
    gpu_memory_bytes: usize,
) -> Result<ScenarioResult> {
    let model = base_model.new_context(
        MAX_BATCH,
        args.max_context_len,
        gpu_memory_bytes,
        GPU_FRACTION,
        kv_type::F16,
        kv_type::F16,
    )?;
    let model_config = model.model_config();
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        GPU_FRACTION,
        BLOCK_SIZE,
        kv_type::F16,
        kv_type::F16,
    ));
    let model: Arc<dyn Model> = Arc::new(model);
    let scheduler = Arc::new(crate::scheduler::Scheduler::new(
        kv_cache.clone(),
        MAX_BATCH,
    ));
    let engine = Arc::new(InferenceEngine::new(
        model,
        scheduler,
        kv_cache,
        model_name.to_string(),
        None,
        chunk, // the knob under test
        None,  // no context rolling (benchmark)
    ));

    let long_tokens = synth_long_prompt(&engine, args.long_prompt_tokens);
    let short_prompt_text = engine
        .apply_chat_template(&[
            (
                "system".to_string(),
                "You are a helpful assistant.".to_string(),
            ),
            ("user".to_string(), SHORT_PROMPT.to_string()),
        ])
        .unwrap_or_else(|_| format!("user: {SHORT_PROMPT}"));
    let short_tokens = engine
        .tokenize(&short_prompt_text)
        .unwrap_or_else(|_| short_prompt_text.bytes().map(|b| b as i32).collect());

    let engine_loop = {
        let e = engine.clone();
        tokio::spawn(async move {
            let _ = e.run_loop().await;
        })
    };

    // Submit the long prompt and the short request back-to-back so both are queued
    // before the first scheduler step — the short request thus competes with the
    // long prefill rather than following it.
    let (long_tx, mut long_rx) = tokio::sync::mpsc::unbounded_channel();
    let long_req = InferenceRequest::new(
        engine.next_request_id(),
        long_tokens,
        LONG_NEW_TOKENS,
        sampling(),
        long_tx,
    );
    let (short_tx, mut short_rx) = tokio::sync::mpsc::unbounded_channel();
    let short_req = InferenceRequest::new(
        engine.next_request_id(),
        short_tokens,
        args.short_new_tokens,
        sampling(),
        short_tx,
    );

    let t0 = Instant::now();
    engine.submit_request(long_req);
    engine.submit_request(short_req);

    // Drain the long request in the background; its first token marks the moment its
    // (chunked or single-shot) prefill finished — a useful reference number.
    let long_task = tokio::spawn(async move {
        let mut ttft: Option<Duration> = None;
        while let Some(tok) = long_rx.recv().await {
            if ttft.is_none() {
                ttft = Some(t0.elapsed());
            }
            if tok.stop_reason.is_some() {
                break;
            }
        }
        ttft
    });

    // Record the arrival time of every short-request token.
    let mut stamps: Vec<Instant> = Vec::new();
    while let Some(tok) = short_rx.recv().await {
        stamps.push(Instant::now());
        if tok.stop_reason.is_some() {
            break;
        }
    }

    let long_prefill_secs = long_task
        .await
        .ok()
        .flatten()
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    engine_loop.abort();

    // ── Short-request latency stats ──────────────────────────────────────────
    // worst_stall = the largest gap the short request ever waited, including the
    // time-to-first-token (its first gap, measured from submit). This is the number
    // chunked prefill is meant to bound.
    let short_ttft = stamps
        .first()
        .map(|t| t.duration_since(t0).as_secs_f64())
        .unwrap_or(0.0);
    let mut worst_stall = short_ttft;
    for pair in stamps.windows(2) {
        let gap = pair[1].duration_since(pair[0]).as_secs_f64();
        if gap > worst_stall {
            worst_stall = gap;
        }
    }
    // Sustained generation speed once the first token has landed.
    let gen_toks_per_sec = if stamps.len() >= 2 {
        let span = stamps
            .last()
            .unwrap()
            .duration_since(stamps[0])
            .as_secs_f64();
        if span > 0.0 {
            (stamps.len() - 1) as f64 / span
        } else {
            0.0
        }
    } else {
        0.0
    };

    Ok(ScenarioResult {
        chunk,
        short_ttft_secs: short_ttft,
        short_worst_stall_secs: worst_stall,
        short_gen_toks_per_sec: gen_toks_per_sec,
        long_prefill_secs,
    })
}

pub async fn run_bench_prefill(args: BenchPrefillArgs) -> Result<()> {
    let (model_name, model_path) = resolve_model_path(&args.model, args.alias_file.as_deref())?;

    let chunks: Vec<usize> = args
        .chunks
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if chunks.is_empty() {
        anyhow::bail!("No valid --chunks values. Example: --chunks 512,0");
    }

    let split_mode = match args.split_mode.as_str() {
        "row" => 2u32,
        "none" => 0u32,
        _ => 1u32,
    };
    let tensor_split: Vec<f32> = args
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

    let gpu_memory_bytes = if split_mode != 0 {
        get_total_gpu_memory_bytes()
    } else {
        get_gpu_memory_bytes()
    };

    // ── Load weights once ────────────────────────────────────────────────────
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .expect("valid template")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message(format!("Loading  {}…", model_name));
    spinner.enable_steady_tick(Duration::from_millis(80));

    let base_model = LlamaCppModel::load(
        &model_path,
        1,
        Some(32), // minimal context — only holds weights; scenarios build their own
        gpu_memory_bytes,
        GPU_FRACTION,
        kv_type::F16,
        kv_type::F16,
        args.main_gpu,
        split_mode,
        &tensor_split,
        args.moe_cpu,
    )?;

    spinner.finish_and_clear();

    // ── Run each chunk setting ───────────────────────────────────────────────
    let mut results: Vec<ScenarioResult> = Vec::new();
    for &chunk in &chunks {
        let label = if chunk == 0 {
            "off".to_string()
        } else {
            chunk.to_string()
        };
        eprint!("  Benchmarking chunk={label}…\r");
        match run_scenario(&base_model, &model_name, chunk, &args, gpu_memory_bytes).await {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("  ✗  chunk={label} failed: {e}"),
        }
        eprint!("                                \r");
    }

    if results.is_empty() {
        anyhow::bail!("All scenarios failed.");
    }

    // ── Print table ──────────────────────────────────────────────────────────
    eprintln!();
    theme::eprint_styled(None, false, false, "  🦊  ");
    theme::eprint_styled(
        Some(crossterm::style::Color::White),
        true,
        false,
        &model_name,
    );
    theme::eprint_styled(None, false, false, "  ·  chunked-prefill validation");
    eprintln!();
    theme::eprint_styled(None, false, true, &format!("  {}\n\n", "─".repeat(62)));

    theme::eprint_kv_pair(
        "Long prompt",
        &format!("{} tokens", args.long_prompt_tokens),
    );
    theme::eprint_kv_pair(
        "Short request",
        &format!("{} generated tokens", args.short_new_tokens),
    );
    eprintln!();

    eprintln!(
        "  {:<8}  {:>11}  {:>13}  {:>11}  {:>12}",
        "chunk", "short-TTFT", "worst-stall", "short-tok/s", "long-prefill"
    );
    theme::eprint_styled(None, false, true, &format!("  {}\n", "─".repeat(62)));

    for r in &results {
        let label = if r.chunk == 0 {
            "off".to_string()
        } else {
            r.chunk.to_string()
        };
        eprintln!(
            "  {:<8}  {:>10.3}s  {:>12.3}s  {:>11.1}  {:>11.3}s",
            format!("  {label}"),
            r.short_ttft_secs,
            r.short_worst_stall_secs,
            r.short_gen_toks_per_sec,
            r.long_prefill_secs,
        );
    }
    eprintln!();

    // ── Headline: how much chunking cut the worst stall ──────────────────────
    let chunked = results.iter().find(|r| r.chunk != 0);
    let single = results.iter().find(|r| r.chunk == 0);
    if let (Some(c), Some(s)) = (chunked, single) {
        if c.short_worst_stall_secs > 0.0 {
            let factor = s.short_worst_stall_secs / c.short_worst_stall_secs;
            eprintln!(
                "  Chunked prefill (chunk={}) cut the short request's worst stall\n  from {:.3}s (single-shot) to {:.3}s — {:.1}× more responsive.",
                c.chunk, s.short_worst_stall_secs, c.short_worst_stall_secs, factor
            );
            eprintln!();
        }
    }

    eprintln!("  worst-stall = largest gap the short request waited (incl. time-to-first-token)");
    eprintln!();

    Ok(())
}
