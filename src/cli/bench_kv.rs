// `fox bench-kv` — compare KV cache quantization types side-by-side.
//
// Loads model weights once, then creates a fresh llama.cpp context for each
// KV type — no weight reload between runs, so GPU memory stays stable.
//
//   fox bench-kv phi
//   fox bench-kv phi --types f16,q8_0,turbo3,turbo4,turbo2
//   fox bench-kv phi --runs 3 --max-context-len 4096

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

const DEFAULT_PROMPT: &str = "Explain what a large language model is in two sentences.";
const DEFAULT_TYPES: &str = "f16,q8_0,turbo3,turbo4,turbo2";
const BLOCK_SIZE: usize = 16;
const GPU_FRACTION: f32 = 0.85;

#[derive(Parser, Debug)]
pub struct BenchKvArgs {
    /// Model name, alias, or path to a GGUF file.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// Comma-separated KV cache types to compare.
    /// Supported: f16, q8_0, q4_0, turbo3, turbo4, turbo2
    #[arg(long, default_value = DEFAULT_TYPES)]
    pub types: String,

    /// Prompt used for each inference run.
    #[arg(long, default_value = DEFAULT_PROMPT)]
    pub prompt: String,

    /// Maximum context length. Defaults to the model's trained context.
    #[arg(long)]
    pub max_context_len: Option<u32>,

    /// Path to aliases TOML file.
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Inference passes per KV type (results are averaged).
    #[arg(long, default_value = "2")]
    pub runs: usize,

    /// Maximum tokens to generate per pass.
    #[arg(long, default_value = "300")]
    pub max_new_tokens: usize,

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

struct RunResult {
    label: &'static str,
    kv_blocks: usize,
    max_context_tokens: usize,
    avg_ttft_secs: f64,
    avg_gen_toks_per_sec: f64,
}

fn parse_kv_type_label(s: &str) -> Option<(&'static str, u32)> {
    match s.trim() {
        "f16" => Some(("f16", kv_type::F16)),
        "q8_0" => Some(("q8_0", kv_type::Q8_0)),
        "q4_0" => Some(("q4_0", kv_type::Q4_0)),
        "turbo3" => Some(("turbo3", kv_type::TURBO3)),
        "turbo4" => Some(("turbo4", kv_type::TURBO4)),
        "turbo2" => Some(("turbo2", kv_type::TURBO2)),
        _ => None,
    }
}

async fn run_one_type(
    base_model: &LlamaCppModel,
    model_name: &str,
    label: &'static str,
    type_id: u32,
    args: &BenchKvArgs,
    gpu_memory_bytes: usize,
) -> Result<RunResult> {
    // Create a fresh context from the already-loaded weights.
    let model = base_model.new_context(
        1,
        args.max_context_len,
        gpu_memory_bytes,
        GPU_FRACTION,
        type_id,
        type_id,
    )?;

    let model_config = model.model_config();
    let kv_cache = Arc::new(KVCacheManager::new(
        &model_config,
        gpu_memory_bytes,
        GPU_FRACTION,
        BLOCK_SIZE,
        type_id,
        type_id,
    ));

    let kv_blocks = kv_cache.total_blocks();
    let max_context_tokens = kv_blocks * BLOCK_SIZE;

    let model: Arc<dyn Model> = Arc::new(model);
    let scheduler = Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));
    let engine = Arc::new(InferenceEngine::new(
        model,
        scheduler,
        kv_cache,
        model_name.to_string(),
        None,
    ));

    let messages = vec![
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

    let engine_loop = {
        let e = engine.clone();
        tokio::spawn(async move {
            let _ = e.run_loop().await;
        })
    };

    let runs = args.runs.max(1);
    let mut ttft_sum = 0.0f64;
    let mut gen_toks_sum = 0usize;
    let mut gen_secs_sum = 0.0f64;

    for _ in 0..runs {
        let sampling = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: Some(42),
            stop: None,
            show_thinking: false,
            initial_in_thinking: false,
            max_thinking_chars: 8192,
        };

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let req = InferenceRequest::new(
            engine.next_request_id(),
            prompt_tokens.clone(),
            args.max_new_tokens,
            sampling,
            tx,
        );

        let t0 = Instant::now();
        engine.submit_request(req);

        let mut ttft: Option<f64> = None;
        let mut gen_start: Option<Instant> = None;
        let mut gen_tokens: usize = 0;
        let mut got_any_token = false;

        while let Some(tok) = rx.recv().await {
            // Count every token (including thinking/empty-text tokens) so that
            // models with a reasoning phase (Qwen3, DeepSeek-R1) report correct
            // throughput rather than showing "—".
            if ttft.is_none() {
                ttft = Some(t0.elapsed().as_secs_f64());
                gen_start = Some(Instant::now());
            } else {
                gen_tokens += 1;
            }
            got_any_token = true;
            if let Some(ref reason) = tok.stop_reason {
                let _ = reason; // stop_reason logged implicitly via break
                break;
            }
        }

        if !got_any_token {
            eprintln!("  [{label}] engine channel closed without sending any token (engine crash or OOM?)");
        }

        ttft_sum += ttft.unwrap_or(0.0);
        gen_toks_sum += gen_tokens;
        if let Some(gs) = gen_start {
            gen_secs_sum += gs.elapsed().as_secs_f64();
        }
    }

    engine_loop.abort();

    let avg_ttft = ttft_sum / runs as f64;
    let avg_gen_toks = gen_toks_sum as f64 / runs as f64;
    let avg_gen_secs = gen_secs_sum / runs as f64;
    let toks_per_sec = if avg_gen_secs > 0.0 {
        avg_gen_toks / avg_gen_secs
    } else {
        0.0
    };

    Ok(RunResult {
        label,
        kv_blocks,
        max_context_tokens,
        avg_ttft_secs: avg_ttft,
        avg_gen_toks_per_sec: toks_per_sec,
    })
}

pub async fn run_bench_kv(args: BenchKvArgs) -> Result<()> {
    let (model_name, model_path) = resolve_model_path(&args.model, args.alias_file.as_deref())?;

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

    let types: Vec<(&'static str, u32)> = args
        .types
        .split(',')
        .filter_map(parse_kv_type_label)
        .collect();

    if types.is_empty() {
        anyhow::bail!("No valid KV types specified. Use: f16, q8_0, q4_0, turbo3, turbo4, turbo2");
    }

    // ── Load model weights once with a minimal context ───────────────────────
    // Using a tiny context (32 tokens) here so the base instance holds only the
    // weights in VRAM. The real contexts (one per KV type) are created via
    // new_context() with the full context length and the requested KV type.
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
        Some(32), // minimal context — only holds weights, frees VRAM for benchmark contexts
        gpu_memory_bytes,
        GPU_FRACTION,
        kv_type::F16,
        kv_type::F16,
        args.main_gpu,
        split_mode,
        &tensor_split,
        args.moe_cpu,
        None,
    )?;

    spinner.finish_and_clear();

    // ── Run each type ────────────────────────────────────────────────────────
    let mut results: Vec<RunResult> = Vec::new();
    for (label, type_id) in &types {
        eprint!("  Benchmarking [{label}]…\r");
        match run_one_type(
            &base_model,
            &model_name,
            label,
            *type_id,
            &args,
            gpu_memory_bytes,
        )
        .await
        {
            Ok(r) => results.push(r),
            Err(e) => eprintln!("  ✗  [{label}] failed: {e}"),
        }
        eprint!("                          \r");
    }

    if results.is_empty() {
        anyhow::bail!("All KV type runs failed.");
    }

    let baseline_blocks = results
        .iter()
        .find(|r| r.label == "f16")
        .map(|r| r.kv_blocks as f64)
        .unwrap_or(results[0].kv_blocks as f64);

    // ── Print table ──────────────────────────────────────────────────────────
    eprintln!();
    theme::eprint_styled(None, false, false, "  🦊  ");
    theme::eprint_styled(
        Some(crossterm::style::Color::White),
        true,
        false,
        &model_name,
    );
    theme::eprint_styled(None, false, false, "  ·  KV cache benchmark");
    eprintln!();
    theme::eprint_styled(None, false, true, &format!("  {}\n\n", "─".repeat(62)));

    eprintln!(
        "  {:<8}  {:>8}  {:>10}  {:>9}  {:>7}  {:>7}",
        "type", "blocks", "ctx-tokens", "tok/s", "TTFT", "vs f16"
    );
    theme::eprint_styled(None, false, true, &format!("  {}\n", "─".repeat(62)));

    for r in &results {
        let ratio = r.kv_blocks as f64 / baseline_blocks;
        let ratio_str = if (ratio - 1.0).abs() < 0.05 {
            "baseline".to_string()
        } else {
            format!("{:.1}×", ratio)
        };

        let ttft_str = if r.avg_ttft_secs > 0.0 {
            format!("{:.3}s", r.avg_ttft_secs)
        } else {
            "—".to_string()
        };

        let tps_str = if r.avg_gen_toks_per_sec > 0.0 {
            format!("{:.1}", r.avg_gen_toks_per_sec)
        } else {
            "—".to_string()
        };

        let is_turbo = r.label.starts_with("turbo");
        let label_display = if is_turbo {
            format!("✦ {}", r.label)
        } else {
            format!("  {}", r.label)
        };

        eprintln!(
            "  {:<8}  {:>8}  {:>10}  {:>9}  {:>7}  {:>7}",
            label_display, r.kv_blocks, r.max_context_tokens, tps_str, ttft_str, ratio_str,
        );
    }

    eprintln!();
    eprintln!("  blocks = available KV cache slots  ·  ctx-tokens = blocks × {BLOCK_SIZE}");
    eprintln!("  ✦ = TurboQuant (requires flash attention + head_dim % 128 == 0)");
    eprintln!();

    Ok(())
}
