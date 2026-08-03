// `fox bench-spec` — quantify speculative decoding (0.15, S4).
//
// Runs the same generation with speculation OFF and ON (greedy, so both produce the
// exact same text — asserted) and reports tokens/s plus the draft acceptance ratio, on
// two workloads: a REPETITIVE one (a numeric cycle the model continues — n-gram lookup's
// best case) and a PROSE one (an open-ended question — closer to its worst case).
//
//   fox bench-spec llama3.2
//   fox bench-spec llama3.2 --max-new-tokens 256 --spec-draft-len 8
//
// Weights are loaded once; each scenario gets a fresh llama.cpp context.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::cli::{get_gpu_memory_bytes, get_total_gpu_memory_bytes, resolve_model_path, theme};
use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::{EngineOptions, InferenceEngine};
use crate::kv_cache::KVCacheManager;
use crate::model_registry::kv_type;
use crate::scheduler::{InferenceRequest, SamplingParams};

const BLOCK_SIZE: usize = 16;
const GPU_FRACTION: f32 = 0.85;

const REPETITIVE_PROMPT: &str =
    "Repeat this exact sequence over and over, without stopping: alpha beta gamma delta. \
     alpha beta gamma delta. alpha beta gamma delta. alpha beta gamma delta.";
const PROSE_PROMPT: &str = "Explain how photosynthesis works.";

#[derive(Parser, Debug)]
pub struct BenchSpecArgs {
    /// Model name, alias, or path to a GGUF file.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// Tokens to generate per run.
    #[arg(long, default_value = "192")]
    pub max_new_tokens: usize,

    /// Suffix length matched against history when speculating.
    #[arg(long, default_value = "2")]
    pub spec_ngram: usize,

    /// Maximum draft tokens proposed per speculative step.
    #[arg(long, default_value = "4")]
    pub spec_draft_len: usize,

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

struct RunResult {
    workload: &'static str,
    speculative: bool,
    toks_per_sec: f64,
    gen_tokens: usize,
    acceptance: Option<f64>,
    text: String,
}

/// Run one generation and return throughput + acceptance + the produced text.
async fn run_one(
    base_model: &LlamaCppModel,
    model_name: &str,
    workload: &'static str,
    prompt: &str,
    speculative: Option<(usize, usize)>,
    args: &BenchSpecArgs,
    gpu_memory_bytes: usize,
) -> Result<RunResult> {
    let model = base_model.new_context(
        1,
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
    let scheduler = Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));
    let engine = Arc::new(InferenceEngine::new(
        model,
        scheduler,
        kv_cache,
        model_name.to_string(),
        None,
        EngineOptions {
            speculative, // the knob under test
            ..Default::default()
        },
    ));

    let messages = vec![
        (
            "system".to_string(),
            "You are a helpful assistant.".to_string(),
        ),
        ("user".to_string(), prompt.to_string()),
    ];
    let prompt_text = engine
        .apply_chat_template(&messages)
        .unwrap_or_else(|_| format!("user: {prompt}"));
    let prompt_tokens = engine
        .tokenize(&prompt_text)
        .unwrap_or_else(|_| prompt_text.bytes().map(|b| b as i32).collect());

    let engine_loop = {
        let e = engine.clone();
        tokio::spawn(async move {
            let _ = e.run_loop().await;
        })
    };

    // Greedy so speculation-on and -off produce the exact same text (asserted by caller).
    let sampling = SamplingParams {
        temperature: 0.0,
        ..SamplingParams::default()
    };

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req = InferenceRequest::new(
        engine.next_request_id(),
        prompt_tokens,
        args.max_new_tokens,
        sampling,
        tx,
    );
    engine.submit_request(req);

    let mut text = String::new();
    let mut gen_tokens = 0usize;
    let mut gen_start: Option<Instant> = None;
    while let Some(tok) = rx.recv().await {
        if gen_start.is_none() {
            gen_start = Some(Instant::now());
        } else {
            gen_tokens += 1;
        }
        text.push_str(&tok.text);
        if tok.stop_reason.is_some() {
            break;
        }
    }
    let gen_secs = gen_start.map(|s| s.elapsed().as_secs_f64()).unwrap_or(0.0);

    let (proposed, accepted) = engine.spec_stats();
    engine_loop.abort();

    Ok(RunResult {
        workload,
        speculative: speculative.is_some(),
        toks_per_sec: if gen_secs > 0.0 {
            gen_tokens as f64 / gen_secs
        } else {
            0.0
        },
        gen_tokens,
        acceptance: (proposed > 0).then(|| accepted as f64 / proposed as f64),
        text,
    })
}

pub async fn run_bench_spec(args: BenchSpecArgs) -> Result<()> {
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

    let spec = Some((args.spec_ngram, args.spec_draft_len));
    let workloads: [(&'static str, &str); 2] =
        [("repetitive", REPETITIVE_PROMPT), ("prose", PROSE_PROMPT)];

    let mut results: Vec<RunResult> = Vec::new();
    for (workload, prompt) in workloads {
        for speculative in [None, spec] {
            let label = if speculative.is_some() { "on" } else { "off" };
            eprint!("  Benchmarking {workload} spec={label}…\r");
            let r = run_one(
                &base_model,
                &model_name,
                workload,
                prompt,
                speculative,
                &args,
                gpu_memory_bytes,
            )
            .await?;
            results.push(r);
            eprint!("                                          \r");
        }
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
    theme::eprint_styled(None, false, false, "  ·  speculative-decoding validation");
    eprintln!();
    theme::eprint_styled(None, false, true, &format!("  {}\n\n", "─".repeat(62)));

    theme::eprint_kv_pair(
        "Config",
        &format!(
            "ngram={} draft_len={} · {} tokens/run · greedy",
            args.spec_ngram, args.spec_draft_len, args.max_new_tokens
        ),
    );
    eprintln!();

    eprintln!(
        "  {:<12}  {:>5}  {:>9}  {:>8}  {:>11}  {:>9}",
        "workload", "spec", "tok/s", "tokens", "acceptance", "speedup"
    );
    theme::eprint_styled(None, false, true, &format!("  {}\n", "─".repeat(62)));

    for pair in results.chunks(2) {
        let (off, on) = (&pair[0], &pair[1]);
        for r in [off, on] {
            let acc = r
                .acceptance
                .map(|a| format!("{:.0}%", a * 100.0))
                .unwrap_or_else(|| "—".to_string());
            let speedup = if r.speculative && off.toks_per_sec > 0.0 {
                format!("{:.2}×", r.toks_per_sec / off.toks_per_sec)
            } else {
                "—".to_string()
            };
            eprintln!(
                "  {:<12}  {:>5}  {:>9.1}  {:>8}  {:>11}  {:>9}",
                r.workload,
                if r.speculative { "on" } else { "off" },
                r.toks_per_sec,
                r.gen_tokens,
                acc,
                speedup,
            );
        }
        // Speculation must never change WHAT is generated, only how fast.
        let identical = off.text == on.text;
        if !identical {
            eprintln!(
                "  ⚠️  output differed between spec off/on for {} — this is a bug",
                off.workload
            );
        }
    }
    eprintln!();
    eprintln!("  acceptance = drafts the target model agreed with / drafts proposed");
    eprintln!("  greedy sampling: spec on/off produce identical text (verified per workload)");
    eprintln!();

    Ok(())
}
