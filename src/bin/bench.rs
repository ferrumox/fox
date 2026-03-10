//! fox-bench — integrated benchmark tool for ferrumox.
//!
//! Launches N concurrent workers, each sending `--requests` chat completions
//! to the target server, and reports:
//!   - TTFT  (time to first token): P50 / P95
//!   - Total latency per request:   P50 / P95 / P99
//!   - Aggregate throughput:        tokens / second
//!
//! Usage (single server):
//!   fox-bench --url http://localhost:8080 \
//!     --model my-model \
//!     --concurrency 8 \
//!     --requests 50 \
//!     --prompt "Write a paragraph about Rust" \
//!     --max-tokens 256
//!
//! Usage (compare vs Ollama):
//!   fox-bench --url http://localhost:8080 \
//!     --compare-url http://localhost:11434 \
//!     --model llama3.2 \
//!     --output json

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::Semaphore;

// ──────────────────────────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, ValueEnum)]
enum OutputFormat {
    Text,
    Json,
}

#[derive(Debug, Parser)]
#[command(name = "fox-bench", about = "Benchmark tool for ferrumox")]
struct Args {
    /// Base URL of the primary server (ferrumox).
    #[arg(long, default_value = "http://localhost:8080")]
    url: String,

    /// Base URL of a second server to compare against (e.g. Ollama).
    /// When set, the same workload is run against both URLs and results are
    /// displayed side-by-side with improvement percentages.
    #[arg(long)]
    compare_url: Option<String>,

    /// Label for the primary server (used in comparison output).
    #[arg(long, default_value = "ferrumox")]
    label: String,

    /// Label for the comparison server.
    #[arg(long, default_value = "ollama")]
    compare_label: String,

    /// Model name to send in each request.
    #[arg(long, default_value = "default")]
    model: String,

    /// Number of concurrent workers.
    #[arg(long, default_value = "4")]
    concurrency: usize,

    /// Total number of requests to send across all workers.
    #[arg(long, default_value = "50")]
    requests: usize,

    /// Prompt to use for every request.
    #[arg(
        long,
        default_value = "Write a short paragraph about the Rust programming language."
    )]
    prompt: String,

    /// Maximum tokens to generate per request.
    #[arg(long, default_value = "128")]
    max_tokens: u32,

    /// Output format: text (default) or json.
    #[arg(long, value_enum, default_value = "text")]
    output: OutputFormat,
}

// ──────────────────────────────────────────────────────────────────────────────
// Measurement types
// ──────────────────────────────────────────────────────────────────────────────

/// Per-request measurements collected during a run.
struct RequestResult {
    ttft: Duration,
    total: Duration,
    tokens_generated: usize,
}

/// Aggregated statistics for a single server run.
#[derive(Debug, Serialize)]
struct RunStats {
    label: String,
    url: String,
    requests_ok: usize,
    requests_err: usize,
    ttft_p50_ms: u128,
    ttft_p95_ms: u128,
    latency_p50_ms: u128,
    latency_p95_ms: u128,
    latency_p99_ms: u128,
    throughput_tokens_per_sec: f64,
    total_time_secs: f64,
    total_tokens: usize,
}

/// Full benchmark report (single or comparison).
#[derive(Debug, Serialize)]
struct BenchReport {
    primary: RunStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    comparison: Option<RunStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    improvement: Option<ImprovementStats>,
}

/// Percentage improvement of primary vs comparison server.
#[derive(Debug, Serialize)]
struct ImprovementStats {
    ttft_p50_pct: f64,
    ttft_p95_pct: f64,
    latency_p50_pct: f64,
    latency_p95_pct: f64,
    latency_p99_pct: f64,
    throughput_pct: f64,
}

// ──────────────────────────────────────────────────────────────────────────────
// SSE chunk (minimal deserialization)
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct SseChunk {
    choices: Vec<SseChoice>,
}

#[derive(Debug, Deserialize)]
struct SseChoice {
    delta: SseDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SseDelta {
    content: Option<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Single request worker
// ──────────────────────────────────────────────────────────────────────────────

async fn run_request(
    client: &Client,
    base_url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Result<RequestResult> {
    let url = format!("{}/v1/chat/completions", base_url);
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1.0,
        "stream": true
    });

    let start = Instant::now();

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("failed to connect to server")?;

    if !response.status().is_success() {
        bail!("server returned {}", response.status());
    }

    let mut stream = response.bytes_stream();
    let mut ttft: Option<Duration> = None;
    let mut tokens_generated: usize = 0;
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.context("stream error")?;
        buf.push_str(&String::from_utf8_lossy(&bytes));

        // Process all complete SSE lines in the buffer
        while let Some(newline) = buf.find('\n') {
            let line = buf[..newline].trim().to_string();
            buf.drain(..=newline);

            if !line.starts_with("data:") {
                continue;
            }
            let data = line.trim_start_matches("data:").trim();
            if data == "[DONE]" {
                break;
            }
            if let Ok(chunk) = serde_json::from_str::<SseChunk>(data) {
                for choice in &chunk.choices {
                    if choice
                        .delta
                        .content
                        .as_deref()
                        .is_some_and(|c| !c.is_empty())
                    {
                        if ttft.is_none() {
                            ttft = Some(start.elapsed());
                        }
                        tokens_generated += 1;
                    }
                    if choice.finish_reason.is_some() {
                        break;
                    }
                }
            }
        }
    }

    let total = start.elapsed();
    let ttft = ttft.unwrap_or(total);

    Ok(RequestResult {
        ttft,
        total,
        tokens_generated,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Percentile helper
// ──────────────────────────────────────────────────────────────────────────────

fn percentile(sorted: &[Duration], pct: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 * pct / 100.0).ceil() as usize).saturating_sub(1);
    sorted[idx.min(sorted.len() - 1)]
}

// ──────────────────────────────────────────────────────────────────────────────
// Run a full workload against one server URL
// ──────────────────────────────────────────────────────────────────────────────

async fn run_workload(
    label: &str,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    concurrency: usize,
    requests: usize,
    show_progress: bool,
) -> Result<RunStats> {
    if show_progress {
        println!("Running against {} ({}) ...", label, url);
    }

    let client = Arc::new(
        Client::builder()
            .timeout(Duration::from_secs(120))
            .build()?,
    );
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let mut handles = Vec::with_capacity(requests);

    let bench_start = Instant::now();

    for _ in 0..requests {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let url = url.to_string();
        let model = model.to_string();
        let prompt = prompt.to_string();

        let handle = tokio::spawn(async move {
            let result = run_request(&client, &url, &model, &prompt, max_tokens).await;
            drop(permit);
            result
        });
        handles.push(handle);
    }

    let mut results: Vec<RequestResult> = Vec::with_capacity(requests);
    let mut errors = 0usize;

    for handle in handles {
        match handle.await {
            Ok(Ok(r)) => results.push(r),
            Ok(Err(e)) => {
                eprintln!("  request error: {}", e);
                errors += 1;
            }
            Err(e) => {
                eprintln!("  task panic: {}", e);
                errors += 1;
            }
        }
    }

    let elapsed = bench_start.elapsed();

    if results.is_empty() {
        bail!("All {} requests failed against {}", errors, url);
    }

    let mut ttfts: Vec<Duration> = results.iter().map(|r| r.ttft).collect();
    let mut totals: Vec<Duration> = results.iter().map(|r| r.total).collect();
    ttfts.sort_unstable();
    totals.sort_unstable();

    let total_tokens: usize = results.iter().map(|r| r.tokens_generated).sum();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();

    Ok(RunStats {
        label: label.to_string(),
        url: url.to_string(),
        requests_ok: results.len(),
        requests_err: errors,
        ttft_p50_ms: percentile(&ttfts, 50.0).as_millis(),
        ttft_p95_ms: percentile(&ttfts, 95.0).as_millis(),
        latency_p50_ms: percentile(&totals, 50.0).as_millis(),
        latency_p95_ms: percentile(&totals, 95.0).as_millis(),
        latency_p99_ms: percentile(&totals, 99.0).as_millis(),
        throughput_tokens_per_sec: throughput,
        total_time_secs: elapsed.as_secs_f64(),
        total_tokens,
    })
}

// ──────────────────────────────────────────────────────────────────────────────
// Improvement calculation
// ──────────────────────────────────────────────────────────────────────────────

fn improvement_pct(primary: f64, compare: f64, lower_is_better: bool) -> f64 {
    if compare == 0.0 {
        return 0.0;
    }
    if lower_is_better {
        (compare - primary) / compare * 100.0
    } else {
        (primary - compare) / compare * 100.0
    }
}

fn compute_improvement(primary: &RunStats, compare: &RunStats) -> ImprovementStats {
    ImprovementStats {
        ttft_p50_pct: improvement_pct(
            primary.ttft_p50_ms as f64,
            compare.ttft_p50_ms as f64,
            true,
        ),
        ttft_p95_pct: improvement_pct(
            primary.ttft_p95_ms as f64,
            compare.ttft_p95_ms as f64,
            true,
        ),
        latency_p50_pct: improvement_pct(
            primary.latency_p50_ms as f64,
            compare.latency_p50_ms as f64,
            true,
        ),
        latency_p95_pct: improvement_pct(
            primary.latency_p95_ms as f64,
            compare.latency_p95_ms as f64,
            true,
        ),
        latency_p99_pct: improvement_pct(
            primary.latency_p99_ms as f64,
            compare.latency_p99_ms as f64,
            true,
        ),
        throughput_pct: improvement_pct(
            primary.throughput_tokens_per_sec,
            compare.throughput_tokens_per_sec,
            false,
        ),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Text output helpers
// ──────────────────────────────────────────────────────────────────────────────

fn print_single(stats: &RunStats) {
    println!(
        "Results ({} ok, {} errors)",
        stats.requests_ok, stats.requests_err
    );
    println!("─────────────────────────────────────────");
    println!(
        "  TTFT        P50: {:>6}ms   P95: {:>6}ms",
        stats.ttft_p50_ms, stats.ttft_p95_ms,
    );
    println!(
        "  Latency     P50: {:>6}ms   P95: {:>6}ms   P99: {:>6}ms",
        stats.latency_p50_ms, stats.latency_p95_ms, stats.latency_p99_ms,
    );
    println!(
        "  Throughput  : {:.1} tokens/sec",
        stats.throughput_tokens_per_sec
    );
    println!("  Total time  : {:.2}s", stats.total_time_secs);
    println!("  Tokens out  : {}", stats.total_tokens);
}

fn fmt_improvement(pct: f64) -> String {
    if pct > 0.0 {
        format!("+{:.0}%", pct)
    } else if pct < 0.0 {
        format!("{:.0}%", pct)
    } else {
        "=".to_string()
    }
}

fn print_comparison(primary: &RunStats, compare: &RunStats, imp: &ImprovementStats) {
    let w = 12usize;
    let col_a = &primary.label;
    let col_b = &compare.label;

    println!(
        "┌─────────────────┬{:─>w$}┬{:─>w$}┬──────────┐",
        "",
        "",
        w = w + 2
    );
    println!(
        "│ Metric          │ {:^w$} │ {:^w$} │ Δ        │",
        col_a,
        col_b,
        w = w
    );
    println!(
        "├─────────────────┼{:─>w$}┼{:─>w$}┼──────────┤",
        "",
        "",
        w = w + 2
    );

    let rows: Vec<(&str, String, String, String)> = vec![
        (
            "TTFT P50",
            format!("{}ms", primary.ttft_p50_ms),
            format!("{}ms", compare.ttft_p50_ms),
            fmt_improvement(imp.ttft_p50_pct),
        ),
        (
            "TTFT P95",
            format!("{}ms", primary.ttft_p95_ms),
            format!("{}ms", compare.ttft_p95_ms),
            fmt_improvement(imp.ttft_p95_pct),
        ),
        (
            "Latency P50",
            format!("{}ms", primary.latency_p50_ms),
            format!("{}ms", compare.latency_p50_ms),
            fmt_improvement(imp.latency_p50_pct),
        ),
        (
            "Latency P95",
            format!("{}ms", primary.latency_p95_ms),
            format!("{}ms", compare.latency_p95_ms),
            fmt_improvement(imp.latency_p95_pct),
        ),
        (
            "Latency P99",
            format!("{}ms", primary.latency_p99_ms),
            format!("{}ms", compare.latency_p99_ms),
            fmt_improvement(imp.latency_p99_pct),
        ),
        (
            "Throughput",
            format!("{:.1} t/s", primary.throughput_tokens_per_sec),
            format!("{:.1} t/s", compare.throughput_tokens_per_sec),
            fmt_improvement(imp.throughput_pct),
        ),
    ];

    for (metric, a, b, delta) in &rows {
        println!(
            "│ {:15} │ {:>w$} │ {:>w$} │ {:8} │",
            metric,
            a,
            b,
            delta,
            w = w
        );
    }

    println!(
        "└─────────────────┴{:─>w$}┴{:─>w$}┴──────────┘",
        "",
        "",
        w = w + 2
    );
    println!();
    println!(
        "  {} ok / {} err (ferrumox)   {} ok / {} err ({})",
        primary.requests_ok,
        primary.requests_err,
        compare.requests_ok,
        compare.requests_err,
        compare.label
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Main
// ──────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let show_progress = matches!(args.output, OutputFormat::Text);

    if show_progress {
        println!("fox-bench");
        println!("  URL         : {}", args.url);
        if let Some(ref cu) = args.compare_url {
            println!("  Compare URL : {}", cu);
        }
        println!("  Model       : {}", args.model);
        println!("  Concurrency : {}", args.concurrency);
        println!("  Requests    : {}", args.requests);
        println!("  Max tokens  : {}", args.max_tokens);
        println!(
            "  Prompt      : \"{}\"",
            &args.prompt[..args.prompt.len().min(60)]
        );
        println!();
    }

    let primary = run_workload(
        &args.label,
        &args.url,
        &args.model,
        &args.prompt,
        args.max_tokens,
        args.concurrency,
        args.requests,
        show_progress,
    )
    .await?;

    let (comparison, improvement) = if let Some(ref cu) = args.compare_url {
        let cmp = run_workload(
            &args.compare_label,
            cu,
            &args.model,
            &args.prompt,
            args.max_tokens,
            args.concurrency,
            args.requests,
            show_progress,
        )
        .await?;
        let imp = compute_improvement(&primary, &cmp);
        (Some(cmp), Some(imp))
    } else {
        (None, None)
    };

    let report = BenchReport {
        primary,
        comparison,
        improvement,
    };

    match args.output {
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        OutputFormat::Text => {
            if show_progress {
                println!();
            }
            if let (Some(ref cmp), Some(ref imp)) = (&report.comparison, &report.improvement) {
                print_comparison(&report.primary, cmp, imp);
            } else {
                print_single(&report.primary);
            }
        }
    }

    Ok(())
}
