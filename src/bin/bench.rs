//! fox-bench — integrated benchmark tool for ferrumox.
//!
//! Launches N concurrent workers, each sending `--requests` chat completions
//! to the target server, and reports:
//!   - TTFT  (time to first token): P50 / P95
//!   - Total latency per request:   P50 / P95 / P99
//!   - Aggregate throughput:        tokens / second
//!   - System metrics:              CPU%, memory MB, GPU util (if available)
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

    /// Model name to use for the comparison server (defaults to --model).
    #[arg(long)]
    compare_model: Option<String>,

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

/// System resource sample taken at one point in time.
struct SystemSample {
    cpu_pct: f64,
    mem_mb: f64,
    gpu_util_pct: Option<f64>,
    gpu_mem_mb: Option<f64>,
}

/// Aggregated system resource statistics over a benchmark run.
#[derive(Debug, Serialize)]
struct SysStats {
    cpu_avg: f64,
    cpu_peak: f64,
    mem_mb_peak: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_util_avg: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_mem_mb_peak: Option<f64>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    sys_metrics: Option<SysStats>,
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
    usage: Option<SseUsage>,
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

#[derive(Debug, Deserialize)]
struct SseUsage {
    completion_tokens: usize,
    #[allow(dead_code)]
    prompt_tokens: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// System metrics sampling
// ──────────────────────────────────────────────────────────────────────────────

/// Read /proc/stat line 0 → (idle, total) jiffies.
#[cfg(target_os = "linux")]
fn read_proc_stat() -> Option<(u64, u64)> {
    let content = std::fs::read_to_string("/proc/stat").ok()?;
    let first = content.lines().next()?;
    let nums: Vec<u64> = first
        .split_whitespace()
        .skip(1)
        .filter_map(|s| s.parse().ok())
        .collect();
    if nums.len() < 4 {
        return None;
    }
    // Fields: user nice system idle iowait irq softirq ...
    let idle = nums.get(3).copied().unwrap_or(0) + nums.get(4).copied().unwrap_or(0);
    let total: u64 = nums.iter().sum();
    Some((idle, total))
}

/// Read /proc/meminfo → used memory in MB.
#[cfg(target_os = "linux")]
fn read_mem_mb() -> f64 {
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return 0.0;
    };
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;
    for line in content.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("MemAvailable:") {
            available_kb = line
                .split_whitespace()
                .nth(1)
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
        }
    }
    (total_kb.saturating_sub(available_kb)) as f64 / 1024.0
}

/// Query nvidia-smi for GPU utilization and memory used.
async fn query_nvidia_smi() -> Option<(f64, f64)> {
    let output = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.lines().next()?;
    let mut parts = line.split(',');
    let util: f64 = parts.next()?.trim().parse().ok()?;
    let mem: f64 = parts.next()?.trim().parse().ok()?;
    Some((util, mem))
}

async fn sample_system() -> SystemSample {
    // CPU: diff two /proc/stat reads 500ms apart (Linux only).
    #[cfg(target_os = "linux")]
    let cpu_pct = {
        if let (Some((idle1, total1)), _) = (read_proc_stat(), ()) {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if let Some((idle2, total2)) = read_proc_stat() {
                let d_total = total2.saturating_sub(total1) as f64;
                let d_idle = idle2.saturating_sub(idle1) as f64;
                if d_total > 0.0 {
                    (1.0 - d_idle / d_total) * 100.0
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            tokio::time::sleep(Duration::from_millis(500)).await;
            0.0
        }
    };
    #[cfg(not(target_os = "linux"))]
    let cpu_pct = 0.0_f64;

    #[cfg(target_os = "linux")]
    let mem_mb = read_mem_mb();
    #[cfg(not(target_os = "linux"))]
    let mem_mb = 0.0_f64;

    let (gpu_util_pct, gpu_mem_mb) = match query_nvidia_smi().await {
        Some((u, m)) => (Some(u), Some(m)),
        None => (None, None),
    };

    SystemSample {
        cpu_pct,
        mem_mb,
        gpu_util_pct,
        gpu_mem_mb,
    }
}

fn aggregate_samples(samples: Vec<SystemSample>) -> Option<SysStats> {
    if samples.is_empty() {
        return None;
    }
    let n = samples.len() as f64;
    let cpu_avg = samples.iter().map(|s| s.cpu_pct).sum::<f64>() / n;
    let cpu_peak = samples.iter().map(|s| s.cpu_pct).fold(0.0_f64, f64::max);
    let mem_mb_peak = samples.iter().map(|s| s.mem_mb).fold(0.0_f64, f64::max);

    let gpu_samples: Vec<f64> = samples.iter().filter_map(|s| s.gpu_util_pct).collect();
    let gpu_util_avg = if gpu_samples.is_empty() {
        None
    } else {
        Some(gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64)
    };

    let gpu_mem_peak: Vec<f64> = samples.iter().filter_map(|s| s.gpu_mem_mb).collect();
    let gpu_mem_mb_peak = if gpu_mem_peak.is_empty() {
        None
    } else {
        Some(gpu_mem_peak.iter().cloned().fold(0.0_f64, f64::max))
    };

    Some(SysStats {
        cpu_avg,
        cpu_peak,
        mem_mb_peak,
        gpu_util_avg,
        gpu_mem_mb_peak,
    })
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
        "stream": true,
        "stream_options": {"include_usage": true}
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
    let mut chunk_count: usize = 0;
    let mut actual_tokens: Option<usize> = None;
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
                // Capture real token count from usage field (final chunk).
                if let Some(ref u) = chunk.usage {
                    actual_tokens = Some(u.completion_tokens);
                }
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
                        chunk_count += 1;
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
    let tokens_generated = actual_tokens.unwrap_or(chunk_count);

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

#[allow(clippy::too_many_arguments)]
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

    // Spawn background system-metrics poller.
    let (sys_tx, mut sys_rx) = tokio::sync::mpsc::unbounded_channel::<SystemSample>();
    let sys_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let sample = sample_system().await;
            if sys_tx.send(sample).is_err() {
                break;
            }
        }
    });

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

    // Stop the metrics poller and collect samples.
    sys_task.abort();
    let _ = sys_task.await;
    let mut samples = Vec::new();
    while let Ok(s) = sys_rx.try_recv() {
        samples.push(s);
    }
    let sys_metrics = aggregate_samples(samples);

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
        sys_metrics,
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
        ttft_p50_pct: improvement_pct(primary.ttft_p50_ms as f64, compare.ttft_p50_ms as f64, true),
        ttft_p95_pct: improvement_pct(primary.ttft_p95_ms as f64, compare.ttft_p95_ms as f64, true),
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

fn print_sys_stats(sys: &SysStats) {
    println!("─────────────────────────────────────────");
    println!("  System metrics (during run)");
    println!(
        "  CPU         avg: {:>5.1}%   peak: {:>5.1}%",
        sys.cpu_avg, sys.cpu_peak
    );
    println!("  Memory peak : {:.0} MB", sys.mem_mb_peak);
    if let (Some(util), Some(mem)) = (sys.gpu_util_avg, sys.gpu_mem_mb_peak) {
        println!("  GPU util avg: {:.1}%   GPU mem peak: {:.0} MB", util, mem);
    }
}

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

    if let Some(ref sys) = stats.sys_metrics {
        print_sys_stats(sys);
    }
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

    // Show system metrics for primary server.
    if let Some(ref sys) = primary.sys_metrics {
        println!();
        println!("  System ({}):", primary.label);
        println!(
            "    CPU avg {:.1}%  peak {:.1}%   Mem peak {:.0} MB",
            sys.cpu_avg, sys.cpu_peak, sys.mem_mb_peak
        );
        if let (Some(util), Some(mem)) = (sys.gpu_util_avg, sys.gpu_mem_mb_peak) {
            println!("    GPU util avg {:.1}%   GPU mem peak {:.0} MB", util, mem);
        }
    }
    if let Some(ref sys) = compare.sys_metrics {
        println!();
        println!("  System ({}):", compare.label);
        println!(
            "    CPU avg {:.1}%  peak {:.1}%   Mem peak {:.0} MB",
            sys.cpu_avg, sys.cpu_peak, sys.mem_mb_peak
        );
        if let (Some(util), Some(mem)) = (sys.gpu_util_avg, sys.gpu_mem_mb_peak) {
            println!("    GPU util avg {:.1}%   GPU mem peak {:.0} MB", util, mem);
        }
    }
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
        let cmp_model = args.compare_model.as_deref().unwrap_or(&args.model);
        let cmp = run_workload(
            &args.compare_label,
            cu,
            cmp_model,
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
