//! fox-bench — integrated benchmark tool for ferrumox.
//!
//! Launches N concurrent workers, each sending `--requests` chat completions
//! to the target server, and reports:
//!   - TTFT  (time to first token): P50 / P95
//!   - Total latency per request:   P50 / P95 / P99
//!   - Aggregate throughput:        tokens / second
//!
//! Usage:
//!   fox-bench --url http://localhost:8080 \
//!     --model my-model \
//!     --concurrency 8 \
//!     --requests 50 \
//!     --prompt "Escribe un párrafo sobre Rust" \
//!     --max-tokens 256

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use clap::Parser;
use futures::StreamExt;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Semaphore;

// ──────────────────────────────────────────────────────────────────────────────
// CLI
// ──────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(name = "fox-bench", about = "Benchmark tool for ferrumox")]
struct Args {
    /// Base URL of the ferrumox server.
    #[arg(long, default_value = "http://localhost:8080")]
    url: String,

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
// Main
// ──────────────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    println!("fox-bench");
    println!("  URL         : {}", args.url);
    println!("  Model       : {}", args.model);
    println!("  Concurrency : {}", args.concurrency);
    println!("  Requests    : {}", args.requests);
    println!("  Max tokens  : {}", args.max_tokens);
    println!(
        "  Prompt      : \"{}\"",
        &args.prompt[..args.prompt.len().min(60)]
    );
    println!();

    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()?;
    let client = Arc::new(client);

    let semaphore = Arc::new(Semaphore::new(args.concurrency));
    let mut handles = Vec::with_capacity(args.requests);

    let bench_start = Instant::now();

    for _ in 0..args.requests {
        let permit = semaphore.clone().acquire_owned().await?;
        let client = client.clone();
        let url = args.url.clone();
        let model = args.model.clone();
        let prompt = args.prompt.clone();
        let max_tokens = args.max_tokens;

        let handle = tokio::spawn(async move {
            let result = run_request(&client, &url, &model, &prompt, max_tokens).await;
            drop(permit);
            result
        });
        handles.push(handle);
    }

    let mut results: Vec<RequestResult> = Vec::with_capacity(args.requests);
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
        eprintln!("All {} requests failed.", errors);
        std::process::exit(1);
    }

    // Sort for percentile computation
    let mut ttfts: Vec<Duration> = results.iter().map(|r| r.ttft).collect();
    let mut totals: Vec<Duration> = results.iter().map(|r| r.total).collect();
    ttfts.sort_unstable();
    totals.sort_unstable();

    let total_tokens: usize = results.iter().map(|r| r.tokens_generated).sum();
    let throughput = total_tokens as f64 / elapsed.as_secs_f64();

    println!("Results ({} ok, {} errors)", results.len(), errors);
    println!("─────────────────────────────────────────");
    println!(
        "  TTFT        P50: {:>6}ms   P95: {:>6}ms",
        percentile(&ttfts, 50.0).as_millis(),
        percentile(&ttfts, 95.0).as_millis(),
    );
    println!(
        "  Latency     P50: {:>6}ms   P95: {:>6}ms   P99: {:>6}ms",
        percentile(&totals, 50.0).as_millis(),
        percentile(&totals, 95.0).as_millis(),
        percentile(&totals, 99.0).as_millis(),
    );
    println!("  Throughput  : {:.1} tokens/sec", throughput);
    println!("  Total time  : {:.2}s", elapsed.as_secs_f64());
    println!("  Tokens out  : {}", total_tokens);

    Ok(())
}
