// `fox ps` — show running model servers.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use clap::Parser;
use serde::Deserialize;

use super::theme;

#[derive(Parser, Debug)]
pub struct PsArgs {
    /// Port the server is running on
    #[arg(long, default_value = "8080")]
    pub port: u16,
}

#[derive(Deserialize)]
struct PsHealthResponse {
    status: String,
    kv_cache_usage: f32,
    queue_depth: usize,
    model_name: String,
    started_at: u64,
}

pub async fn run_ps(args: PsArgs) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let url = format!("http://127.0.0.1:{}/health", args.port);
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(_) => {
            println!("No models running.");
            return Ok(());
        }
    };

    if !resp.status().is_success() {
        println!("No models running.");
        return Ok(());
    }

    let health: PsHealthResponse = match resp.json().await {
        Ok(h) => h,
        Err(_) => {
            println!("No models running.");
            return Ok(());
        }
    };

    let uptime = format_uptime(health.started_at);
    let name_col = health.model_name.len().max(4) + 2;

    theme::print_table_header(&[
        ("NAME", name_col),
        ("STATUS", 10),
        ("PORT", 8),
        ("KV CACHE", 12),
        ("QUEUE", 9),
        ("UPTIME", 10),
    ]);
    theme::print_separator(name_col + 49);

    print!("{:<width$} ", health.model_name, width = name_col);
    theme::print_status(&health.status, 10);
    print!("{:<8} ", args.port);
    theme::print_kv_cache(health.kv_cache_usage, 12);
    print!("{:<9} ", health.queue_depth);
    println!("{}", uptime);

    Ok(())
}

fn format_uptime(started_at: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let elapsed = now.saturating_sub(started_at);
    let hours = elapsed / 3600;
    let mins = (elapsed % 3600) / 60;
    if hours > 0 {
        format!("{}h {}m", hours, mins)
    } else if mins > 0 {
        format!("{}m", mins)
    } else {
        format!("{}s", elapsed)
    }
}
