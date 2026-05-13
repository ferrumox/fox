// `fox probe` — print a one-shot environment report so the operator can
// verify GPU drivers, model directory, and configuration before serving.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::system_probe::{GpuProbe, GpuVendor};

use super::{format_size, get_ram_info, list_models, models_dir, theme};

#[derive(Parser, Debug)]
pub struct ProbeArgs {
    /// Models directory to inventory (defaults to ~/.cache/ferrumox/models).
    #[arg(long)]
    pub path: Option<PathBuf>,

    /// Emit the report as JSON instead of human-readable text. Useful for
    /// scripting (e.g. `fox probe --json | jq '.gpu.devices[0].vendor'`).
    #[arg(long)]
    pub json: bool,
}

pub async fn run_probe(args: ProbeArgs) -> Result<()> {
    let dir = args.path.unwrap_or_else(models_dir);
    let gpu = GpuProbe::detect();
    let ram = get_ram_info();
    let model_count = list_models(&dir).map(|v| v.len()).unwrap_or(0);

    if args.json {
        let report = serde_json::json!({
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "ram_total_bytes": ram.total_bytes,
            "ram_used_bytes": ram.used_bytes,
            "models_dir": dir.display().to_string(),
            "model_count": model_count,
            "gpu": gpu,
        });
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    theme::print_kv_pair("OS", std::env::consts::OS);
    theme::print_kv_pair("Arch", std::env::consts::ARCH);
    if ram.total_bytes > 0 {
        theme::print_kv_pair("RAM total", &format_size(ram.total_bytes as u64));
    }
    theme::print_kv_pair("Models dir", &dir.display().to_string());
    theme::print_kv_pair("Model count", &model_count.to_string());

    println!();
    if gpu.devices.is_empty() {
        theme::print_kv_pair("GPUs", "none detected (CPU-only)");
    } else {
        theme::print_kv_pair("GPUs", &format!("{} detected", gpu.devices.len()));
        for (i, device) in gpu.devices.iter().enumerate() {
            println!();
            theme::print_kv_pair(&format!("  [{i}] vendor"), vendor_label(device.vendor));
            theme::print_kv_pair(&format!("  [{i}] name"), &device.name);
            if let Some(total) = device.vram_total_mib {
                theme::print_kv_pair(&format!("  [{i}] vram total"), &format!("{} MiB", total));
            }
            if let Some(free) = device.vram_free_mib {
                theme::print_kv_pair(&format!("  [{i}] vram free"), &format!("{} MiB", free));
            }
            if let Some(driver) = &device.driver {
                theme::print_kv_pair(&format!("  [{i}] driver"), driver);
            }
        }
    }

    Ok(())
}

fn vendor_label(v: GpuVendor) -> &'static str {
    match v {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Amd => "AMD",
        GpuVendor::Intel => "Intel",
        GpuVendor::Apple => "Apple",
        GpuVendor::Unknown => "unknown",
    }
}
