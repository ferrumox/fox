//! Detect GPUs on the host without panicking, blocking forever, or assuming
//! a particular vendor.
//!
//! Each vendor probe shells out to its own CLI tool, parses the structured
//! output (CSV for NVIDIA, JSON for AMD's `rocm-smi`, plist/JSON for
//! `system_profiler` on macOS) and bails on any failure. The shape of the
//! output is the same regardless of vendor, so callers can iterate
//! `probe.devices` without per-platform branching.
//!
//! Probes are short — bounded by a 2 s timeout per vendor tool, so calling
//! [`GpuProbe::detect`] on startup or on `/metrics` adds at most a few
//! seconds in the worst case (every tool installed but slow to spawn) and a
//! few milliseconds in the common case (only the host's own vendor tool
//! responds, the others fail to launch immediately).

use std::process::{Command, Stdio};
use std::time::Duration;

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

impl GpuVendor {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Nvidia => "nvidia",
            Self::Amd => "amd",
            Self::Intel => "intel",
            Self::Apple => "apple",
            Self::Unknown => "unknown",
        }
    }
}

/// One detected GPU. Memory fields are in MiB to match the units returned by
/// the underlying tools — convert at the call site if you need bytes.
#[derive(Debug, Clone, Serialize)]
pub struct GpuDevice {
    pub vendor: GpuVendor,
    pub name: String,
    pub vram_total_mib: Option<u64>,
    pub vram_free_mib: Option<u64>,
    pub driver: Option<String>,
}

/// Snapshot of every GPU the host exposes.
#[derive(Debug, Clone, Default, Serialize)]
pub struct GpuProbe {
    pub devices: Vec<GpuDevice>,
}

impl GpuProbe {
    /// Probe every supported vendor in turn. Order: NVIDIA → AMD → Apple →
    /// Intel. The list is built lazily, so a host with only an AMD card
    /// pays for at most one failed `nvidia-smi` lookup.
    pub fn detect() -> Self {
        let mut devices = Vec::new();
        devices.extend(probe_nvidia());
        devices.extend(probe_amd());
        devices.extend(probe_apple());
        // Intel iGPU detection is currently a no-op — Intel doesn't ship a
        // host-side query tool comparable to nvidia-smi.
        Self { devices }
    }

    pub fn primary(&self) -> Option<&GpuDevice> {
        self.devices.first()
    }

    pub fn primary_vendor(&self) -> GpuVendor {
        self.primary().map(|d| d.vendor).unwrap_or(GpuVendor::Unknown)
    }

    pub fn vendors(&self) -> Vec<GpuVendor> {
        let mut v: Vec<GpuVendor> = self.devices.iter().map(|d| d.vendor).collect();
        v.sort_by_key(|x| *x as u8);
        v.dedup();
        v
    }
}

// ---------------------------------------------------------------------------
// NVIDIA — `nvidia-smi --query-gpu=...`
// ---------------------------------------------------------------------------

fn probe_nvidia() -> Vec<GpuDevice> {
    let bin = nvidia_smi_binary();
    let Some(output) = run_with_timeout(
        bin,
        &[
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        Duration::from_millis(2_000),
    ) else {
        return Vec::new();
    };
    parse_nvidia_csv(&output)
}

fn parse_nvidia_csv(s: &str) -> Vec<GpuDevice> {
    s.lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').map(str::trim).collect();
            if parts.len() < 4 {
                return None;
            }
            Some(GpuDevice {
                vendor: GpuVendor::Nvidia,
                name: parts[0].to_string(),
                vram_total_mib: parts[1].parse().ok(),
                vram_free_mib: parts[2].parse().ok(),
                driver: Some(parts[3].to_string()),
            })
        })
        .collect()
}

fn nvidia_smi_binary() -> &'static str {
    if cfg!(target_os = "windows") {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    }
}

// ---------------------------------------------------------------------------
// AMD — `rocm-smi --showproductname --showmeminfo vram --json`
// ---------------------------------------------------------------------------

fn probe_amd() -> Vec<GpuDevice> {
    let Some(output) = run_with_timeout(
        "rocm-smi",
        &["--showproductname", "--showmeminfo", "vram", "--json"],
        Duration::from_millis(2_000),
    ) else {
        return Vec::new();
    };
    parse_amd_json(&output)
}

fn parse_amd_json(s: &str) -> Vec<GpuDevice> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(s) else {
        return Vec::new();
    };
    let Some(obj) = value.as_object() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (key, fields) in obj {
        // Per-card sub-objects are keyed `card0`, `card1`, …
        if !key.to_lowercase().starts_with("card") {
            continue;
        }
        let name = fields
            .get("Card Series")
            .or_else(|| fields.get("Card series"))
            .or_else(|| fields.get("Card model"))
            .and_then(|v| v.as_str())
            .unwrap_or("AMD GPU")
            .to_string();
        let total = fields
            .get("VRAM Total Memory (B)")
            .or_else(|| fields.get("VRAM Total (B)"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|b| b / (1024 * 1024));
        let used = fields
            .get("VRAM Total Used Memory (B)")
            .or_else(|| fields.get("VRAM Used (B)"))
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<u64>().ok())
            .map(|b| b / (1024 * 1024));
        let free = match (total, used) {
            (Some(t), Some(u)) => Some(t.saturating_sub(u)),
            _ => None,
        };
        out.push(GpuDevice {
            vendor: GpuVendor::Amd,
            name,
            vram_total_mib: total,
            vram_free_mib: free,
            driver: None,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Apple — `system_profiler SPDisplaysDataType -json` (macOS only)
// ---------------------------------------------------------------------------

fn probe_apple() -> Vec<GpuDevice> {
    if !cfg!(target_os = "macos") {
        return Vec::new();
    }
    let Some(output) = run_with_timeout(
        "system_profiler",
        &["SPDisplaysDataType", "-json"],
        Duration::from_millis(2_000),
    ) else {
        return Vec::new();
    };
    parse_apple_json(&output)
}

fn parse_apple_json(s: &str) -> Vec<GpuDevice> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(s) else {
        return Vec::new();
    };
    let cards = value
        .get("SPDisplaysDataType")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    cards
        .into_iter()
        .map(|card| {
            let name = card
                .get("sppci_model")
                .and_then(|v| v.as_str())
                .unwrap_or("Apple GPU")
                .to_string();
            // `sppci_vram` looks like "8 GB"; strip the unit, multiply.
            let vram_total_mib = card
                .get("sppci_vram")
                .and_then(|v| v.as_str())
                .and_then(parse_size_to_mib);
            GpuDevice {
                vendor: GpuVendor::Apple,
                name,
                vram_total_mib,
                vram_free_mib: None,
                driver: None,
            }
        })
        .collect()
}

fn parse_size_to_mib(s: &str) -> Option<u64> {
    let trimmed = s.trim();
    let (num_part, unit_part) = trimmed.split_at(
        trimmed
            .find(|c: char| !c.is_ascii_digit() && c != '.' && c != ' ')
            .unwrap_or(trimmed.len()),
    );
    let n: f64 = num_part.trim().parse().ok()?;
    let unit = unit_part.trim().to_uppercase();
    let mib = match unit.as_str() {
        "GB" | "GIB" => n * 1024.0,
        "MB" | "MIB" => n,
        "" => n,
        _ => return None,
    };
    Some(mib as u64)
}

// ---------------------------------------------------------------------------
// Process runner with a deadline
// ---------------------------------------------------------------------------

/// Spawn `bin args…`, kill it after `timeout`, and return its stdout when it
/// exits successfully. Returns `None` for any failure mode (binary not
/// found, non-zero exit, timeout, non-UTF-8 output).
fn run_with_timeout(bin: &str, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = Command::new(bin)
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .stdin(Stdio::null())
        .spawn()
        .ok()?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if !status.success() {
                    return None;
                }
                let mut buf = Vec::new();
                if let Some(mut out) = child.stdout.take() {
                    use std::io::Read;
                    let _ = out.read_to_end(&mut buf);
                }
                return String::from_utf8(buf).ok();
            }
            Ok(None) => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                // Yield without sleeping a long stretch — the process is
                // either fast (most cases) or genuinely stuck.
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(_) => return None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nvidia_csv_handles_multi_line_output() {
        let out = "NVIDIA GeForce RTX 4090, 24564, 23000, 550.54.14\n\
                   NVIDIA GeForce RTX 3060, 12288, 11500, 550.54.14\n";
        let devices = parse_nvidia_csv(out);
        assert_eq!(devices.len(), 2);
        assert_eq!(devices[0].vendor, GpuVendor::Nvidia);
        assert_eq!(devices[0].name, "NVIDIA GeForce RTX 4090");
        assert_eq!(devices[0].vram_total_mib, Some(24564));
        assert_eq!(devices[0].vram_free_mib, Some(23000));
        assert_eq!(devices[0].driver.as_deref(), Some("550.54.14"));
        assert_eq!(devices[1].name, "NVIDIA GeForce RTX 3060");
    }

    #[test]
    fn parse_nvidia_csv_skips_short_lines() {
        let out = "NVIDIA RTX, 24564\n";
        assert!(parse_nvidia_csv(out).is_empty());
    }

    #[test]
    fn parse_amd_json_extracts_per_card_fields() {
        // Shape simplified from a real `rocm-smi --showproductname --showmeminfo vram --json` dump.
        let out = r#"{
            "card0": {
                "Card Series": "Radeon RX 7900 XTX",
                "VRAM Total Memory (B)": "25753026560",
                "VRAM Total Used Memory (B)": "1572864000"
            }
        }"#;
        let devices = parse_amd_json(out);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor, GpuVendor::Amd);
        assert_eq!(devices[0].name, "Radeon RX 7900 XTX");
        // 25753026560 / 1MiB = 24560
        assert_eq!(devices[0].vram_total_mib, Some(24560));
        // free = 24560 − 1500
        assert_eq!(devices[0].vram_free_mib, Some(23060));
    }

    #[test]
    fn parse_amd_json_ignores_non_card_keys() {
        let out = r#"{ "system": {"foo": "bar"} }"#;
        assert!(parse_amd_json(out).is_empty());
    }

    #[test]
    fn parse_amd_json_handles_invalid_json() {
        assert!(parse_amd_json("not json").is_empty());
    }

    #[test]
    fn parse_apple_json_reads_sppci_vram_and_model() {
        let out = r#"{
            "SPDisplaysDataType": [
                {"sppci_model": "Apple M2 Pro", "sppci_vram": "16 GB"}
            ]
        }"#;
        let devices = parse_apple_json(out);
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].vendor, GpuVendor::Apple);
        assert_eq!(devices[0].name, "Apple M2 Pro");
        assert_eq!(devices[0].vram_total_mib, Some(16 * 1024));
    }

    #[test]
    fn parse_size_to_mib_handles_common_units() {
        assert_eq!(parse_size_to_mib("8 GB"), Some(8 * 1024));
        assert_eq!(parse_size_to_mib("512 MB"), Some(512));
        assert_eq!(parse_size_to_mib("1.5 GB"), Some(1536));
        assert_eq!(parse_size_to_mib("nonsense"), None);
    }

    #[test]
    fn run_with_timeout_returns_none_for_missing_binary() {
        let v = run_with_timeout(
            "this-binary-definitely-does-not-exist-1234",
            &[],
            Duration::from_millis(50),
        );
        assert!(v.is_none());
    }

    #[test]
    fn primary_vendor_falls_back_to_unknown_on_empty_probe() {
        let p = GpuProbe::default();
        assert_eq!(p.primary_vendor(), GpuVendor::Unknown);
    }

    #[test]
    fn vendors_lists_each_vendor_once() {
        let p = GpuProbe {
            devices: vec![
                GpuDevice {
                    vendor: GpuVendor::Nvidia,
                    name: "a".into(),
                    vram_total_mib: None,
                    vram_free_mib: None,
                    driver: None,
                },
                GpuDevice {
                    vendor: GpuVendor::Nvidia,
                    name: "b".into(),
                    vram_total_mib: None,
                    vram_free_mib: None,
                    driver: None,
                },
                GpuDevice {
                    vendor: GpuVendor::Amd,
                    name: "c".into(),
                    vram_total_mib: None,
                    vram_free_mib: None,
                    driver: None,
                },
            ],
        };
        let v = p.vendors();
        assert!(v.contains(&GpuVendor::Nvidia));
        assert!(v.contains(&GpuVendor::Amd));
        assert_eq!(v.len(), 2);
    }
}
