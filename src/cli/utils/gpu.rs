/// GPU info: name, VRAM used and total (in bytes).
pub struct GpuInfo {
    pub name: String,
    pub used_bytes: usize,
    pub total_bytes: usize,
}

/// System RAM info: used and total (in bytes).
pub struct RamInfo {
    pub used_bytes: usize,
    pub total_bytes: usize,
}

/// Minimum VRAM usage to consider the GPU as actively running the model.
/// Below this threshold (e.g. driver-only overhead) we return `None` so the
/// GPU line is omitted from the status display.
const GPU_ACTIVE_THRESHOLD_BYTES: usize = 256 * 1024 * 1024; // 256 MiB

/// Query GPU name, used VRAM, and total VRAM via nvidia-smi.
/// Returns `None` if no NVIDIA GPU is found, nvidia-smi is not available,
/// or used VRAM is below the active threshold (model is on CPU/Vulkan).
pub fn get_gpu_info() -> Option<GpuInfo> {
    let nvidia_smi = if cfg!(target_os = "windows") {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    };
    let out = std::process::Command::new(nvidia_smi)
        .args([
            "--query-gpu=name,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
    let parts: Vec<&str> = s.splitn(3, ',').collect();
    if parts.len() < 3 {
        return None;
    }
    let name = parts[0].trim().to_string();
    let used_mib: usize = parts[1].trim().parse().ok()?;
    let total_mib: usize = parts[2].trim().parse().ok()?;
    let used_bytes = used_mib * 1024 * 1024;
    // Skip when GPU is idle — model is running on CPU or Vulkan (not tracked by nvidia-smi).
    if used_bytes < GPU_ACTIVE_THRESHOLD_BYTES {
        return None;
    }
    Some(GpuInfo {
        name,
        used_bytes,
        total_bytes: total_mib * 1024 * 1024,
    })
}

/// Query RAM info.
///
/// - `used_bytes`: RSS of **this process** (`/proc/self/status` VmRSS on Linux).
///   This reflects the actual memory footprint of the loaded model + KV cache,
///   and is more meaningful than system-wide used RAM in a status bar.
/// - `total_bytes`: total system RAM from `/proc/meminfo` (capacity reference).
///
/// Falls back to zeros on unsupported platforms.
pub fn get_ram_info() -> RamInfo {
    #[cfg(target_os = "linux")]
    {
        // Process RSS — reflects model weights + KV cache in RAM.
        let rss_bytes = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<usize>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0);

        // System total — capacity reference shown in the startup banner.
        let total_bytes = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("MemTotal:"))
                    .and_then(|l| l.split_whitespace().nth(1))
                    .and_then(|v| v.parse::<usize>().ok())
                    .map(|kb| kb * 1024)
            })
            .unwrap_or(0);

        RamInfo {
            used_bytes: rss_bytes,
            total_bytes,
        }
    }
    #[cfg(target_os = "macos")]
    {
        let total = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        if total > 0 {
            let free_pages = std::process::Command::new("vm_stat")
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .and_then(|s| {
                    s.lines()
                        .find(|l| l.contains("Pages free:"))
                        .and_then(|l| l.split(':').nth(1))
                        .and_then(|v| v.trim().trim_end_matches('.').parse::<usize>().ok())
                })
                .unwrap_or(0);
            return RamInfo {
                used_bytes: total.saturating_sub(free_pages * 4096),
                total_bytes: total,
            };
        }
    }
    #[cfg(not(target_os = "linux"))]
    RamInfo {
        used_bytes: 0,
        total_bytes: 0,
    }
}

/// Query total GPU memory via nvidia-smi. Falls back to 8 GiB if no GPU is found.
/// Returns only the first GPU's memory (used for single-GPU budget calculations).
pub fn get_gpu_memory_bytes() -> usize {
    get_all_gpu_memory_bytes()
        .into_iter()
        .next()
        .unwrap_or(8 * 1024 * 1024 * 1024)
}

/// Query memory for all available GPUs via nvidia-smi.
/// Returns a Vec with one entry per GPU (in bytes). Empty if nvidia-smi is unavailable.
pub fn get_all_gpu_memory_bytes() -> Vec<usize> {
    let nvidia_smi = if cfg!(target_os = "windows") {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    };
    if let Ok(out) = std::process::Command::new(nvidia_smi)
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if out.status.success() {
            if let Ok(s) = std::str::from_utf8(&out.stdout) {
                let gpus: Vec<usize> = s
                    .lines()
                    .filter_map(|line| line.trim().parse::<usize>().ok())
                    .map(|mib| mib * 1024 * 1024)
                    .collect();
                if !gpus.is_empty() {
                    return gpus;
                }
            }
        }
    }
    vec![]
}

/// Sum of memory across all GPUs. Falls back to 8 GiB if no GPU is found.
pub fn get_total_gpu_memory_bytes() -> usize {
    let gpus = get_all_gpu_memory_bytes();
    if gpus.is_empty() {
        8 * 1024 * 1024 * 1024
    } else {
        gpus.iter().sum()
    }
}
