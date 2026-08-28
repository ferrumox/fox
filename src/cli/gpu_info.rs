// `fox gpu-info` — display GPU backend, VRAM, and driver details.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
pub struct GpuInfoArgs {}

pub async fn run_gpu_info(_args: GpuInfoArgs) -> Result<()> {
    println!("GPU Information");

    #[cfg(fox_stub)]
    {
        println!("  Backend:        CPU only (stub build)");
        println!("  No GPU acceleration detected");
    }

    #[cfg(not(fox_stub))]
    {
        if let Some(info) = detect_cuda() {
            print_cuda_info(&info);
        } else if cfg!(target_os = "macos") {
            print_metal_info();
        } else {
            println!("  Backend:        CPU only");
            println!("  No GPU acceleration detected");
        }
    }

    println!();
    println!("Build Info");
    print_build_info();

    Ok(())
}

#[cfg(not(fox_stub))]
struct CudaInfo {
    devices: Vec<CudaDevice>,
    driver_version: Option<String>,
    cuda_version: Option<String>,
}

#[cfg(not(fox_stub))]
struct CudaDevice {
    name: String,
    memory_total_mib: usize,
    memory_free_mib: usize,
    compute_cap: Option<String>,
}

#[cfg(not(fox_stub))]
fn detect_cuda() -> Option<CudaInfo> {
    let nvidia_smi = if cfg!(target_os = "windows") {
        "nvidia-smi.exe"
    } else {
        "nvidia-smi"
    };

    let out = std::process::Command::new(nvidia_smi)
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !out.status.success() {
        return None;
    }

    let stdout = std::str::from_utf8(&out.stdout).ok()?.trim().to_string();
    if stdout.is_empty() {
        return None;
    }

    let mut devices = Vec::new();
    let mut driver_version = None;

    for line in stdout.lines() {
        let parts: Vec<&str> = line.splitn(5, ',').collect();
        if parts.len() < 4 {
            continue;
        }
        let name = parts[0].trim().to_string();
        let total_mib: usize = parts[1].trim().parse().unwrap_or(0);
        let free_mib: usize = parts[2].trim().parse().unwrap_or(0);
        let drv = parts[3].trim().to_string();
        let compute = parts.get(4).map(|s| s.trim().to_string());

        if driver_version.is_none() && !drv.is_empty() {
            driver_version = Some(drv);
        }

        devices.push(CudaDevice {
            name,
            memory_total_mib: total_mib,
            memory_free_mib: free_mib,
            compute_cap: compute.filter(|s| !s.is_empty()),
        });
    }

    if devices.is_empty() {
        return None;
    }

    let cuda_version = detect_cuda_version(nvidia_smi);

    Some(CudaInfo {
        devices,
        driver_version,
        cuda_version,
    })
}

#[cfg(not(fox_stub))]
fn detect_cuda_version(nvidia_smi: &str) -> Option<String> {
    let out = std::process::Command::new(nvidia_smi).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = std::str::from_utf8(&out.stdout).ok()?;
    for line in stdout.lines() {
        if let Some(idx) = line.find("CUDA Version:") {
            let rest = line[idx + "CUDA Version:".len()..].trim();
            let ver = rest.split_whitespace().next().unwrap_or(rest);
            return Some(ver.to_string());
        }
    }
    None
}

#[cfg(not(fox_stub))]
fn print_cuda_info(info: &CudaInfo) {
    println!("  Backend:        CUDA");
    for (i, dev) in info.devices.iter().enumerate() {
        println!("  Device {}:       {}", i, dev.name);
        println!("  VRAM Total:     {} MB", dev.memory_total_mib);
        println!("  VRAM Free:      {} MB", dev.memory_free_mib);
        if let Some(ref cap) = dev.compute_cap {
            println!("  Compute Cap:    {}", cap);
        }
    }
    if let Some(ref drv) = info.driver_version {
        println!("  Driver Version: {}", drv);
    }
    if let Some(ref ver) = info.cuda_version {
        println!("  CUDA Version:   {}", ver);
    }
}

#[cfg(all(not(fox_stub), target_os = "macos"))]
fn print_metal_info() {
    println!("  Backend:        Metal");

    if let Some(gpu_name) = detect_macos_gpu_name() {
        println!("  Device 0:       {}", gpu_name);
    }
    if let Some(mem_mb) = detect_macos_memory_mb() {
        println!("  Unified Memory: {} MB", mem_mb);
    }
    if let Some(metal_ver) = detect_macos_metal_version() {
        println!("  Metal Version:  {}", metal_ver);
    }
}

#[cfg(all(not(fox_stub), not(target_os = "macos")))]
fn print_metal_info() {
    println!("  Backend:        CPU only");
    println!("  No GPU acceleration detected");
}

#[cfg(all(not(fox_stub), target_os = "macos"))]
fn detect_macos_gpu_name() -> Option<String> {
    let out = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let displays = json.get("SPDisplaysDataType")?.as_array()?;
    let first = displays.first()?;
    first
        .get("sppci_model")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

#[cfg(all(not(fox_stub), target_os = "macos"))]
fn detect_macos_memory_mb() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let bytes: usize = std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

#[cfg(all(not(fox_stub), target_os = "macos"))]
fn detect_macos_metal_version() -> Option<String> {
    let out = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    let displays = json.get("SPDisplaysDataType")?.as_array()?;
    let first = displays.first()?;
    first
        .get("spdisplays_metal_supported")
        .or_else(|| first.get("spdisplays_mtlgpufamilysupport"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

fn print_build_info() {
    #[cfg(fox_stub)]
    {
        println!("  llama.cpp:      stub build (no native backend)");
        println!("  Flash Attn:     not available");
        println!("  CUDA:           not available");
        println!("  Metal:          not available");
    }

    #[cfg(not(fox_stub))]
    {
        let has_cuda =
            cfg!(any(target_os = "linux", target_os = "windows",)) && cuda_backend_present();

        let has_metal = cfg!(target_os = "macos");

        if has_cuda {
            println!("  llama.cpp:      built with CUDA support");
        } else if has_metal {
            println!("  llama.cpp:      built with Metal support");
        } else {
            println!("  llama.cpp:      CPU only");
        }

        println!(
            "  Flash Attn:     {}",
            if has_cuda || has_metal {
                "available"
            } else {
                "not available"
            }
        );
        println!(
            "  CUDA:           {}",
            if has_cuda {
                "available"
            } else {
                "not available"
            }
        );
        println!(
            "  Metal:          {}",
            if has_metal {
                "available"
            } else {
                "not available"
            }
        );
    }
}

#[cfg(not(fox_stub))]
fn cuda_backend_present() -> bool {
    let exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return false,
    };
    let dir = match exe.parent() {
        Some(d) => d,
        None => return false,
    };
    let so_name = if cfg!(target_os = "windows") {
        "ggml-cuda.dll"
    } else {
        "libggml-cuda.so"
    };
    dir.join(so_name).exists()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_info_runs_without_panic() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let result = run_gpu_info(GpuInfoArgs {}).await;
            assert!(result.is_ok());
        });
    }
}
