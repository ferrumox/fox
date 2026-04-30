// Build llama.cpp and generate Rust FFI bindings.
//
// Strategy: GGML_BACKEND_DL=ON → GPU backends (CUDA, Metal, Vulkan, ROCm) are
// compiled as shared libraries (.so/.dylib/.dll) and loaded at runtime via dlopen.
// The fox binary itself has zero hard dependency on GPU runtime libraries — it
// works on any system and auto-detects the GPU at startup.
//
// GPU detection is automatic at build time:
//   - CUDA:  nvcc found in PATH or CUDACXX env var → builds libggml-cuda.so
//   - ROCm:  hipcc found in PATH or HIPCC env var  → builds libggml-hip.so
//   - Metal: macOS target                          → builds libggml-metal.dylib
//   - Vulkan (Linux): glslc/VULKAN_SDK/vulkan.h    → builds libggml-vulkan.so
//   - Vulkan (Windows): VULKAN_SDK env var          → builds ggml-vulkan.dll
//   No Cargo features needed; users just run `cargo build --release`.

use std::env;
use std::path::PathBuf;

/// Returns the absolute path to a command found in PATH, or None.
/// Used to detect GPU toolchains (nvcc, hipcc, glslc) at build time.
fn which_cmd(cmd: &str) -> Option<String> {
    std::process::Command::new("which")
        .arg(cmd)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(fox_stub)");
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let llama_root = PathBuf::from(&manifest_dir)
        .join("vendor")
        .join("llama.cpp");

    if env::var("FOX_SKIP_LLAMA").is_ok() || !llama_root.exists() {
        if !llama_root.exists() {
            println!(
                "cargo:warning=llama.cpp not found at vendor/llama.cpp. \
                 Clone it or set FOX_SKIP_LLAMA=1 to build with stubs."
            );
        }
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(
            out.join("llama_bindings.rs"),
            "// Stub — llama.cpp not built.\n#[allow(dead_code)] const _STUB: () = ();\n",
        )
        .unwrap();
        println!("cargo:rustc-cfg=fox_stub");
        return;
    }

    // ── patch vendor sources ──────────────────────────────────────────────────
    // ROCm 6.2.x ships with HIP_VERSION >= 60200000 but does NOT include
    // hip/hip_fp8.h or __hip_fp8_e4m3. The guard in llama.cpp's vendors/hip.h
    // incorrectly assumes 6.2 has FP8 support. Patch the file in-place before
    // invoking cmake so we don't need to maintain a fork of llama.cpp.
    let hip_compat_h = llama_root.join("ggml/src/ggml-cuda/vendors/hip.h");
    if hip_compat_h.exists() {
        if let Ok(src) = std::fs::read_to_string(&hip_compat_h) {
            let patched = src.replace(
                "#if HIP_VERSION >= 60200000\n#include <hip/hip_fp8.h>\ntypedef __hip_fp8_e4m3 __nv_fp8_e4m3;\n#define FP8_AVAILABLE\n#endif // HIP_VERSION >= 60200000",
                "#if HIP_VERSION >= 60300000\n#include <hip/hip_fp8.h>\ntypedef __hip_fp8_e4m3 __nv_fp8_e4m3;\n#define FP8_AVAILABLE\n#endif // HIP_VERSION >= 60300000",
            );
            if patched != src {
                let _ = std::fs::write(&hip_compat_h, &patched);
                println!("cargo:warning=Patched vendors/hip.h: FP8 guard raised to ROCm 6.3 (HIP_VERSION >= 60300000)");
            }
        }
    }

    // ── cmake configuration ───────────────────────────────────────────────────
    let mut cmake_config = cmake::Config::new(&llama_root);
    cmake_config
        // Backends become shared libraries; core (ggml-base, llama) stays static.
        .define("BUILD_SHARED_LIBS", "ON")
        // Each backend (.so) is dlopen-ed at runtime — zero CUDA dep in the binary.
        .define("GGML_BACKEND_DL", "ON")
        // GGML_NATIVE (CPU arch-specific optimizations) is incompatible with GGML_BACKEND_DL.
        // The CPU backend is selected generically; GGML_CPU_ALL_VARIANTS would build
        // arch-specific variants as separate .so files (optional future improvement).
        .define("GGML_NATIVE", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_WEBUI", "OFF")
        .profile("Release");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // ── CUDA auto-detection ───────────────────────────────────────────────────
    // Check CUDACXX env var first, then PATH.
    let nvcc = env::var("CUDACXX").ok().or_else(|| which_cmd("nvcc"));

    let cuda_enabled = if let Some(ref nvcc_path) = nvcc {
        if std::path::Path::new(nvcc_path).exists() {
            println!("cargo:warning=CUDA found at {nvcc_path} — building libggml-cuda.so");
            cmake_config.define("GGML_CUDA", "ON");
            cmake_config.define("CMAKE_CUDA_COMPILER", nvcc_path);
            true
        } else {
            false
        }
    } else {
        false
    };

    if !cuda_enabled {
        match target_os.as_str() {
            "macos" => {
                // Metal is always available on macOS — no tool detection needed.
                cmake_config.define("GGML_METAL", "ON");
            }
            "linux" => {
                // ── ROCm/HIP auto-detection ────────────────────────────────────
                // Check HIPCC env var first, then PATH. Mutually exclusive with CUDA.
                let hipcc = env::var("HIPCC").ok().or_else(|| which_cmd("hipcc"));
                let rocm_enabled = if let Some(ref hipcc_path) = hipcc {
                    if std::path::Path::new(hipcc_path).exists() {
                        // CMake >= 3.21 requires a real Clang, not the hipcc wrapper.
                        // Try several candidate paths in order of preference:
                        //   1. Derived from hipcc's location (works for old and new ROCm layouts)
                        //   2. /opt/rocm/lib/llvm/bin/clang  (ROCm 6.x)
                        //   3. /opt/rocm/llvm/bin/clang      (ROCm 5.x)
                        //   4. amdclang in PATH              (ROCm 6.x alias)
                        let hipcc_p = std::path::Path::new(hipcc_path);
                        let derived_clang = hipcc_p.parent().and_then(|bin| {
                            // bin/../llvm/bin/clang   →  /opt/rocm/llvm/bin/clang
                            // bin/../lib/llvm/bin/clang → /opt/rocm/lib/llvm/bin/clang
                            for rel in &["../llvm/bin/clang", "../lib/llvm/bin/clang"] {
                                let candidate = bin.join(rel);
                                if let Ok(p) = candidate.canonicalize() {
                                    if p.exists() {
                                        return Some(p.to_string_lossy().into_owned());
                                    }
                                }
                            }
                            None
                        });
                        let rocm_clang = derived_clang
                            .or_else(|| {
                                let p = std::path::Path::new("/opt/rocm/lib/llvm/bin/clang");
                                p.exists().then(|| p.to_string_lossy().into_owned())
                            })
                            .or_else(|| {
                                let p = std::path::Path::new("/opt/rocm/llvm/bin/clang");
                                p.exists().then(|| p.to_string_lossy().into_owned())
                            })
                            .or_else(|| which_cmd("amdclang"));

                        match rocm_clang {
                            Some(ref clang_path) => {
                                println!(
                                    "cargo:warning=ROCm/HIP found at {hipcc_path} — building libggml-hip.so (HIP compiler: {clang_path})"
                                );
                                cmake_config.define("GGML_HIP", "ON");
                                cmake_config.define("CMAKE_HIP_COMPILER", clang_path);
                                true
                            }
                            None => {
                                println!(
                                    "cargo:warning=ROCm/HIP found at {hipcc_path} but no Clang compiler located \
                                     (CMake 3.21+ requires Clang, not hipcc). Skipping ROCm backend. \
                                     Set AMDCLANG or ensure /opt/rocm/lib/llvm/bin/clang exists."
                                );
                                false
                            }
                        }
                    } else {
                        false
                    }
                } else {
                    false
                };

                // ── Vulkan auto-detection (Linux) ──────────────────────────────
                // Enable when ROCm is unavailable and the Vulkan toolchain is present.
                // glslc (shader compiler) is required: apt install glslc libvulkan-dev
                if !rocm_enabled {
                    let has_vulkan = env::var("VULKAN_SDK").is_ok()
                        || which_cmd("glslc").is_some()
                        || std::path::Path::new("/usr/include/vulkan/vulkan.h").exists();
                    if has_vulkan {
                        println!(
                            "cargo:warning=Vulkan toolchain detected — building libggml-vulkan.so"
                        );
                        cmake_config.define("GGML_VULKAN", "ON");
                    }
                }
            }
            "windows" => {
                // Use Ninja to avoid MSBuild's FileTracker MAX_PATH limit.
                cmake_config.generator("Ninja");

                // Vulkan works on any modern GPU (NVIDIA, AMD, Intel) via DirectX 12 drivers.
                // Requires CARGO_TARGET_DIR=C:\t (or similarly short) in the workflow to keep
                // the vulkan-shaders-gen ExternalProject paths under Windows MAX_PATH.
                if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
                    println!(
                        "cargo:warning=Vulkan SDK found at {vulkan_sdk} — building ggml-vulkan.dll"
                    );
                    cmake_config.define("GGML_VULKAN", "ON");
                }
            }
            _ => {}
        }
    }

    // ── build ─────────────────────────────────────────────────────────────────
    let dst = cmake_config.build();
    let build_dir = dst.join("build");

    // Core libraries: llama (static) + ggml-base (shared, contains backend registry).
    // With GGML_BACKEND_DL + BUILD_SHARED_LIBS=ON, the backend-registry symbols
    // (ggml_backend_load_all_from_path etc.) live in libggml-base.so only.
    let llama_lib = build_dir.join("src");
    let ggml_src = build_dir.join("ggml").join("src");
    let bin_out = build_dir.join("bin");

    // bin_out must come FIRST so the linker finds libggml.so / libggml-base.so
    // before it encounters libggml.a in ggml/src/. With GGML_BACKEND_DL=ON the
    // static archives reference ggml_backend_cpu_reg directly, causing link
    // errors if the .a is resolved instead of the .so.
    println!("cargo:rustc-link-search=native={}", bin_out.display());
    println!("cargo:rustc-link-search=native={}", llama_lib.display());
    println!("cargo:rustc-link-search=native={}", ggml_src.display());

    // llama.cpp's add_library(llama) has no explicit STATIC/SHARED, so with
    // BUILD_SHARED_LIBS=ON it becomes a shared library on all platforms.
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml-base"); // shared: core ggml types
    println!("cargo:rustc-link-lib=dylib=ggml"); // shared: backend dlopen registry
                                                 // NOTE: do NOT link ggml-cpu / ggml-metal / ggml-cuda explicitly.
                                                 // With GGML_BACKEND_DL=ON they are MODULE libraries loaded at runtime via
                                                 // dlopen. On macOS, cmake MODULEs use .so which the macOS linker rejects.

    // ── Copy backend .so/.dylib files next to the fox binary ─────────────────
    // OUT_DIR is target/{profile}/build/ferrumox-<hash>/out — three levels up
    // gives target/{profile}/, which is where the fox binary lands.
    // This ensures both `cargo build` (debug) and `cargo build --release` have
    // the backend shared libraries in the same directory as the binary.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_dest = out_dir
        .parent() // ferrumox-<hash>
        .and_then(|p| p.parent()) // build/
        .and_then(|p| p.parent()) // target/{profile}/
        .map(|p| p.to_path_buf());

    if let Some(ref dest) = bin_dest {
        // On macOS, SHARED libs (.dylib) and MODULE libs (.so) coexist:
        //   .dylib → llama, ggml-base, ggml (linked at compile time)
        //   .so    → ggml-cpu, ggml-metal, ... (MODULE, dlopen-ed at runtime)
        // Both must be copied next to the binary.
        let exts: &[&str] = if target_os == "macos" {
            &["dylib", "so"]
        } else {
            &["so"]
        };
        for search_dir in &[&llama_lib, &ggml_src, &bin_out] {
            if let Ok(entries) = std::fs::read_dir(search_dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let p = entry.path();
                    let fname = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    // Match both plain .so and versioned soname files like
                    // libllama.so.0 / libllama.so.0.9.7 (p.extension() returns
                    // the last component, not "so", for versioned names).
                    for ext in exts {
                        let is_backend = fname.contains(&format!(".{ext}"))
                        && (fname.starts_with("libggml")
                            || fname.starts_with("libllama.")
                            || fname == format!("llama.{ext}"));
                        if is_backend {
                            let dst = dest.join(p.file_name().unwrap());
                            let _ = std::fs::copy(&p, &dst);
                        }
                    }
                }
            }
        }
    }

    // ── RPATH: find backend .so files next to the fox binary ($ORIGIN) ────────
    // This lets users copy fox + libggml-cuda.so to any directory and have it
    // just work without LD_LIBRARY_PATH.
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=dylib=m");
        }
        "macos" => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
            println!("cargo:rustc-link-lib=dylib=c++");
            if nvcc.is_none() {
                // Metal backend shared lib (built by cmake above).
                println!(
                    "cargo:rustc-link-search=native={}",
                    ggml_src.join("ggml-metal").display()
                );
            }
        }
        _ => {
            // Windows MSVC: the C++ runtime is linked automatically by the toolchain.
        }
    }

    // ── bindgen ───────────────────────────────────────────────────────────────
    let llama_include = llama_root.join("include");
    let ggml_include = llama_root.join("ggml").join("include");
    let ggml_build_include = build_dir.join("ggml").join("include");

    let mut include_paths = vec![llama_include.clone(), ggml_include];
    if ggml_build_include.exists() {
        include_paths.push(ggml_build_include);
    }

    let clang_args: Vec<String> = include_paths
        .iter()
        .flat_map(|p| vec!["-I".to_string(), p.to_string_lossy().into_owned()])
        .collect();

    let bindings = bindgen::Builder::default()
        .header(llama_include.join("llama.h").to_string_lossy())
        .clang_args(clang_args)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_var("LLAMA_.*")
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("llama_bindings.rs"))
        .expect("Couldn't write bindings");
}
