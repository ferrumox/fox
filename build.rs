// Build llama.cpp and generate Rust FFI bindings.
//
// Strategy: GGML_BACKEND_DL=ON → GPU backends (CUDA, Metal) are compiled as
// shared libraries (.so/.dylib) and loaded at runtime via dlopen. The fox binary
// itself has zero hard dependency on CUDA runtime libraries — it works on any
// system and auto-detects the GPU at startup.
//
// GPU detection is automatic at build time:
//   - CUDA:  nvcc found in PATH or CUDACXX env var → builds libggml-cuda.so
//   - Metal: macOS target → builds libggml-metal.dylib
//   No Cargo features needed; users just run `cargo build --release`.

use std::env;
use std::path::PathBuf;

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
        .profile("Release");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    // ── CUDA auto-detection ───────────────────────────────────────────────────
    // Check CUDACXX env var first, then PATH.
    let nvcc = env::var("CUDACXX").ok().or_else(|| {
        std::process::Command::new("which")
            .arg("nvcc")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
    });

    if let Some(ref nvcc_path) = nvcc {
        if std::path::Path::new(nvcc_path).exists() {
            println!("cargo:warning=CUDA found at {nvcc_path} — building libggml-cuda.so");
            cmake_config.define("GGML_CUDA", "ON");
            cmake_config.define("CMAKE_CUDA_COMPILER", nvcc_path);
        }
    } else if target_os == "macos" {
        // Metal is always available on macOS — no tool detection needed.
        cmake_config.define("GGML_METAL", "ON");
    } else if target_os == "windows" {
        // Use Ninja to avoid MSBuild's FileTracker MAX_PATH limit.
        cmake_config.generator("Ninja");

        // Vulkan works on any modern GPU (NVIDIA, AMD, Intel) via DirectX 12 drivers.
        // Requires CARGO_TARGET_DIR=C:\t (or similarly short) in the workflow to keep
        // the vulkan-shaders-gen ExternalProject paths under Windows MAX_PATH.
        if let Ok(vulkan_sdk) = env::var("VULKAN_SDK") {
            println!("cargo:warning=Vulkan SDK found at {vulkan_sdk} — building ggml-vulkan.dll");
            cmake_config.define("GGML_VULKAN", "ON");
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

    println!("cargo:rustc-link-search=native={}", llama_lib.display());
    println!("cargo:rustc-link-search=native={}", ggml_src.display());
    println!("cargo:rustc-link-search=native={}", bin_out.display());

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
                    let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                    let is_backend = exts.contains(&ext)
                        && (fname.starts_with("libggml-")
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
