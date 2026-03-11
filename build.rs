// ============ MÓDULO 8: build.rs ============
//
// Build llama.cpp and generate Rust FFI bindings.

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
                "cargo:warning=llama.cpp not found. Set FOX_SKIP_LLAMA=1 to build with stubs only."
            );
        }
        // Generate minimal stub bindings (empty but valid module)
        let out = PathBuf::from(env::var("OUT_DIR").unwrap());
        std::fs::write(
            out.join("llama_bindings.rs"),
            "// Stub - llama.cpp not built. Build with vendor/llama.cpp present for full functionality.\n#[allow(dead_code)] const _STUB: () = ();\n",
        )
        .unwrap();
        println!("cargo:rustc-cfg=fox_stub");
        return;
    }

    // Build llama.cpp with cmake
    let mut cmake_config = cmake::Config::new(&llama_root);
    cmake_config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .profile("Release");

    // Feature-based backend selection
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        cmake_config.define("GGML_CUDA", "ON");

        // Resolve nvcc: CUDACXX env var → `which nvcc` → /usr/local/cuda fallback.
        let nvcc = env::var("CUDACXX").unwrap_or_else(|_| {
            std::process::Command::new("which")
                .arg("nvcc")
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout)
                            .ok()
                            .map(|s| s.trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "/usr/local/cuda/bin/nvcc".to_string())
        });
        if std::path::Path::new(&nvcc).exists() {
            cmake_config.define("CMAKE_CUDA_COMPILER", &nvcc);
        }
    } else if env::var("CARGO_FEATURE_METAL").is_ok() {
        cmake_config.define("GGML_METAL", "ON");
    }
    // else: cpu-only (default)

    let dst = cmake_config.build();
    let build_dir = dst.join("build");

    // llama.cpp puts libllama.a in build/src, ggml libs in build/ggml/src
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("src").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        build_dir.join("ggml").join("src").display()
    );

    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    println!("cargo:rustc-link-lib=static=ggml-base");

    // Platform-conditional system libraries.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=pthread");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=gomp");
            println!("cargo:rustc-link-lib=dylib=m");
        }
        "macos" => {
            // macOS uses libc++ (not libstdc++); pthread is part of libSystem.
            println!("cargo:rustc-link-lib=dylib=c++");
            if env::var("CARGO_FEATURE_METAL").is_ok() {
                println!("cargo:rustc-link-lib=static=ggml-metal");
                println!("cargo:rustc-link-lib=framework=Foundation");
                println!("cargo:rustc-link-lib=framework=Metal");
                println!("cargo:rustc-link-lib=framework=MetalKit");
            }
        }
        "windows" => {
            // MSVC links the C++ runtime automatically; no extra flags needed.
        }
        _ => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
        }
    }

    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        // CUDA backend static library (produced by llama.cpp cmake with GGML_CUDA=ON)
        println!(
            "cargo:rustc-link-search=native={}",
            build_dir
                .join("ggml")
                .join("src")
                .join("ggml-cuda")
                .display()
        );
        println!("cargo:rustc-link-lib=static=ggml-cuda");

        // Link CUDA runtime and cuBLAS.
        // Derive cuda_root from nvcc location (strip /bin/nvcc suffix).
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| {
            std::process::Command::new("which")
                .arg("nvcc")
                .output()
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout).ok().and_then(|s| {
                            std::path::Path::new(s.trim())
                                .parent() // .../bin
                                .and_then(|p| p.parent()) // cuda root
                                .map(|p| p.to_string_lossy().into_owned())
                        })
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "/usr/local/cuda".to_string())
        });
        // Support both /cuda/lib64 and /cuda/targets/x86_64-linux/lib layouts
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            cuda_path
        );
        println!("cargo:rustc-link-lib=dylib=cuda"); // Driver API (cuDeviceGet, cuMemCreate…)
        println!("cargo:rustc-link-lib=dylib=cudart");
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }

    // Generate bindings with bindgen
    let llama_include = llama_root.join("include");
    let ggml_include = llama_root.join("ggml").join("include");
    let ggml_build_include = llama_root.join("build").join("ggml").join("include");

    let mut include_paths = vec![llama_include.clone(), ggml_include.clone()];
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
