// `fox probe` — load a model and print its resolved `ModelInfo` (the source of
// truth), flagging any internal contradictions between the model's metadata and
// the formulas fox uses elsewhere.
//
// Unlike `fox show` (which guesses architecture/quant from the *filename*), probe
// actually loads the model and reads the llama.cpp API + GGUF metadata.

use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;

use super::theme;
use super::{list_models, models_dir};
use crate::engine::model::{LlamaCppModel, Model};
use crate::model_registry::kv_type;

#[derive(Parser, Debug)]
pub struct ProbeArgs {
    /// Model name (stem), filename, or path to a .gguf file
    pub model: String,

    /// Directory to search for models (defaults to ~/.cache/ferrumox/models)
    #[arg(long)]
    pub path: Option<PathBuf>,
}

pub async fn run_probe(args: ProbeArgs) -> Result<()> {
    let path = resolve(&args)?;

    // Minimal load just to introspect. We cap the context small so this probe's
    // KV allocation stays tiny; `n_ctx_train` (the model's real trained context)
    // is read from metadata and is independent of this small context.
    let model = LlamaCppModel::load(
        &path,
        1,                       // max_batch_size
        Some(1024),              // max_context_len — keep the probe's KV small
        24 * 1024 * 1024 * 1024, // gpu_memory_bytes budget hint
        0.9,                     // gpu_memory_fraction
        kv_type::F16,
        kv_type::F16,
        0,     // main_gpu
        0,     // split_mode = none
        &[],   // tensor_split
        false, // moe_offload_cpu
    )?;

    let info = model.model_info();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?");

    println!();
    theme::print_kv_pair("Model", stem);
    theme::print_kv_pair("Architecture", &info.arch_name);
    theme::print_kv_pair("n_embd", &info.n_embd.to_string());
    theme::print_kv_pair(
        "Heads",
        &format!("{} (kv: {})", info.n_head, info.n_head_kv),
    );
    theme::print_kv_pair("Head dim", &info.head_dim.to_string());
    theme::print_kv_pair("Layers", &info.n_layer.to_string());
    theme::print_kv_pair("Vocab size", &info.vocab_size.to_string());
    theme::print_kv_pair("Trained ctx", &info.n_ctx_train.to_string());
    theme::print_kv_pair("EOS token", &info.eos_token_id.to_string());
    theme::print_kv_pair(
        "Chat template",
        if info.has_chat_template {
            "embedded"
        } else {
            "none (built-in fallback)"
        },
    );
    theme::print_kv_pair(
        "Native thinking",
        if info.supports_thinking { "yes" } else { "no" },
    );
    theme::print_kv_pair(
        "KV seq-copy",
        if info.supports_seq_copy {
            "yes"
        } else {
            "no (prefix cache disabled)"
        },
    );
    theme::print_kv_pair("Stop tokens", &info.stop_token_count.to_string());
    match &info.recommended_sampling {
        Some(r) => {
            let fmt = |v: Option<f32>| v.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string());
            let fmt_k =
                |v: Option<u32>| v.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string());
            theme::print_kv_pair(
                "Rec. sampling",
                &format!(
                    "temp={} top_p={} top_k={}",
                    fmt(r.temperature),
                    fmt(r.top_p),
                    fmt_k(r.top_k)
                ),
            );
        }
        None => theme::print_kv_pair("Rec. sampling", "none in metadata"),
    }

    let contradictions = info.contradictions();
    println!();
    if contradictions.is_empty() {
        println!("✓ No contradictions — model is coherent with fox's assumptions.");
    } else {
        println!("⚠ {} contradiction(s) detected:", contradictions.len());
        for c in &contradictions {
            println!("  - {c}");
        }
    }
    println!();
    Ok(())
}

/// Resolve the argument to a GGUF path: accept a direct file path, else look the
/// name/stem up in the models directory (like `fox show`).
fn resolve(args: &ProbeArgs) -> Result<PathBuf> {
    let direct = Path::new(&args.model);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    let dir = args.path.clone().unwrap_or_else(models_dir);
    let models = list_models(&dir)?;
    models
        .into_iter()
        .find(|(p, _)| {
            let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            stem == args.model || name == args.model
        })
        .map(|(p, _)| p)
        .ok_or_else(|| anyhow::anyhow!("model '{}' not found in {}", args.model, dir.display()))
}
