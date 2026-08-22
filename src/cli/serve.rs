// `fox serve` — start the OpenAI-compatible HTTP inference server.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// Default values for ServeArgs — centralised so grep/docs find them in one place.
const DEFAULT_GPU_MEMORY_FRACTION: &str = "0.85";
const DEFAULT_MAX_BATCH_SIZE: &str = "32";
const DEFAULT_MAX_QUEUE_DEPTH: &str = "0";
const DEFAULT_MAX_PREFILL_CHUNK: &str = "512";
const DEFAULT_RS_ROLLBACK: &str = "4";
const DEFAULT_BLOCK_SIZE: &str = "16";
const DEFAULT_HOST: &str = "0.0.0.0";
const DEFAULT_PORT: &str = "8080";
const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const DEFAULT_SWAP_FRACTION: &str = "0.0";
const DEFAULT_MAX_MODELS: &str = "1";
const DEFAULT_KEEP_ALIVE_SECS: &str = "300";
const DEFAULT_TYPE_KV: &str = "f16";
const DEFAULT_TOOL_CALL_PARSER: &str = "auto";
const DEFAULT_REPEAT_LAST_N: &str = "-1";
const DEFAULT_SLOT_PROMPT_SIMILARITY: &str = "0.1";

use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::api::router;
use crate::metrics::Metrics;
use crate::model_registry::{ModelRegistry, RegistryConfig};

use super::get_gpu_memory_bytes;
use super::get_total_gpu_memory_bytes;
use super::list_models;
use super::load_aliases;
use super::models_dir as default_models_dir;
use super::theme;

#[derive(Parser, Debug, Clone)]
pub struct ServeArgs {
    /// Path to the GGUF model file (optional; if omitted models are loaded on demand)
    #[arg(long, env = "FOX_MODEL_PATH")]
    pub model_path: Option<PathBuf>,

    /// Fraction of GPU memory to use for KV cache (0.0-1.0)
    #[arg(long, default_value = DEFAULT_GPU_MEMORY_FRACTION, env = "FOX_GPU_MEMORY_FRACTION")]
    pub gpu_memory_fraction: f32,

    /// Maximum batch size for inference
    #[arg(long, default_value = DEFAULT_MAX_BATCH_SIZE, env = "FOX_MAX_BATCH_SIZE")]
    pub max_batch_size: usize,

    /// Maximum requests allowed to wait in the scheduler queue before new ones are
    /// rejected with HTTP 429 (0 = unbounded).
    #[arg(long, default_value = DEFAULT_MAX_QUEUE_DEPTH, env = "FOX_MAX_QUEUE_DEPTH")]
    pub max_queue_depth: usize,

    /// Max prompt tokens prefilled per request per scheduler step (0 = single-shot).
    /// Chunking a long prompt lets it interleave with other requests' token generation
    /// instead of blocking the engine loop for the whole prefill.
    #[arg(long, default_value = DEFAULT_MAX_PREFILL_CHUNK, env = "FOX_MAX_PREFILL_CHUNK")]
    pub max_prefill_chunk: usize,

    /// Recurrent-state snapshots kept per sequence so a hybrid/recurrent model can roll
    /// its cache back and reuse a prompt prefix (0 = no rollback, no prompt reuse).
    ///
    /// Ignored by dense models — llama.cpp allocates nothing for architectures that do
    /// not support rollback. For the ones that do (Qwen3.5, Qwen3-Next, Falcon-H1,
    /// Jamba) it is what makes prompt reuse possible at all, and it is not free: each
    /// snapshot is a full recurrent state per sequence. Measured on Qwen3.5-9B with 8
    /// sequences, ~453 MB per snapshot — 4 costs ~1.8 GB and covers multi-turn, where
    /// the rollback is a single token; 64 costs ~30 GB and covers re-sending a whole
    /// prompt. Raise it only for the second case.
    #[arg(long, default_value = DEFAULT_RS_ROLLBACK, env = "FOX_RS_ROLLBACK")]
    pub rs_rollback: u32,

    /// Roll the context window when a sequence fills n_ctx: discard the oldest KV window
    /// and keep generating, instead of stopping the request with `length`. Automatically
    /// skipped for recurrent/hybrid models whose KV cache cannot shift.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, env = "FOX_CONTEXT_SHIFT")]
    pub context_shift: bool,

    /// Tokens preserved at the front (BOS + system prompt) when the context is rolled.
    #[arg(long, default_value = "0", env = "FOX_CONTEXT_KEEP")]
    pub context_keep: usize,

    /// How far back the repetition, frequency and presence penalties look, counted in
    /// generated tokens: `-1` = the whole history, `0` = penalties disabled, `n` = the
    /// last `n`. Requests may override it per call (`repeat_last_n` / `options.repeat_last_n`).
    ///
    /// Defaults to `-1`, fox's historical behaviour. llama.cpp defaults to 64; a bounded
    /// window both stops penalising tokens from thousands of positions back and turns the
    /// per-step penalty pass from O(generated) into O(window).
    #[arg(long, default_value = DEFAULT_REPEAT_LAST_N, env = "FOX_REPEAT_LAST_N")]
    pub repeat_last_n: i32,

    /// Load the model as a reranker: create its context with RANK pooling so
    /// `POST /rerank` can read the relevance head. Required — a reranker GGUF does not
    /// reliably declare its pooling type, so fox cannot detect it (llama-server takes a
    /// flag for the same reason). A model loaded this way scores query/document pairs;
    /// it does not generate text.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, env = "FOX_RERANKING")]
    pub reranking: bool,

    /// Host-RAM budget in MiB for serialised sequence states. Complements
    /// `--kv-reuse`: a slot keeps a conversation warm by holding GPU blocks, this keeps
    /// one warm without holding any — a reclaimed sequence is copied to host memory and
    /// restored later instead of being re-prefilled. Worth it when VRAM is the scarce
    /// resource and prompts are long. `0` (the default) disables it.
    /// `0` leaves it to the model: hybrid and recurrent architectures get
    /// `--cache-ram 2048` implicitly, because for them it is not an optimisation but
    /// the only way prompt reuse works at all — their KV cannot be trimmed back to a
    /// past position, so a serialised checkpoint is the only route. Dense models get
    /// nothing, since they reach the same prefix by trimming, for free.
    #[arg(long, default_value = "0", env = "FOX_CACHE_RAM")]
    pub cache_ram: usize,

    /// Keep a finished request's KV cache resident so a later prompt sharing a prefix
    /// with it skips re-prefilling that much. Each sequence remembers the tokens it
    /// holds — prompt *and* generated reply — so the next turn of a conversation
    /// matches well past where the previous prompt ended.
    ///
    /// Pass `--kv-reuse false` to restore the previous behaviour (every sequence
    /// cleared on completion, every prompt prefilled from token 0). That is also the
    /// baseline arm when A/B-measuring this with `scripts/ab_bench.sh`.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set, env = "FOX_KV_REUSE")]
    pub kv_reuse: bool,

    /// Minimum fraction of an incoming prompt that must already be resident in an idle
    /// sequence before that sequence's KV is inherited rather than started fresh
    /// (0.0–1.0). Mirrors llama-server's `--slot-prompt-similarity`. Ignored when
    /// `--kv-reuse false`.
    #[arg(long, default_value = DEFAULT_SLOT_PROMPT_SIMILARITY, env = "FOX_SLOT_PROMPT_SIMILARITY")]
    pub slot_prompt_similarity: f32,

    /// Enable n-gram / prompt-lookup speculative decoding — verify several guessed tokens
    /// per forward pass for single-request decode steps. Output is unchanged; only speed.
    /// Off by default. Automatically skipped while a request uses guided decoding.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set, env = "FOX_SPECULATIVE")]
    pub speculative: bool,

    /// Suffix length matched against the request's history when speculating.
    #[arg(long, default_value = "2", env = "FOX_SPEC_NGRAM")]
    pub spec_ngram: usize,

    /// Maximum draft tokens proposed per speculative step.
    #[arg(long, default_value = "4", env = "FOX_SPEC_DRAFT_LEN")]
    pub spec_draft_len: usize,

    /// Model name/path for a smaller "draft" model used to propose tokens for
    /// verification by the main model (classic draft-model speculative decoding),
    /// instead of n-gram lookup. Requires --speculative true (a value set without it
    /// is ignored, with a startup warning). The draft and target must share the same
    /// tokenizer — checked at load time, fails loudly on mismatch. Loaded once
    /// alongside the target and kept resident for the process lifetime; NOT subject
    /// to LRU eviction or VRAM budgeting in this release — size both models to fit.
    /// `--spec-ngram` is ignored in this mode.
    #[arg(long, env = "FOX_DRAFT_MODEL")]
    pub draft_model: Option<String>,

    /// Name/path of the mmproj (vision projector) GGUF paired with the main model,
    /// enabling image input (OpenAI `image_url` / Ollama `images`) via llama.cpp's
    /// `mtmd` library. Like `--draft-model`, this is a single global pairing — one
    /// mmproj active at a time, matched against whatever model is currently loaded.
    #[arg(long, env = "FOX_MMPROJ")]
    pub mmproj: Option<String>,

    /// EXPERIMENTAL, opt-in, and currently SLOWER than the default n-gram proposer —
    /// do not enable it except to work on it. Acceptance is 4-17% against
    /// `llama-server`'s ~60% with the same head, because a second driver desync
    /// remains unfixed; the server warns about this at load. Deliberately absent from
    /// `docs/cli/`, so it carries no Tier 1 compatibility promise and may change or be
    /// removed in any release.
    ///
    /// Name/path of the multi-token-prediction head GGUF paired with the main model
    /// (published as `mtp-<model>.gguf`), enabling MTP speculative decoding. Requires
    /// `--speculative true`; a value set without it is ignored, with a startup warning.
    /// Unlike `--draft-model` this is not a second model that generates on its own: it
    /// is a trained NextN block reading the target's hidden states, which is why it is
    /// expected to beat n-gram drafting on text that does not repeat once the desync is
    /// fixed. A head belonging to a different model is rejected at load time by
    /// hidden-state width.
    #[arg(long, env = "FOX_MTP_MODEL")]
    pub mtp_model: Option<String>,

    /// Named LoRA adapters to load alongside the primary model, so a client can
    /// select one by name in the `model` field (OpenAI/Ollama) instead of the
    /// base model — e.g. `--lora-modules support-es=adapters/es.gguf,legal=
    /// adapters/legal.gguf:0.8`. Format: `name=path[:scale]` (scale default
    /// `1.0`, mirrors llama.cpp's own `--lora-scaled FNAME:SCALE`), comma-
    /// separated for multiple adapters. All adapters apply to the single
    /// primary model (like `--draft-model`/`--mmproj`, this is one global
    /// pairing, not per-request-selectable across multiple base models).
    #[arg(long, env = "FOX_LORA_MODULES")]
    pub lora_modules: Option<String>,

    /// Tokens per KV block
    #[arg(long, default_value = DEFAULT_BLOCK_SIZE, env = "FOX_BLOCK_SIZE")]
    pub block_size: usize,

    /// Host to bind the server to
    #[arg(long, default_value = DEFAULT_HOST, env = "FOX_HOST")]
    pub host: String,

    /// Port to bind the server to
    #[arg(long, default_value = DEFAULT_PORT, env = "FOX_PORT")]
    pub port: u16,

    /// Maximum context length per sequence (tokens).
    /// If omitted, fox auto-detects the model's trained context length.
    #[arg(long, env = "FOX_MAX_CONTEXT_LEN")]
    pub max_context_len: Option<u32>,

    /// Default system prompt injected when no system message is present.
    /// Pass an empty string to disable injection.
    #[arg(
        long,
        default_value = DEFAULT_SYSTEM_PROMPT,
        env = "FOX_SYSTEM_PROMPT"
    )]
    pub system_prompt: String,

    /// Which format to parse tool calls from the model's raw output: `auto` (default
    /// — picks `hermes` when the loaded model's own chat template natively formats
    /// tool calls via `<tool_call>` tags, `mistral` via a `[TOOL_CALLS]` marker,
    /// generic prompt-based JSON otherwise), `generic`, `hermes`, `mistral`, or
    /// `llama3` (explicit-opt-in only — never auto-selected, since most GGUF chat
    /// templates for Llama3 models don't retain a detectable tool-call convention).
    #[arg(long, default_value = DEFAULT_TOOL_CALL_PARSER, env = "FOX_TOOL_CALL_PARSER")]
    pub tool_call_parser: String,

    /// [reserved — not yet implemented] Fraction of GPU memory for CPU↔GPU KV-cache
    /// swap space (0.0-1.0). Accepted for forward compatibility; currently a no-op.
    #[arg(long, default_value = DEFAULT_SWAP_FRACTION, env = "FOX_SWAP_FRACTION")]
    pub swap_fraction: f32,

    /// Use JSON log format (for production)
    #[arg(long, env = "FOX_JSON_LOGS")]
    pub json_logs: bool,

    /// HuggingFace API token for authenticated model pulls via POST /api/pull.
    /// Can also be set with the HF_TOKEN environment variable.
    #[arg(long, env = "HF_TOKEN")]
    pub hf_token: Option<String>,

    /// Maximum number of models to keep in memory simultaneously (LRU eviction).
    /// Defaults to 1: a request for a second model evicts the first (logged), which
    /// is the safe choice for the small-VRAM iGPUs fox targets. Raise it if you have
    /// the VRAM to serve several models at once.
    #[arg(long, default_value = DEFAULT_MAX_MODELS, env = "FOX_MAX_MODELS")]
    pub max_models: usize,

    /// Path to aliases TOML file. Default: ~/.config/ferrumox/aliases.toml
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Seconds a model stays in memory after its last request (0 = never evict by time).
    #[arg(long, default_value = DEFAULT_KEEP_ALIVE_SECS, env = "FOX_KEEP_ALIVE_SECS")]
    pub keep_alive_secs: u64,

    /// KV cache quantization for both K and V: f16 (default), q8_0, q4_0.
    /// Use --type-k / --type-v to set K and V independently (asymmetric configs).
    #[arg(long, default_value = DEFAULT_TYPE_KV, env = "FOX_TYPE_KV")]
    pub type_kv: String,

    /// Key cache element type (overrides --type-kv for K).
    /// Values: f16, q8_0, q4_0.
    #[arg(long, env = "FOX_TYPE_K")]
    pub type_k: Option<String>,

    /// Value cache element type (overrides --type-kv for V).
    /// Values: f16, q8_0, q4_0.
    #[arg(long, env = "FOX_TYPE_V")]
    pub type_v: Option<String>,

    /// Require `Authorization: Bearer <key>` on every API request.
    /// Omit to run without authentication.
    #[arg(long, env = "FOX_API_KEY")]
    pub api_key: Option<String>,

    /// Transformer layers to offload to the GPU. `-1` (default) offloads all of them,
    /// `0` keeps the model on the CPU, and anything in between splits it: the first N
    /// layers run on the GPU and the rest on the CPU. Mirrors `llama-server -ngl`.
    ///
    /// Set this when a model's weights do not fit in VRAM. With the default `-1` such a
    /// model fails to load outright — llama.cpp is asked for every layer, and refuses.
    /// Only activations cross PCIe per token, a few KB, so a narrow link does not
    /// penalise the split; the cost is that CPU-side layers run at system-RAM bandwidth.
    #[arg(long, default_value = "-1", env = "FOX_N_GPU_LAYERS")]
    pub n_gpu_layers: i32,

    /// Primary GPU index (0-based) used when split_mode=none, or as the main GPU for splits.
    #[arg(long, default_value = "0", env = "FOX_MAIN_GPU")]
    pub main_gpu: i32,

    /// How to split the model across multiple GPUs: none, layer (default), row.
    /// - none:  single GPU (main_gpu)
    /// - layer: distribute transformer layers across GPUs (most common)
    /// - row:   tensor-parallel row split (requires model support)
    #[arg(long, default_value = "layer", env = "FOX_SPLIT_MODE")]
    pub split_mode: String,

    /// Comma-separated VRAM proportions for tensor splitting (e.g. "3,1" for 75%/25%).
    /// When omitted, llama.cpp splits proportionally to available VRAM.
    #[arg(long, env = "FOX_TENSOR_SPLIT")]
    pub tensor_split: Option<String>,

    /// Offload MoE expert tensors to CPU RAM instead of VRAM.
    /// Useful for MoE models (DeepSeek, Mixtral) that don't fully fit in VRAM.
    /// Attention layers remain on GPU; expert weights are read from RAM on demand.
    #[arg(long, env = "FOX_MOE_CPU")]
    pub moe_cpu: bool,
}

// Both of these used to answer an unrecognised value with the default and no comment:
// `type_kv = "turbo3"` in config.toml quantised nothing and said nothing, and the user
// spent their time looking elsewhere. A silently ignored setting is worse than a
// rejected one — the config file has no completion, no type checking and no feedback
// except this.
fn parse_kv_type(s: &str) -> Result<u32> {
    use crate::model_registry::kv_type;
    match s {
        "f16" => Ok(kv_type::F16),
        "q8_0" => Ok(kv_type::Q8_0),
        "q4_0" => Ok(kv_type::Q4_0),
        other => anyhow::bail!("unknown KV cache type '{other}' — expected f16, q8_0 or q4_0"),
    }
}

fn parse_split_mode(s: &str) -> Result<u32> {
    match s {
        "layer" => Ok(1),
        "row" => Ok(2),
        "none" => Ok(0),
        other => anyhow::bail!("unknown split mode '{other}' — expected layer, row or none"),
    }
}

/// Parse `name=path[:scale][,name=path[:scale]...]` into `(name, path, scale)`
/// triples. Scale defaults to `1.0` when omitted. Entries that don't contain
/// `=` are skipped (malformed input just yields fewer adapters rather than a
/// startup crash — same leniency `parse_tensor_split` already has for a bad
/// value).
fn parse_lora_modules(s: &str) -> Vec<(String, PathBuf, f32)> {
    s.split(',')
        .filter_map(|entry| {
            let entry = entry.trim();
            let (name, rest) = entry.split_once('=')?;
            // Only treat a trailing `:N` as a scale if it actually parses as a
            // number — a bare colon otherwise belongs to the path (e.g. a
            // Windows drive letter, `C:\path\to\file.gguf`), not a scale.
            let (path, scale) = match rest.rsplit_once(':') {
                Some((p, sc)) if sc.parse::<f32>().is_ok() => (p, sc.parse().unwrap()),
                _ => (rest, 1.0),
            };
            if name.is_empty() || path.is_empty() {
                return None;
            }
            Some((name.to_string(), PathBuf::from(path), scale))
        })
        .collect()
}

/// Parse "3,1" → [0.75, 0.25] (normalizes so values sum to 1.0).
fn parse_tensor_split(s: &str) -> Vec<f32> {
    let raw: Vec<f32> = s
        .split(',')
        .filter_map(|p| p.trim().parse::<f32>().ok())
        .collect();
    let sum: f32 = raw.iter().sum();
    if sum <= 0.0 {
        return vec![];
    }
    raw.iter().map(|&v| v / sum).collect()
}

/// One-time startup note when `--max-models` is left at its default of 1.
///
/// The default trades resident-model churn (a second model's request evicts
/// the first, once idle) for VRAM safety: fox has no cross-model VRAM
/// accounting today — the per-load "does this fit" check compares against a
/// static, whole-GPU figure captured at startup, never subtracting what's
/// already resident, so raising the default without that accounting would
/// risk silently trading a churn footgun for an OOM footgun. Making the
/// trade-off visible here (not just in `--help`) is the fix; see STATUS.md's
/// "Known issues" for the max_models item and why the default itself stays 1.
fn max_models_default_hint(max_models: usize) -> Option<&'static str> {
    (max_models <= 1).then_some(
        // Kept visible at startup on purpose, but said in one line: this fires on the
        // default configuration, so every user reads it on every start, and a paragraph
        // there costs more attention than the trade-off is worth. The VRAM-accounting
        // caveat lives in STATUS.md's known issues, where someone hitting the limit
        // will look.
        "--max-models 1 (default): loading a second model evicts the first once idle",
    )
}

/// Warn once if `--swap-fraction` was explicitly set to a nonzero value — the
/// flag is accepted for forward compatibility but is not wired to anything
/// (CPU↔GPU KV transfer needs a llama.cpp API that doesn't exist yet).
/// Silently dropping a value the user explicitly asked for is exactly the
/// kind of silent failure CLAUDE.md's conventions ask fox to avoid.
pub(crate) fn swap_fraction_unused_warning(swap_fraction: f32) -> Option<String> {
    (swap_fraction != 0.0).then(|| {
        format!(
            "--swap-fraction {swap_fraction} was set but is not implemented yet (no-op) — \
             CPU↔GPU KV swap needs a llama.cpp API that doesn't exist yet; this value is ignored."
        )
    })
}

impl ServeArgs {
    fn validate(&self) -> Result<()> {
        if self.gpu_memory_fraction <= 0.0 || self.gpu_memory_fraction > 1.0 {
            anyhow::bail!(
                "gpu_memory_fraction must be in range (0, 1], got {}",
                self.gpu_memory_fraction
            );
        }
        if self.max_context_len == Some(0) {
            anyhow::bail!("max_context_len must be greater than 0");
        }
        if self.max_batch_size == 0 {
            anyhow::bail!("max_batch_size must be greater than 0");
        }
        if self.block_size == 0 {
            anyhow::bail!("block_size must be greater than 0");
        }
        // Reject unknown enum-ish values here, at startup, rather than at the point of
        // use — `validate()` runs before anything is loaded, so the message is the first
        // thing the operator sees instead of the last.
        parse_kv_type(&self.type_kv)?;
        if let Some(t) = &self.type_k {
            parse_kv_type(t)?;
        }
        if let Some(t) = &self.type_v {
            parse_kv_type(t)?;
        }
        parse_split_mode(&self.split_mode)?;
        Ok(())
    }
}

fn setup_logging(json: bool) {
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("ferrumox=info,warn"));

    if json {
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        // `.compact()`, not `.pretty()`. The pretty formatter is built for reading a
        // debug session: it prints an indented `at src/file.rs:line` under every event
        // and a blank line between them, so starting the server produced three lines
        // per log entry and a screenful of source paths before it had served anything.
        // That is the first thing a new user sees. Source locations belong in a
        // debugger or in `--json-logs`, which carries them as fields for anything
        // machine-read.
        tracing_subscriber::registry()
            .with(filter)
            .with(tracing_subscriber::fmt::layer().compact())
            .init();
    }
}

pub async fn run_serve(args: ServeArgs) -> Result<()> {
    args.validate()?;
    setup_logging(args.json_logs);

    let effective_max_models = args.max_models.max(1);
    if let Some(hint) = max_models_default_hint(effective_max_models) {
        tracing::info!("{hint}");
    }
    if let Some(warning) = swap_fraction_unused_warning(args.swap_fraction) {
        tracing::warn!("{warning}");
    }

    let split_mode = parse_split_mode(&args.split_mode)?;
    // Use total VRAM across all GPUs when splitting across multiple GPUs.
    let gpu_memory_bytes = if split_mode != 0 {
        get_total_gpu_memory_bytes()
    } else {
        get_gpu_memory_bytes()
    };

    let metrics = match Metrics::new() {
        Ok(m) => {
            tracing::info!("Prometheus metrics registered; scrape at /metrics");
            Some(std::sync::Arc::new(m))
        }
        Err(e) => {
            tracing::warn!("failed to register Prometheus metrics: {}", e);
            None
        }
    };

    // Load model aliases from TOML file (optional).
    let aliases = load_aliases(args.alias_file.clone());

    let models_dir = default_models_dir();

    // Determine the primary model's NAME up front (no registry needed for this —
    // just file_stem / directory discovery). Needed both to pre-load below and to
    // tell LoRA-alias resolution which model `--lora-modules` adapters attach to
    // (mirrors `--draft-model`/`--mmproj`: one global pairing against whatever the
    // primary model is, not a per-request-selectable base model).
    let primary_model = match &args.model_path {
        Some(path) => path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("default")
            .to_string(),
        None => {
            // Lazy mode: discover the first model in models_dir as the primary.
            let first = match list_models(&models_dir) {
                Ok(models) => models
                    .into_iter()
                    .next()
                    .and_then(|(p, _)| p.file_stem().and_then(|s| s.to_str()).map(str::to_string)),
                Err(e) => {
                    tracing::error!("failed to list models in {}: {}", models_dir.display(), e);
                    None
                }
            };
            if let Some(ref name) = first {
                tracing::info!(
                    "lazy mode: primary model set to '{}' (not pre-loaded)",
                    name
                );
            } else {
                tracing::warn!(
                    "no model specified and no .gguf files found in {}; \
                     requests will fail until a model is available",
                    models_dir.display()
                );
            }
            first.unwrap_or_default()
        }
    };

    let registry_cfg = RegistryConfig {
        models_dir: models_dir.clone(),
        max_models: effective_max_models,
        max_batch_size: args.max_batch_size,
        max_queue_depth: args.max_queue_depth,
        max_prefill_chunk: args.max_prefill_chunk,
        rs_rollback: args.rs_rollback,
        context_shift: args.context_shift,
        context_keep: args.context_keep,
        reranking: args.reranking,
        cache_ram_bytes: args.cache_ram.saturating_mul(1024 * 1024),
        kv_reuse: args.kv_reuse,
        slot_prompt_similarity: args.slot_prompt_similarity,
        speculative: args.speculative,
        spec_ngram: args.spec_ngram,
        spec_draft_len: args.spec_draft_len,
        draft_model: args.draft_model,
        mmproj: args.mmproj,
        mtp_model: args.mtp_model,
        lora_modules: args
            .lora_modules
            .as_deref()
            .map(parse_lora_modules)
            .unwrap_or_default(),
        primary_model: (!primary_model.is_empty()).then(|| primary_model.clone()),
        max_context_len: args.max_context_len,
        block_size: args.block_size,
        gpu_memory_bytes,
        gpu_memory_fraction: args.gpu_memory_fraction,
        metrics,
        keep_alive_secs: args.keep_alive_secs,
        type_k: parse_kv_type(args.type_k.as_deref().unwrap_or(&args.type_kv))?,
        type_v: parse_kv_type(args.type_v.as_deref().unwrap_or(&args.type_kv))?,
        main_gpu: args.main_gpu,
        split_mode,
        tensor_split: args
            .tensor_split
            .as_deref()
            .map(parse_tensor_split)
            .unwrap_or_default(),
        moe_offload_cpu: args.moe_cpu,
        n_gpu_layers: args.n_gpu_layers,
    };

    let registry = std::sync::Arc::new(ModelRegistry::new(registry_cfg, aliases));

    // Actually pre-load the initial model now that the registry exists.
    if let Some(path) = &args.model_path {
        tracing::info!("pre-loading model from {:?}", path);
        registry
            .get_or_load(path.to_string_lossy().as_ref())
            .await?;
    }

    // Start background keep-alive eviction task.
    std::sync::Arc::clone(&registry).start_eviction_task();

    let addr: std::net::SocketAddr =
        format!("{}:{}", args.host, args.port)
            .parse()
            .map_err(|e| {
                anyhow::anyhow!("invalid bind address '{}:{}': {}", args.host, args.port, e)
            })?;

    let system_prompt = if args.system_prompt.is_empty() {
        None
    } else {
        Some(args.system_prompt)
    };

    let started_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if args.api_key.is_some() {
        tracing::info!("API key authentication enabled");
    }

    let app = router(
        registry,
        primary_model.clone(),
        system_prompt,
        started_at,
        models_dir,
        args.hf_token,
        args.api_key,
        args.tool_call_parser,
        args.repeat_last_n,
    )
    .layer(tower_http::cors::CorsLayer::permissive());

    tracing::info!("listening on {}", addr);
    let display_name = if primary_model.is_empty() {
        "none (lazy)"
    } else {
        &primary_model
    };
    theme::print_serve_ready(display_name, &addr.to_string());

    let server = axum::serve(tokio::net::TcpListener::bind(addr).await?, app);

    tokio::select! {
        r = server => { r?; }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received, exiting");
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            tracing::warn!("failed to listen for Ctrl-C: {}", e);
            // Block forever — server stays up if signal setup fails.
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = sigterm.recv() => {}
                }
            }
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {}", e);
                ctrl_c.await;
            }
        }
    }

    #[cfg(not(unix))]
    ctrl_c.await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_lora_modules_single_no_scale_defaults_to_one() {
        let got = parse_lora_modules("mylora=/models/lora.gguf");
        assert_eq!(
            got,
            vec![(
                "mylora".to_string(),
                PathBuf::from("/models/lora.gguf"),
                1.0
            )]
        );
    }

    #[test]
    fn parse_lora_modules_with_scale() {
        let got = parse_lora_modules("legal=/models/legal.gguf:0.8");
        assert_eq!(
            got,
            vec![(
                "legal".to_string(),
                PathBuf::from("/models/legal.gguf"),
                0.8
            )]
        );
    }

    #[test]
    fn parse_lora_modules_multiple_mixed() {
        let got = parse_lora_modules("a=/x/a.gguf:0.5,b=/x/b.gguf");
        assert_eq!(
            got,
            vec![
                ("a".to_string(), PathBuf::from("/x/a.gguf"), 0.5),
                ("b".to_string(), PathBuf::from("/x/b.gguf"), 1.0),
            ]
        );
    }

    #[test]
    fn parse_lora_modules_windows_drive_letter_path_not_mistaken_for_scale() {
        let got = parse_lora_modules(r"win=C:\models\lora.gguf");
        assert_eq!(
            got,
            vec![(
                "win".to_string(),
                PathBuf::from(r"C:\models\lora.gguf"),
                1.0
            )]
        );
    }

    #[test]
    fn parse_lora_modules_skips_malformed_entries() {
        let got = parse_lora_modules("noequals,=missingname.gguf,name=");
        assert!(got.is_empty());
    }

    #[test]
    fn parse_lora_modules_empty_string_yields_no_adapters() {
        assert!(parse_lora_modules("").is_empty());
    }

    #[test]
    fn max_models_default_hint_fires_at_default() {
        assert!(max_models_default_hint(1).is_some());
    }

    #[test]
    fn max_models_default_hint_fires_below_default_too() {
        // args.max_models.max(1) clamps 0 up to 1 before this is called in
        // practice, but the function itself should still treat "0" as "1".
        assert!(max_models_default_hint(0).is_some());
    }

    #[test]
    fn max_models_default_hint_silent_when_raised() {
        assert!(max_models_default_hint(2).is_none());
        assert!(max_models_default_hint(8).is_none());
    }

    #[test]
    fn swap_fraction_unused_warning_silent_at_zero() {
        assert!(swap_fraction_unused_warning(0.0).is_none());
    }

    #[test]
    fn swap_fraction_unused_warning_fires_when_set() {
        let msg = swap_fraction_unused_warning(0.3).expect("nonzero should warn");
        assert!(msg.contains("0.3"));
        assert!(msg.contains("not implemented"));
    }
}
