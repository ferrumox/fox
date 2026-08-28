// `fox run` — single-shot inference or interactive REPL, streaming output to stdout.
// Reuses the full engine stack (Scheduler + InferenceEngine) without an HTTP server.
//
// fox run llama "Hello"   → one-shot (resolved from models_dir)
// fox run llama           → opens interactive REPL
// fox run /abs/path/to/model.gguf "Hello"  → direct path

use std::io::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};

use crate::engine::model::{LlamaCppModel, Model};
use crate::engine::{EngineOptions, InferenceEngine};
use crate::kv_cache::KVCacheManager;
use crate::scheduler::{InferenceRequest, SamplingParams};

use super::resolve_model_path;
use super::theme;
use super::{get_gpu_info, get_gpu_memory_bytes, get_ram_info, get_total_gpu_memory_bytes};

#[derive(Parser, Debug)]
pub struct RunArgs {
    /// Model name, alias, or path to a GGUF file.
    /// Resolved against ~/.cache/ferrumox/models with alias → exact → prefix → contains fallback.
    #[arg(env = "FOX_MODEL_PATH")]
    pub model: String,

    /// The prompt to send to the model.
    /// If omitted, an interactive chat session is started.
    pub prompt: Option<String>,

    /// Path to aliases TOML file (default: ~/.config/ferrumox/aliases.toml)
    #[arg(long, env = "FOX_ALIAS_FILE")]
    pub alias_file: Option<PathBuf>,

    /// Maximum number of tokens to generate per turn
    #[arg(long, default_value = "4096")]
    pub max_new_tokens: usize,

    /// Sampling temperature (0 = greedy). Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.8.
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p nucleus sampling threshold. Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.9.
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Top-K filter (0 = disabled). Defaults to the model's recommended value if
    /// present in its metadata, otherwise 0.
    #[arg(long)]
    pub top_k: Option<u32>,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value = "1.0")]
    pub repetition_penalty: f32,

    /// How far back the penalties look, in generated tokens: -1 = whole history,
    /// 0 = disabled, n = last n.
    #[arg(long, default_value = "-1")]
    pub repeat_last_n: i32,

    /// RNG seed for reproducible output
    #[arg(long)]
    pub seed: Option<u64>,

    /// System prompt prepended to the conversation
    #[arg(
        long,
        default_value = "You are a helpful assistant.",
        env = "FOX_SYSTEM_PROMPT"
    )]
    pub system_prompt: String,

    /// Disable system prompt injection entirely
    #[arg(long)]
    pub no_system_prompt: bool,

    /// Maximum context length per sequence (tokens).
    /// If omitted, fox auto-detects the model's trained context length.
    #[arg(long)]
    pub max_context_len: Option<u32>,

    /// Fraction of GPU/RAM to use for KV cache
    #[arg(long, default_value = "0.85")]
    pub gpu_memory_fraction: f32,

    /// Tokens per KV block
    #[arg(long, default_value = "16")]
    pub block_size: usize,

    /// Fraction of GPU memory reserved for CPU↔GPU KV-cache swap space (0.0-1.0).
    /// Set to 0 to disable (default). Currently a placeholder — see `fox serve --help`.
    #[arg(long, default_value = "0.0")]
    pub swap_fraction: f32,

    /// Transformer layers to offload to the GPU. `-1` (default) offloads all of them,
    /// `0` keeps the model on the CPU, anything in between splits it. Mirrors
    /// `llama-server -ngl`. Needed when a model's weights do not fit in VRAM: with `-1`
    /// such a model fails to load rather than running partly on the CPU.
    #[arg(long, default_value = "-1", env = "FOX_N_GPU_LAYERS")]
    pub n_gpu_layers: i32,

    /// Primary GPU index (0-based). Used when split_mode=none, or as main GPU for splits.
    #[arg(long, default_value = "0", env = "FOX_MAIN_GPU")]
    pub main_gpu: i32,

    /// How to split the model across multiple GPUs: none, layer (default), row.
    #[arg(long, default_value = "layer", env = "FOX_SPLIT_MODE")]
    pub split_mode: String,

    /// Comma-separated VRAM proportions for tensor splitting (e.g. "3,1" for 75%/25%).
    #[arg(long, env = "FOX_TENSOR_SPLIT")]
    pub tensor_split: Option<String>,

    /// Offload MoE expert tensors to CPU RAM instead of VRAM.
    #[arg(long, env = "FOX_MOE_CPU")]
    pub moe_cpu: bool,

    /// Show the model's internal <think>…</think> reasoning block in the output.
    /// By default reasoning is suppressed; only the final answer is printed.
    #[arg(long)]
    pub show_thinking: bool,

    /// Show engine logs (hidden by default for cleaner output)
    #[arg(long)]
    pub verbose: bool,

    /// Enable n-gram / prompt-lookup speculative decoding (faster on repetitive output;
    /// output is unchanged). Off by default.
    #[arg(long, env = "FOX_SPECULATIVE")]
    pub speculative: bool,
}

pub async fn run_run(args: RunArgs) -> Result<()> {
    if args.verbose {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }

    if let Some(warning) = super::serve::swap_fraction_unused_warning(args.swap_fraction) {
        eprintln!("Warning: {warning}");
    }

    // Resolve model — auto-pull from HuggingFace if not found locally.
    let (model_name, model_path) = match resolve_model_path(&args.model, args.alias_file.as_deref())
    {
        Ok(r) => r,
        Err(_) => {
            eprintln!(
                "Model '{}' not found locally. Pulling from HuggingFace…",
                args.model
            );
            super::pull::run_pull(super::pull::PullArgs {
                model_id: args.model.clone(),
                filename: None,
                output_dir: None,
                hf_token: std::env::var("HF_TOKEN").ok(),
            })
            .await?;
            resolve_model_path(&args.model, args.alias_file.as_deref())?
        }
    };

    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("  {spinner:.cyan} {msg}")
            .expect("valid template")
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("Loading model…");
    spinner.enable_steady_tick(Duration::from_millis(80));

    let split_mode = match args.split_mode.as_str() {
        "row" => 2u32,
        "none" => 0u32,
        _ => 1u32, // layer
    };
    let tensor_split_parsed: Vec<f32> = args
        .tensor_split
        .as_deref()
        .map(|s| {
            let raw: Vec<f32> = s
                .split(',')
                .filter_map(|p| p.trim().parse::<f32>().ok())
                .collect();
            let sum: f32 = raw.iter().sum();
            if sum > 0.0 {
                raw.iter().map(|&v| v / sum).collect()
            } else {
                vec![]
            }
        })
        .unwrap_or_default();
    let gpu_memory_bytes_load = if split_mode != 0 {
        get_total_gpu_memory_bytes()
    } else {
        get_gpu_memory_bytes()
    };
    let model = LlamaCppModel::load(
        &model_path,
        1,
        args.max_context_len,
        gpu_memory_bytes_load,
        args.gpu_memory_fraction,
        1,
        1,
        args.n_gpu_layers,
        args.main_gpu,
        split_mode,
        &tensor_split_parsed,
        args.moe_cpu,
        None,  // mmproj_path — `fox run` has no --mmproj flag yet; use `fox serve` for vision
        1,     // vision_contexts — no mmproj, single (unused) pool slot
        &[],   // lora_modules — same: fox run has no --lora-modules flag yet
        false, // reranking — benches generate, never score,
        0,     // rs_rollback — no prompt reuse in this path
    )?;
    spinner.finish_and_clear();
    theme::print_success("Model loaded.");
    theme::print_kv_pair("Backend", &model.active_backend());

    // Size the block pool from the backend's actual KV capacity (llama_n_ctx).
    let kv_cache = std::sync::Arc::new(KVCacheManager::from_kv_tokens(
        model.kv_cache_capacity(),
        args.block_size,
    ));
    let scheduler = std::sync::Arc::new(crate::scheduler::Scheduler::new(kv_cache.clone(), 1));

    let model = std::sync::Arc::new(model);
    let engine = std::sync::Arc::new(InferenceEngine::new(
        model.clone(),
        scheduler.clone(),
        kv_cache,
        model_name,
        None,
        EngineOptions {
            // Roll context on full so long single-shot generations don't stop early.
            context_shift: Some(0),
            speculative: args
                .speculative
                .then_some(crate::engine::SpeculativeConfig::Ngram {
                    ngram: 2,
                    draft_len: 4,
                }),
            ..Default::default()
        },
        None,
    ));

    match args.prompt.clone() {
        Some(prompt) => run_oneshot(&args, &engine, prompt).await,
        None => run_repl(&args, &engine).await,
    }
}

/// One-shot mode: send a single prompt and stream the response to stdout.
async fn run_oneshot(args: &RunArgs, engine: &Arc<InferenceEngine>, prompt: String) -> Result<()> {
    let mut messages: Vec<(String, String)> = Vec::new();
    if !args.no_system_prompt && !args.system_prompt.is_empty() {
        messages.push(("system".to_string(), args.system_prompt.clone()));
    }
    messages.push(("user".to_string(), prompt));

    stream_turn(args, engine, &messages).await?;
    println!();
    Ok(())
}

/// Interactive REPL mode: maintain conversation history across multiple turns.
async fn run_repl(args: &RunArgs, engine: &Arc<InferenceEngine>) -> Result<()> {
    let model_name = engine.model_name();

    let effective_ctx = engine.context_len();
    let supports_thinking = engine.supports_thinking();
    let mut show_thinking = supports_thinking;

    theme::print_banner(model_name, effective_ctx, supports_thinking);
    let startup_gpu = get_gpu_info();
    let startup_ram = get_ram_info();
    theme::print_system_info(
        startup_gpu.as_ref(),
        &startup_ram,
        effective_ctx,
        supports_thinking,
        show_thinking,
    );

    // Keep the engine loop running for the lifetime of the session.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };
    let mut messages: Vec<(String, String)> = Vec::new();
    if !args.no_system_prompt && !args.system_prompt.is_empty() {
        messages.push(("system".to_string(), args.system_prompt.clone()));
    }

    // Line editing comes from rustyline. Plain `read_line()` leaves the terminal in
    // canonical mode, where the line discipline does not interpret arrow keys: pressing
    // Up to recall the previous message typed a literal `^[[A` into the prompt, and a
    // typo could only be fixed by backspacing to it.
    let history_path = chat_history_path();
    let mut editor = new_editor(history_path.as_deref());

    loop {
        // Reading blocks, so it runs off the runtime thread — otherwise it would starve
        // the engine loop task running concurrently. The editor is moved in and handed
        // back because it owns terminal state that must survive across turns.
        let (returned, result) = tokio::task::spawn_blocking(move || {
            let outcome = read_turn(editor.as_mut());
            (editor, outcome)
        })
        .await
        .expect("spawn_blocking panicked");
        editor = returned;

        let line = match result {
            Ok(Input::Line(l)) => l,
            // Ctrl+C abandons the half-typed line, it does not end the session.
            Ok(Input::Interrupted) => continue,
            Ok(Input::Eof) => break,
            Err(e) => {
                eprintln!("\nError reading input: {}", e);
                break;
            }
        };

        eprintln!();

        let input = line.trim().to_string();

        if input.is_empty() {
            continue;
        }

        if input == "/bye" || input == "/exit" || input == "exit" || input == "quit" {
            break;
        }

        // Recorded after the exit check: recalling "how do I quit" helps nobody.
        if let Some(ed) = editor.as_mut() {
            let _ = ed.add_history_entry(input.as_str());
        }

        if input == "/help" || input == "/?" {
            theme::print_repl_help(supports_thinking);
            continue;
        }

        if input == "/clear" {
            // Drop the history but keep the system prompt, which is configuration
            // rather than conversation.
            messages.truncate(if args.no_system_prompt { 0 } else { 1 });
            theme::eprint_styled(None, false, true, "  context cleared\n\n");
            continue;
        }

        if input == "/think" {
            if !supports_thinking {
                theme::eprint_styled(
                    Some(crossterm::style::Color::Yellow),
                    false,
                    false,
                    "  This model has no native reasoning support (<think> token not found)\n\n",
                );
            } else {
                show_thinking = !show_thinking;
                let status = if show_thinking { "on" } else { "off" };
                theme::eprint_styled(None, false, true, &format!("  think · {status}\n\n"));
            }
            continue;
        }

        messages.push(("user".to_string(), input));

        // Thinking spinner
        let spinner = ProgressBar::new_spinner();
        spinner.set_style(
            ProgressStyle::with_template("  {spinner:.dim} {msg:.dim}")
                .expect("valid template")
                .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
        );
        spinner.set_message("Thinking…");
        spinner.enable_steady_tick(Duration::from_millis(80));

        let start = Instant::now();
        let turn = stream_turn_collecting(args, engine, &messages, spinner, show_thinking).await?;
        let (response, token_count, cancelled) = (turn.text, turn.visible, turn.cancelled);
        let elapsed = start.elapsed();

        println!();
        let secs = elapsed.as_secs_f64();
        let toks_per_sec = if secs > 0.0 {
            token_count as f64 / secs
        } else {
            0.0
        };

        if cancelled {
            // Keep what was generated, cut back to a clean boundary: someone who stops a
            // reply has usually read enough of it to build on, but a mid-word tail
            // derails the next turn (see `trim_to_clean_boundary`). With nothing kept
            // there is no reply at all, so the question goes with it — leaving a user
            // turn unanswered would corrupt the next template render.
            //
            // This branch has to come first. An interrupted turn also produces an empty
            // response, and the test below reads that as a full context window.
            let kept = trim_to_clean_boundary(&response);
            if kept.is_empty() {
                messages.pop();
            } else {
                messages.push(("assistant".to_string(), kept.to_string()));
            }
        } else if response.is_empty() {
            // An empty reply used to be reported as a full context window and the whole
            // conversation was thrown away. That diagnosis was a guess, and usually a
            // wrong one: the engine emits tokens whose text is filtered out (control
            // markers, a reasoning block with thinking hidden), which produces no text
            // while the window sits nearly empty. Say what actually happened and keep
            // the conversation — a session should not lose its history to a guess.
            let ctx_used = engine
                .build_prompt_tokens(&messages, show_thinking, None)
                .map(|t| t.len())
                .unwrap_or(0);
            let full = ctx_used as u32 >= engine.context_len().saturating_sub(64);
            if full {
                theme::eprint_styled(
                    None,
                    false,
                    true,
                    &format!(
                        "  context window full ({ctx_used}/{}) — clearing the conversation\n\n",
                        engine.context_len()
                    ),
                );
                messages.truncate(if args.no_system_prompt { 0 } else { 1 });
            } else {
                theme::eprint_styled(
                    None,
                    false,
                    true,
                    &format!(
                        "  the model produced no text ({} token(s), stop: {:?}) — \
                         conversation kept, try rephrasing\n\n",
                        turn.tokens, turn.stop_reason
                    ),
                );
                // The user turn stays unanswered otherwise, which breaks the strict
                // user/assistant alternation some chat templates require.
                messages.pop();
            }
        } else {
            messages.push(("assistant".to_string(), response));
        }

        // Counted the way the next turn will actually be tokenised, or the status line
        // reports a context size the engine never sees.
        let ctx_tokens = engine
            .build_prompt_tokens(&messages, show_thinking, None)
            .map(|t| t.len())
            .unwrap_or(0);
        let gpu_info = get_gpu_info();
        let ram_info = get_ram_info();
        theme::print_status_line(
            ctx_tokens,
            engine.context_len(),
            gpu_info.as_ref(),
            &ram_info,
            toks_per_sec,
        );
    }

    if let (Some(ed), Some(path)) = (editor.as_mut(), history_path.as_deref()) {
        let _ = ed.save_history(&path);
    }

    engine_loop.abort();
    Ok(())
}

/// Trim an interrupted reply back to a boundary a model will accept as a finished turn.
///
/// Stopping generation cuts mid-word, and `"…methods for baking larg"` is not something a
/// model ever wrote as a finished turn, so sending it back as one puts the conversation
/// in a state the model was never trained on. Cutting at the last sentence end — or
/// failing that the last word — costs a few words of a reply the user chose to abandon
/// and leaves the history looking like an ordinary short answer.
///
/// This is hygiene, not a fix for anything measured: it was written while chasing empty
/// replies after a cancelled turn, and a control run reproduced those with no
/// cancellation involved, so the mid-word tail was not their cause.
fn trim_to_clean_boundary(text: &str) -> &str {
    let trimmed = text.trim_end();
    // Prefer a sentence end, but only a late one: cutting a 300-word reply back to its
    // first full stop would throw away most of what the user just read.
    if let Some(idx) = trimmed.rfind(['.', '!', '?', '\n']) {
        if idx + 1 >= trimmed.len() / 2 {
            return trimmed[..idx + 1].trim_end();
        }
    }
    match trimmed.rfind(char::is_whitespace) {
        Some(idx) => trimmed[..idx].trim_end(),
        // A single unbroken word: there is no clean cut, so keep it whole.
        None => trimmed,
    }
}

/// What one turn of input produced.
enum Input {
    Line(String),
    /// Ctrl+C — abandon the line, keep the session.
    Interrupted,
    /// Ctrl+D or a closed stdin.
    Eof,
}

/// Where the chat history is kept between sessions, beside the other config.
/// `None` disables persistence rather than failing the session over it.
fn chat_history_path() -> Option<PathBuf> {
    let dir = dirs::config_dir()?.join("ferrumox");
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir.join("chat_history"))
}

/// Build the line editor, loading previous history. Returns `None` when the terminal
/// cannot be driven, in which case the caller falls back to plain reads and the session
/// still works — just without editing.
fn new_editor(history: Option<&std::path::Path>) -> Option<rustyline::DefaultEditor> {
    let mut ed = rustyline::DefaultEditor::new().ok()?;
    if let Some(path) = history {
        // A missing file on first run is the normal case, not an error.
        let _ = ed.load_history(path);
    }
    Some(ed)
}

/// Read one line, from the editor when there is one and from stdin when there is not.
fn read_one(
    editor: Option<&mut rustyline::DefaultEditor>,
    prompt: &str,
    plain: impl FnOnce(),
) -> Result<Input, rustyline::error::ReadlineError> {
    match editor {
        Some(ed) => match ed.readline(prompt) {
            Ok(l) => Ok(Input::Line(l)),
            Err(rustyline::error::ReadlineError::Interrupted) => Ok(Input::Interrupted),
            Err(rustyline::error::ReadlineError::Eof) => Ok(Input::Eof),
            Err(e) => Err(e),
        },
        None => {
            plain();
            let mut buf = String::new();
            let n = std::io::stdin().read_line(&mut buf)?;
            Ok(if n == 0 { Input::Eof } else { Input::Line(buf) })
        }
    }
}

/// Read a full turn. Typing `"""` alone enters multiline mode; a second `"""` submits.
fn read_turn(
    mut editor: Option<&mut rustyline::DefaultEditor>,
) -> Result<Input, rustyline::error::ReadlineError> {
    const PROMPT: &str = "\x1b[1;36m  ❯ \x1b[0m";
    const CONT: &str = "\x1b[1;36m  · \x1b[0m";

    let first = match read_one(editor.as_deref_mut(), PROMPT, theme::print_prompt_glyph)? {
        Input::Line(l) => l,
        other => return Ok(other),
    };
    if first.trim() != "\"\"\"" {
        return Ok(Input::Line(first));
    }

    let mut buf = String::new();
    loop {
        match read_one(editor.as_deref_mut(), CONT, || {
            eprint!("  · ");
            let _ = std::io::stderr().flush();
        })? {
            Input::Line(l) if l.trim() == "\"\"\"" => break,
            Input::Line(l) => {
                buf.push_str(&l);
                // The editor strips the newline; plain reads keep it.
                if !buf.ends_with('\n') {
                    buf.push('\n');
                }
            }
            Input::Eof => break,
            Input::Interrupted => return Ok(Input::Interrupted),
        }
    }
    Ok(Input::Line(buf))
}

/// What one turn produced. `tokens` counts tokens the engine emitted, which is not the
/// same as text: a token whose text is filtered away (a control marker, half a UTF-8
/// sequence) still counts. The two disagreeing is exactly the case worth reporting.
struct Turn {
    text: String,
    tokens: usize,
    /// Emitted tokens whose text survived filtering.
    visible: usize,
    cancelled: bool,
    stop_reason: Option<crate::scheduler::StopReason>,
}

/// Run one inference turn, streaming tokens to stdout.
async fn stream_turn_collecting(
    args: &RunArgs,
    engine: &Arc<InferenceEngine>,
    messages: &[(String, String)],
    spinner: ProgressBar,
    show_thinking: bool,
) -> Result<Turn> {
    // Build the prompt exactly the way a `/v1/chat/completions` request does. Doing it
    // by hand here — render the template, then hand the result to `tokenize()` — looks
    // equivalent and is not: `tokenize()` is the *raw text* tokenizer, so it takes the
    // template's `<start_of_turn>` markers as literal text instead of the control tokens
    // they are, and prepends a second BOS on top of the one the template already emits.
    // The model then sees a conversation with no turn structure, and answers often enough
    // by writing a literal `<start_of_turn>model` — which the output filter holds back as
    // a control pattern, so the user gets an empty reply. `build_prompt_tokens` picks the
    // add_special/parse_special pair that matches how the prompt was rendered, and also
    // handles the thinking activation, so the manual `<think>` append goes with it.
    let prompt_tokens = engine
        .build_prompt_tokens(messages, show_thinking, None)
        .unwrap_or_else(|_| {
            let flat = messages
                .iter()
                .map(|(r, c)| format!("{}: {}", r, c))
                .collect::<Vec<_>>()
                .join("\n");
            engine
                .tokenize(&flat)
                .unwrap_or_else(|_| flat.bytes().map(|b| b as i32).take(4096).collect())
        });

    let recommended = engine.recommended_sampling();
    let mut sampling = build_sampling_params(args, recommended.as_ref());
    sampling.show_thinking = show_thinking;
    sampling.initial_in_thinking = show_thinking;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine
        .submit_request(req)
        .expect("submit: single request against a freshly-sized queue should never be rejected");

    let stdout = std::io::stdout();
    let mut response = String::new();
    let mut token_count: usize = 0;
    let mut first_token = true;
    // Track whether we are currently inside a <think>…</think> block so we
    // can apply ANSI dim styling to the reasoning section.
    let mut in_thinking_display = show_thinking;

    // Ctrl+C stops the reply instead of killing the session. While the editor is reading
    // the terminal is raw with ISIG off, so Ctrl+C arrives there as a byte and rustyline
    // handles it; only during generation does it become a real SIGINT, which is exactly
    // the window this covers. The listener is created once per turn rather than per token
    // — a fresh one each iteration would re-register a receiver thousands of times, and
    // creating it per turn also means a signal from an earlier turn cannot leak into this
    // one. Dropping `rx` on the way out is what cancels the work: the engine sees its
    // `send()` fail, preempts the request and frees the KV blocks.
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    let mut cancelled = false;
    let mut emitted: usize = 0;
    let mut last_stop: Option<crate::scheduler::StopReason> = None;

    loop {
        let token = tokio::select! {
            // Tokens win ties: a fast stream should never be starved by signal polling.
            biased;
            received = rx.recv() => match received {
                Some(t) => t,
                None => break,
            },
            _ = &mut interrupt => {
                cancelled = true;
                break;
            }
        };

        emitted += 1;
        if token.stop_reason.is_some() {
            last_stop = token.stop_reason.clone();
        }

        if !token.text.is_empty() {
            if first_token {
                spinner.finish_and_clear();
                eprintln!();
                theme::print_fox_label();
                let _ = std::io::stderr().flush();
                // The <think> tag was injected into the prompt; emit it
                // synthetically with dim styling so the user sees it.
                if show_thinking {
                    println!("\x1b[2m<think>");
                    let _ = stdout.lock().flush();
                }
                first_token = false;
            }

            if in_thinking_display {
                if let Some(idx) = token.text.find("</think>") {
                    // Print everything up to and including </think> in dim,
                    // then reset and print any text that follows normally.
                    let end = idx + "</think>".len();
                    print!("{}\x1b[0m{}", &token.text[..end], &token.text[end..]);
                    in_thinking_display = false;
                } else {
                    // Still inside thinking block — dim mode stays active.
                    print!("{}", token.text);
                }
            } else {
                print!("{}", token.text);
            }
            let _ = stdout.lock().flush();
            response.push_str(&token.text);
            token_count += 1;
        }
        if token.stop_reason.is_some() {
            if first_token {
                spinner.finish_and_clear();
            }
            break;
        }
    }

    if cancelled {
        // The spinner may still be spinning if nothing was generated yet.
        spinner.finish_and_clear();
        // Reset any dim styling left open by an interrupted <think> block, or the rest
        // of the session would render dim.
        if in_thinking_display {
            print!("\x1b[0m");
        }
        let _ = stdout.lock().flush();
        theme::eprint_styled(None, false, true, "\n  stopped\n");
    }

    Ok(Turn {
        text: response,
        tokens: emitted,
        visible: token_count,
        cancelled,
        stop_reason: last_stop,
    })
}

/// Run one inference turn streaming to stdout (no response collection — for one-shot mode).
async fn stream_turn(
    args: &RunArgs,
    engine: &Arc<InferenceEngine>,
    messages: &[(String, String)],
) -> Result<()> {
    // Same prompt-building path as the REPL and the HTTP handlers — see the comment in
    // `stream_turn_collecting` for why the manual render-then-tokenize is not equivalent.
    let prompt_tokens = engine
        .build_prompt_tokens(messages, args.show_thinking, None)
        .unwrap_or_else(|_| {
            let flat = messages
                .iter()
                .map(|(r, c)| format!("{}: {}", r, c))
                .collect::<Vec<_>>()
                .join("\n");
            engine
                .tokenize(&flat)
                .unwrap_or_else(|_| flat.bytes().map(|b| b as i32).take(4096).collect())
        });

    let recommended = engine.recommended_sampling();
    let mut sampling = build_sampling_params(args, recommended.as_ref());
    sampling.initial_in_thinking = args.show_thinking;

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let req_id = engine.next_request_id();
    let req = InferenceRequest::new(req_id, prompt_tokens, args.max_new_tokens, sampling, tx);
    engine
        .submit_request(req)
        .expect("submit: single request against a freshly-sized queue should never be rejected");

    // Drive the engine loop in the background for this single request.
    let engine_loop = {
        let engine = engine.clone();
        tokio::spawn(async move {
            let _ = engine.run_loop().await;
        })
    };

    let stdout = std::io::stdout();
    if args.show_thinking {
        println!("<think>");
        let _ = stdout.lock().flush();
    }
    while let Some(token) = rx.recv().await {
        if !token.text.is_empty() {
            print!("{}", token.text);
            let _ = stdout.lock().flush();
        }
        if token.stop_reason.is_some() {
            break;
        }
    }

    engine_loop.abort();
    Ok(())
}

fn build_sampling_params(
    args: &RunArgs,
    recommended: Option<&crate::engine::model::RecommendedSampling>,
) -> SamplingParams {
    // Priority: user flag > model metadata > hardcoded default.
    let temperature = args
        .temperature
        .or_else(|| recommended.and_then(|r| r.temperature))
        .unwrap_or(0.8);
    let top_p = args
        .top_p
        .or_else(|| recommended.and_then(|r| r.top_p))
        .unwrap_or(0.9);
    let top_k = args
        .top_k
        .or_else(|| recommended.and_then(|r| r.top_k))
        .unwrap_or(0);

    SamplingParams {
        temperature,
        top_p,
        top_k,
        repetition_penalty: args.repetition_penalty,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repeat_last_n: args.repeat_last_n,
        seed: args.seed,
        stop: None,
        show_thinking: args.show_thinking,
        initial_in_thinking: false, // set by callers that force thinking mode
        max_thinking_chars: 8192,
        grammar: None,
        logprobs: None,
        min_p: 0.0,
        min_tokens: 0,
        top_n_sigma: 0.0,
        min_keep: 0,
        logit_bias: None,
    }
}

#[cfg(test)]
mod tests {
    use super::trim_to_clean_boundary;

    #[test]
    fn an_interrupted_reply_is_cut_at_the_last_sentence() {
        let cut = trim_to_clean_boundary(
            "Bread is ancient. The Romans refined it. They introduced methods for baking larg",
        );
        assert_eq!(cut, "Bread is ancient. The Romans refined it.");
    }

    #[test]
    fn a_reply_with_no_late_sentence_end_is_cut_at_the_last_word() {
        // The only full stop sits in the first half, and cutting there would discard most
        // of what the user read, so the word boundary wins instead.
        let cut = trim_to_clean_boundary("Yes. Bread has been baked for millennia across ma");
        assert_eq!(cut, "Yes. Bread has been baked for millennia across");
    }

    #[test]
    fn a_single_unbroken_word_survives_whole() {
        // There is no clean cut available; returning "" would drop the turn entirely.
        assert_eq!(trim_to_clean_boundary("Constantinopl"), "Constantinopl");
    }

    #[test]
    fn stopping_before_the_first_word_ends_keeps_nothing_to_push() {
        assert_eq!(trim_to_clean_boundary("   "), "");
    }
}
