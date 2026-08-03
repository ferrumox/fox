// POST /v1/chat/completions handler.

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    Json,
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use uuid::Uuid;

use crate::api::error::{load_model_or_respond, AppError};
use crate::api::router::AppState;
use crate::api::shared::inference::{
    parse_tool_call, prepare_multimodal_prompt, prepare_prompt, resolve_tool_call_parser,
    resolve_tool_choice, MessageForTemplate,
};
use crate::api::shared::sampling_defaults as defaults;
use crate::api::shared::streaming::finish_reason_str;
use crate::api::types::{
    ChatCompletionChoice, ChatCompletionChunk, ChatCompletionChunkChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatLogprobEntry, ChatLogprobs, ChatMessageDelta, ChatMessageResponse,
    ToolCall, ToolCallDelta, ToolCallFunctionDelta, Usage,
};
use crate::engine::model::MEDIA_MARKER;
use crate::scheduler::{InferenceRequest, SamplingParams, Token};

/// One fully-buffered generation branch: `(text, completion_tokens, stop_reason,
/// logprobs, cached_prompt_tokens)`. Named because `n`/`best_of` fan-out collects a
/// `Vec` of these in three separate places.
type BufferedBranch = (
    String,
    u32,
    Option<crate::scheduler::StopReason>,
    Vec<ChatLogprobEntry>,
    u32,
);

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    if let Err(msg) = req.validate() {
        return AppError::BadRequest(msg).into_response();
    }
    let start = Instant::now();
    let (entry, lora) = match load_model_or_respond(&state.registry, &req.model).await {
        Ok(e) => e,
        Err(r) => return r,
    };

    let id = Uuid::new_v4().to_string();
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // fox has no audio support: warn loudly instead of silently dropping
    // input_audio (and similar) content blocks. Image blocks are handled below —
    // dropped only when the model has no vision support.
    let dropped_blocks: usize = req
        .messages
        .iter()
        .filter_map(|m| m.content.as_ref())
        .map(|c| c.non_text_blocks())
        .sum();
    if dropped_blocks > 0 {
        tracing::warn!(
            blocks = dropped_blocks,
            "dropped {dropped_blocks} non-text content block(s) — fox has no audio support"
        );
    }

    let has_images = req
        .messages
        .iter()
        .any(|m| m.content.as_ref().is_some_and(|c| c.has_images()));
    let use_vision = has_images && entry.engine.supports_vision();
    if has_images && !use_vision {
        tracing::warn!(
            model = %req.model,
            "dropped image content block(s) — model has no vision support (no mmproj loaded)"
        );
    }

    // Resolve tool_choice: filters tools and determines required/specific constraints.
    let tc = resolve_tool_choice(req.tools.as_deref(), req.tool_choice.as_ref());
    let eff_tools = tc.tools.as_deref();
    let tool_required = tc.required;
    let specific_tool = tc.specific.as_deref();

    // Thinking is opt-in via the `think` request field, and only when the model
    // actually supports it. Default off (no reasoning latency unless requested).
    let enable_thinking = req.think.unwrap_or(false) && entry.engine.supports_thinking();

    let (prompt_tokens, prompt_tokens_len, multimodal) = if use_vision {
        // Keep image blocks: each becomes a MEDIA_MARKER in the flattened text,
        // with its decoded bytes collected alongside for mtmd_tokenize.
        let mut images = Vec::new();
        let mut messages = Vec::with_capacity(req.messages.len());
        for m in &req.messages {
            let content = match m.content.as_ref() {
                Some(c) => match c.as_text_with_media_marker(MEDIA_MARKER) {
                    Ok((text, imgs)) => {
                        images.extend(imgs);
                        (!text.is_empty()).then_some(text)
                    }
                    Err(e) => return AppError::BadRequest(e).into_response(),
                },
                None => None,
            };
            messages.push(MessageForTemplate {
                role: m.role.clone(),
                content,
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            });
        }
        match prepare_multimodal_prompt(
            &entry,
            messages,
            state.system_prompt.as_deref(),
            eff_tools,
            tool_required,
            specific_tool,
            req.response_format.as_ref(),
            enable_thinking,
            &images,
        ) {
            Ok(chunks) => {
                let n = chunks.n_positions();
                (Vec::new(), n, Some(chunks))
            }
            Err(e) => {
                return AppError::BadRequest(format!("failed to encode image(s): {e}"))
                    .into_response()
            }
        }
    } else {
        // Build MessageForTemplate, extracting text from MessageContent (image
        // blocks dropped — either unsupported by this model, or none present).
        let messages: Vec<MessageForTemplate> = req
            .messages
            .iter()
            .map(|m| MessageForTemplate {
                role: m.role.clone(),
                content: m
                    .content
                    .as_ref()
                    .map(|c| c.as_text())
                    .filter(|s| !s.is_empty()),
                tool_calls: m.tool_calls.clone(),
                tool_call_id: m.tool_call_id.clone(),
            })
            .collect();
        let (tokens, len) = prepare_prompt(
            &entry,
            messages,
            state.system_prompt.as_deref(),
            eff_tools,
            tool_required,
            specific_tool,
            req.response_format.as_ref(),
            enable_thinking,
        );
        (tokens, len, None)
    };

    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .unwrap_or(defaults::openai::MAX_TOKENS as u32) as usize;

    // Guided decoding: convert `response_format` into a GBNF grammar. A json_schema
    // that can't be converted is a 400, not a silent unconstrained fallback.
    let grammar = match req.response_format.as_ref() {
        Some(rf) => match crate::api::shared::json_schema::grammar_from_response_format(rf) {
            Ok(g) => g,
            Err(e) => {
                return AppError::BadRequest(format!("invalid response_format: {e}"))
                    .into_response()
            }
        },
        None => None,
    };

    // Raw GBNF (`grammar`), a fox extension mirroring llama-server's field. The engine
    // has had full GBNF support since 0.14; until now the only way to reach it was
    // `response_format`/`format`, which can only express JSON. Setting both is an
    // error rather than a silent precedence rule — a caller who sends two conflicting
    // constraints has a bug, and picking one for them hides it.
    let grammar = match (&grammar, req.grammar.as_ref()) {
        (Some(_), Some(_)) => {
            return AppError::BadRequest(
                "`grammar` and `response_format` are mutually exclusive".to_string(),
            )
            .into_response()
        }
        (None, Some(g)) if !g.trim().is_empty() => Some(std::sync::Arc::from(g.as_str())),
        _ => grammar,
    };

    // `stream_options.include_usage` was previously parsed and ignored — usage always
    // rode the final chunk. Honour an explicit `false` (OpenAI's semantics: usage is
    // opt-in on streams); with `stream_options` absent, keep fox's historical
    // always-attach behaviour, which is a harmless superset and breaks no existing
    // caller.
    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(true);

    // Per-token logprobs: OpenAI caps top_logprobs at 20.
    let want_logprobs = req.logprobs == Some(true);
    let logprobs_top_n = req.top_logprobs.unwrap_or(0).min(20);

    // logit_bias arrives as string token ids (OpenAI); parse to a numeric map, dropping
    // any non-integer keys.
    let logit_bias = req.logit_bias.as_ref().and_then(|m| {
        let parsed: std::collections::HashMap<i32, f32> = m
            .iter()
            .filter_map(|(k, &v)| k.parse::<i32>().ok().map(|id| (id, v)))
            .collect();
        (!parsed.is_empty()).then(|| std::sync::Arc::new(parsed))
    });

    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(defaults::TEMPERATURE).max(0.0),
        top_p: req.top_p.unwrap_or(defaults::TOP_P).clamp(0.0, 1.0),
        top_k: req.top_k.unwrap_or(defaults::openai::TOP_K),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(defaults::openai::REPETITION_PENALTY)
            .max(1.0),
        frequency_penalty: req
            .frequency_penalty
            .unwrap_or(defaults::openai::FREQUENCY_PENALTY),
        presence_penalty: req
            .presence_penalty
            .unwrap_or(defaults::openai::PRESENCE_PENALTY),
        repeat_last_n: req.repeat_last_n.unwrap_or(state.repeat_last_n),
        seed: req.seed,
        stop: req.stop.clone(),
        show_thinking: false,
        initial_in_thinking: enable_thinking,
        max_thinking_chars: defaults::MAX_THINKING_CHARS,
        grammar,
        logprobs: if want_logprobs {
            Some(logprobs_top_n)
        } else {
            None
        },
        min_p: req.min_p.unwrap_or(0.0).clamp(0.0, 1.0),
        min_tokens: req.min_tokens.unwrap_or(0),
        top_n_sigma: req.top_n_sigma.unwrap_or(0.0).max(0.0),
        min_keep: req.min_keep.unwrap_or(0),
        logit_bias,
    };

    // n/best_of: each branch is a fully independent generation over the same
    // prompt (fan-out, not a shared-prefill fork — see
    // docs/design/n-best-of-support.md). `req.validate()` already guarantees
    // best_of >= n and (best_of == n whenever stream: true), so streaming
    // callers below never need to rank/discard.
    let n = req.n.unwrap_or(1).clamp(1, defaults::openai::MAX_N);
    let effective_best_of = req
        .best_of
        .unwrap_or(n)
        .max(n)
        .clamp(1, defaults::openai::MAX_N);
    let branch_logprobs = if want_logprobs {
        Some(logprobs_top_n)
    } else if effective_best_of > n {
        // best_of ranking needs each branch's total log-likelihood even when
        // the caller didn't request logprobs — Some(0) is cheap (just the
        // sampled token, no top-k alternatives).
        Some(0)
    } else {
        None
    };

    let mut branch_rxs: Vec<tokio::sync::mpsc::UnboundedReceiver<Token>> =
        Vec::with_capacity(effective_best_of as usize);
    // Branch 0 prefills the shared prompt; the rest copy its KV instead of
    // recomputing the identical thing N times. The scheduler holds them back until
    // branch 0 is decoding and falls back to a normal prefill if it never gets there,
    // so this is a speed optimisation with no correctness edge of its own.
    let mut fork_parent: Option<u64> = None;
    for branch_idx in 0..effective_best_of {
        let mut branch_sampling = sampling.clone();
        branch_sampling.logprobs = branch_logprobs;
        // Perturb the seed per branch so n>1/best_of doesn't collapse to
        // identical output under an explicit seed (StdRng is a pure function
        // of seed + position, no per-request salt). Branch 0 keeps the
        // caller's literal seed, so a plain seed + n:1 request is unaffected.
        if branch_idx > 0 {
            branch_sampling.seed = branch_sampling
                .seed
                .map(|s| s.wrapping_add(u64::from(branch_idx)));
        }

        let req_id = entry.engine.next_request_id();
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Token>();
        let mut inference_req = InferenceRequest::new(
            req_id,
            prompt_tokens.clone(),
            max_tokens,
            branch_sampling,
            tx,
        );
        if let Some(chunks) = multimodal.clone() {
            inference_req = inference_req.with_multimodal(chunks);
        }
        if let Some(selection) = lora.clone() {
            inference_req = inference_req.with_lora(selection);
        }
        if let Some(parent) = fork_parent {
            inference_req = inference_req.with_fork_parent(parent);
        } else {
            fork_parent = Some(req_id); // branch 0 is everyone else's parent
        }
        // Admission failure is safe by construction: bailing here drops every
        // already-submitted branch's `rx` (and its paired `tx`), which is
        // exactly the existing client-disconnect preemption path
        // (`send().is_err()` → preempt, frees GPU memory) — no separate
        // cancellation mechanism needed.
        if let Err(e) = entry.engine.submit_request(inference_req) {
            entry
                .engine
                .record_rejection(crate::api::error::rejection_reason_label(&e));
            return AppError::from(e).into_response();
        }
        branch_rxs.push(rx);
    }

    let has_tools = eff_tools.is_some();
    let allow_parallel = req.parallel_tool_calls.unwrap_or(true);
    let tool_parser = resolve_tool_call_parser(
        &state.tool_call_parser,
        entry.engine.native_tool_call_format(),
    );

    tracing::info!(
        model = %req.model,
        stream = req.stream,
        prompt_tokens = prompt_tokens_len,
        thinking = enable_thinking,
        has_tools,
        "request"
    );

    if req.stream {
        if has_tools {
            // With tools: buffer every branch, parse each one's tool call,
            // emit index-tagged SSE deltas. (logprobs are not surfaced for
            // tool-call responses.) stream: true guarantees best_of == n
            // (enforced in ChatCompletionRequest::validate()), so every
            // submitted branch is returned — no ranking needed here.
            let branches: Vec<BufferedBranch> =
                futures::future::join_all(branch_rxs.iter_mut().map(buffer_tokens)).await;
            let n_branches = branches.len();
            // Shared prompt across branches — see the non-streaming path.
            let tool_cached_tokens = branches.first().map(|b| b.4).unwrap_or(0);

            let mut completion_tokens_total = 0u32;
            let parsed: Vec<(String, Option<Vec<ToolCall>>, String)> = branches
                .into_iter()
                .map(
                    |(full_content, completion_tokens, stop_reason, _lp, _cached)| {
                        completion_tokens_total += completion_tokens;
                        let (content, mut tool_calls) =
                            parse_tool_call(&full_content, eff_tools, tool_parser);
                        if !allow_parallel {
                            if let Some(ref mut calls) = tool_calls {
                                calls.truncate(1);
                            }
                        }
                        let finish_reason = if tool_calls.is_some() {
                            "tool_calls".to_string()
                        } else {
                            stop_reason
                                .as_ref()
                                .map(finish_reason_str)
                                .unwrap_or("stop")
                                .to_string()
                        };
                        (content, tool_calls, finish_reason)
                    },
                )
                .collect();

            tracing::info!(
                model = %req.model,
                stream = true,
                prompt_tokens = prompt_tokens_len as u32,
                completion_tokens = completion_tokens_total,
                n = n_branches,
                duration_ms = start.elapsed().as_millis() as u64,
                "done"
            );

            let id_c = id.clone();
            let model_c = req.model.clone();
            let prompt_tokens_u32 = prompt_tokens_len as u32;
            let stream = async_stream::stream! {
                for (branch_idx, (content, tool_calls, finish_reason)) in parsed.into_iter().enumerate() {
                    // Chunk 1: role announcement
                    let first = ChatCompletionChunk {
                        id: id_c.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_c.clone(),
                        choices: vec![ChatCompletionChunkChoice {
                            index: branch_idx as u32,
                            delta: ChatMessageDelta {
                                role: Some("assistant".to_string()),
                                content: None,
                                tool_calls: None,
                            },
                            finish_reason: None,
                            logprobs: None,
                        }],
                        usage: None,
                        system_fingerprint: None,
                    };
                    yield Ok::<_, std::convert::Infallible>(
                        Event::default().json_data(first).unwrap_or_else(|_| Event::default().data(""))
                    );
                    tokio::task::yield_now().await;

                    // Chunk 2: tool_calls delta or content, + summed usage on
                    // the very last chunk of the very last branch.
                    let delta = if let Some(ref tcs) = tool_calls {
                        ChatMessageDelta {
                            role: None,
                            content: None,
                            tool_calls: Some(
                                tcs.iter()
                                    .enumerate()
                                    .map(|(i, tc)| ToolCallDelta {
                                        index: i as u32,
                                        id: Some(tc.id.clone()),
                                        call_type: Some("function".to_string()),
                                        function: ToolCallFunctionDelta {
                                            name: Some(tc.function.name.clone()),
                                            arguments: Some(tc.function.arguments.clone()),
                                        },
                                    })
                                    .collect(),
                            ),
                        }
                    } else {
                        ChatMessageDelta {
                            role: None,
                            content: Some(content),
                            tool_calls: None,
                        }
                    };

                    let is_last = branch_idx + 1 == n_branches;
                    let final_chunk = ChatCompletionChunk {
                        id: id_c.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model_c.clone(),
                        choices: vec![ChatCompletionChunkChoice {
                            index: branch_idx as u32,
                            delta,
                            finish_reason: Some(finish_reason),
                            logprobs: None,
                        }],
                        usage: (is_last && include_usage).then(|| {
                            Usage::new(
                                prompt_tokens_u32,
                                completion_tokens_total,
                                tool_cached_tokens,
                            )
                        }),
                        system_fingerprint: None,
                    };
                    yield Ok::<_, std::convert::Infallible>(
                        Event::default().json_data(final_chunk).unwrap_or_else(|_| Event::default().data(""))
                    );
                }

                // OpenAI-compatible stream terminator
                yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
            };

            return Sse::new(stream)
                .keep_alive(KeepAlive::default())
                .into_response();
        }

        // Normal streaming path (no tools). stream: true guarantees best_of
        // == n (validated above), so every branch in `branch_rxs` is
        // returned — no ranking needed. Branches are merged into one SSE
        // stream via StreamMap, tagged by branch index, in arrival order.
        let log_model = req.model.clone();
        let log_prompt = prompt_tokens_len;
        let n_branches = branch_rxs.len();
        let stream = async_stream::stream! {
            use tokio_stream::StreamExt as _;

            let mut merged: tokio_stream::StreamMap<
                usize,
                tokio_stream::wrappers::UnboundedReceiverStream<Token>,
            > = tokio_stream::StreamMap::new();
            for (branch_idx, rx) in branch_rxs.into_iter().enumerate() {
                merged.insert(branch_idx, tokio_stream::wrappers::UnboundedReceiverStream::new(rx));
            }

            let mut first_chunk = vec![true; n_branches];
            let mut completion_tokens = vec![0u32; n_branches];
            let mut done_count = 0usize;
            // Carried on every token; the prompt is shared, so any branch reports the
            // same value and the last write wins harmlessly.
            let mut stream_cached_tokens;
            while let Some((branch_idx, token)) = merged.next().await {
                stream_cached_tokens = token.cached_tokens;
                let is_done = token.stop_reason.is_some();
                let finish_reason = token.stop_reason.as_ref().map(finish_reason_str).map(str::to_string);
                completion_tokens[branch_idx] += 1;
                if is_done {
                    done_count += 1;
                    tracing::info!(
                        model = %log_model,
                        stream = true,
                        prompt_tokens = log_prompt,
                        completion_tokens = completion_tokens[branch_idx],
                        branch = branch_idx,
                        duration_ms = start.elapsed().as_millis() as u64,
                        finish_reason = %finish_reason.as_deref().unwrap_or("stop"),
                        "done"
                    );
                }

                // Summed usage attaches once, on the very last chunk overall.
                let usage = (is_done && done_count == n_branches && include_usage).then(|| {
                    Usage::new(
                        log_prompt as u32,
                        completion_tokens.iter().sum(),
                        stream_cached_tokens,
                    )
                });

                // First chunk of each branch carries role; subsequent chunks carry content.
                let (role, content) = if first_chunk[branch_idx] {
                    first_chunk[branch_idx] = false;
                    (Some("assistant".to_string()), Some(token.text.clone()))
                } else {
                    (None, Some(token.text.clone()))
                };

                let chunk_logprobs = token
                    .logprob
                    .map(|l| ChatLogprobs { content: vec![l.into()] });
                let chunk = ChatCompletionChunk {
                    id: id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created,
                    model: req.model.clone(),
                    choices: vec![ChatCompletionChunkChoice {
                        index: branch_idx as u32,
                        delta: ChatMessageDelta {
                            role,
                            content,
                            tool_calls: None,
                        },
                        finish_reason,
                        logprobs: chunk_logprobs,
                    }],
                    usage,
                    system_fingerprint: None,
                };
                let event = Event::default()
                    .json_data(chunk)
                    .unwrap_or_else(|_| Event::default().data(""));
                tokio::task::yield_now().await;
                yield Ok::<_, std::convert::Infallible>(event);
                if done_count == n_branches {
                    break;
                }
            }
            // OpenAI-compatible stream terminator
            yield Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"));
        };
        Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        let mut branches: Vec<BufferedBranch> =
            futures::future::join_all(branch_rxs.iter_mut().map(buffer_tokens)).await;
        // Every branch shares one prompt, so its cached-prefix count is a property of
        // the request, not of a branch. Branch 0 is the canonical one (with n == 1,
        // the overwhelmingly common case, it is the only one).
        let cached_tokens = branches.first().map(|b| b.4).unwrap_or(0);

        // best_of > n: rank by total log-likelihood (sum of per-token
        // logprobs) and keep only the top n. A no-op when best_of == n.
        if branches.len() > n as usize {
            let scored = branches
                .into_iter()
                .map(|b| (b.3.iter().map(|e| e.logprob).sum::<f32>(), b))
                .collect();
            branches = select_best_of(scored, n as usize);
        }

        let mut completion_tokens_total = 0u32;
        let mut choices = Vec::with_capacity(branches.len());
        for (index, (full_content, completion_tokens, stop_reason, logprob_entries, _cached)) in
            branches.into_iter().enumerate()
        {
            completion_tokens_total += completion_tokens;
            let stop_str = stop_reason
                .as_ref()
                .map(finish_reason_str)
                .unwrap_or("stop")
                .to_string();

            let (content, mut tool_calls) = if has_tools {
                parse_tool_call(&full_content, eff_tools, tool_parser)
            } else {
                (full_content, None)
            };

            // Enforce parallel_tool_calls: false
            if !allow_parallel {
                if let Some(ref mut calls) = tool_calls {
                    calls.truncate(1);
                }
            }

            let finish_reason = if tool_calls.is_some() {
                "tool_calls".to_string()
            } else {
                stop_str
            };

            choices.push(ChatCompletionChoice {
                index: index as u32,
                message: ChatMessageResponse {
                    role: "assistant".to_string(),
                    content: if tool_calls.is_some() {
                        None
                    } else {
                        Some(content)
                    },
                    tool_calls,
                },
                finish_reason: Some(finish_reason),
                logprobs: if want_logprobs {
                    Some(ChatLogprobs {
                        content: logprob_entries,
                    })
                } else {
                    None
                },
            });
        }

        tracing::info!(
            model = %req.model,
            stream = false,
            prompt_tokens = prompt_tokens_len as u32,
            completion_tokens = completion_tokens_total,
            n = choices.len(),
            duration_ms = start.elapsed().as_millis() as u64,
            "done"
        );

        Json(ChatCompletionResponse {
            id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
            choices,
            usage: Some(Usage::new(
                prompt_tokens_len as u32,
                completion_tokens_total,
                cached_tokens,
            )),
            system_fingerprint: None,
        })
        .into_response()
    }
}

/// Rank `(score, item)` pairs by score descending and keep the first `n`. Used
/// for `best_of`: score is a branch's total log-likelihood (sum of per-token
/// logprobs) — higher is a more confident completion.
fn select_best_of<T>(mut candidates: Vec<(f32, T)>, n: usize) -> Vec<T> {
    candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
    candidates.truncate(n);
    candidates.into_iter().map(|(_, item)| item).collect()
}

/// Buffer all tokens from the receiver into a [`BufferedBranch`].
/// `logprobs` is empty unless the request asked for them.
async fn buffer_tokens(rx: &mut tokio::sync::mpsc::UnboundedReceiver<Token>) -> BufferedBranch {
    let mut text = String::new();
    let mut count = 0u32;
    let mut stop_reason = None;
    let mut logprobs = Vec::new();
    let mut cached_tokens = 0u32;
    while let Some(token) = rx.recv().await {
        text.push_str(&token.text);
        cached_tokens = token.cached_tokens;
        count += 1;
        if let Some(lp) = token.logprob {
            logprobs.push(lp.into());
        }
        if token.stop_reason.is_some() {
            stop_reason = token.stop_reason;
            break;
        }
    }
    (text, count, stop_reason, logprobs, cached_tokens)
}

#[cfg(test)]
mod tests {
    use super::select_best_of;
    use crate::api::test_helpers::*;

    /// Parse SSE body bytes into a list of JSON values from "data: " lines.
    /// Skips the "[DONE]" sentinel (mirrors `tests/integration.rs`'s helper).
    fn parse_sse_chunks(bytes: &[u8]) -> Vec<serde_json::Value> {
        let body = std::str::from_utf8(bytes).expect("SSE body is not UTF-8");
        body.lines()
            .filter(|l| l.starts_with("data: "))
            .filter_map(|l| {
                let payload = &l["data: ".len()..];
                if payload == "[DONE]" {
                    return None;
                }
                serde_json::from_str(payload).ok()
            })
            .collect()
    }

    #[test]
    fn select_best_of_keeps_highest_scored() {
        let candidates = vec![(-5.0, "worst"), (-1.0, "best"), (-3.0, "middle")];
        assert_eq!(select_best_of(candidates, 2), vec!["best", "middle"]);
    }

    #[test]
    fn select_best_of_n_equal_len_is_noop_order_by_score() {
        let candidates = vec![(-2.0, "a"), (-1.0, "b")];
        assert_eq!(select_best_of(candidates, 2), vec!["b", "a"]);
    }

    #[test]
    fn select_best_of_n_larger_than_candidates_returns_all() {
        let candidates = vec![(-1.0, "only")];
        assert_eq!(select_best_of(candidates, 5), vec!["only"]);
    }

    #[tokio::test]
    async fn test_n_returns_multiple_choices() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 4,
            "n": 3
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let choices = v["choices"].as_array().unwrap();
        assert_eq!(choices.len(), 3);
        let mut indices: Vec<u64> = choices
            .iter()
            .map(|c| c["index"].as_u64().unwrap())
            .collect();
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1, 2]);
        for c in choices {
            assert!(!c["message"]["content"].as_str().unwrap().is_empty());
        }
        let completion_tokens = v["usage"]["completion_tokens"].as_u64().unwrap();
        assert!(
            completion_tokens >= 3,
            "expected tokens summed across 3 branches, got {completion_tokens}"
        );
    }

    #[tokio::test]
    async fn test_best_of_greater_than_n_returns_n_choices() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 4,
            "n": 1,
            "best_of": 3
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 1);
        assert_eq!(v["choices"][0]["index"], 0);
    }

    #[tokio::test]
    async fn test_best_of_greater_than_n_with_stream_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "n": 1,
            "best_of": 3
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn test_n_over_max_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "n": 9
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn test_n_streaming_returns_interleaved_indexed_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": true,
            "max_tokens": 4,
            "n": 2
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let chunks = parse_sse_chunks(&bytes);
        assert!(!chunks.is_empty());

        let indices_seen: std::collections::HashSet<u64> = chunks
            .iter()
            .map(|c| c["choices"][0]["index"].as_u64().unwrap())
            .collect();
        assert_eq!(indices_seen, std::collections::HashSet::from([0, 1]));

        // Each branch must terminate with a non-null finish_reason.
        for branch_idx in [0u64, 1u64] {
            let finished = chunks.iter().any(|c| {
                c["choices"][0]["index"].as_u64() == Some(branch_idx)
                    && !c["choices"][0]["finish_reason"].is_null()
            });
            assert!(finished, "branch {branch_idx} never sent a finish_reason");
        }

        // Usage (summed across both branches) appears exactly once, on the last chunk.
        let usage_chunks: Vec<_> = chunks.iter().filter(|c| !c["usage"].is_null()).collect();
        assert_eq!(usage_chunks.len(), 1);
        assert!(
            usage_chunks[0]["usage"]["completion_tokens"]
                .as_u64()
                .unwrap()
                >= 2
        );
    }

    #[tokio::test]
    async fn test_chat_completions_non_streaming() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false,
            "max_tokens": 4
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
        assert!(!v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty());
        assert_eq!(v["choices"][0]["finish_reason"].as_str().unwrap(), "stop");
        // system_fingerprint present as null
        assert!(v["system_fingerprint"].is_null());
    }

    #[tokio::test]
    async fn test_unknown_model_returns_404() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "nonexistent-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 404);
        // Error response matches OpenAI format
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(v["error"]["message"].is_string());
        assert!(v["error"]["type"].is_string());
    }

    #[tokio::test]
    async fn test_chat_completions_json_mode() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "response_format": {"type": "json_object"}
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_chat_completions_json_schema() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "stream": false,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": true, "schema": {"type": "object"}}
            }
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_chat_completions_system_prompt_injected() {
        use crate::api::router::router;

        let dir = tempfile::tempdir().unwrap();
        let (registry, _entry) = make_test_registry("stub", dir.path());
        let app = router(
            registry,
            "stub".to_string(),
            Some("You are a helpful assistant.".to_string()),
            0,
            dir.path().to_path_buf(),
            None,
            None,
            "auto".to_string(),
            -1,
        );
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_chat_with_tool_result_in_history() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny, 22°C"}
            ],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"].as_str().unwrap(), "chat.completion");
    }

    #[tokio::test]
    async fn test_tool_choice_none_bypasses_tools() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "tools": [{"type":"function","function":{"name":"f","parameters":{}}}],
            "tool_choice": "none"
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
        let bytes = body_bytes(resp).await;
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        // finish_reason should not be "tool_calls" since tool_choice = "none"
        let fr = v["choices"][0]["finish_reason"].as_str().unwrap();
        assert_ne!(fr, "tool_calls");
    }

    #[tokio::test]
    async fn test_multimodal_content_array() {
        // Content as array of blocks should be accepted and text extracted.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": "Hello from array"}]
            }],
            "stream": false
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn test_compat_fields_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "parallel_tool_calls": false,
            "stream_options": {"include_usage": true},
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "user": "test-user"
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn usage_reports_cached_tokens_after_a_kv_reuse_hit() {
        // The second identical request inherits the first's resident sequence, so
        // `usage.prompt_tokens_details.cached_tokens` is how a client sees that.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let body = || {
            serde_json::json!({
                "model": "stub",
                "messages": [{"role": "user", "content": "A reasonably long prompt to reuse"}],
                "stream": false,
                "max_tokens": 3,
            })
        };

        // First request: nothing resident yet, so the field is omitted entirely
        // (rather than reported as a misleading zero).
        let v1: serde_json::Value = serde_json::from_slice(
            &body_bytes(post_json(make_router(&state), "/v1/chat/completions", body()).await).await,
        )
        .unwrap();
        assert!(
            v1["usage"]["prompt_tokens_details"].is_null(),
            "nothing should be cached on the first request: {v1}"
        );

        // Second, identical request: the sequence parked by the first is reused.
        let v2: serde_json::Value = serde_json::from_slice(
            &body_bytes(post_json(make_router(&state), "/v1/chat/completions", body()).await).await,
        )
        .unwrap();
        let cached = v2["usage"]["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap_or(0);
        assert!(cached > 0, "the repeated prompt should report reuse: {v2}");
        assert!(
            cached < v2["usage"]["prompt_tokens"].as_u64().unwrap(),
            "one token is always left to decode ([TAG_PROMPT_LOGITS]): {v2}"
        );
    }

    #[tokio::test]
    async fn raw_gbnf_grammar_is_accepted() {
        // The engine has had GBNF support since 0.14; until now the only door was
        // `response_format`, which can only describe JSON.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "max_tokens": 4,
            "grammar": "root ::= \"yes\" | \"no\""
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }

    #[tokio::test]
    async fn grammar_and_response_format_together_are_rejected() {
        // Two conflicting constraints is a caller bug; silently picking one hides it.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "grammar": "root ::= \"x\"",
            "response_format": {"type": "json_object"}
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 400);
    }

    #[tokio::test]
    async fn stream_options_include_usage_is_honoured() {
        // Regression: this field was parsed and ignored — usage always rode the last
        // chunk regardless.
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());

        let chunks_for = |include: Option<bool>| {
            let state = state.clone();
            async move {
                let mut body = serde_json::json!({
                    "model": "stub",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": true,
                    "max_tokens": 3,
                });
                if let Some(v) = include {
                    body["stream_options"] = serde_json::json!({ "include_usage": v });
                }
                let resp = post_json(make_router(&state), "/v1/chat/completions", body).await;
                assert_eq!(resp.status(), 200);
                String::from_utf8(body_bytes(resp).await.to_vec()).unwrap()
            }
        };

        assert!(
            chunks_for(Some(true)).await.contains("\"usage\""),
            "include_usage:true must attach usage"
        );
        assert!(
            !chunks_for(Some(false)).await.contains("\"usage\":{"),
            "include_usage:false must suppress usage"
        );
        // Absent stream_options keeps fox's historical always-attach behaviour.
        assert!(
            chunks_for(None).await.contains("\"usage\""),
            "omitting stream_options must not change existing behaviour"
        );
    }

    #[tokio::test]
    async fn test_max_completion_tokens_alias() {
        let dir = tempfile::tempdir().unwrap();
        let (state, _entry) = make_test_state("stub", dir.path());
        let app = make_router(&state);
        let body = serde_json::json!({
            "model": "stub",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": false,
            "max_completion_tokens": 4
        });
        let resp = post_json(app, "/v1/chat/completions", body).await;
        assert_eq!(resp.status(), 200);
    }
}
