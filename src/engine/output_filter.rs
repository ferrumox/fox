// Output filtering: thinking-block suppression, control-token holdback,
// and user-supplied stop-sequence detection.
//
// Extracted from engine/mod.rs so the logic can be tested without the full
// inference engine. `InferenceEngine` uses `PerRequestState` and the
// free functions directly through `use super::output_filter::*`.

/// Special token patterns that mark end-of-turn or other control sequences.
/// These are detected in `check_stop_sequences` using a rolling buffer so that
/// patterns spanning multiple tokens (e.g. `<`, `|`, `im_end`, `|`, `>`) are
/// also caught.  Never emitted to the user.
pub(super) const CONTROL_TOKEN_PATTERNS: &[&str] = &[
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<|endofthought|>",
];

// ---------------------------------------------------------------------------
// Per-request output state
// ---------------------------------------------------------------------------

/// Per-request mutable state for output processing.
#[derive(Default)]
pub(super) struct PerRequestState {
    /// True while we are inside a `<think>…</think>` block.
    pub(super) in_thinking: bool,
    /// When true the `<think>…</think>` block is forwarded to the caller instead
    /// of being silently discarded.  Set from `SamplingParams::show_thinking`.
    pub(super) show_thinking: bool,
    /// Text that has passed the thinking filter but is being held back until
    /// we know it is not the start of a control-token pattern (e.g. `<|im_end|>`
    /// arriving across several BPE tokens: `<`, `|`, `im_end`, `|`, `>`).
    pub(super) pending_output: String,
    /// Rolling suffix of recently emitted text (length ≤ 2 × max_stop_len).
    /// Used to detect *user-supplied* stop strings that span multiple tokens.
    pub(super) text_buffer: String,
    /// Model-native stop token strings (EOS/EOT text forms) treated exactly like
    /// `CONTROL_TOKEN_PATTERNS`: held back in `pending_output` until the full
    /// pattern is assembled, then suppressed and generation stops.
    pub(super) model_control_patterns: Vec<String>,
}

// ---------------------------------------------------------------------------
// Pure helpers (no self, no lock)
// ---------------------------------------------------------------------------

/// Apply output filtering rules.
///
/// Returns `(text_to_pass_downstream, control_stop)`.
/// * `text_to_pass_downstream` — safe visible text (empty while thinking or holding back
///   a partial control-token prefix).
/// * `control_stop` — `true` when a complete control-token pattern (`<|im_end|>` etc.)
///   was detected, meaning generation should stop.
///
/// **Why two stages?**
/// Models like Qwen3.5 often generate `<|im_end|>` as 5–6 separate BPE tokens
/// (`<`, `|`, `im`, `_end`, `|`, `>`).  A single-token `contains("<|")` check
/// misses this.  We therefore buffer the output in `state.pending_output` and
/// only flush text that cannot be the start of a control pattern.
pub(super) fn apply_output_filter(state: &mut PerRequestState, raw: &str) -> (String, bool) {
    if raw.is_empty() {
        return (String::new(), false);
    }

    // Build combined pattern list: static patterns + model-native stop token strings.
    // SAFETY: the Vec<&str> borrows from `state.model_control_patterns` which lives for
    // the duration of this call.  We reborrow `state` fields individually below to
    // avoid a borrow-checker conflict with `state.pending_output`.
    let patterns = all_control_patterns(&state.model_control_patterns);

    // 1. Enter <think> block (usually a single special token like 248068 for Qwen3.5).
    if raw.contains("<think>") {
        state.in_thinking = true;
        if state.show_thinking {
            // Emit the <think> tag so the user can see when reasoning starts.
            return (raw.to_string(), false);
        }
        return (String::new(), false);
    }

    // 2. Exit <think> block; text *after* the closing tag goes to the pending buffer.
    if raw.contains("</think>") {
        state.in_thinking = false;
        if state.show_thinking {
            // Emit the closing tag, then flush whatever was pending.
            let after_tag = raw
                .find("</think>")
                .map(|i| i + "</think>".len())
                .unwrap_or(raw.len());
            let mut out = raw[..after_tag].to_string();
            // Any text after </think> in the same token also needs to be flushed.
            if after_tag < raw.len() {
                state.pending_output.push_str(&raw[after_tag..]);
                let (rest, stop) = flush_pending_output(&mut state.pending_output, &patterns);
                out.push_str(&rest);
                if stop {
                    return (out, true);
                }
            }
            return (out, false);
        }
        // Normal mode: discard the tag, keep text after it.
        if let Some(idx) = raw.find("</think>") {
            let after = idx + "</think>".len();
            if after < raw.len() {
                state.pending_output.push_str(&raw[after..]);
            }
        }
        return flush_pending_output(&mut state.pending_output, &patterns);
    }

    // 3. Inside a thinking block.
    if state.in_thinking {
        if state.show_thinking {
            // Emit thinking tokens directly (no holdback needed — control patterns
            // like <|im_end|> should not appear inside a thinking block).
            return (raw.to_string(), false);
        }
        return (String::new(), false);
    }

    // 4. Normal text: push through the pending buffer; hold back any partial
    //    control-token prefix (e.g. `<` that could be the start of `<|im_end|>`).
    state.pending_output.push_str(raw);
    flush_pending_output(&mut state.pending_output, &patterns)
}

/// Flush as much of `pending` as is safe.
///
/// 1. If `pending` contains a complete control-token pattern, emit everything
///    *before* the pattern and signal stop (the pattern itself is discarded).
/// 2. Otherwise, hold back the longest suffix that is a strict prefix of any
///    control pattern (could be the start of `<|im_end|>` etc.).
///
/// `patterns` is the combined list of static `CONTROL_TOKEN_PATTERNS` plus any
/// model-native stop token strings for the current request.
pub(super) fn flush_pending_output(pending: &mut String, patterns: &[&str]) -> (String, bool) {
    // Check for complete control-token patterns.
    for &pat in patterns {
        if let Some(idx) = pending.find(pat) {
            let emit = pending[..idx].to_string();
            pending.clear();
            return (emit, true); // stop generation
        }
    }

    // Find the earliest `<` that could be the start of a control pattern.
    let holdback_start = find_holdback_start(pending, patterns);
    let emit = pending[..holdback_start].to_string();
    *pending = pending[holdback_start..].to_string();
    (emit, false)
}

/// Returns the byte offset of the first `<` in `text` from which a control-token
/// pattern *could* still begin (i.e. some pattern starts with the suffix
/// `text[offset..]`).  Returns `text.len()` when nothing needs to be held back.
fn find_holdback_start(text: &str, patterns: &[&str]) -> usize {
    for (i, c) in text.char_indices() {
        if c != '<' {
            continue;
        }
        let suffix = &text[i..];
        if patterns.iter().any(|p| p.starts_with(suffix)) {
            return i;
        }
    }
    text.len()
}

/// Build the combined pattern list for a request: static CONTROL_TOKEN_PATTERNS
/// plus any model-native stop token strings stored in state.
pub(super) fn all_control_patterns(model_pats: &[String]) -> Vec<&str> {
    let mut v: Vec<&str> = CONTROL_TOKEN_PATTERNS.to_vec();
    for p in model_pats {
        if !v.contains(&p.as_str()) {
            v.push(p.as_str());
        }
    }
    v
}

/// Check whether the rolling text buffer (extended with `new_text`) ends with
/// any of the user-supplied stop strings.
///
/// Returns `(text_to_emit, was_stopped)`. When stopped, `text_to_emit` is
/// the prefix of `new_text` that appears *before* the stop string; the stop
/// string itself is NOT emitted (OpenAI spec behaviour).
///
/// Note: built-in control-token patterns (`<|im_end|>` etc.) are already handled
/// upstream in `apply_output_filter` / `flush_pending_output`.  This function
/// only deals with user-supplied stop strings.
pub(super) fn check_stop_sequences(
    state: &mut PerRequestState,
    new_text: String,
    stop: &Option<Vec<String>>,
) -> (String, bool) {
    let stops = match stop.as_deref() {
        Some(s) if !s.is_empty() => s,
        _ => {
            // No stop strings — just maintain the buffer for future calls.
            state.text_buffer.push_str(&new_text);
            trim_text_buffer(&mut state.text_buffer, 0);
            return (new_text, false);
        }
    };

    let max_stop_len: usize = stops.iter().map(|s| s.len()).max().unwrap_or(0);

    // Extend the buffer with the new token text.
    state.text_buffer.push_str(&new_text);

    // Check every stop string.
    for stop_str in stops {
        if stop_str.is_empty() {
            continue;
        }
        if state.text_buffer.ends_with(stop_str.as_str()) {
            // Find how much of `new_text` to emit (the part before the stop string).
            let buf_len = state.text_buffer.len();
            let stop_start_in_buf = buf_len - stop_str.len();
            // `new_text` starts at `buf_len - new_text.len()` within the buffer.
            let text_start_in_buf = buf_len.saturating_sub(new_text.len());

            let emit = if stop_start_in_buf >= text_start_in_buf {
                let offset = stop_start_in_buf - text_start_in_buf;
                new_text[..offset].to_string()
            } else {
                // Stop string began in a previously-emitted token — emit nothing.
                String::new()
            };

            state.text_buffer.clear();
            return (emit, true);
        }
    }

    // No match — trim the buffer to avoid unbounded growth.
    trim_text_buffer(&mut state.text_buffer, max_stop_len);
    (new_text, false)
}

/// Keep only the trailing `max_stop_len` characters of the buffer (aligned to a char boundary).
pub(super) fn trim_text_buffer(buf: &mut String, max_stop_len: usize) {
    let keep = (max_stop_len * 2).max(128);
    if buf.len() > keep {
        let trim_byte = buf.len() - keep;
        // Walk forward to the next valid char boundary.
        let trim_at = buf
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| i >= trim_byte)
            .unwrap_or(trim_byte);
        *buf = buf[trim_at..].to_string();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(v: &[&str]) -> Option<Vec<String>> {
        Some(v.iter().map(|s| s.to_string()).collect())
    }

    // Helper: unwrap the text from apply_output_filter (ignores the control_stop bool)
    fn aof(state: &mut PerRequestState, raw: &str) -> String {
        apply_output_filter(state, raw).0
    }

    // --- apply_output_filter ---

    #[test]
    fn test_filter_think_block() {
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "<think>"), "");
        assert!(s.in_thinking);
        assert_eq!(aof(&mut s, "internal thought"), "");
        assert_eq!(aof(&mut s, "</think> hello"), " hello");
        assert!(!s.in_thinking);
        assert_eq!(aof(&mut s, " world"), " world");
    }

    #[test]
    fn test_filter_passthrough() {
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "hello"), "hello");
        assert_eq!(aof(&mut s, " world"), " world");
    }

    #[test]
    fn test_filter_control_single_token_stopped() {
        // Single-token <|im_end|> (e.g. EOS token decoded to its text form) must stop.
        let mut s = PerRequestState::default();
        let (text, stop) = apply_output_filter(&mut s, "<|im_end|>");
        assert_eq!(text, "");
        assert!(stop, "<|im_end|> single token must trigger control stop");
    }

    #[test]
    fn test_filter_control_multi_token_im_end() {
        // Qwen3.5 emits <|im_end|> as 5 separate BPE tokens.
        let mut s = PerRequestState::default();
        for tok in &["<", "|", "im", "_end", "|"] {
            let (text, stop) = apply_output_filter(&mut s, tok);
            assert_eq!(text, "", "partial token '{tok}' must be held back");
            assert!(!stop, "no stop on partial token '{tok}'");
        }
        let (text, stop) = apply_output_filter(&mut s, ">");
        assert_eq!(text, "", "closing '>' must not leak");
        assert!(stop, "closing '>' completes <|im_end|> → must stop");
    }

    #[test]
    fn test_filter_holdback_released_on_non_pattern() {
        // A lone `<` is held back until the next token confirms it is not a control pattern.
        let mut s = PerRequestState::default();
        let (t1, _) = apply_output_filter(&mut s, "<");
        assert_eq!(t1, "", "< must be held back");
        // `x` cannot extend any control pattern starting with `<` → release both.
        let (t2, stop) = apply_output_filter(&mut s, "x");
        assert_eq!(t2, "<x", "< and x must be released together");
        assert!(!stop);
    }

    #[test]
    fn test_filter_text_before_control_token_emitted() {
        // Normal text followed by <|im_end|>: text emitted, pattern stops generation.
        let mut s = PerRequestState::default();
        assert_eq!(aof(&mut s, "Hello!"), "Hello!");
        let (text, stop) = apply_output_filter(&mut s, "<|im_end|>");
        assert_eq!(text, "");
        assert!(stop);
    }

    // --- check_stop_sequences ---

    #[test]
    fn test_stop_no_stops_configured() {
        let mut s = PerRequestState::default();
        let (text, hit) = check_stop_sequences(&mut s, "hello world".to_string(), &None);
        assert_eq!(text, "hello world");
        assert!(!hit);
    }

    #[test]
    fn test_stop_exact_single_token() {
        let mut s = PerRequestState::default();
        let (text, hit) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["User:"]));
        assert_eq!(text, "", "stop string itself must not be emitted");
        assert!(hit);
    }

    #[test]
    fn test_stop_partial_current_token_emitted() {
        let mut s = PerRequestState::default();
        let (text, hit) =
            check_stop_sequences(&mut s, "Hello\nUser:".to_string(), &stops(&["\nUser:"]));
        assert_eq!(text, "Hello");
        assert!(hit);
    }

    #[test]
    fn test_stop_multi_token_span() {
        let mut s = PerRequestState::default();
        let (t1, h1) = check_stop_sequences(&mut s, "Hello\n".to_string(), &stops(&["\nUser:"]));
        assert_eq!(t1, "Hello\n");
        assert!(!h1);
        let (t2, h2) = check_stop_sequences(&mut s, "User:".to_string(), &stops(&["\nUser:"]));
        assert_eq!(t2, "");
        assert!(h2);
    }

    #[test]
    fn test_stop_multiple_candidates_first_wins() {
        let mut s = PerRequestState::default();
        let (text, hit) =
            check_stop_sequences(&mut s, "STOP".to_string(), &stops(&["STOP", "OTHER"]));
        assert_eq!(text, "");
        assert!(hit);
    }

    #[test]
    fn test_stop_no_match_passes_through() {
        let mut s = PerRequestState::default();
        let (t1, h1) = check_stop_sequences(&mut s, "hello ".to_string(), &stops(&["User:"]));
        assert_eq!(t1, "hello ");
        assert!(!h1);
        let (t2, h2) = check_stop_sequences(&mut s, "world".to_string(), &stops(&["User:"]));
        assert_eq!(t2, "world");
        assert!(!h2);
    }

    // --- trim_text_buffer ---

    #[test]
    fn test_trim_buffer_below_limit_unchanged() {
        let mut buf = "hello".to_string();
        trim_text_buffer(&mut buf, 10); // keep = 20, buf.len() = 5 < 20
        assert_eq!(buf, "hello");
    }

    #[test]
    fn test_trim_buffer_trims_to_trailing_content() {
        let mut buf = "a".repeat(200);
        trim_text_buffer(&mut buf, 10); // keep = 20
        assert!(buf.len() <= 200);
        assert!(buf.len() >= 20);
        // Verify suffix is preserved
        assert!(buf.chars().all(|c| c == 'a'));
    }
}
