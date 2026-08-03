// Reusable streaming primitives for NDJSON and SSE responses.

use axum::http::header;
use bytes::Bytes;
use serde::Serialize;
use std::convert::Infallible;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::scheduler::{StopReason, Token};

/// Returns the current time as a minimal RFC 3339 UTC string.
pub fn now_rfc3339() -> String {
    let s = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let sec = s % 60;
    let min = (s / 60) % 60;
    let hour = (s / 3600) % 24;
    let days = s / 86400;
    let year = 1970u64 + days / 365;
    let doy = days % 365;
    let month = doy / 30 + 1;
    let day = doy % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z")
}

/// Convert a `StopReason` to the Ollama `done_reason` string.
pub fn ollama_done_reason(reason: &Option<StopReason>) -> String {
    match reason {
        Some(StopReason::Length) => "length".to_string(),
        Some(StopReason::EngineError) => "error".to_string(),
        _ => "stop".to_string(),
    }
}

/// Convert a `StopReason` to the OpenAI `finish_reason` string.
///
/// `"error"` is a fox-specific extension (not part of the OpenAI spec) for a request
/// that failed mid-generation (e.g. a `llama_decode` failure) — there is no existing
/// standard finish_reason for that, and returning `"stop"`/`"length"` would misreport
/// a crash as a normal completion.
pub fn finish_reason_str(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Eos => "stop",
        StopReason::Length => "length",
        StopReason::Preempt => "stop",
        StopReason::StopSequence => "stop",
        StopReason::EngineError => "error",
    }
}

/// Wrap a bytes stream in an `application/x-ndjson` HTTP response.
pub fn ndjson_response(
    stream: impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static,
) -> axum::response::Response {
    axum::response::Response::builder()
        .status(200)
        .header(header::CONTENT_TYPE, "application/x-ndjson")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}

/// Wall-clock breakdown of one generation, in nanoseconds — the Ollama API's
/// `total_duration` / `load_duration` / `prompt_eval_duration` / `eval_duration`.
///
/// fox previously reported `load_duration` and `prompt_eval_duration` as a literal
/// `0`, and `total_duration` and `eval_duration` as the same wall clock, so a client
/// could not tell prefill cost from decode cost at all. These are measured:
///
/// * `load_ns` — time spent getting the model resident for this request. Naturally
///   ~0 when it was already loaded, which is the honest answer.
/// * `prompt_eval_ns` — submission to first token. This includes scheduler queueing,
///   not just the prefill compute; on a busy server that queue wait is real latency
///   the client paid, and fox has no cheaper place to separate the two.
/// * `eval_ns` — first token to last, i.e. pure decode.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenTimings {
    pub total_ns: u64,
    pub load_ns: u64,
    pub prompt_eval_ns: u64,
    pub eval_ns: u64,
}

impl GenTimings {
    /// Build from the pieces the handlers can actually observe.
    pub fn new(load_ns: u64, prompt_eval_ns: u64, total_since_submit_ns: u64) -> Self {
        Self {
            total_ns: load_ns + total_since_submit_ns,
            load_ns,
            prompt_eval_ns,
            eval_ns: total_since_submit_ns.saturating_sub(prompt_eval_ns),
        }
    }
}

/// Build an NDJSON bytes stream from a token receiver.
///
/// `make_chunk` receives `(token, eval_count, timings)` and returns a serialisable
/// chunk. The stream yields one newline-terminated JSON line per token and
/// terminates when a done token is received. `load_ns` is threaded in by the caller
/// since model loading happens before the stream exists.
pub fn ndjson_stream<F, T>(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Token>,
    load_ns: u64,
    make_chunk: F,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static
where
    F: Fn(Token, u32, GenTimings) -> T + Send + 'static,
    T: Serialize + Send,
{
    let start = Instant::now();
    async_stream::stream! {
        let mut eval_count: u32 = 0;
        let mut prompt_eval_ns: u64 = 0;
        while let Some(token) = rx.recv().await {
            let is_done = token.stop_reason.is_some();
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            // The first token to arrive is what prefill produced; everything after
            // it is decode.
            if eval_count == 0 {
                prompt_eval_ns = elapsed_ns;
            }
            let timings = GenTimings::new(load_ns, prompt_eval_ns, elapsed_ns);
            let chunk = make_chunk(token, eval_count, timings);
            eval_count += 1;
            let mut line = serde_json::to_string(&chunk).unwrap_or_default();
            line.push('\n');
            yield Ok::<_, Infallible>(Bytes::from(line.into_bytes()));
            if is_done {
                break;
            }
        }
    }
}

/// Collect all tokens from `rx` into `(full_text, token_count, stop_reason)`.
pub async fn collect_tokens(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Token>,
) -> (String, u32, Option<StopReason>) {
    let (text, count, stop_reason, _) = collect_tokens_timed(rx).await;
    (text, count, stop_reason)
}

/// Like [`collect_tokens`], but also reports how long the first token took to
/// arrive (nanoseconds from this call) — the prefill half of the split.
pub async fn collect_tokens_timed(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<Token>,
) -> (String, u32, Option<StopReason>, u64) {
    let start = Instant::now();
    let mut text = String::new();
    let mut count = 0u32;
    let mut stop_reason = None;
    let mut prompt_eval_ns = 0u64;
    while let Some(token) = rx.recv().await {
        if count == 0 {
            prompt_eval_ns = start.elapsed().as_nanos() as u64;
        }
        text.push_str(&token.text);
        count += 1;
        if token.stop_reason.is_some() {
            stop_reason = token.stop_reason;
            break;
        }
    }
    (text, count, stop_reason, prompt_eval_ns)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::StopReason;

    #[test]
    fn test_now_rfc3339_format() {
        let s = now_rfc3339();
        assert_eq!(s.len(), 20);
        assert!(s.ends_with('Z'));
        assert!(s.contains('T'));
        let parts: Vec<&str> = s.splitn(2, 'T').collect();
        assert_eq!(parts[0].split('-').count(), 3);
    }

    #[test]
    fn test_done_reason_eos() {
        assert_eq!(ollama_done_reason(&Some(StopReason::Eos)), "stop");
    }

    #[test]
    fn test_done_reason_length() {
        assert_eq!(ollama_done_reason(&Some(StopReason::Length)), "length");
    }

    #[test]
    fn test_done_reason_stop_sequence() {
        assert_eq!(ollama_done_reason(&Some(StopReason::StopSequence)), "stop");
    }

    #[test]
    fn test_done_reason_none() {
        assert_eq!(ollama_done_reason(&None), "stop");
    }
}
