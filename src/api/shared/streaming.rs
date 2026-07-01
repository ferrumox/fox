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
        _ => "stop".to_string(),
    }
}

/// Convert a `StopReason` to the OpenAI `finish_reason` string.
pub fn finish_reason_str(reason: &StopReason) -> &'static str {
    match reason {
        StopReason::Eos => "stop",
        StopReason::Length => "length",
        StopReason::Preempt => "stop",
        StopReason::StopSequence => "stop",
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

/// Build an NDJSON bytes stream from a token receiver.
///
/// `make_chunk` receives `(token, eval_count, elapsed_ns)` and returns a
/// serialisable chunk. The stream yields one newline-terminated JSON line per
/// token and terminates when a done token is received.
pub fn ndjson_stream<F, T>(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<Token>,
    make_chunk: F,
) -> impl futures::Stream<Item = Result<Bytes, Infallible>> + Send + 'static
where
    F: Fn(Token, u32, u64) -> T + Send + 'static,
    T: Serialize + Send,
{
    let start = Instant::now();
    async_stream::stream! {
        let mut eval_count: u32 = 0;
        while let Some(token) = rx.recv().await {
            let is_done = token.stop_reason.is_some();
            let elapsed_ns = start.elapsed().as_nanos() as u64;
            let chunk = make_chunk(token, eval_count, elapsed_ns);
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
    let mut text = String::new();
    let mut count = 0u32;
    let mut stop_reason = None;
    while let Some(token) = rx.recv().await {
        text.push_str(&token.text);
        count += 1;
        if token.stop_reason.is_some() {
            stop_reason = token.stop_reason;
            break;
        }
    }
    (text, count, stop_reason)
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
