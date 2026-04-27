//! Bundled [`ToolHandler`] implementations registered by [`super::default_board`].
//!
//! Keep these intentionally small and dependency-free — operators can always
//! register richer tools at startup. The two builtins here cover the two
//! most common chat-tool needs:
//!
//! * [`json_extract::JsonExtract`] — pure-Rust JSON Pointer access. No I/O,
//!   safe to call without sandboxing.
//! * [`http_fetch::HttpFetch`] — bounded GET against an http(s) URL. Useful
//!   for retrieval-augmented chats; comes with a per-call byte budget.
//!
//! A shell exec builtin is intentionally **not** shipped here. Adding one
//! requires a security model fox does not yet have (sandbox, allowlist,
//! redaction of secrets).

pub mod http_fetch;
pub mod json_extract;
