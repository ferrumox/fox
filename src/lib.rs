// Ferrum Engine - library exports

pub mod api;
pub mod cli;
pub mod config;
pub(crate) mod engine;
pub(crate) mod kv_cache;
pub mod metrics;
pub mod model_registry;
pub mod registry;
pub(crate) mod scheduler;

/// Public re-export of the candle backend's reader / tokenizer / template
/// helpers, so integration tests and external embedders can drive them
/// without exposing the rest of the `engine` internals.
#[cfg(feature = "backend-candle")]
pub mod candle {
    pub use crate::engine::model::candle::{
        chat_template, gguf_loader, gguf_metadata, llama_arch, llama_model, tokenizer,
    };
}
