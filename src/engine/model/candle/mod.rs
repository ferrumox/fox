//! In-process candle backend.
//!
//! Rollout state (per the multi-backend refactor plan):
//!   - C.1 (this commit): tensor-index reader + module skeleton. No model
//!     instances are constructed yet — `CandleBackend::supports` declines
//!     every architecture so the router falls through to llama.cpp.
//!   - C.2: tokeniser + chat templates from GGUF metadata.
//!   - C.3: Llama-family forward pass (RMSNorm, RoPE, GQA, SwiGLU) and Q4_K
//!     dequantisation.
//!   - C.4: full `Model` trait wiring + smoke generation.
//!   - C.5: Gemma 4 and Qwen 3.5 architectures.

pub mod chat_template;
pub mod gguf_loader;
pub mod gguf_metadata;
pub mod tokenizer;
