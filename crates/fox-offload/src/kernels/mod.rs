//! Kernels fox would actually want.
//!
//! Each one is written once against [`crate::Kernel`] and is differentially tested
//! against the CPU implementation in fox that it would replace. None of them are
//! wired into fox — see the crate docs for why.

pub mod argmax;
