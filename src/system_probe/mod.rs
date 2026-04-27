//! Lightweight environment probes the operator can use to verify a host
//! before serving traffic.
//!
//! Wired into:
//! * `fox probe` — pretty-printed report on stdout.
//! * `/metrics` — gauge labels for `gpu_vendor`, `gpu_name`, `gpu_driver`.
//! * Future health checks — surface when CUDA is missing but the binary was
//!   built with it, etc.

pub mod gpu;

pub use gpu::{GpuDevice, GpuProbe, GpuVendor};
