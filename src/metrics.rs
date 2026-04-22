// Prometheus metrics registry for ferrumox.
// All metrics are registered once at startup via Metrics::new()
// and exposed on GET /metrics in the Prometheus text exposition format.

use anyhow::Result;
use prometheus::{
    register_gauge, register_histogram, register_histogram_vec, register_int_counter,
    register_int_counter_vec, register_int_gauge, Gauge, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge,
};

/// All Prometheus metrics for a single server instance.
pub struct Metrics {
    /// Total inference requests completed, labelled by finish_reason.
    pub requests_total: IntCounterVec,
    /// Total tokens generated across all requests.
    pub tokens_generated_total: IntCounter,
    /// End-to-end request latency in seconds (from submit to last token).
    pub request_latency_seconds: Histogram,
    /// Current KV cache memory usage as a ratio [0.0, 1.0].
    pub kv_cache_usage_ratio: Gauge,
    /// Number of requests currently waiting in the queue.
    pub queue_depth: IntGauge,
    /// Number of requests currently running (prefill or decode).
    pub active_requests: IntGauge,
    /// Total prefix cache hits (prompt already in KV cache).
    pub prefix_cache_hits_total: IntCounter,
    /// Total prefix cache misses.
    pub prefix_cache_misses_total: IntCounter,
    /// Time to first token in seconds, labelled by model name.
    pub ttft_seconds: HistogramVec,
}

impl Metrics {
    /// Register all metrics with the default Prometheus registry.
    pub fn new() -> Result<Self> {
        Ok(Self {
            requests_total: register_int_counter_vec!(
                "ferrumox_requests_total",
                "Total inference requests completed",
                &["finish_reason"]
            )?,
            tokens_generated_total: register_int_counter!(
                "ferrumox_tokens_generated_total",
                "Total tokens generated across all requests"
            )?,
            request_latency_seconds: register_histogram!(HistogramOpts::new(
                "ferrumox_request_latency_seconds",
                "End-to-end request latency in seconds (submit → last token)"
            )
            .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,]))?,
            kv_cache_usage_ratio: register_gauge!(
                "ferrumox_kv_cache_usage_ratio",
                "KV cache memory usage ratio [0.0, 1.0]"
            )?,
            queue_depth: register_int_gauge!(
                "ferrumox_queue_depth",
                "Number of requests waiting in the scheduler queue"
            )?,
            active_requests: register_int_gauge!(
                "ferrumox_active_requests",
                "Number of requests currently being processed (prefill + decode)"
            )?,
            prefix_cache_hits_total: register_int_counter!(
                "ferrumox_prefix_cache_hits_total",
                "Prefix cache hits (prompt KV data reused from a previous request)"
            )?,
            prefix_cache_misses_total: register_int_counter!(
                "ferrumox_prefix_cache_misses_total",
                "Prefix cache misses (full prefill required)"
            )?,
            ttft_seconds: register_histogram_vec!(
                HistogramOpts::new(
                    "ferrumox_ttft_seconds",
                    "Time to first token in seconds (submit → first token)"
                )
                .buckets(vec![0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
                &["model"]
            )?,
        })
    }
}
