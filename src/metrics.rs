// Prometheus metrics registry for fox.
// All metrics are registered once at startup via Metrics::new()
// and exposed on GET /metrics in the Prometheus text exposition format.
//
// Every metric carries a `model` label. Fox serves several models at once
// (`--max-models`) and until 0.21 nothing on /metrics said which one was
// responsible: a saturated KV cache, a deep queue and a slow p99 all looked
// like properties of the server rather than of one model in it.
//
// The label cannot be added without a bound. Model names are whatever a client
// asks for — `fox pull` takes arbitrary HuggingFace repos — so the label set is
// influenced from outside the server, and an unbounded one turns /metrics into a
// memory leak that a scrape then has to serialise. See `ModelLabels`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use prometheus::{
    register_gauge_vec, register_histogram_vec, register_int_counter_vec, register_int_gauge_vec,
    GaugeVec, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec,
};

/// Label applied to every metric, naming the model the observation belongs to.
const MODEL_LABEL: &str = "model";

/// Maximum distinct `model` label values ever admitted. Past this, every further
/// model collapses into [`OVERFLOW_MODEL_LABEL`].
///
/// 32 is well above any real `--max-models` (default 1) while keeping the worst
/// case small: the widest metric here is the latency histogram, so the ceiling is
/// 32 models x (10 buckets + sum + count) = 384 series, plus a couple of hundred
/// across everything else.
const MAX_MODEL_LABELS: usize = 32;

/// Where observations go once [`MAX_MODEL_LABELS`] is exhausted. Deliberately a
/// value no real model can collide with by accident — a GGUF basename never has
/// angle brackets.
const OVERFLOW_MODEL_LABEL: &str = "<other>";

/// Placeholder for an engine built with metrics disabled. Never reaches a
/// registry: nothing observes when `Metrics` is `None`.
pub const UNOBSERVED_MODEL_LABEL: &str = "<unobserved>";

/// Bounded interner mapping a model name to the `&'static str` used as its label.
///
/// Interning rather than passing `&str` around is what keeps the hot paths free:
/// a label is resolved once when an engine is built, and every later observation
/// — including the per-token counter — uses it with no allocation and no lock.
///
/// The bound counts models *ever seen*, not models currently loaded. That is the
/// safe direction: a load/evict/load cycle reuses its slot instead of consuming a
/// new one, and nothing can walk the cap upward by churning models.
struct ModelLabels {
    interned: Mutex<HashMap<Box<str>, &'static str>>,
    overflow_warned: AtomicBool,
}

impl ModelLabels {
    fn new() -> Self {
        Self {
            interned: Mutex::new(HashMap::new()),
            overflow_warned: AtomicBool::new(false),
        }
    }

    fn resolve(&self, model: &str) -> &'static str {
        // A poisoned lock here must not take the server down: telemetry is not
        // worth a panic on the request path, so recover the map and carry on.
        let mut interned = match self.interned.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(&label) = interned.get(model) {
            return label;
        }
        if interned.len() >= MAX_MODEL_LABELS {
            // Once per process, not once per model: a server being walked through
            // a thousand names must not be able to turn its own logs into the
            // flood the cap exists to prevent.
            if !self.overflow_warned.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    limit = MAX_MODEL_LABELS,
                    overflow_label = OVERFLOW_MODEL_LABEL,
                    "distinct model names on /metrics hit the cardinality cap; \
                     further models are reported under the overflow label \
                     (serving is unaffected)"
                );
            }
            return OVERFLOW_MODEL_LABEL;
        }
        // Bounded by MAX_MODEL_LABELS, so this leaks at most 32 short strings for
        // the life of the process — the price of a `&'static str` label.
        let label: &'static str = Box::leak(model.to_owned().into_boxed_str());
        interned.insert(model.into(), label);
        label
    }
}

/// All Prometheus metrics for a single server instance.
pub struct Metrics {
    /// Total inference requests completed, by model and finish_reason.
    pub requests_total: IntCounterVec,
    /// Total tokens generated, by model.
    pub tokens_generated_total: IntCounterVec,
    /// End-to-end request latency in seconds (from submit to last token), by model.
    pub request_latency_seconds: HistogramVec,
    /// Current KV cache memory usage as a ratio [0.0, 1.0], by model.
    pub kv_cache_usage_ratio: GaugeVec,
    /// Requests currently waiting in the queue, by model.
    pub queue_depth: IntGaugeVec,
    /// Requests currently running (prefill or decode), by model.
    pub active_requests: IntGaugeVec,
    /// Prefix cache hits (prompt already in KV cache), by model.
    pub prefix_cache_hits_total: IntCounterVec,
    /// Prefix cache misses, by model.
    pub prefix_cache_misses_total: IntCounterVec,
    /// Draft tokens proposed by speculative decoding, by model.
    pub spec_tokens_proposed_total: IntCounterVec,
    /// Draft tokens the target model accepted during verification, by model.
    pub spec_tokens_accepted_total: IntCounterVec,
    /// Lifetime accepted/proposed ratio of speculative decoding, by model.
    pub spec_acceptance_ratio: GaugeVec,
    /// Requests rejected before admission, by model and reason
    /// (`queue_full`, `too_large`).
    pub requests_rejected_total: IntCounterVec,
    /// Batch-size-bisection retries triggered by a recoverable llama_decode failure
    /// ("no KV slot for batch") during prefill/decode, by model.
    pub decode_bisection_retries_total: IntCounterVec,
    labels: ModelLabels,
}

impl Metrics {
    /// Register all metrics with the default Prometheus registry.
    pub fn new() -> Result<Self> {
        Ok(Self {
            requests_total: register_int_counter_vec!(
                "fox_requests_total",
                "Total inference requests completed",
                &[MODEL_LABEL, "finish_reason"]
            )?,
            tokens_generated_total: register_int_counter_vec!(
                "fox_tokens_generated_total",
                "Total tokens generated across all requests",
                &[MODEL_LABEL]
            )?,
            request_latency_seconds: register_histogram_vec!(
                HistogramOpts::new(
                    "fox_request_latency_seconds",
                    "End-to-end request latency in seconds (submit → last token)"
                )
                .buckets(vec![0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,]),
                &[MODEL_LABEL]
            )?,
            kv_cache_usage_ratio: register_gauge_vec!(
                "fox_kv_cache_usage_ratio",
                "KV cache memory usage ratio [0.0, 1.0]",
                &[MODEL_LABEL]
            )?,
            queue_depth: register_int_gauge_vec!(
                "fox_queue_depth",
                "Number of requests waiting in the scheduler queue",
                &[MODEL_LABEL]
            )?,
            active_requests: register_int_gauge_vec!(
                "fox_active_requests",
                "Number of requests currently being processed (prefill + decode)",
                &[MODEL_LABEL]
            )?,
            prefix_cache_hits_total: register_int_counter_vec!(
                "fox_prefix_cache_hits_total",
                "Prefix cache hits (prompt KV data reused from a previous request)",
                &[MODEL_LABEL]
            )?,
            prefix_cache_misses_total: register_int_counter_vec!(
                "fox_prefix_cache_misses_total",
                "Prefix cache misses (full prefill required)",
                &[MODEL_LABEL]
            )?,
            spec_tokens_proposed_total: register_int_counter_vec!(
                "fox_spec_tokens_proposed_total",
                "Draft tokens proposed by speculative decoding (n-gram lookup)",
                &[MODEL_LABEL]
            )?,
            spec_tokens_accepted_total: register_int_counter_vec!(
                "fox_spec_tokens_accepted_total",
                "Draft tokens accepted by the target model during speculative verification",
                &[MODEL_LABEL]
            )?,
            spec_acceptance_ratio: register_gauge_vec!(
                "fox_spec_acceptance_ratio",
                "Lifetime accepted/proposed ratio of speculative decoding (0 when unused)",
                &[MODEL_LABEL]
            )?,
            requests_rejected_total: register_int_counter_vec!(
                "fox_requests_rejected_total",
                "Requests rejected before admission",
                &[MODEL_LABEL, "reason"]
            )?,
            decode_bisection_retries_total: register_int_counter_vec!(
                "fox_decode_bisection_retries_total",
                "Batch-size-bisection retries triggered by a recoverable llama_decode failure",
                &[MODEL_LABEL]
            )?,
            labels: ModelLabels::new(),
        })
    }

    /// The `model` label for `model_name`, or the overflow label once the
    /// cardinality cap is exhausted. Called once per engine construction.
    pub fn model_label(&self, model_name: &str) -> &'static str {
        self.labels.resolve(model_name)
    }

    /// Drop every series carrying `label`, called when a model is evicted.
    ///
    /// Counters could be left alone — a monotonic total that stops advancing is
    /// still true — but the gauges could not: `kv_cache_usage_ratio` for an
    /// evicted model would sit at its last value forever, and a dashboard would
    /// go on reporting a full KV cache for a model that no longer occupies any.
    /// So all of them are removed together, and a reload starts the series again.
    ///
    /// The overflow bucket is never removed: it aggregates models that are not
    /// individually tracked, so one of them being evicted says nothing about the
    /// rest.
    pub fn forget_model(&self, label: &str) {
        if label == OVERFLOW_MODEL_LABEL || label == UNOBSERVED_MODEL_LABEL {
            return;
        }
        // Every metric here has `model` first, so removing by that one label
        // takes the whole family for this model — including the two-label
        // metrics, whose `finish_reason` / `reason` values are unknown here.
        let _ = self.kv_cache_usage_ratio.remove_label_values(&[label]);
        let _ = self.queue_depth.remove_label_values(&[label]);
        let _ = self.active_requests.remove_label_values(&[label]);
        let _ = self.spec_acceptance_ratio.remove_label_values(&[label]);
        let _ = self.tokens_generated_total.remove_label_values(&[label]);
        let _ = self.request_latency_seconds.remove_label_values(&[label]);
        let _ = self.prefix_cache_hits_total.remove_label_values(&[label]);
        let _ = self.prefix_cache_misses_total.remove_label_values(&[label]);
        let _ = self
            .spec_tokens_proposed_total
            .remove_label_values(&[label]);
        let _ = self
            .spec_tokens_accepted_total
            .remove_label_values(&[label]);
        let _ = self
            .decode_bisection_retries_total
            .remove_label_values(&[label]);
        // `requests_total{model,finish_reason}` and
        // `requests_rejected_total{model,reason}` need the second label to
        // address a series, so they are swept by prefix instead.
        remove_by_model(&self.requests_total, label);
        remove_by_model(&self.requests_rejected_total, label);
    }
}

/// Remove every series of a two-label counter whose first label is `model`.
///
/// `remove_label_values` needs the full label tuple and the second value is not
/// known at eviction time, so the live children are enumerated and matched.
fn remove_by_model(vec: &IntCounterVec, model: &str) {
    use prometheus::core::Collector;
    let seconds: Vec<String> = vec
        .collect()
        .into_iter()
        .flat_map(|family| family.get_metric().to_vec())
        .filter_map(|metric| {
            let pairs = metric.get_label();
            let is_ours = pairs
                .iter()
                .any(|p| p.name() == MODEL_LABEL && p.value() == model);
            is_ours.then(|| {
                pairs
                    .iter()
                    .find(|p| p.name() != MODEL_LABEL)
                    .map(|p| p.value().to_string())
                    .unwrap_or_default()
            })
        })
        .collect();
    for second in seconds {
        let _ = vec.remove_label_values(&[model, &second]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_models_get_distinct_labels() {
        let labels = ModelLabels::new();
        let a = labels.resolve("qwen3.6");
        let b = labels.resolve("gemma3");
        assert_ne!(a, b);
        assert_eq!(a, "qwen3.6");
        assert_eq!(b, "gemma3");
    }

    #[test]
    fn the_same_model_interns_to_the_same_pointer() {
        let labels = ModelLabels::new();
        let a = labels.resolve("qwen3.6");
        let b = labels.resolve("qwen3.6");
        // Pointer equality, not just string equality: resolving twice must not
        // leak a second copy, or the cap would bound nothing.
        assert!(std::ptr::eq(a, b));
    }

    /// The property the cap exists for: a client naming endless models cannot
    /// grow the label set past the bound.
    #[test]
    fn models_past_the_cap_collapse_into_the_overflow_bucket() {
        let labels = ModelLabels::new();
        for i in 0..MAX_MODEL_LABELS {
            assert_eq!(labels.resolve(&format!("model-{i}")), format!("model-{i}"));
        }
        for i in 0..1000 {
            assert_eq!(
                labels.resolve(&format!("flood-{i}")),
                OVERFLOW_MODEL_LABEL,
                "everything past the cap must share one series"
            );
        }
        assert_eq!(labels.interned.lock().unwrap().len(), MAX_MODEL_LABELS);
    }

    /// A model admitted before the cap filled keeps its own series afterwards —
    /// the flood must not evict the models an operator actually runs.
    #[test]
    fn admitted_models_survive_a_flood() {
        let labels = ModelLabels::new();
        let real = labels.resolve("qwen3.6");
        for i in 0..MAX_MODEL_LABELS * 4 {
            labels.resolve(&format!("flood-{i}"));
        }
        assert!(std::ptr::eq(labels.resolve("qwen3.6"), real));
    }

    /// Eviction of one model must not disturb another's series. Uses the real
    /// registry, so it also proves the label tuples line up — a mismatched arity
    /// makes `with_label_values` panic rather than fail quietly.
    #[test]
    fn forgetting_one_model_leaves_the_others_intact() {
        let m = match Metrics::new() {
            Ok(m) => m,
            // The default registry is process-global and these metric names are
            // registered once; if another test in this binary got there first,
            // there is nothing to assert here.
            Err(_) => return,
        };
        let a = m.model_label("model-a");
        let b = m.model_label("model-b");

        m.kv_cache_usage_ratio.with_label_values(&[a]).set(0.9);
        m.kv_cache_usage_ratio.with_label_values(&[b]).set(0.1);
        m.requests_total.with_label_values(&[a, "stop"]).inc();
        m.requests_total.with_label_values(&[b, "stop"]).inc();
        m.tokens_generated_total.with_label_values(&[a]).inc();
        m.tokens_generated_total.with_label_values(&[b]).inc();

        m.forget_model(a);

        assert_eq!(
            m.kv_cache_usage_ratio.with_label_values(&[b]).get(),
            0.1,
            "the surviving model keeps its gauge"
        );
        assert_eq!(
            m.tokens_generated_total.with_label_values(&[b]).get(),
            1,
            "the surviving model keeps its counter"
        );
        assert_eq!(
            m.requests_total.with_label_values(&[b, "stop"]).get(),
            1,
            "the surviving model keeps its two-label counter"
        );
        // Re-addressing the evicted model creates a fresh series at zero, which
        // is the point: a reloaded model starts over rather than resuming a
        // stale value.
        assert_eq!(m.kv_cache_usage_ratio.with_label_values(&[a]).get(), 0.0);
        assert_eq!(m.requests_total.with_label_values(&[a, "stop"]).get(), 0);
    }
}
