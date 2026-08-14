//! Prometheus metrics for the VK Cloud deployment (migration T9).
//!
//! This binary had none, while `deploy/k8s/service-staging.yaml` has shipped a
//! ServiceMonitor scraping `/metrics` since T7 — so the target has been down in
//! Prometheus for the entire migration.
//!
//! The load-bearing metric here is [`retries_exhausted`]. There is **no
//! dead-letter queue by design** (`cutl-infra/docs/kafka-contract.md` §4): when a
//! message exhausts the retry ladder it is dropped, and this counter plus an ERROR
//! log are the *only* record that it ever existed. cutl-infra already ships the
//! alert (`CutlRetriesExhausted`, `modules-vk/observability/main.tf:540-568`);
//! until now nothing produced the series it fires on.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use prometheus::{
    Encoder, Histogram, HistogramOpts, HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts,
    Registry, TextEncoder,
};

/// The consumer group this worker runs under. Also the `group` label value on
/// [`retries_exhausted`], which the alert's description template reads back as
/// `{{ $labels.group }}`.
pub const CONSUMER_GROUP: &str = "jagua-nesting";

/// Everything is registered on a private registry rather than the crate-global
/// default, so a test can assert on exact output without picking up whatever else
/// a linked crate happens to have registered.
pub struct Metrics {
    registry: Registry,

    /// **The one metric that is a correctness requirement rather than telemetry.**
    ///
    /// Labels are exactly `group` and `origin_topic` — no more. The alert
    /// annotation reads `$labels.group` and `$labels.origin_topic`, and its PromQL
    /// (`increase(cutl_retries_exhausted_total[5m]) > 0`) has no selector, so extra
    /// labels are harmless but these two must be present and spelled this way.
    ///
    /// Deliberately NOT prefixed `jagua_`: it is an estate-wide contract shared
    /// with the other workers, hence `cutl_`.
    pub retries_exhausted: IntCounterVec,

    pub messages_consumed: IntCounterVec,
    pub messages_failed: IntCounterVec,
    pub retry_republished: IntCounterVec,
    pub nesting_duration: Histogram,
    pub operation_retries: HistogramVec,
    pub inflight_jobs: IntGauge,
    pub committed_offset: IntGaugeVec,
}

impl Metrics {
    /// Build an independent registry.
    ///
    /// Public so tests can each own one: the process-wide handle from [`metrics`]
    /// is shared, and Rust runs tests in the same process in parallel, so
    /// asserting "this counter reads 0" against the global would race any test
    /// that increments it.
    pub fn new() -> Result<Self> {
        let registry = Registry::new();

        let retries_exhausted = IntCounterVec::new(
            Opts::new(
                "cutl_retries_exhausted_total",
                "Messages dropped after exhausting the retry ladder. There is no DLQ, \
                 so each increment is a message that is permanently gone.",
            ),
            &["group", "origin_topic"],
        )?;

        let messages_consumed = IntCounterVec::new(
            Opts::new(
                "jagua_messages_consumed_total",
                "Records consumed, by topic.",
            ),
            &["topic"],
        )?;

        let messages_failed = IntCounterVec::new(
            Opts::new(
                "jagua_messages_failed_total",
                "Records whose handler returned an error, by topic.",
            ),
            &["topic"],
        )?;

        let retry_republished = IntCounterVec::new(
            Opts::new(
                "jagua_retry_republished_total",
                "Records republished to a retry tier, by destination tier.",
            ),
            &["tier"],
        )?;

        // Nesting runs are minutes long and capped at 600s by `maxSeconds`, so the
        // default buckets (which top out at 10s) would put everything in +Inf.
        let nesting_duration = Histogram::with_opts(
            HistogramOpts::new(
                "jagua_nesting_duration_seconds",
                "Wall-clock duration of a nesting run.",
            )
            .buckets(vec![1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0]),
        )?;

        let operation_retries = HistogramVec::new(
            HistogramOpts::new(
                "jagua_operation_attempts",
                "Attempts taken by a retried operation before it succeeded or gave up.",
            )
            .buckets(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            &["operation", "outcome"],
        )?;

        let inflight_jobs = IntGauge::new(
            "jagua_inflight_jobs",
            "Nesting jobs currently running. Bounded by MAX_CONCURRENT_TASKS.",
        )?;

        // Makes the contiguous-completed watermark observable from outside the
        // process. Without it, an over- or under-advancing watermark is invisible
        // until messages are already lost.
        let committed_offset = IntGaugeVec::new(
            Opts::new(
                "jagua_committed_offset",
                "Last offset committed per topic-partition (the contiguous-completed watermark).",
            ),
            &["topic", "partition"],
        )?;

        registry.register(Box::new(retries_exhausted.clone()))?;
        registry.register(Box::new(messages_consumed.clone()))?;
        registry.register(Box::new(messages_failed.clone()))?;
        registry.register(Box::new(retry_republished.clone()))?;
        registry.register(Box::new(nesting_duration.clone()))?;
        registry.register(Box::new(operation_retries.clone()))?;
        registry.register(Box::new(inflight_jobs.clone()))?;
        registry.register(Box::new(committed_offset.clone()))?;

        Ok(Self {
            registry,
            retries_exhausted,
            messages_consumed,
            messages_failed,
            retry_republished,
            nesting_duration,
            operation_retries,
            inflight_jobs,
            committed_offset,
        })
    }

    /// Render the registry in Prometheus text exposition format.
    pub fn encode(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let mut buf = Vec::new();
        encoder
            .encode(&self.registry.gather(), &mut buf)
            .context("failed to encode Prometheus metrics")?;
        String::from_utf8(buf).context("Prometheus encoder emitted non-UTF-8")
    }

    /// Record that a message was dropped after exhausting its attempt budget.
    ///
    /// Pair every call with an ERROR log carrying the key and
    /// `topic-partition-offset`: the counter says *that* something was lost, the
    /// log is the only thing that says *what*.
    pub fn record_retries_exhausted(&self, origin_topic: &str) {
        self.retries_exhausted
            .with_label_values(&[CONSUMER_GROUP, origin_topic])
            .inc();
    }

    /// Create the label combinations that must exist before the first failure.
    ///
    /// A labelled counter does not exist as a series until its child is first
    /// created, and `increase(cutl_retries_exhausted_total[5m]) > 0` can never
    /// evaluate on a series that has never been reported. The alert would sit
    /// green not because nothing was lost but because nothing was *observable* —
    /// the exact failure that killed `cutl_log_errors_total`
    /// (`cutl-infra/modules-vk/observability/main.tf:602-607`), an alert that
    /// reported healthy forever because nothing emitted its metric.
    ///
    /// Call once at startup with the topics this worker consumes.
    pub fn preregister(&self, origin_topics: &[&str]) {
        for topic in origin_topics {
            // `with_label_values` creates the child at zero without incrementing.
            self.retries_exhausted
                .with_label_values(&[CONSUMER_GROUP, topic]);
            self.messages_consumed.with_label_values(&[topic]);
            self.messages_failed.with_label_values(&[topic]);
        }
        for tier in ["1", "2", "3"] {
            self.retry_republished.with_label_values(&[tier]);
        }
    }
}

static METRICS: OnceLock<Metrics> = OnceLock::new();

/// Process-wide metrics handle.
///
/// Panics only if metric construction itself fails, which can happen for exactly
/// one reason: a duplicate or malformed metric definition in this file. That is a
/// programming error caught by the first test that touches the registry, not a
/// runtime condition.
pub fn metrics() -> &'static Metrics {
    METRICS.get_or_init(|| Metrics::new().expect("metric definitions are invalid"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The alert fires on `increase(...) > 0`, which needs the series to exist
    /// before the first drop. This is the regression test for the
    /// `cutl_log_errors_total` failure mode: an alert that is green because its
    /// metric is absent rather than because nothing is wrong.
    #[test]
    fn retries_exhausted_series_exists_at_zero_before_any_failure() {
        let m = Metrics::new().unwrap();
        m.preregister(&["nesting-request"]);
        let out = m.encode().unwrap();

        assert!(
            out.contains(
                r#"cutl_retries_exhausted_total{group="jagua-nesting",origin_topic="nesting-request"} 0"#
            ),
            "counter must be exposed at zero before the first drop, got:\n{out}"
        );
    }

    /// Label names are contractual: the alert's description template reads
    /// `$labels.group` and `$labels.origin_topic` by those exact names.
    #[test]
    fn retries_exhausted_carries_group_and_origin_topic_labels() {
        let m = Metrics::new().unwrap();
        m.record_retries_exhausted("nesting-request");
        let out = m.encode().unwrap();

        let line = out
            .lines()
            .find(|l| l.starts_with("cutl_retries_exhausted_total{"))
            .expect("counter should be present");

        assert!(line.contains(r#"group="jagua-nesting""#), "got: {line}");
        assert!(
            line.contains(r#"origin_topic="nesting-request""#),
            "got: {line}"
        );
        assert!(
            line.ends_with(" 1"),
            "expected one drop recorded, got: {line}"
        );
    }

    /// The metric is estate-wide, shared with the other workers' alerts, so it
    /// must not pick up this crate's `jagua_` prefix.
    #[test]
    fn exhaustion_counter_is_not_jagua_prefixed() {
        let m = Metrics::new().unwrap();
        m.preregister(&["nesting-request"]);
        let out = m.encode().unwrap();
        assert!(!out.contains("jagua_cutl_retries"));
        assert!(out.contains("cutl_retries_exhausted_total"));
    }

    #[test]
    fn encodes_prometheus_text_format() {
        let m = Metrics::new().unwrap();
        m.preregister(&["nesting-request"]);
        let out = m.encode().unwrap();
        assert!(out.contains("# TYPE cutl_retries_exhausted_total counter"));
        assert!(out.contains("# HELP"));
    }

    /// The process-wide handle must build — a duplicate or malformed metric
    /// definition would otherwise only panic at startup in production.
    #[test]
    fn global_registry_initialises() {
        assert!(!metrics().encode().unwrap().is_empty());
    }
}
