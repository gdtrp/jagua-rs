//! W3C trace-context propagation over Kafka headers (VK migration T9).
//!
//! Without this, every nesting job starts a brand-new trace. Grafana can still
//! join a span to its logs via `service.name`, but the request that *caused* the
//! job — the backend call that published to `nesting-request` — is a separate,
//! unlinked trace. The interesting question ("why was this job slow?") spans both.
//!
//! Propagation is the standard W3C pair carried as Kafka record headers:
//!
//! - `traceparent` — version, trace id, parent span id, flags.
//! - `tracestate` — vendor-specific continuation, passed through untouched.
//!
//! **Headers, not a body field.** The RFC notes the existing envelope carries a
//! hand-rolled `traceparent` *inside the JSON*; putting it in headers instead
//! keeps the payload governed purely by the AsyncAPI schema, and means the
//! producer does not have to touch a message it is only forwarding — which is
//! exactly the retry-ladder case, where a republish must preserve the original
//! trace rather than start a new one.
//!
//! Interop: dxf-processor and interpreter already do W3C propagation, and
//! Spring's Kafka instrumentation reads and writes these same two headers, so a
//! trace started in cutl-backend will continue here once T8 lands.

use opentelemetry::propagation::{Extractor, Injector};
use rdkafka::message::{BorrowedMessage, Header, Headers, OwnedHeaders};
use rdkafka::Message;

pub const TRACEPARENT: &str = "traceparent";
pub const TRACESTATE: &str = "tracestate";

/// Writes trace context into a header list being built for an outgoing record.
pub struct HeaderInjector {
    pub headers: OwnedHeaders,
}

impl HeaderInjector {
    pub fn new(headers: OwnedHeaders) -> Self {
        Self { headers }
    }

    pub fn into_headers(self) -> OwnedHeaders {
        self.headers
    }
}

impl Injector for HeaderInjector {
    fn set(&mut self, key: &str, value: String) {
        // `OwnedHeaders::insert` consumes and returns, so swap through a temporary
        // rather than requiring &mut self semantics rdkafka does not offer.
        let current = std::mem::replace(&mut self.headers, OwnedHeaders::new());
        self.headers = current.insert(Header {
            key,
            value: Some(&value),
        });
    }
}

/// Reads trace context off an incoming record.
///
/// Headers are collected eagerly into owned strings because the propagator's
/// `get` returns a borrow tied to `&self`, which a `BorrowedMessage`'s lifetime
/// cannot satisfy once the message is dropped.
pub struct HeaderExtractor {
    entries: Vec<(String, String)>,
}

impl HeaderExtractor {
    pub fn from_message(msg: &BorrowedMessage<'_>) -> Self {
        let mut entries = Vec::new();
        if let Some(headers) = msg.headers() {
            for header in headers.iter() {
                if let Some(value) = header.value {
                    if let Ok(text) = std::str::from_utf8(value) {
                        entries.push((header.key.to_string(), text.to_string()));
                    }
                }
            }
        }
        Self { entries }
    }

    /// Build from already-extracted header pairs, for a record that has been
    /// lifted out of rdkafka's buffer.
    pub fn from_pairs(entries: Vec<(String, String)>) -> Self {
        Self { entries }
    }

    /// The raw `traceparent`, for logging and for forwarding on a republish.
    pub fn traceparent(&self) -> Option<&str> {
        self.get(TRACEPARENT)
    }
}

impl Extractor for HeaderExtractor {
    fn get(&self, key: &str) -> Option<&str> {
        // Header names are case-insensitive in the W3C spec; Kafka does not
        // normalise them, and Spring writes them lowercase while some clients do
        // not, so compare case-insensitively rather than trusting the producer.
        self.entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    fn keys(&self) -> Vec<&str> {
        self.entries.iter().map(|(k, _)| k.as_str()).collect()
    }
}

/// Extract the parent context from a record's headers.
///
/// Returns a context with no remote parent when the headers carry none, which is
/// the correct behaviour rather than an error: a message produced by a service
/// with no tracing simply starts a new trace here.
pub fn extract_context(extractor: &HeaderExtractor) -> opentelemetry::Context {
    opentelemetry::global::get_text_map_propagator(|propagator| propagator.extract(extractor))
}

/// Inject the given context into a header list.
pub fn inject_context(cx: &opentelemetry::Context, headers: OwnedHeaders) -> OwnedHeaders {
    let mut injector = HeaderInjector::new(headers);
    opentelemetry::global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut injector)
    });
    injector.into_headers()
}

/// Inject the *currently active* span's context, which is what a produce inside a
/// traced handler wants.
pub fn inject_current(headers: OwnedHeaders) -> OwnedHeaders {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;
    let cx = tracing::Span::current().context();
    tracing::debug!(
        valid = cx.span().span_context().is_valid(),
        trace_id = %cx.span().span_context().trace_id(),
        "injecting trace context into kafka headers"
    );
    inject_context(&cx, headers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::propagation::TextMapPropagator;
    use opentelemetry_sdk::propagation::TraceContextPropagator;

    fn header_pairs(headers: &OwnedHeaders) -> Vec<(String, String)> {
        headers
            .iter()
            .filter_map(|h| {
                h.value
                    .and_then(|v| std::str::from_utf8(v).ok())
                    .map(|v| (h.key.to_string(), v.to_string()))
            })
            .collect()
    }

    /// The round trip that matters: a context injected into headers must extract
    /// back to the *same trace id*, or spans land in unrelated traces and the
    /// propagation is silently useless.
    #[test]
    fn context_round_trips_through_headers() {
        let propagator = TraceContextPropagator::new();

        // A concrete, valid traceparent standing in for an upstream service.
        let incoming = HeaderExtractor::from_pairs(vec![(
            TRACEPARENT.to_string(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
        )]);
        let cx = propagator.extract(&incoming);

        let mut injector = HeaderInjector::new(OwnedHeaders::new());
        propagator.inject_context(&cx, &mut injector);
        let pairs = header_pairs(&injector.into_headers());

        let traceparent = pairs
            .iter()
            .find(|(k, _)| k == TRACEPARENT)
            .map(|(_, v)| v.as_str())
            .expect("traceparent must be re-emitted");

        assert!(
            traceparent.contains("4bf92f3577b34da6a3ce929d0e0e4736"),
            "trace id must survive the round trip, got {traceparent}"
        );
    }

    /// Producers vary in header casing and Kafka does not normalise it. Matching
    /// case-sensitively would drop the parent and silently start a new trace.
    #[test]
    fn extraction_is_case_insensitive() {
        let e = HeaderExtractor::from_pairs(vec![(
            "TraceParent".to_string(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
        )]);
        assert!(e.get(TRACEPARENT).is_some(), "casing must not matter");
    }

    /// A message from a service with no tracing must not error — it just starts a
    /// new trace.
    #[test]
    fn missing_traceparent_yields_no_remote_parent() {
        let e = HeaderExtractor::from_pairs(vec![("x-cutl-attempt".to_string(), "2".to_string())]);
        assert!(e.traceparent().is_none());

        let propagator = TraceContextPropagator::new();
        let cx = propagator.extract(&e);
        use opentelemetry::trace::TraceContextExt;
        assert!(
            !cx.span().span_context().is_valid(),
            "absent headers must give an invalid (new-trace) context, not a bogus one"
        );
    }

    /// The production path, end to end: with a real subscriber installed, a span
    /// parented from an incoming `traceparent` must make `inject_current` emit a
    /// `traceparent` in the SAME trace.
    ///
    /// The unit tests above drive the propagator directly, which passes even when
    /// the wiring between `tracing` and OpenTelemetry is wrong — this one is what
    /// catches "spans exist but headers come out empty".
    #[test]
    fn inject_current_emits_the_active_span_context() {
        use opentelemetry::trace::TraceContextExt;
        use opentelemetry::trace::TracerProvider as _;
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        use tracing_subscriber::layer::SubscriberExt;

        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());

        // A provider with no exporter still assigns real span contexts, which is
        // all propagation needs.
        let provider = opentelemetry_sdk::trace::TracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let subscriber =
            tracing_subscriber::registry().with(tracing_opentelemetry::layer().with_tracer(tracer));

        let trace_id = "4bf92f3577b34da6a3ce929d0e0e4736";
        let incoming = HeaderExtractor::from_pairs(vec![(
            TRACEPARENT.to_string(),
            format!("00-{trace_id}-00f067aa0ba902b7-01"),
        )]);
        let parent = TraceContextPropagator::new().extract(&incoming);

        let pairs = tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("test.handle");
            span.set_parent(parent);
            let _guard = span.enter();

            assert!(
                tracing::Span::current()
                    .context()
                    .span()
                    .span_context()
                    .is_valid(),
                "the active span must carry a valid OTel context, or nothing can be injected"
            );

            header_pairs(&inject_current(OwnedHeaders::new()))
        });

        let emitted = pairs
            .iter()
            .find(|(k, _)| k == TRACEPARENT)
            .map(|(_, v)| v.as_str())
            .expect("inject_current must emit a traceparent while a span is active");

        assert!(
            emitted.contains(trace_id),
            "response must stay in the caller's trace; got {emitted}"
        );
    }

    /// Injection must not clobber the retry headers sharing the same list — a
    /// republish carries both, and losing `x-cutl-attempt` would reset the
    /// attempt budget and loop forever.
    #[test]
    fn injection_preserves_existing_headers() {
        let existing = OwnedHeaders::new()
            .insert(Header {
                key: "x-cutl-attempt",
                value: Some("2"),
            })
            .insert(Header {
                key: "x-cutl-origin-topic",
                value: Some("nesting-request"),
            });

        let propagator = TraceContextPropagator::new();
        let incoming = HeaderExtractor::from_pairs(vec![(
            TRACEPARENT.to_string(),
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01".to_string(),
        )]);
        let cx = propagator.extract(&incoming);

        let mut injector = HeaderInjector::new(existing);
        propagator.inject_context(&cx, &mut injector);
        let pairs = header_pairs(&injector.into_headers());

        assert!(pairs.iter().any(|(k, v)| k == "x-cutl-attempt" && v == "2"));
        assert!(pairs
            .iter()
            .any(|(k, v)| k == "x-cutl-origin-topic" && v == "nesting-request"));
        assert!(pairs.iter().any(|(k, _)| k == TRACEPARENT));
    }
}
