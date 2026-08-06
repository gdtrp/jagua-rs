//! Health endpoints and tracing for the VK Cloud deployment (migration T7).
//!
//! This binary had neither. On ECS that was survivable because the SQS consumer
//! was the only thing that mattered and nothing probed it; on Kubernetes it is
//! not, because without a readiness probe a wedged worker keeps its pod Ready
//! forever and simply stops consuming.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use axum::{extract::State, http::StatusCode, routing::get, Router};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
// with_endpoint lives on this trait, not on the builder itself.
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{runtime, trace::TracerProvider, Resource};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

/// Liveness/readiness state shared between the HTTP server and the processor.
///
/// Liveness is deliberately NOT derived from this: a process that is running at
/// all answers `/health`. Only readiness is gated, so a failing dependency stops
/// traffic (and, later, stops the rollout) without triggering a restart loop
/// that cannot fix it.
#[derive(Clone, Default)]
pub struct Health {
    ready: Arc<AtomicBool>,
}

impl Health {
    pub fn new() -> Self {
        Self::default()
    }

    /// Flipped once the processor has its clients and is about to poll.
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::SeqCst);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }
}

async fn health() -> StatusCode {
    StatusCode::OK
}

async fn ready(State(health): State<Health>) -> StatusCode {
    if health.is_ready() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Serves `/health` and `/ready` until `shutdown` resolves.
///
/// Returns as soon as the listener is bound and the server is spawned, so
/// startup order is not a race: the endpoints answer before the processor
/// begins its first poll.
pub async fn serve_health(
    port: u16,
    health: Health,
    mut shutdown: tokio::sync::broadcast::Receiver<()>,
) -> Result<tokio::task::JoinHandle<()>> {
    let app = Router::new()
        .route("/health", get(health_handler_marker))
        .route("/ready", get(ready))
        .with_state(health);

    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port))
        .await
        .with_context(|| format!("failed to bind health server on port {port}"))?;

    tracing::info!("health endpoints listening on 0.0.0.0:{port}");

    Ok(tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = shutdown.recv().await;
        });
        if let Err(e) = server.await {
            tracing::error!("health server error: {e}");
        }
    }))
}

// Named wrapper so the route above reads symmetrically with `ready`.
async fn health_handler_marker() -> StatusCode {
    health().await
}

/// Installs the tracing subscriber, exporting spans over OTLP when
/// `OTEL_EXPORTER_OTLP_ENDPOINT` is set.
///
/// Returns the provider so the caller can flush it on shutdown. Dropping it
/// without flushing loses whatever is still buffered, which is precisely the
/// spans from a crash — the ones worth having.
///
/// `service.name` must match what the Java services declare, because Grafana
/// joins a trace to its logs on that exact string. A mismatch breaks the links
/// silently rather than erroring.
pub fn init_tracing(service_name: &str) -> Result<Option<TracerProvider>> {
    // NO explicit LogTracer::init() here. tracing-subscriber's default features
    // include `tracing-log`, so its own `.init()` installs the log bridge — and
    // calling LogTracer::init() first claims the global logger, making that
    // second call PANIC at startup with SetLoggerError. The existing log::info!
    // call sites are bridged either way, so nothing is lost by leaving it out.
    //
    // Found by running the built image; it compiles cleanly either way.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    match endpoint {
        Some(endpoint) if !endpoint.is_empty() => {
            let exporter = opentelemetry_otlp::SpanExporter::builder()
                .with_http()
                .with_endpoint(format!("{}/v1/traces", endpoint.trim_end_matches('/')))
                .build()
                .context("failed to build the OTLP span exporter")?;

            let provider = TracerProvider::builder()
                .with_batch_exporter(exporter, runtime::Tokio)
                .with_resource(Resource::new(vec![KeyValue::new(
                    opentelemetry_semantic_conventions::resource::SERVICE_NAME,
                    service_name.to_string(),
                )]))
                .build();

            let tracer = provider.tracer(service_name.to_string());

            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .with(tracing_opentelemetry::layer().with_tracer(tracer))
                .init();

            tracing::info!("OTLP tracing enabled, exporting to {endpoint}");
            Ok(Some(provider))
        }
        _ => {
            // No endpoint configured: log to stdout only. Deliberately not an
            // error — local runs and the ECS deployment have no collector, and
            // failing to start over a missing trace exporter would be absurd.
            tracing_subscriber::registry()
                .with(filter)
                .with(tracing_subscriber::fmt::layer())
                .init();

            tracing::info!("OTEL_EXPORTER_OTLP_ENDPOINT unset; tracing to stdout only");
            Ok(None)
        }
    }
}
