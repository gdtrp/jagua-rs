use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client as S3Client;
use jagua_sqs_processor::kafka::KafkaSettings;
use jagua_sqs_processor::metrics::metrics;
use jagua_sqs_processor::observability::{init_tracing, serve_health, Health};
use jagua_sqs_processor::NestingProcessor;
use log::{error, info};
// `fetch_metadata` lives on the Consumer trait, not on StreamConsumer itself.
use rdkafka::consumer::Consumer;
use std::env;
use tokio::signal;

/// Must match what the Java services declare, since Grafana joins a trace to
/// its logs on this exact string.
const SERVICE_NAME: &str = "jagua-nesting";

/// Everything the worker needs from the environment.
struct Config {
    kafka: KafkaSettings,
    s3_bucket: String,
    aws_region: String,
}

/// Read and validate configuration.
///
/// Returns `Err` rather than exiting so the caller can withhold readiness instead
/// of crash-looping — see the call site for why that distinction matters here.
///
/// The Kafka variables are supplied verbatim by the `kafka-jagua-nesting` Secret,
/// which cutl-infra mounts with `envFrom`; there is nothing to parse or decode.
/// `S3_BUCKET` has no default on purpose: guessing a bucket name would write
/// results somewhere nobody is reading.
fn load_config() -> Result<Config> {
    let kafka = KafkaSettings::from_env()?;
    let s3_bucket = env::var("S3_BUCKET")
        .ok()
        .filter(|v| !v.is_empty())
        .context("S3_BUCKET is required")?;
    // eu-north-1 is where the buckets live. Reaching them from VK needs the relay
    // (AWS_ENDPOINT_URL); the region stays the bucket's real region either way.
    let aws_region = env::var("AWS_REGION").unwrap_or_else(|_| "eu-north-1".to_string());

    Ok(Config {
        kafka,
        s3_bucket,
        aws_region,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    // Replaces env_logger (VK migration T7). tracing-log bridges the existing
    // log::info! call sites, so every line below still works unchanged and
    // RUST_LOG keeps behaving the same way.
    let tracer_provider = init_tracing(SERVICE_NAME)?;

    info!("Starting jagua-sqs-processor");

    // The shutdown channel is created HERE rather than just before the
    // processor loop, because the health server needs a receiver and must be
    // bound before any of the slow startup work below. Binding early is what
    // makes the probes meaningful: if AWS config or client construction hangs,
    // /health still answers (the process is alive) while /ready keeps returning
    // 503, so Kubernetes withholds traffic instead of restarting a pod that a
    // restart would not fix.
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    let health = Health::new();
    let health_port: u16 = env::var("HEALTH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let health_server = serve_health(health_port, health.clone(), shutdown_tx.subscribe()).await?;

    // ── Configuration ──
    //
    // Missing config does NOT exit. Before T9 a missing S3_BUCKET killed the
    // process, and the resulting crash-loop held a CPU reservation on a saturated
    // cluster until the workload was scaled to 0 — which is why this service was
    // down. A misconfigured pod that stays up with /ready returning 503 is both
    // more honest and less damaging: Kubernetes withholds traffic and the
    // Deployment stops rolling, instead of restarting forever over something a
    // restart cannot fix. This is the same reasoning that put /health outside the
    // readiness gate in T7.
    let config = match load_config() {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Configuration is incomplete, staying unready: {e:#}");
            // Park until SIGTERM. The health server is already bound, so /health
            // answers 200 and /ready answers 503 for as long as this lasts.
            let _ = shutdown_rx.recv().await;
            info!("Shutting down without ever becoming ready");
            return Ok(());
        }
    };

    info!("Configuration:");
    info!(
        "  KAFKA_BOOTSTRAP_SERVERS: {}",
        config.kafka.bootstrap_servers
    );
    info!("  KAFKA_CONSUMER_GROUP: {}", config.kafka.consumer_group);
    info!("  KAFKA_REQUEST_TOPIC: {}", config.kafka.request_topic);
    info!("  KAFKA_RESPONSE_TOPIC: {}", config.kafka.response_topic);
    info!("  KAFKA_SASL_MECHANISM: {}", config.kafka.sasl_mechanism);
    info!("  KAFKA_USERNAME: {}", config.kafka.username);
    info!("  S3_BUCKET: {}", config.s3_bucket);
    info!("  AWS_REGION: {}", config.aws_region);

    let (kafka_settings, s3_bucket, aws_region) =
        (config.kafka, config.s3_bucket, config.aws_region);

    // Create the `cutl_retries_exhausted_total` series before the first failure.
    // A labelled counter has no series until its child exists, and the alert's
    // `increase(...[5m]) > 0` can never fire on a series that was never reported —
    // it would read as healthy rather than as blind.
    metrics().preregister(&[
        kafka_settings.request_topic.as_str(),
        &kafka_settings.retry_topic(1),
        &kafka_settings.retry_topic(2),
        &kafka_settings.retry_topic(3),
    ]);

    // Log AWS configuration
    info!("AWS Configuration:");
    info!("  AWS_REGION: {:?}", env::var("AWS_REGION"));
    info!("  AWS_ENDPOINT_URL: {:?}", env::var("AWS_ENDPOINT_URL"));
    info!(
        "  AWS_ACCESS_KEY_ID: {:?}",
        env::var("AWS_ACCESS_KEY_ID").map(|s| format!("{}...", &s[..10.min(s.len())]))
    );
    info!(
        "  AWS_SECRET_ACCESS_KEY: {:?}",
        env::var("AWS_SECRET_ACCESS_KEY").map(|_| "***")
    );

    // Initialize AWS clients - both use LocalStack endpoint if provided
    let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

    // AWS_ENDPOINT_URL is what points S3 at MinIO locally — and it is also the
    // hook the future eu-west-1 relay will use, since `s3.eu-north-1` is
    // unreachable from a VK pod. Not test-only scaffolding.
    let use_path_style = env::var("AWS_ENDPOINT_URL").is_ok();

    if let Ok(endpoint_url) = env::var("AWS_ENDPOINT_URL") {
        config_loader = config_loader.endpoint_url(&endpoint_url);
        info!("Using S3 endpoint override: {}", endpoint_url);
    } else {
        info!("No AWS_ENDPOINT_URL set, using default AWS S3 endpoints");
    }

    let aws_config = config_loader.load().await;

    // Path-style addressing for MinIO/LocalStack and for a relay: virtual-hosted
    // style (bucket.host) cannot work against a host that is not the real S3.
    let s3_client = if use_path_style {
        info!("Using path-style S3 addressing");
        let s3_config = aws_sdk_s3::config::Builder::from(&aws_config)
            .force_path_style(true)
            .build();
        S3Client::from_conf(s3_config)
    } else {
        S3Client::new(&aws_config)
    };

    let endpoint_url = env::var("AWS_ENDPOINT_URL").ok();

    // Build the Kafka clients. Failure here is a dependency problem, not a code
    // problem, so it withholds readiness rather than exiting — same reasoning as
    // the config block above.
    let (producer, consumer) = match (kafka_settings.producer(), {
        let group = kafka_settings.consumer_group.clone();
        kafka_settings.consumer(&group)
    }) {
        (Ok(p), Ok(c)) => (p, c),
        (p, c) => {
            let e = p.err().or(c.err()).expect("at least one client failed");
            error!("Failed to build Kafka clients, staying unready: {e:#}");
            let _ = shutdown_rx.recv().await;
            return Ok(());
        }
    };

    // Cloned before the move into the processor; the tier consumers build their
    // own clients from the same settings.
    let kafka_settings_for_tiers = kafka_settings.clone();

    let processor = NestingProcessor::new(
        producer,
        s3_client,
        s3_bucket,
        aws_region,
        kafka_settings,
        endpoint_url,
    );

    // Spawn signal handler
    let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
        .context("Failed to register SIGTERM handler")?;
    let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
        .context("Failed to register SIGINT handler")?;

    let shutdown_tx_clone = shutdown_tx.clone();
    tokio::spawn(async move {
        tokio::select! {
            _ = sigterm.recv() => {
                info!("Received SIGTERM, initiating graceful shutdown...");
                let _ = shutdown_tx_clone.send(());
            }
            _ = sigint.recv() => {
                info!("Received SIGINT, initiating graceful shutdown...");
                let _ = shutdown_tx_clone.send(());
            }
        }
    });

    // Readiness now means "the broker answered", not merely "the clients were
    // constructed". Before T9 this flag flipped true before any connection was
    // attempted, so a pod with bad credentials reported Ready and then silently
    // consumed nothing. Fetching metadata is the cheapest call that actually
    // exercises the SASL/SCRAM handshake, and a failure here is precisely the case
    // where traffic should be withheld without restarting.
    match tokio::task::block_in_place(|| {
        consumer.fetch_metadata(None, std::time::Duration::from_secs(10))
    }) {
        Ok(md) => {
            info!(
                "Connected to Kafka: {} broker(s), {} topic(s) visible",
                md.brokers().len(),
                md.topics().len()
            );
            health.set_ready(true);
        }
        Err(e) => {
            error!("Kafka metadata fetch failed, staying unready: {e}");
            let _ = shutdown_rx.recv().await;
            let _ = shutdown_tx.send(());
            let _ = health_server.await;
            return Ok(());
        }
    }

    // One consumer per retry tier, each in its own group, applying its delay by
    // pausing the partition. Without these the ladder is write-only: a failure
    // reaches tier 1 and stops there instead of escalating to exhaustion.
    let mut tier_handles = Vec::new();
    for tier in 1..=3u32 {
        let processor = processor.clone();
        let settings = kafka_settings_for_tiers.clone();
        let shutdown = shutdown_tx.subscribe();
        tier_handles.push(tokio::spawn(async move {
            if let Err(e) =
                jagua_sqs_processor::retry_consumer::run_tier(processor, settings, tier, shutdown)
                    .await
            {
                error!("Tier {tier} consumer exited with error: {e:#}");
            }
        }));
    }

    // Start listening and processing
    let result = processor.listen_and_process(consumer, shutdown_rx).await;

    // Tiers drain on the same broadcast signal the main loop just observed.
    for handle in tier_handles {
        let _ = handle.await;
    }

    // No longer serving: fail readiness before the pod actually goes away, so
    // it is removed from any endpoint list while draining.
    health.set_ready(false);

    // Give a moment for any final cleanup
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Release the health server's graceful-shutdown future. Without this the
    // task lives until the process exits and the await below never returns.
    let _ = shutdown_tx.send(());
    let _ = health_server.await;

    // Flush buffered spans BEFORE returning. Dropping the provider without this
    // discards whatever is still in the batch queue — which is exactly the
    // spans from a crash, the ones actually worth having.
    if let Some(provider) = tracer_provider {
        if let Err(e) = provider.shutdown() {
            log::warn!("Failed to flush traces on shutdown: {}", e);
        }
    }

    if let Err(e) = &result {
        log::warn!("Processor exited with error: {}", e);
    }

    result
}
