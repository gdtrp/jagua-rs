use anyhow::{Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client as S3Client;
use aws_sdk_sqs::Client as SqsClient;
use jagua_sqs_processor::observability::{init_tracing, serve_health, Health};
use jagua_sqs_processor::SqsProcessor;
use log::info;
use std::env;
use tokio::signal;

/// Must match what the Java services declare, since Grafana joins a trace to
/// its logs on this exact string.
const SERVICE_NAME: &str = "jagua-nesting";

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
    let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel::<()>(1);

    let health = Health::new();
    let health_port: u16 = env::var("HEALTH_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let health_server = serve_health(health_port, health.clone(), shutdown_tx.subscribe()).await?;

    // Get configuration from environment variables
    let input_queue_url =
        env::var("INPUT_QUEUE_URL").context("INPUT_QUEUE_URL environment variable is required")?;
    let output_queue_url = env::var("OUTPUT_QUEUE_URL")
        .context("OUTPUT_QUEUE_URL environment variable is required")?;
    let s3_bucket = env::var("S3_BUCKET").context("S3_BUCKET environment variable is required")?;
    let aws_region = env::var("AWS_REGION").unwrap_or_else(|_| "eu-north-1".to_string());

    info!("Configuration:");
    info!("  INPUT_QUEUE_URL: {}", input_queue_url);
    info!("  OUTPUT_QUEUE_URL: {}", output_queue_url);
    info!("  S3_BUCKET: {}", s3_bucket);
    info!("  AWS_REGION: {}", aws_region);

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

    // Check if using custom endpoint (LocalStack/Minio)
    let use_path_style =
        env::var("AWS_ENDPOINT_URL").is_ok() || env::var("AWS_ENDPOINT_URL_SQS").is_ok();

    // Configure LocalStack endpoint if provided (applies to both SQS and S3)
    if let Ok(endpoint_url) = env::var("AWS_ENDPOINT_URL") {
        config_loader = config_loader.endpoint_url(&endpoint_url);
        info!(
            "Using AWS endpoint: {} (applies to both SQS and S3)",
            endpoint_url
        );
    } else if let Ok(endpoint_url) = env::var("AWS_ENDPOINT_URL_SQS") {
        config_loader = config_loader.endpoint_url(&endpoint_url);
        info!(
            "Using SQS endpoint: {} (applies to both SQS and S3)",
            endpoint_url
        );
    } else {
        info!("No AWS_ENDPOINT_URL set, using default AWS endpoints");
    }

    let config = config_loader.load().await;
    let sqs_client = SqsClient::new(&config);

    // Create S3 client - use path-style addressing for LocalStack/Minio
    // (virtual-hosted style like bucket.localstack:4566 won't work with local services)
    let s3_client = if use_path_style {
        info!("Using path-style S3 addressing for LocalStack/Minio compatibility");
        let s3_config = aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(true)
            .build();
        S3Client::from_conf(s3_config)
    } else {
        S3Client::new(&config)
    };

    // Detect custom endpoint URL for S3 URL generation (LocalStack/Minio)
    let endpoint_url = env::var("AWS_ENDPOINT_URL").ok();

    // Create processor
    let processor = SqsProcessor::new(
        sqs_client,
        s3_client,
        s3_bucket,
        aws_region,
        input_queue_url,
        output_queue_url,
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

    // Everything the processor needs exists; accept traffic.
    health.set_ready(true);

    // Start listening and processing
    let result = processor.listen_and_process(shutdown_rx).await;

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
