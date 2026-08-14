use anyhow::{anyhow, Context, Result};
use aws_sdk_s3::Client as S3Client;
use base64::{engine::general_purpose, Engine as _};
use jagua_utils::svg_nesting::{AdaptiveNestingStrategy, NestingResult, PageResult, PartInput};
use log::{debug, error, info, warn};
use rdkafka::consumer::{CommitMode, Consumer, StreamConsumer};
use rdkafka::message::{BorrowedMessage, Headers, Message};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::{Offset, TopicPartitionList};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::Instrument;

use crate::kafka::{retry_headers, KafkaSettings, OffsetWatermark, RecordHeaders};
use crate::metrics::metrics;

/// Default maximum concurrent processing tasks
const DEFAULT_MAX_CONCURRENT_TASKS: usize = 20;

/// Maximum retry attempts for AWS operations
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Base delay for exponential backoff (in milliseconds)
const RETRY_BASE_DELAY_MS: u64 = 100;

/// TTL for orphaned cancellation registry entries (in seconds)
const CANCELLATION_REGISTRY_TTL_SECS: u64 = 300; // 5 minutes

/// Default execution timeout for nesting operations (in seconds)
const DEFAULT_EXECUTION_TIMEOUT_SECS: u64 = 600; // 10 minutes

/// A single part specification in a multi-part nesting request.
///
/// Ergonomic view of the generated `NestingRequestPart`; (de)serialization is governed by the
/// generated wire types via [`crate::wire`].
#[derive(Debug, Clone)]
pub struct SvgPartSpec {
    /// User-provided correlation ID for this part type
    pub item_id: String,
    /// HTTPS S3 URL to the SVG file
    pub svg_url: String,
    /// Number of copies of this part to nest
    pub amount_of_parts: usize,
    /// Optional grain-direction constraint: the exact set of rotations (in **whole
    /// degrees**) this part may be placed at, e.g. `[0, 180]` to keep the grain on one
    /// axis while allowing a 180° flip, or `[0]` to lock it fully. When present, only
    /// these orientations are used for this part, overriding the request-level
    /// `amount_of_rotations`. Absent ⇒ the part follows `amount_of_rotations` (today's
    /// behaviour).
    pub allowed_rotations: Option<Vec<i32>>,
}

/// Trait for downloading SVG bytes from a URL.
/// Abstracts S3 access so strategies and tests stay S3-agnostic.
#[async_trait::async_trait]
#[allow(dead_code)]
pub trait SvgDownloader: Send + Sync {
    /// Download SVG bytes from the given URL
    async fn download(&self, url: &str) -> Result<Vec<u8>>;
}

/// Request message structure for SQS queue.
///
/// Ergonomic view of the generated `NestingRequest` (engine-friendly `f32`/`usize` widths, worker
/// defaults applied). (De)serialization is governed by the generated wire type via [`crate::wire`]:
/// the JSON contract lives in the AsyncAPI spec, not in this struct.
///
/// For cancellation requests, only `correlation_id` and `cancelled: true` are required.
/// All other fields are required only when `cancelled` is false or not present.
#[derive(Debug, Clone)]
pub struct SqsNestingRequest {
    /// Unique identifier for tracking the request
    pub correlation_id: String,
    /// Base64-encoded SVG payload (deprecated, use svg_url instead)
    pub svg_base64: Option<String>,
    /// S3 URL to the input SVG file (format: s3://bucket/key or https://bucket.s3.region.amazonaws.com/key)
    pub svg_url: Option<String>,
    /// Bin width for nesting (required if not cancelled)
    pub bin_width: Option<f32>,
    /// Bin height for nesting (required if not cancelled)
    pub bin_height: Option<f32>,
    /// Spacing between parts (required if not cancelled)
    pub spacing: Option<f32>,
    /// Number of parts to nest (required for legacy single-part format, ignored if `parts` is set)
    pub amount_of_parts: Option<usize>,
    /// Multi-part specification: array of different SVG parts with counts.
    /// When present and non-empty, takes precedence over legacy svg_url/svg_base64 + amount_of_parts.
    pub parts: Option<Vec<SvgPartSpec>>,
    /// Number of rotations to try (absent / null ⇒ default 8, applied in [`crate::wire`])
    pub amount_of_rotations: usize,
    /// Output queue URL for results (falls back to default if omitted)
    pub output_queue_url: Option<String>,
    /// Whether this is a cancellation request
    pub cancelled: bool,
    /// When `Some(true)`, compute the maximum number of copies of a single part
    /// that fit on one sheet. Requires exactly one part type in the request.
    pub max_fit: Option<bool>,
    /// Bucket the worker must write outputs into. None ⇒ legacy default
    /// bucket (BE data-CDN path disabled).
    pub bucket: Option<String>,
    /// Key prefix under `bucket`; None ⇒ fall back to `nesting/{correlation_id}`.
    pub s3_prefix: Option<String>,
    /// Optional offcut (free-space) detection policy (JG-OFF-2). Absent ⇒ detection skipped
    /// and the response is byte-identical to today. Ignored for `maxFit` requests.
    pub offcut_policy: Option<jagua_utils::OffcutPolicy>,
    /// Optional per-request wall-clock cap (seconds) for the nesting optimization. When set,
    /// it overrides the strategy's default time budget (the 42s max_fit budget / 600s normal
    /// budget) and the cooperative execution timeout. Clamped to a 600s ceiling. Absent ⇒
    /// today's behaviour (max 600s).
    pub max_seconds: Option<u64>,
}

/// Generate an empty page SVG (used when all parts are placed)
fn decode_svg(encoded: &str) -> Result<Vec<u8>> {
    general_purpose::STANDARD
        .decode(encoded)
        .map_err(|e| anyhow!("Failed to decode svg_base64: {}", e))
}

fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Helper to safely lock a mutex, recovering from poison errors
fn safe_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Marks a failure as *infrastructure* trouble rather than a bad request.
///
/// The distinction decides what happens to the message. A malformed request will
/// fail identically on every attempt, so it gets an error response and is done
/// with. An S3 outage might not, so it belongs on the retry ladder.
///
/// This exists because the two were previously indistinguishable: an upload that
/// exhausted its attempts was logged and swallowed, and the worker then published
/// a `final: true` response carrying `partsPlaced: 4` and no page URLs — a
/// success as far as the schema is concerned, for a job that produced nothing
/// the caller can use. Observed on VK staging, where S3 eu-north-1 is unreachable.
#[derive(Debug)]
pub struct RetryableError(pub String);

impl std::fmt::Display for RetryableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for RetryableError {}

/// Whether an error should be escalated to the retry ladder.
///
/// Walks the whole chain, so a `RetryableError` keeps its meaning through any
/// number of `.context(...)` wrappings on the way up.
pub fn is_retryable(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|cause| cause.downcast_ref::<RetryableError>().is_some())
}

/// Retry an async operation with exponential backoff
async fn retry_with_backoff<F, Fut, T, E>(
    operation_name: &str,
    mut operation: F,
) -> std::result::Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
    E: std::fmt::Display,
{
    let mut attempts = 0;
    loop {
        attempts += 1;
        match operation().await {
            Ok(result) => {
                metrics()
                    .operation_retries
                    .with_label_values(&[operation_name, "success"])
                    .observe(attempts as f64);
                return Ok(result);
            }
            Err(e) if attempts < MAX_RETRY_ATTEMPTS => {
                let delay = Duration::from_millis(RETRY_BASE_DELAY_MS * 2u64.pow(attempts - 1));
                warn!(
                    "{} failed (attempt {}/{}): {}. Retrying in {:?}...",
                    operation_name, attempts, MAX_RETRY_ATTEMPTS, e, delay
                );
                tokio::time::sleep(delay).await;
            }
            Err(e) => {
                error!(
                    "{} failed after {} attempts: {}",
                    operation_name, attempts, e
                );
                // Every retried operation in this file funnels through here, so
                // one instrumentation point covers all of them. This is the
                // *intra-operation* budget running out (an S3 upload, a produce);
                // the message-level retry ladder and its
                // `cutl_retries_exhausted_total` counter sit one level above.
                metrics()
                    .operation_retries
                    .with_label_values(&[operation_name, "exhausted"])
                    .observe(attempts as f64);
                return Err(e);
            }
        }
    }
}

/// Determine the last page SVG bytes based on nesting result
/// Returns None if all parts placed (no unplaced sheet needed),
/// unplaced parts SVG if available, otherwise the last filled page
fn determine_last_page_svg(result: &NestingResult, first_page_bytes: &[u8]) -> Option<Vec<u8>> {
    if result.parts_placed == result.total_parts_requested {
        // All parts placed - no last page needed
        info!(
            "All parts placed ({}), no unplaced parts sheet needed",
            result.parts_placed
        );
        None
    } else if let Some(ref unplaced_svg) = result.unplaced_parts_svg {
        // Some parts unplaced - use unplaced parts SVG
        info!(
            "Some parts unplaced ({} of {}), using unplaced parts SVG",
            result.parts_placed, result.total_parts_requested
        );
        Some(unplaced_svg.clone())
    } else {
        // No unplaced parts SVG - use last filled page or first page
        info!(
            "No unplaced parts SVG available, using last filled page (parts_placed: {} of {})",
            result.parts_placed, result.total_parts_requested
        );
        Some(
            result
                .page_svgs
                .last()
                .unwrap_or(&first_page_bytes.to_vec())
                .clone(),
        )
    }
}

/// Get the maximum concurrent tasks from environment or use default
fn get_max_concurrent_tasks() -> usize {
    std::env::var("MAX_CONCURRENT_TASKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_TASKS)
}

/// Get the execution timeout from environment or use default (10 minutes)
fn get_execution_timeout() -> Duration {
    let secs = std::env::var("EXECUTION_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_EXECUTION_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Cancellation registry entry with timestamp for TTL-based cleanup
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CancellationEntry {
    pub(crate) cancelled: bool,
    pub(crate) created_at: Instant,
}

/// Response message structure for SQS queue.
///
/// Ergonomic view of the generated `NestingResponse`; serialization is governed by the generated
/// wire type via [`crate::wire`] (camelCase, `improvement`/`final` literals, spec numeric widths).
#[derive(Debug, Clone, PartialEq)]
pub struct SqsNestingResponse {
    /// Correlation ID from request
    pub correlation_id: String,
    /// S3 URL to the first page SVG (format: s3://bucket/nesting/{requestId}/first-page.svg)
    pub first_page_svg_url: Option<String>,
    /// S3 URL to the last page SVG (format: s3://bucket/nesting/{requestId}/last-page.svg)
    pub last_page_svg_url: Option<String>,
    /// Number of sheets/pages used
    pub sheets: Option<usize>,
    /// Estimated total sheets the run will produce, when known up front (deterministic fast paths).
    /// Absent for the general LBF path. Enables a determinate progress bar (CUTL-160 #1).
    pub sheets_total: Option<usize>,
    /// S3 URLs to all page SVGs (ordered by page index)
    pub page_svg_urls: Option<Vec<String>>,
    /// Per-page results: utilisation and placements grouped by page
    pub pages: Option<Vec<PageResult>>,
    /// Number of parts placed
    pub parts_placed: usize,
    /// Average bin utilisation ratio (0.0 to 1.0) across all pages
    pub utilisation: f32,
    /// Whether this is an intermediate improvement (always false for simple strategy)
    pub is_improvement: bool,
    /// Whether this is the final result (always true for simple strategy)
    pub is_final: bool,
    /// Timestamp in seconds since epoch
    pub timestamp: u64,
    /// Error message if processing failed
    pub error_message: Option<String>,
}

/// Processor for handling SVG nesting requests off Kafka.
///
/// Per the T9 directive in `cutl-infra/docs/rfc/001-vk-cloud-migration.md`, the SQS
/// fields were swapped in place rather than hidden behind a transport trait: the
/// receive loop's shape is itself load-bearing (see [`Self::listen_and_process`]),
/// and an abstraction over it would obscure exactly the part that matters.
///
/// S3 is untouched by the port. It stays on AWS, reached through a relay in a
/// region VK can actually see — the payload path is orthogonal to the queue.
#[derive(Clone)]
pub struct NestingProcessor {
    producer: FutureProducer,
    s3_client: S3Client,
    s3_bucket: String,
    aws_region: String,
    /// Topic responses go to unless a request overrides it.
    response_topic: String,
    kafka: KafkaSettings,
    endpoint_url: Option<String>,
    cancellation_registry: Arc<Mutex<HashMap<String, CancellationEntry>>>,
}

impl NestingProcessor {
    /// Mark a correlation_id as cancelled. Returns true if it was already registered.
    fn mark_cancelled(&self, correlation_id: &str) -> bool {
        let mut registry = safe_lock(&self.cancellation_registry);
        if let Some(entry) = registry.get_mut(correlation_id) {
            entry.cancelled = true;
            true
        } else {
            // Insert new entry for future cancellation check
            registry.insert(
                correlation_id.to_string(),
                CancellationEntry {
                    cancelled: true,
                    created_at: Instant::now(),
                },
            );
            false
        }
    }

    /// Check if a correlation_id is cancelled
    fn is_cancelled(&self, correlation_id: &str) -> bool {
        let registry = safe_lock(&self.cancellation_registry);
        registry
            .get(correlation_id)
            .map(|e| e.cancelled)
            .unwrap_or(false)
    }

    /// Register a correlation_id in the cancellation registry
    fn register_correlation_id(&self, correlation_id: &str) {
        let mut registry = safe_lock(&self.cancellation_registry);
        registry.insert(
            correlation_id.to_string(),
            CancellationEntry {
                cancelled: false,
                created_at: Instant::now(),
            },
        );
    }

    /// Remove a correlation_id from the cancellation registry
    fn unregister_correlation_id(&self, correlation_id: &str) {
        let mut registry = safe_lock(&self.cancellation_registry);
        registry.remove(correlation_id);
    }

    /// Clean up expired entries from the cancellation registry
    fn cleanup_expired_entries(&self) {
        let mut registry = safe_lock(&self.cancellation_registry);
        let now = Instant::now();
        let ttl = Duration::from_secs(CANCELLATION_REGISTRY_TTL_SECS);

        let expired_keys: Vec<String> = registry
            .iter()
            .filter(|(_, entry)| now.duration_since(entry.created_at) > ttl)
            .map(|(key, _)| key.clone())
            .collect();

        for key in &expired_keys {
            registry.remove(key);
        }

        if !expired_keys.is_empty() {
            info!(
                "Cleaned up {} expired entries from cancellation registry",
                expired_keys.len()
            );
        }
    }

    /// Create a new processor.
    pub fn new(
        producer: FutureProducer,
        s3_client: S3Client,
        s3_bucket: String,
        aws_region: String,
        kafka: KafkaSettings,
        endpoint_url: Option<String>,
    ) -> Self {
        Self {
            producer,
            s3_client,
            s3_bucket,
            aws_region,
            response_topic: kafka.response_topic.clone(),
            kafka,
            endpoint_url,
            cancellation_registry: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Download SVG from S3 URL
    async fn download_svg_from_s3(&self, s3_url: &str) -> Result<Vec<u8>> {
        // Parse S3 URL (supports both s3://bucket/key and https://bucket.s3.region.amazonaws.com/key)
        let (bucket, key) = parse_s3_url(s3_url)?;

        info!(
            "Downloading SVG from S3: url={}, bucket={}, key={}",
            s3_url, bucket, key
        );

        let response = match self
            .s3_client
            .get_object()
            .bucket(&bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(resp) => resp,
            Err(e) => {
                // Log detailed error information
                error!("S3 GetObject failed: {}", e);
                error!("S3 URL: {}, bucket: {}, key: {}", s3_url, bucket, key);

                // Try to extract more error details
                use aws_sdk_s3::error::ProvideErrorMetadata;
                if let Some(code) = e.code() {
                    error!("S3 error code: {}", code);
                }
                if let Some(message) = e.message() {
                    error!("S3 error message: {}", message);
                }

                // Log the full error
                error!("Full error details: {}", e);

                // Retryable: a download failure is the network or S3, not the
                // request. With the eu-west-1 relay adding a hop, this path gets
                // more likely rather than less.
                return Err(anyhow!(RetryableError(format!(
                    "Failed to download SVG from S3: bucket={bucket}, key={key}, error={e}"
                ))));
            }
        };

        // Collect the body stream into bytes
        let svg_bytes = response
            .body
            .collect()
            .await
            .context("Failed to read S3 object body")?
            .into_bytes()
            .to_vec();
        info!("Downloaded SVG from S3: {} bytes", svg_bytes.len());
        Ok(svg_bytes)
    }
}

#[async_trait::async_trait]
impl SvgDownloader for NestingProcessor {
    async fn download(&self, url: &str) -> Result<Vec<u8>> {
        self.download_svg_from_s3(url).await
    }
}

/// Parse S3 URL and extract bucket and key
/// Supports multiple formats:
/// - s3://bucket/key
/// - https://bucket.s3.region.amazonaws.com/key (virtual-hosted style)
/// - https://bucket.s3-region.amazonaws.com/key (virtual-hosted style with dash)
/// - https://s3.region.amazonaws.com/bucket/key (path-style)
/// - https://s3-region.amazonaws.com/bucket/key (path-style with dash)
/// - http://hostname:port/bucket/key (path-style, for localstack/minio)
/// - https://hostname:port/bucket/key (path-style, for localstack/minio)
fn parse_s3_url(s3_url: &str) -> Result<(String, String)> {
    // Handle s3://bucket/key format
    if let Some(path) = s3_url.strip_prefix("s3://") {
        if let Some(slash_pos) = path.find('/') {
            let bucket = path[..slash_pos].to_string();
            let key = path[slash_pos + 1..].to_string();
            if bucket.is_empty() || key.is_empty() {
                return Err(anyhow!(
                    "Invalid S3 URL: bucket or key is empty: {}",
                    s3_url
                ));
            }
            return Ok((bucket, key));
        }
        return Err(anyhow!("Invalid S3 URL format (missing key): {}", s3_url));
    }

    // Handle HTTP/HTTPS formats
    let url = if let Some(stripped) = s3_url.strip_prefix("https://") {
        stripped
    } else if let Some(stripped) = s3_url.strip_prefix("http://") {
        stripped
    } else {
        return Err(anyhow!(
            "Unsupported S3 URL format (must start with s3://, http://, or https://): {}",
            s3_url
        ));
    };

    // Check for AWS path-style URL: https://s3.region.amazonaws.com/bucket/key
    // or https://s3-region.amazonaws.com/bucket/key
    if url.starts_with("s3.") || url.starts_with("s3-") {
        // Path-style URL
        if let Some(aws_pos) = url.find(".amazonaws.com/") {
            let path = &url[aws_pos + 15..];
            if let Some(slash_pos) = path.find('/') {
                let bucket = path[..slash_pos].to_string();
                let key = path[slash_pos + 1..].to_string();
                if bucket.is_empty() || key.is_empty() {
                    return Err(anyhow!(
                        "Invalid S3 path-style URL: bucket or key is empty: {}",
                        s3_url
                    ));
                }
                return Ok((bucket, key));
            }
            return Err(anyhow!(
                "Invalid S3 path-style URL (missing key): {}",
                s3_url
            ));
        }
    }

    // Virtual-hosted style: https://bucket.s3.region.amazonaws.com/key
    // or https://bucket.s3-region.amazonaws.com/key
    if let Some(s3_pos) = url.find(".s3") {
        let bucket = url[..s3_pos].to_string();
        // Extract key (everything after .amazonaws.com/)
        if let Some(aws_pos) = url.find(".amazonaws.com/") {
            let key = url[aws_pos + 15..].to_string();
            if bucket.is_empty() || key.is_empty() {
                return Err(anyhow!(
                    "Invalid S3 virtual-hosted URL: bucket or key is empty: {}",
                    s3_url
                ));
            }
            return Ok((bucket, key));
        }
        return Err(anyhow!(
            "Invalid S3 virtual-hosted URL (missing .amazonaws.com): {}",
            s3_url
        ));
    }

    // Path-style URL for non-AWS S3-compatible services (e.g., localstack, minio):
    // http://hostname:port/bucket/key or https://hostname:port/bucket/key
    // Find the first slash after the host (and optional port)
    if let Some(first_slash) = url.find('/') {
        let path = &url[first_slash + 1..];
        if let Some(second_slash) = path.find('/') {
            let bucket = path[..second_slash].to_string();
            let key = path[second_slash + 1..].to_string();
            if bucket.is_empty() || key.is_empty() {
                return Err(anyhow!(
                    "Invalid S3-compatible URL: bucket or key is empty: {}",
                    s3_url
                ));
            }
            return Ok((bucket, key));
        }
        return Err(anyhow!(
            "Invalid S3-compatible URL (missing key after bucket): {}",
            s3_url
        ));
    }

    Err(anyhow!("Invalid S3-compatible URL format: {}", s3_url))
}

/// Internal helper function to upload SVG to S3 (used by both improvement and final responses).
/// `s3_prefix` is the full key prefix (without trailing slash) — callers should pass either the
/// BE-provided `request.s3_prefix` or the legacy default `nesting/{correlation_id}`.
async fn upload_svg_to_s3_internal(
    s3_client: &S3Client,
    s3_bucket: &str,
    aws_region: &str,
    svg_bytes: &[u8],
    s3_prefix: &str,
    filename: &str,
    endpoint_url: Option<&str>,
) -> Result<String> {
    let s3_key = format!("{}/{}", s3_prefix.trim_end_matches('/'), filename);
    let s3_url = if let Some(endpoint) = endpoint_url {
        // Path-style URL for LocalStack/Minio
        format!(
            "{}/{}/{}",
            endpoint.trim_end_matches('/'),
            s3_bucket,
            s3_key
        )
    } else {
        // Virtual-hosted style for AWS
        format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            s3_bucket, aws_region, s3_key
        )
    };

    info!(
        "Uploading SVG to S3: bucket={}, key={}, size={} bytes",
        s3_bucket,
        s3_key,
        svg_bytes.len()
    );

    s3_client
        .put_object()
        .bucket(s3_bucket)
        .key(&s3_key)
        .body(aws_sdk_s3::primitives::ByteStream::from(svg_bytes.to_vec()))
        .content_type("image/svg+xml")
        .send()
        .await
        .with_context(|| {
            format!(
                "Failed to upload SVG to S3: bucket={}, key={}",
                s3_bucket, s3_key
            )
        })?;

    info!("Successfully uploaded SVG to S3: {}", s3_url);
    Ok(s3_url)
}

/// Producer-side guard mirroring the broker's `message.max.bytes`.
///
/// The old constant was named for SQS's 1 MiB body limit; Kafka's default happens
/// to be the same order, so the guard survives the port — but it is now a *broker*
/// limit, and exceeding it fails the produce rather than the API call.
const KAFKA_MAX_MESSAGE_BYTES: usize = 1024 * 1024;

impl NestingProcessor {
    /// Publish one response record.
    ///
    /// **Keyed by `correlationId`, always.** Keying is what gives a job's responses
    /// a single partition and therefore real ordering, and on the request side it
    /// is what makes cancellation work at all. An unkeyed produce here would round-
    /// robin a job's improvement responses across partitions and let the consumer
    /// observe them out of order.
    async fn publish_response(
        producer: &FutureProducer,
        topic: &str,
        response: &SqsNestingResponse,
    ) -> Result<()> {
        let payload = serde_json::to_string(response).context("Failed to serialize response")?;

        let size_kb = payload.len() / 1024;
        if payload.len() > KAFKA_MAX_MESSAGE_BYTES {
            return Err(anyhow!(
                "Message size {} KB exceeds the broker's message.max.bytes of {} KB",
                size_kb,
                KAFKA_MAX_MESSAGE_BYTES / 1024
            ));
        }

        debug!(
            "Producing to {}: correlation_id={}, is_final={}, size={} KB",
            topic, response.correlation_id, response.is_final, size_kb
        );

        // Carry the current trace out with the response, so the backend's consumer
        // links back to the job that produced it instead of opening a new trace.
        let headers = crate::trace_context::inject_current(rdkafka::message::OwnedHeaders::new());

        let record = FutureRecord::to(topic)
            .key(&response.correlation_id)
            .payload(&payload)
            .headers(headers);

        // `Duration::ZERO` is the *enqueue* timeout, not the delivery timeout: it
        // means "fail immediately if the producer queue is full" rather than
        // blocking the caller. Delivery itself is bounded by `message.timeout.ms`.
        producer
            .send(record, Duration::ZERO)
            .await
            .map_err(|(e, _)| {
                anyhow!(
                    "Failed to produce to {}: correlation_id={}, size={} KB, error={}",
                    topic,
                    response.correlation_id,
                    size_kb,
                    e
                )
            })?;

        Ok(())
    }

    /// Publish a response with retries. Retains the original method name so the
    /// ~8 call sites in this file are untouched by the transport swap.
    pub async fn send_to_output_queue(
        &self,
        topic: &str,
        response: &SqsNestingResponse,
    ) -> Result<()> {
        let producer = self.producer.clone();
        let topic_owned = topic.to_string();
        let response_clone = response.clone();

        retry_with_backoff("send_to_output_queue", || {
            let producer = producer.clone();
            let topic = topic_owned.clone();
            let resp = response_clone.clone();
            async move { Self::publish_response(&producer, &topic, &resp).await }
        })
        .await?;

        debug!(
            "Emitted response to {}: correlation_id={}, parts_placed={}, is_final={}",
            topic, response.correlation_id, response.parts_placed, response.is_final
        );

        Ok(())
    }

    /// Process a single message from the queue
    /// Returns Ok(()) on success, or sends error response and returns Ok(()) on error
    /// (message should always be acknowledged after calling this)
    pub async fn process_message(&self, _receipt_handle: &str, body: &str) -> Result<()> {
        // Parse request - if this fails, we can't get correlation_id, so we'll log and return error
        let request: SqsNestingRequest = match serde_json::from_str(body) {
            Ok(req) => req,
            Err(e) => {
                let error_msg = format!(
                    "Failed to parse request message: {}. Body (first 200 chars): {}",
                    e,
                    body.chars().take(200).collect::<String>()
                );
                error!("{}", error_msg);
                // Try to extract correlation_id from body if possible
                if let Ok(partial) = serde_json::from_str::<serde_json::Value>(body) {
                    if let Some(corr_id) = partial.get("correlationId").and_then(|v| v.as_str()) {
                        let output_queue_url = partial
                            .get("outputQueueUrl")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| self.response_topic.clone());

                        let error_response = SqsNestingResponse {
                            correlation_id: corr_id.to_string(),
                            first_page_svg_url: None,
                            last_page_svg_url: None,
                            sheets: None,
                            sheets_total: None,
                            page_svg_urls: None,
                            pages: None,
                            parts_placed: 0,
                            utilisation: 0.0,
                            is_improvement: false,
                            is_final: true,
                            timestamp: current_timestamp(),
                            error_message: Some(error_msg.clone()),
                        };

                        if let Err(send_err) = self
                            .send_to_output_queue(&output_queue_url, &error_response)
                            .await
                        {
                            error!("Failed to send error response: {}", send_err);
                        }
                    }
                }
                return Err(anyhow::anyhow!(error_msg));
            }
        };

        // Calculate SVG size info for logging
        let svg_size_info = if let Some(ref svg_url) = request.svg_url {
            format!("S3 URL: {}", svg_url)
        } else if let Some(ref svg_b64) = request.svg_base64 {
            let base64_len = svg_b64.len();
            // Try to decode to get exact size, fall back to approximation if decoding fails
            match general_purpose::STANDARD.decode(svg_b64) {
                Ok(decoded) => format!("{} bytes (base64: {} bytes)", decoded.len(), base64_len),
                Err(_) => {
                    // Base64 encoding increases size by ~33%, so approximate decoded size
                    let approx_decoded_size = (base64_len * 3) / 4;
                    format!(
                        "~{} bytes (base64: {} bytes, decode failed)",
                        approx_decoded_size, base64_len
                    )
                }
            }
        } else {
            "N/A".to_string()
        };

        info!(
            "Processing request: correlation_id={}, bin_width={:?}, bin_height={:?}, spacing={:?}, amount_of_parts={:?}, amount_of_rotations={}, cancelled={}, svg_size={}, output_queue_url={:?}",
            request.correlation_id,
            request.bin_width,
            request.bin_height,
            request.spacing,
            request.amount_of_parts,
            request.amount_of_rotations,
            request.cancelled,
            svg_size_info,
            request.output_queue_url.as_deref().unwrap_or("default")
        );

        // Handle cancellation requests
        if request.cancelled {
            let was_running = self.mark_cancelled(&request.correlation_id);
            if was_running {
                info!(
                    "Cancellation request received and forwarded to running optimizer: correlation_id={}",
                    request.correlation_id
                );
            } else {
                info!(
                    "Cancellation request received for idle correlation_id={}, future runs will be skipped",
                    request.correlation_id
                );
            }
            return Ok(());
        }

        // Determine output queue (use request override if provided)
        let output_queue_url = request
            .output_queue_url
            .clone()
            .unwrap_or_else(|| self.response_topic.clone());

        // Validate required fields for non-cancellation requests
        let has_multi_parts = request.parts.as_ref().is_some_and(|p| !p.is_empty());
        let has_legacy_svg = request.svg_base64.is_some() || request.svg_url.is_some();

        // Must have either multi-part or legacy SVG source
        if !has_multi_parts && !has_legacy_svg {
            let error_msg = "Missing required field: either 'parts' array or 'svgBase64'/'svgUrl' must be provided";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                sheets: None,
                sheets_total: None,
                page_svg_urls: None,
                pages: None,
                parts_placed: 0,
                utilisation: 0.0,
                is_improvement: false,
                is_final: true,
                timestamp: current_timestamp(),
                error_message: Some(error_msg.to_string()),
            };
            if let Err(send_err) = self
                .send_to_output_queue(&output_queue_url, &error_response)
                .await
            {
                error!("Failed to send error response: {}", send_err);
            }
            return Ok(());
        }

        // Legacy format requires amount_of_parts
        if !has_multi_parts && request.amount_of_parts.is_none() {
            let error_msg =
                "Missing required field: amount_of_parts (required for legacy single-part format)";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                sheets: None,
                sheets_total: None,
                page_svg_urls: None,
                pages: None,
                parts_placed: 0,
                utilisation: 0.0,
                is_improvement: false,
                is_final: true,
                timestamp: current_timestamp(),
                error_message: Some(error_msg.to_string()),
            };
            if let Err(send_err) = self
                .send_to_output_queue(&output_queue_url, &error_response)
                .await
            {
                error!("Failed to send error response: {}", send_err);
            }
            return Ok(());
        }

        // Validate bin dimensions and spacing
        for (field_name, field_value) in [
            ("bin_width", &request.bin_width),
            ("bin_height", &request.bin_height),
            ("spacing", &request.spacing),
        ] {
            if field_value.is_none() {
                let error_msg = format!("Missing required field: {}", field_name);
                error!("{}", error_msg);
                let error_response = SqsNestingResponse {
                    correlation_id: request.correlation_id.clone(),
                    first_page_svg_url: None,
                    last_page_svg_url: None,
                    sheets: None,
                    sheets_total: None,
                    page_svg_urls: None,
                    pages: None,
                    parts_placed: 0,
                    utilisation: 0.0,
                    is_improvement: false,
                    is_final: true,
                    timestamp: current_timestamp(),
                    error_message: Some(error_msg),
                };
                if let Err(send_err) = self
                    .send_to_output_queue(&output_queue_url, &error_response)
                    .await
                {
                    error!("Failed to send error response: {}", send_err);
                }
                return Ok(());
            }
        }

        // Process the request and handle errors by sending error response
        let result = self
            .process_nesting_request(&request, &output_queue_url)
            .await;

        if let Err(e) = &result {
            let error_msg = format!("{:#}", e);
            error!(
                "Failed to process message for correlation_id={}: {:?}",
                request.correlation_id, e
            );

            // Infrastructure failures go to the retry ladder instead of becoming a
            // response. Two reasons not to answer here: a `final: true` error tells
            // the caller the job is permanently done, which a retry then
            // contradicts with a second final response; and answering would make
            // the handler look successful, so nothing would ever escalate.
            //
            // If every tier is exhausted the message is dropped and
            // cutl_retries_exhausted_total fires — that alert is the contract's
            // substitute for a DLQ, and it is deliberately the only signal.
            if is_retryable(e) {
                warn!(
                    "Infrastructure failure for correlation_id={}, escalating to the retry ladder: {}",
                    request.correlation_id, error_msg
                );
                return Err(anyhow!(RetryableError(error_msg)));
            }

            // Send error response for internal processing errors
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                sheets: None,
                sheets_total: None,
                page_svg_urls: None,
                pages: None,
                parts_placed: 0,
                utilisation: 0.0,
                is_improvement: false,
                is_final: true,
                timestamp: current_timestamp(),
                error_message: Some(error_msg),
            };

            if let Err(send_err) = self
                .send_to_output_queue(&output_queue_url, &error_response)
                .await
            {
                error!("Failed to send error response: {}", send_err);
            } else {
                info!(
                    "Sent error response to queue for correlation_id={}",
                    request.correlation_id
                );
            }
        }

        // Always return Ok so message gets acknowledged
        Ok(())
    }

    /// Internal method to process nesting request
    async fn process_nesting_request(
        &self,
        request: &SqsNestingRequest,
        output_queue_url: &str,
    ) -> Result<()> {
        // Register correlation_id in cancellation registry
        self.register_correlation_id(&request.correlation_id);

        // Ensure cleanup happens even on error
        let result = async {
            // Unwrap required fields (validation already done in process_message)
            let bin_width = request.bin_width.unwrap();
            let bin_height = request.bin_height.unwrap();
            let spacing = request.spacing.unwrap();

            // Normalize into Vec<PartInput>
            let decode_start = Instant::now();
            let part_inputs: Vec<PartInput> = if let Some(ref parts) = request.parts {
                if !parts.is_empty() {
                    // Multi-part format: download each SVG from its URL
                    let mut inputs = Vec::with_capacity(parts.len());
                    for spec in parts {
                        info!("Downloading SVG for multi-part from: {}", spec.svg_url);
                        let svg_bytes = self.download_svg_from_s3(&spec.svg_url).await?;
                        inputs.push(PartInput {
                            svg_bytes,
                            count: spec.amount_of_parts,
                            item_id: Some(spec.item_id.clone()),
                            // Integer request degrees → f32 degrees for the geometry layer.
                            allowed_rotations: spec
                                .allowed_rotations
                                .as_ref()
                                .map(|angles| angles.iter().map(|&d| d as f32).collect()),
                        });
                    }
                    inputs
                } else {
                    return Err(anyhow!("'parts' array is empty"));
                }
            } else {
                // Legacy single-part format
                let amount_of_parts = request.amount_of_parts.unwrap();
                let svg_bytes = if let Some(ref svg_url) = request.svg_url {
                    info!("Downloading SVG from S3: {}", svg_url);
                    self.download_svg_from_s3(svg_url).await?
                } else if let Some(ref svg_base64) = request.svg_base64 {
                    decode_svg(svg_base64)?
                } else {
                    return Err(anyhow!("Neither svg_url nor svg_base64 provided"));
                };
                vec![PartInput {
                    svg_bytes,
                    count: amount_of_parts,
                    item_id: None,
                    // Legacy single-part format has no per-part grain field.
                    allowed_rotations: None,
                }]
            };

            let total_parts: usize = part_inputs.iter().map(|p| p.count).sum();
            info!(
                "SVG payload ready: {} part type(s), {} total parts (took {:?})",
                part_inputs.len(),
                total_parts,
                decode_start.elapsed()
            );

            let max_fit = request.max_fit.unwrap_or(false);
            if max_fit && part_inputs.len() != 1 {
                return Err(anyhow!(
                    "max_fit requires exactly one part type, got {}",
                    part_inputs.len()
                ));
            }

            // Create cancellation checker closure using the helper method
            // The checker also monitors for execution timeout.
            // A per-request `maxSeconds` (clamped to a 600s ceiling) overrides both the
            // cooperative execution timeout and the strategy's internal time budget; absent
            // ⇒ today's behaviour (the env default, max 600s).
            let max_seconds = request.max_seconds.map(|s| s.min(600));
            let execution_timeout = match max_seconds {
                Some(s) => Duration::from_secs(s),
                None => get_execution_timeout(),
            };
            let execution_start = Instant::now();
            let timed_out = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let timed_out_for_checker = timed_out.clone();
            let processor_clone = self.clone();
            let correlation_id_clone = request.correlation_id.clone();
            let cancellation_check_count = Arc::new(AtomicU64::new(0));
            let cancellation_check_count_for_log = cancellation_check_count.clone();
            let cancellation_checker = move || {
                let count = cancellation_check_count_for_log.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(1000) {
                    log::debug!("Cancellation checker called {} times", count);
                }
                // Check for timeout
                if execution_start.elapsed() > execution_timeout {
                    timed_out_for_checker.store(true, Ordering::SeqCst);
                    log::warn!("Execution timeout reached after {:?}", execution_start.elapsed());
                    return true;
                }
                processor_clone.is_cancelled(&correlation_id_clone)
            };

            // Create channel for sending improvement results from sync callback to async task
            let (tx, mut rx) = mpsc::unbounded_channel::<NestingResult>();

            // Spawn async task to handle improvement messages
            info!("Spawning async task to handle improvement messages");
            let producer_for_task = self.producer.clone();
            let s3_client_for_task = self.s3_client.clone();
            // BE may override bucket / key-prefix per request; fall back to the
            // worker's defaults so legacy callers keep working unchanged.
            let s3_bucket_for_task = request
                .bucket
                .clone()
                .unwrap_or_else(|| self.s3_bucket.clone());
            let s3_prefix_for_task = request
                .s3_prefix
                .clone()
                .unwrap_or_else(|| format!("nesting/{}", request.correlation_id));
            let aws_region_for_task = self.aws_region.clone();
            let endpoint_url_for_task = self.endpoint_url.clone();
            let output_queue_url_for_task = output_queue_url.to_string();
            let correlation_id_for_task = request.correlation_id.clone();

            // `tokio::spawn` does NOT inherit the caller's span — the spawned future
            // starts with an empty span stack. Without this, `inject_current` inside
            // `publish_response` sees an invalid context and the propagator writes no
            // headers at all, so every improvement lands in its own orphan trace while
            // the final response stays in the caller's. Created here, before the spawn,
            // so it is a child of `nesting.handle` and carries the same trace id.
            let improvement_span = tracing::info_span!("nesting.improvements");

            let improvement_task_handle = tokio::spawn(async move {
                info!("Improvement task started, waiting for messages...");
                while let Some(result) = rx.recv().await {
                    info!("Improvement task received message: {} parts placed, {} pages", result.parts_placed, result.page_svgs.len());

                    // Upload all page SVGs to S3
                    let mut page_svg_urls: Vec<String> = Vec::new();
                    for (page_idx, page_bytes) in result.page_svgs.iter().enumerate() {
                        let filename = format!("page-{}.svg", page_idx);
                        match retry_with_backoff(&format!("upload improvement page {}", page_idx), || {
                            let client = s3_client_for_task.clone();
                            let bucket = s3_bucket_for_task.clone();
                            let region = aws_region_for_task.clone();
                            let endpoint = endpoint_url_for_task.clone();
                            let bytes = page_bytes.clone();
                            let prefix = s3_prefix_for_task.clone();
                            let fname = filename.clone();
                            async move {
                                upload_svg_to_s3_internal(&client, &bucket, &region, &bytes, &prefix, &fname, endpoint.as_deref()).await
                            }
                        }).await {
                            Ok(url) => {
                                info!("Uploaded improvement page {} SVG to S3: {}", page_idx, url);
                                page_svg_urls.push(url);
                            }
                            Err(e) => {
                                error!("Failed to upload improvement page {} SVG to S3 after retries: {}", page_idx, e);
                            }
                        }
                    }

                    // Upload last page (unplaced parts) only if there are unplaced parts
                    let first_page_bytes = result.page_svgs.first()
                        .unwrap_or(&result.combined_svg);
                    let last_page_svg_url = if let Some(last_page_bytes) = determine_last_page_svg(
                        &result,
                        first_page_bytes,
                    ) {
                        match retry_with_backoff("upload improvement last page", || {
                            let client = s3_client_for_task.clone();
                            let bucket = s3_bucket_for_task.clone();
                            let region = aws_region_for_task.clone();
                            let endpoint = endpoint_url_for_task.clone();
                            let bytes = last_page_bytes.clone();
                            let prefix = s3_prefix_for_task.clone();
                            async move {
                                upload_svg_to_s3_internal(&client, &bucket, &region, &bytes, &prefix, "last-page.svg", endpoint.as_deref()).await
                            }
                        }).await {
                            Ok(url) => {
                                info!("Uploaded improvement last page SVG to S3: {}", url);
                                Some(url)
                            }
                            Err(e) => {
                                error!("Failed to upload improvement last page SVG to S3 after retries: {}", e);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let first_page_svg_url = page_svg_urls.first().cloned();

                    // Build pages with S3 URLs populated
                    let response_pages: Vec<PageResult> = result.pages.iter().map(|p| {
                        let mut page = p.clone();
                        page.svg_url = page_svg_urls.get(p.page_index).cloned();
                        page
                    }).collect();

                    let sheets = Some(response_pages.len());

                    // Create improvement response with S3 URLs
                    let response = SqsNestingResponse {
                        correlation_id: correlation_id_for_task.clone(),
                        first_page_svg_url,
                        last_page_svg_url,
                        sheets,
                        sheets_total: result.sheets_total_estimate,
                        page_svg_urls: Some(page_svg_urls),
                        pages: Some(response_pages),
                        parts_placed: result.parts_placed,
                        utilisation: result.utilisation,
                        is_improvement: true,
                        is_final: false,
                        timestamp: current_timestamp(),
                        error_message: None,
                    };

                    info!("Sending improvement response: {} parts placed", response.parts_placed);

                    // Produce with retry. A dropped improvement is not fatal — the
                    // final response still carries the complete layout — so this
                    // logs and continues rather than failing the job.
                    if let Err(e) = retry_with_backoff("send improvement to Kafka", || {
                        let producer = producer_for_task.clone();
                        let topic = output_queue_url_for_task.clone();
                        let resp = response.clone();
                        async move { Self::publish_response(&producer, &topic, &resp).await }
                    }).await {
                        error!("Failed to produce improvement after retries: {}", e);
                    } else {
                        info!("Successfully produced improvement response");
                    }
                }
                info!("Improvement task finished (channel closed)");
            }
            // Instrument the task, so the produce inside it injects the job's span
            // instead of an empty one.
            .instrument(improvement_span));

            // Create improvement callback that sends to channel
            info!("Creating improvement callback");
            let tx_for_callback = tx.clone();
            let improvement_callback: Option<jagua_utils::svg_nesting::ImprovementCallback> =
                Some(Box::new(move |result: NestingResult| -> Result<()> {
                    info!("Improvement callback called from blocking thread: {} parts placed, {} pages", result.parts_placed, result.page_svgs.len());
                    // Send result to channel (non-blocking for unbounded channel)
                    tx_for_callback.send(result)
                        .map_err(|e| anyhow!("Failed to send improvement result to channel: {}", e))
                }));

            // Use adaptive nesting strategy with cancellation checker
            info!("Creating AdaptiveNestingStrategy with cancellation checker");
            let strategy_start = Instant::now();
            let mut strategy = AdaptiveNestingStrategy::with_cancellation_checker(Box::new(cancellation_checker));
            // Offcut detection (JG-OFF-2) runs only on the normal nesting path; it is
            // deliberately omitted for max_fit (a maximally-packed sheet has little free
            // space and runs under a tight wall-clock budget).
            if !max_fit {
                if let Some(policy) = request.offcut_policy {
                    strategy = strategy.with_offcut_policy(policy);
                    info!("Offcut detection enabled: {:?}", policy);
                }
            }
            if let Some(s) = max_seconds {
                strategy = strategy.with_time_budget(Duration::from_secs(s));
                info!("Time budget overridden to {}s via maxSeconds", s);
            }
            info!("Strategy created (took {:?})", strategy_start.elapsed());

            // Clone cancellation_check_count for logging after spawn_blocking
            let cancellation_check_count_for_final_log = cancellation_check_count.clone();
            let timed_out_for_check = timed_out.clone();

            // Run nest() in a blocking task to avoid blocking the async runtime
            // The cancellation checker handles timeout detection cooperatively
            info!("Starting nesting optimization in spawn_blocking task (timeout: {:?})", execution_timeout);
            let nest_start = Instant::now();
            let part_inputs_for_nest = part_inputs.clone();
            let amount_of_rotations = request.amount_of_rotations;
            // The packer is chosen by jagua-rs from the incoming part shapes (the classifier),
            // not by the caller — always AUTO.
            let packing_mode = jagua_utils::PackingMode::Auto;
            let correlation_id_for_error = request.correlation_id.clone();
            let correlation_id_for_timeout = request.correlation_id.clone();

            // Use a longer failsafe timeout (execution_timeout + 60s buffer) in case cooperative cancellation is slow
            let failsafe_timeout = execution_timeout + Duration::from_secs(60);

            let nesting_future = tokio::task::spawn_blocking(move || {
                info!("Inside spawn_blocking: calling strategy.nest() (max_fit={})", max_fit);
                let nest_call_start = Instant::now();
                let result = if max_fit {
                    jagua_utils::nest_max_fit_auto(
                        &strategy,
                        bin_width,
                        bin_height,
                        spacing,
                        &part_inputs_for_nest[0],
                        amount_of_rotations,
                        packing_mode,
                        improvement_callback,
                    )
                } else {
                    jagua_utils::nest_auto(
                        &strategy,
                        bin_width,
                        bin_height,
                        spacing,
                        &part_inputs_for_nest,
                        amount_of_rotations,
                        packing_mode,
                        improvement_callback,
                    )
                };
                info!("Inside spawn_blocking: strategy.nest() completed (took {:?})", nest_call_start.elapsed());
                result
            });

            // Apply failsafe timeout to the blocking task (cooperative timeout via cancellation checker is primary)
            let nesting_result = match tokio::time::timeout(failsafe_timeout, nesting_future).await {
                Ok(spawn_result) => {
                    spawn_result.context("Failed to spawn blocking task for nesting")?
                }
                Err(_) => {
                    // Failsafe timeout occurred (cooperative timeout should have triggered first)
                    error!(
                        "Failsafe execution timeout after {:?} for correlation_id={}",
                        failsafe_timeout, correlation_id_for_timeout
                    );
                    return Err(anyhow!("execution timeout"));
                }
            };

            info!("spawn_blocking task completed (took {:?})", nest_start.elapsed());

            // Check if nesting succeeded
            let nesting_result = nesting_result.with_context(|| {
                format!(
                    "Failed to process SVG nesting for correlation_id={}",
                    correlation_id_for_error
                )
            })?;

            // Log if timeout was triggered but we still got a result
            if timed_out_for_check.load(Ordering::SeqCst) {
                info!(
                    "Timeout triggered but nesting completed with {} parts placed for correlation_id={}",
                    nesting_result.parts_placed, correlation_id_for_error
                );
            }
            info!("Nesting result obtained successfully");
            info!("Cancellation checker was called {} times total", cancellation_check_count_for_final_log.load(Ordering::Relaxed));

            // Drop the sender to signal the async task that no more improvements will come
            drop(tx);

            // Wait for improvement task to complete (properly await instead of fixed sleep)
            info!("Waiting for improvement task to complete...");
            if let Err(e) = improvement_task_handle.await {
                error!("Improvement task panicked: {}", e);
            }
            info!("Improvement task completed");

            info!(
                "Nesting complete: {} parts placed out of {} requested ({} page SVGs generated)",
                nesting_result.parts_placed,
                nesting_result.total_parts_requested,
                nesting_result.page_svgs.len()
            );

            // Resolve bucket / key-prefix for the final-response uploads. BE may
            // override; otherwise we fall back to the worker default and the
            // legacy `nesting/{correlation_id}` prefix.
            let final_bucket = request
                .bucket
                .clone()
                .unwrap_or_else(|| self.s3_bucket.clone());
            let final_prefix = request
                .s3_prefix
                .clone()
                .unwrap_or_else(|| format!("nesting/{}", request.correlation_id));

            // Upload all page SVGs to S3
            let mut page_svg_urls: Vec<String> = Vec::new();
            for (page_idx, page_bytes) in nesting_result.page_svgs.iter().enumerate() {
                let filename = format!("page-{}.svg", page_idx);
                match retry_with_backoff(&format!("upload final page {}", page_idx), || {
                    let s3_client = self.s3_client.clone();
                    let bucket = final_bucket.clone();
                    let region = self.aws_region.clone();
                    let endpoint = self.endpoint_url.clone();
                    let bytes = page_bytes.clone();
                    let prefix = final_prefix.clone();
                    let fname = filename.clone();
                    async move {
                        upload_svg_to_s3_internal(&s3_client, &bucket, &region, &bytes, &prefix, &fname, endpoint.as_deref()).await
                    }
                }).await {
                    Ok(url) => {
                        info!("Uploaded final page {} SVG to S3: {}", page_idx, url);
                        page_svg_urls.push(url);
                    }
                    Err(e) => {
                        // Escalate rather than swallow. Publishing a response
                        // without this page would report success for a layout the
                        // caller cannot fetch; the retry ladder at least gives S3
                        // a chance to come back, and failing that the exhaustion
                        // alert says the message was lost.
                        error!("Failed to upload final page {} SVG to S3 after retries: {}", page_idx, e);
                        return Err(anyhow!(RetryableError(format!(
                            "S3 upload of final page {} failed after retries: {}",
                            page_idx, e
                        ))));
                    }
                }
            }

            // Upload last page (unplaced parts) only if there are unplaced parts
            let first_page_bytes = nesting_result.page_svgs.first()
                .unwrap_or(&nesting_result.combined_svg);
            let last_page_svg_url = if let Some(last_page_bytes) = determine_last_page_svg(
                &nesting_result,
                first_page_bytes,
            ) {
                match retry_with_backoff("upload final last page", || {
                    let s3_client = self.s3_client.clone();
                    let bucket = final_bucket.clone();
                    let region = self.aws_region.clone();
                    let endpoint = self.endpoint_url.clone();
                    let bytes = last_page_bytes.clone();
                    let prefix = final_prefix.clone();
                    async move {
                        upload_svg_to_s3_internal(&s3_client, &bucket, &region, &bytes, &prefix, "last-page.svg", endpoint.as_deref()).await
                    }
                }).await {
                    Ok(url) => {
                        info!("Uploaded final result last page SVG to S3: {}", url);
                        Some(url)
                    }
                    Err(e) => {
                        error!("Failed to upload final result last page SVG to S3 after retries: {}", e);
                        return Err(anyhow!(RetryableError(format!(
                            "S3 upload of the final last page failed after retries: {e}"
                        ))));
                    }
                }
            } else {
                None
            };

            let first_page_svg_url = page_svg_urls.first().cloned();

            // Build pages with S3 URLs populated
            let response_pages: Vec<PageResult> = nesting_result.pages.iter().map(|p| {
                let mut page = p.clone();
                page.svg_url = page_svg_urls.get(p.page_index).cloned();
                page
            }).collect();

            let sheets = Some(response_pages.len());

            // Send final result to queue (with S3 URLs)
            let response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url,
                last_page_svg_url,
                sheets,
                sheets_total: nesting_result.sheets_total_estimate,
                page_svg_urls: Some(page_svg_urls),
                pages: Some(response_pages),
                parts_placed: nesting_result.parts_placed,
                utilisation: nesting_result.utilisation,
                is_improvement: false,
                is_final: true,
                timestamp: current_timestamp(),
                error_message: None,
            };

            info!(
                "Sending final response with parts_placed: {} (from nesting_result.parts_placed: {})",
                response.parts_placed, nesting_result.parts_placed
            );

            self.send_to_output_queue(output_queue_url, &response)
                .await
                .context("Failed to send final result to queue")?;

            info!("Sent final result to queue");

            Ok(())
        }.await;

        // Cleanup: remove correlation_id from cancellation registry (always happens)
        self.unregister_correlation_id(&request.correlation_id);

        result
    }

    /// Consume the request topic and process records concurrently.
    ///
    /// **The shape of this loop is load-bearing.** Three things about it are
    /// mandated by `cutl-infra/docs/kafka-contract.md` and each of them fails
    /// silently rather than loudly if changed:
    ///
    /// - **Fetching never stops while jobs run.** The obvious throttle — pause the
    ///   partition while busy — is wrong here. Requests are keyed by
    ///   `correlationId`, so a cancellation lands on the *same partition* as the
    ///   job it cancels, at a later offset. A paused partition means the cancel is
    ///   only read once the job has already finished, turning cancellation into a
    ///   no-op that nothing reports.
    /// - **Cancellations are handled inline, before the semaphore.** Under SQS the
    ///   permit was taken first and the cancel dealt with inside the handler; that
    ///   was survivable because SQS long-polling kept delivering. Here a cancel
    ///   queued behind twenty running jobs defeats the point of keying.
    ///   Cancellation is a cheap registry write, so it happens on the poll thread.
    /// - **Offsets advance through a contiguous watermark, not on completion.** A
    ///   commit of offset 105 implicitly acknowledges 100..=104, so committing when
    ///   a job finishes would acknowledge its still-running predecessors. With no
    ///   DLQ, over-acknowledging is unrecoverable data loss.
    pub async fn listen_and_process(
        &self,
        consumer: StreamConsumer,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<()> {
        let max_concurrent = get_max_concurrent_tasks();
        let topic = self.kafka.request_topic.clone();

        consumer
            .subscribe(&[topic.as_str()])
            .with_context(|| format!("failed to subscribe to {topic}"))?;

        info!(
            "Consuming topic {} as group {} (max {} concurrent tasks)",
            topic, self.kafka.consumer_group, max_concurrent
        );

        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        // Completion carries the topic-partition-offset so the watermark can be
        // advanced on the loop thread, where the single mutable copy lives.
        let mut active_tasks: JoinSet<CompletedRecord> = JoinSet::new();
        let mut watermark = OffsetWatermark::new();

        let mut last_cleanup = Instant::now();
        let cleanup_interval = Duration::from_secs(60);

        loop {
            if last_cleanup.elapsed() > cleanup_interval {
                self.cleanup_expired_entries();
                last_cleanup = Instant::now();
            }

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!(
                        "Received shutdown signal, waiting for {} active tasks to complete...",
                        active_tasks.len()
                    );
                    break;
                }

                Some(joined) = active_tasks.join_next(), if !active_tasks.is_empty() => {
                    match joined {
                        Ok(done) => {
                            metrics().inflight_jobs.dec();
                            if !done.success {
                                warn!(
                                    "Handler failed for {}-{}-{}",
                                    done.topic, done.partition, done.offset
                                );
                            }
                            self.advance_and_commit(&consumer, &mut watermark, &done);
                        }
                        Err(e) => {
                            // A panicked task never reports its offset, so the
                            // watermark stalls behind it and the record is
                            // redelivered on the next rebalance. That is the safe
                            // direction: reprocessing beats acknowledging work
                            // that never happened.
                            metrics().inflight_jobs.dec();
                            error!("Task panicked, watermark will hold: {}", e);
                        }
                    }
                }

                received = consumer.recv() => {
                    let borrowed = match received {
                        Ok(m) => m,
                        Err(e) => {
                            // Do NOT propagate: `?` here would kill the loop and
                            // the process on a transient broker hiccup, which is
                            // exactly the crash-loop the T7 handoff warns about.
                            error!("Kafka receive error: {}", e);
                            continue;
                        }
                    };

                    let record = match OwnedRecord::from_borrowed(&borrowed) {
                        Some(r) => r,
                        None => {
                            // Nothing to do but move past it: a record with no
                            // payload can never be handled, and leaving it
                            // uncommitted would block the partition forever.
                            error!(
                                "Record {}-{}-{} has no payload, skipping",
                                borrowed.topic(), borrowed.partition(), borrowed.offset()
                            );
                            continue;
                        }
                    };

                    metrics().messages_consumed.with_label_values(&[&record.topic]).inc();
                    watermark.observe(&record.topic, record.partition, record.offset);

                    // ── Cancellations short-circuit here, before the semaphore ──
                    if let Some(correlation_id) = peek_cancellation(&record.payload) {
                        let was_running = self.mark_cancelled(&correlation_id);
                        info!(
                            "Cancellation for correlation_id={} ({}), handled inline at {}-{}-{}",
                            correlation_id,
                            if was_running { "job running" } else { "not yet started" },
                            record.topic, record.partition, record.offset
                        );
                        let done = CompletedRecord::ok(&record);
                        self.advance_and_commit(&consumer, &mut watermark, &done);
                        continue;
                    }

                    let processor = self.clone();
                    let semaphore_clone = semaphore.clone();
                    metrics().inflight_jobs.inc();

                    // Continue the producer's trace rather than starting a new one.
                    // The span is created here, on the poll thread, so the parent
                    // link is established from the record's own headers before the
                    // work moves to another task.
                    let span = tracing::info_span!(
                        "nesting.handle",
                        messaging.system = "kafka",
                        messaging.source.name = %record.topic,
                        messaging.kafka.partition = record.partition,
                        messaging.kafka.offset = record.offset,
                    );
                    {
                        use tracing_opentelemetry::OpenTelemetrySpanExt;
                        span.set_parent(record.parent_context());
                    }

                    active_tasks.spawn(async move {
                        let _permit = match semaphore_clone.acquire().await {
                            Ok(permit) => permit,
                            Err(e) => {
                                error!("Failed to acquire semaphore permit: {}", e);
                                return CompletedRecord::failed(&record);
                            }
                        };

                        let result = processor
                            .process_message(&record.key.clone().unwrap_or_default(), &record.payload)
                            .await;

                        match result {
                            Ok(()) => CompletedRecord::ok(&record),
                            Err(e) => {
                                error!(
                                    "Error handling {}-{}-{}: {}",
                                    record.topic, record.partition, record.offset, e
                                );
                                metrics()
                                    .messages_failed
                                    .with_label_values(&[&record.topic])
                                    .inc();
                                processor.route_to_retry(&record).await;
                                // The record is acknowledged either way. A failure
                                // that stayed uncommitted would head-of-line-block
                                // its partition for the whole attempt budget.
                                CompletedRecord::failed(&record)
                            }
                        }
                    }
                    // Instrument the whole task, so the response produced inside it
                    // injects THIS span's context and the backend's consumer links
                    // straight back to the job that produced it.
                    .instrument(span));
                }
            }
        }

        // Graceful shutdown. `terminationGracePeriodSeconds: 600` in the Deployment
        // exists for exactly this drain: a nesting run is minutes long, and the
        // default 30s would SIGKILL one mid-flight on every rollout.
        info!(
            "Waiting for {} active tasks to complete...",
            active_tasks.len()
        );
        while let Some(joined) = active_tasks.join_next().await {
            match joined {
                Ok(done) => {
                    metrics().inflight_jobs.dec();
                    self.advance_and_commit(&consumer, &mut watermark, &done);
                }
                Err(e) => {
                    metrics().inflight_jobs.dec();
                    error!("Shutdown: task panicked: {}", e);
                }
            }
        }

        // One last synchronous commit. The async commits above may still be in
        // flight, and dropping the consumer without this loses the final offsets —
        // which would replay a job that has already published its result.
        if let Err(e) = consumer.commit_consumer_state(CommitMode::Sync) {
            warn!("Final offset commit failed: {}", e);
        }

        info!("Worker exiting gracefully, all tasks completed");
        Ok(())
    }

    /// Send a failed record to the next retry tier, or drop it and raise the alert.
    ///
    /// **Republish-and-commit, never retry in place.** Retrying by seeking back
    /// head-of-line-blocks the partition on a poison message for the whole attempt
    /// budget, and sleeping before a retry trips `max.poll.interval.ms` and gets
    /// the consumer evicted mid-job.
    ///
    /// Tier delays are applied by the tier's own consumer, not here.
    async fn route_to_retry(&self, record: &OwnedRecord) {
        self.escalate(
            &record.payload,
            record.key.as_deref(),
            &record.headers,
            &record.topic,
            record.partition,
            record.offset,
        )
        .await
    }

    /// Escalate a failed message, from either the main loop or a retry tier.
    ///
    /// Takes the parts rather than an `OwnedRecord` so the tier consumers, which
    /// never build one, can share the same ladder and the same exhaustion
    /// reporting. Two code paths reporting loss differently would be worse than
    /// one: the counter is the only record a message existed.
    pub async fn escalate(
        &self,
        payload: &str,
        key: Option<&str>,
        headers: &RecordHeaders,
        source_topic: &str,
        partition: i32,
        offset: i64,
    ) {
        let origin_topic = headers.origin_or(source_topic);
        let attempt = headers.attempt;

        if attempt >= self.kafka.attempt_budget {
            // ── The end of the line. There is no DLQ: this message is now gone. ──
            //
            // The counter says a message was lost; this log is the only thing that
            // says *which*. Both are required — cutl-infra's CutlRetriesExhausted
            // alert fires on the counter and its description points the on-call at
            // these fields.
            error!(
                "RETRIES EXHAUSTED after {} attempts, message DROPPED (no DLQ): \
                 origin_topic={} key={} at {}-{}-{}",
                attempt,
                origin_topic,
                key.unwrap_or("<none>"),
                source_topic,
                partition,
                offset,
            );
            metrics().record_retries_exhausted(&origin_topic);
            return;
        }

        let tier = attempt;
        let tier_topic = self.kafka.retry_topic(tier);
        let next_attempt = attempt + 1;

        let key = key.unwrap_or_default().to_string();
        // Trace context rides along with the retry headers, so all three ladder
        // hops and the eventual drop stay in ONE trace. Without it each tier looks
        // like an unrelated failure and the "why was this message lost" question
        // needs manual correlation-id grepping across three topics.
        let out_headers =
            crate::trace_context::inject_current(retry_headers(next_attempt, &origin_topic));

        let record_to_send = FutureRecord::to(&tier_topic)
            .key(&key)
            .payload(payload)
            .headers(out_headers);

        match self.producer.send(record_to_send, Duration::ZERO).await {
            Ok(_) => {
                metrics()
                    .retry_republished
                    .with_label_values(&[&tier.to_string()])
                    .inc();
                info!(
                    "Republished to {} (attempt {} of {}): origin_topic={} key={}",
                    tier_topic, next_attempt, self.kafka.attempt_budget, origin_topic, key
                );
            }
            Err((e, _)) => {
                // The ladder itself is broken, so this message is lost exactly as
                // if it had exhausted its budget — and must be reported the same
                // way, or the loss is silent.
                error!(
                    "Failed to republish to {}, message DROPPED (no DLQ): \
                     origin_topic={} key={} at {}-{}-{}: {}",
                    tier_topic, origin_topic, key, source_topic, partition, offset, e
                );
                metrics().record_retries_exhausted(&origin_topic);
            }
        }
    }

    /// Advance the watermark for a finished record and commit if the contiguous
    /// prefix moved.
    fn advance_and_commit(
        &self,
        consumer: &StreamConsumer,
        watermark: &mut OffsetWatermark,
        done: &CompletedRecord,
    ) {
        let Some(commit_at) = watermark.complete(&done.topic, done.partition, done.offset) else {
            debug!(
                "{}-{}-{} done but an earlier offset is still in flight; holding the commit",
                done.topic, done.partition, done.offset
            );
            return;
        };

        let mut tpl = TopicPartitionList::new();
        if let Err(e) =
            tpl.add_partition_offset(&done.topic, done.partition, Offset::Offset(commit_at))
        {
            error!("Failed to build commit list for {}: {}", done.topic, e);
            return;
        }

        // Async: a synchronous commit here would block the receive loop, and
        // blocking the receive loop is what makes cancellation stop working.
        if let Err(e) = consumer.commit(&tpl, CommitMode::Async) {
            // Not fatal. The watermark keeps its position and the next commit
            // carries this offset too; worst case a record is redelivered.
            warn!("Commit of {}@{} failed: {}", done.topic, commit_at, e);
            return;
        }

        metrics()
            .committed_offset
            .with_label_values(&[&done.topic, &done.partition.to_string()])
            .set(commit_at);
    }
}

/// A record lifted out of rdkafka's borrowed message so it can cross a task
/// boundary. `BorrowedMessage` is tied to the consumer's buffer and cannot be sent
/// to a spawned task.
#[derive(Debug, Clone)]
struct OwnedRecord {
    topic: String,
    partition: i32,
    offset: i64,
    key: Option<String>,
    payload: String,
    headers: RecordHeaders,
    /// Every header as a string pair, kept so the W3C trace context can be
    /// re-extracted inside the spawned task: `BorrowedMessage` is tied to the
    /// consumer's buffer and cannot cross a task boundary.
    raw_headers: Vec<(String, String)>,
}

impl OwnedRecord {
    fn from_borrowed(msg: &BorrowedMessage<'_>) -> Option<Self> {
        let payload = msg
            .payload()
            .map(|b| String::from_utf8_lossy(b).into_owned())?;

        let mut raw_headers = Vec::new();
        if let Some(headers) = msg.headers() {
            for header in headers.iter() {
                if let Some(value) = header.value {
                    if let Ok(text) = std::str::from_utf8(value) {
                        raw_headers.push((header.key.to_string(), text.to_string()));
                    }
                }
            }
        }

        Some(Self {
            topic: msg.topic().to_string(),
            partition: msg.partition(),
            offset: msg.offset(),
            key: msg.key().map(|k| String::from_utf8_lossy(k).into_owned()),
            payload,
            headers: RecordHeaders::from_message(msg),
            raw_headers,
        })
    }

    /// The upstream trace context, when the producer sent one. An absent
    /// `traceparent` yields an invalid context, which starts a new trace rather
    /// than failing — a producer without tracing is not an error.
    fn parent_context(&self) -> opentelemetry::Context {
        crate::trace_context::extract_context(&crate::trace_context::HeaderExtractor::from_pairs(
            self.raw_headers.clone(),
        ))
    }
}

/// Outcome of a handled record, carrying enough to advance the watermark.
#[derive(Debug, Clone)]
struct CompletedRecord {
    topic: String,
    partition: i32,
    offset: i64,
    success: bool,
}

impl CompletedRecord {
    fn ok(r: &OwnedRecord) -> Self {
        Self {
            topic: r.topic.clone(),
            partition: r.partition,
            offset: r.offset,
            success: true,
        }
    }

    fn failed(r: &OwnedRecord) -> Self {
        Self {
            topic: r.topic.clone(),
            partition: r.partition,
            offset: r.offset,
            success: false,
        }
    }
}

/// Cheaply detect a cancellation message without paying for full deserialization.
///
/// Returns the `correlationId` when `cancelled` is true. Cancellation messages
/// arrive with every nesting field set to explicit `null`, so this deliberately
/// parses to `serde_json::Value` rather than the strict DTO — a cancel must be
/// honoured even if the rest of the message would not deserialize.
fn peek_cancellation(payload: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    if value.get("cancelled")?.as_bool()? {
        value
            .get("correlationId")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        None
    }
}

#[cfg(test)]
impl NestingProcessor {
    pub(crate) fn cancellation_registry_handle(
        &self,
    ) -> Arc<Mutex<HashMap<String, CancellationEntry>>> {
        self.cancellation_registry.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_config::BehaviorVersion;
    use tokio::time::Duration;

    /// Build a processor with no live dependencies.
    ///
    /// rdkafka connects lazily, so a producer aimed at an address nothing is
    /// listening on constructs fine and is never used by the tests below — they
    /// exercise the cancellation registry and the request DTOs, not the transport.
    async fn test_processor() -> NestingProcessor {
        let aws = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let kafka = KafkaSettings {
            bootstrap_servers: "127.0.0.1:1".to_string(),
            username: "test".to_string(),
            password: "test".to_string(),
            sasl_mechanism: "SCRAM-SHA-512".to_string(),
            consumer_group: crate::kafka::DEFAULT_CONSUMER_GROUP.to_string(),
            request_topic: crate::kafka::DEFAULT_REQUEST_TOPIC.to_string(),
            response_topic: crate::kafka::DEFAULT_RESPONSE_TOPIC.to_string(),
            attempt_budget: crate::kafka::DEFAULT_ATTEMPT_BUDGET,
            retry_delays: crate::kafka::DEFAULT_RETRY_DELAYS_MS
                .iter()
                .map(|ms| Duration::from_millis(*ms))
                .collect(),
        };
        NestingProcessor::new(
            kafka.producer().expect("producer builds without a broker"),
            aws_sdk_s3::Client::new(&aws),
            "test-bucket".to_string(),
            "us-east-1".to_string(),
            kafka,
            None,
        )
    }

    #[test]
    fn test_cancellation_registry_insert_and_get() {
        let registry: Arc<Mutex<HashMap<String, CancellationEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Insert a cancellation flag
        {
            let mut reg = registry.lock().unwrap();
            reg.insert(
                "test-id-1".to_string(),
                CancellationEntry {
                    cancelled: true,
                    created_at: Instant::now(),
                },
            );
        }

        // Check that it's set
        {
            let reg = registry.lock().unwrap();
            assert_eq!(reg.get("test-id-1").map(|e| e.cancelled), Some(true));
            assert_eq!(reg.get("test-id-2"), None);
        }
    }

    #[test]
    fn test_cancellation_registry_remove() {
        let registry: Arc<Mutex<HashMap<String, CancellationEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Insert and then remove
        {
            let mut reg = registry.lock().unwrap();
            reg.insert(
                "test-id-1".to_string(),
                CancellationEntry {
                    cancelled: false,
                    created_at: Instant::now(),
                },
            );
        }

        {
            let mut reg = registry.lock().unwrap();
            reg.remove("test-id-1");
        }

        // Verify it's gone
        {
            let reg = registry.lock().unwrap();
            assert_eq!(reg.get("test-id-1"), None);
        }
    }

    #[test]
    fn test_safe_lock_recovers_from_poison() {
        // This test verifies that safe_lock recovers from poisoned mutexes
        let mutex = Mutex::new(42);

        // Poison the mutex by panicking while holding the lock
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = mutex.lock().unwrap();
            panic!("intentional panic to poison mutex");
        }));
        assert!(result.is_err(), "Should have panicked");

        // Normal lock() would fail, but safe_lock should recover
        let value = safe_lock(&mutex);
        assert_eq!(*value, 42);
    }

    #[test]
    fn test_parse_s3_url_s3_scheme() {
        let (bucket, key) = parse_s3_url("s3://my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_virtual_hosted() {
        let (bucket, key) =
            parse_s3_url("https://my-bucket.s3.us-east-1.amazonaws.com/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_virtual_hosted_dash_region() {
        let (bucket, key) =
            parse_s3_url("https://my-bucket.s3-us-east-1.amazonaws.com/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_path_style() {
        let (bucket, key) =
            parse_s3_url("https://s3.us-east-1.amazonaws.com/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_path_style_dash_region() {
        let (bucket, key) =
            parse_s3_url("https://s3-us-east-1.amazonaws.com/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_invalid() {
        assert!(parse_s3_url("ftp://example.com/file.svg").is_err());
        assert!(parse_s3_url("s3://").is_err());
        assert!(parse_s3_url("s3://bucket").is_err());
        assert!(parse_s3_url("s3://bucket/").is_err());
        // http:// with only bucket, no key
        assert!(parse_s3_url("http://localhost:4566/bucket").is_err());
        assert!(parse_s3_url("http://localhost:4566/bucket/").is_err());
    }

    #[test]
    fn test_parse_s3_url_localstack() {
        // LocalStack/Minio path-style URL
        let (bucket, key) =
            parse_s3_url("http://localstack:4566/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_localstack_https() {
        // LocalStack/Minio path-style URL with HTTPS
        let (bucket, key) =
            parse_s3_url("https://localhost:4566/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_minio() {
        // Minio without port
        let (bucket, key) = parse_s3_url("http://minio.local/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_get_max_concurrent_tasks() {
        // These tests must run sequentially to avoid race conditions with env vars
        // Test 1: Default value when env var is not set
        std::env::remove_var("MAX_CONCURRENT_TASKS");
        assert_eq!(
            get_max_concurrent_tasks(),
            DEFAULT_MAX_CONCURRENT_TASKS,
            "Should use default when env var not set"
        );

        // Test 2: Valid value from env var
        std::env::set_var("MAX_CONCURRENT_TASKS", "50");
        assert_eq!(
            get_max_concurrent_tasks(),
            50,
            "Should use value from env var"
        );

        // Test 3: Invalid value falls back to default
        std::env::set_var("MAX_CONCURRENT_TASKS", "not-a-number");
        assert_eq!(
            get_max_concurrent_tasks(),
            DEFAULT_MAX_CONCURRENT_TASKS,
            "Should fall back to default for invalid value"
        );

        // Cleanup
        std::env::remove_var("MAX_CONCURRENT_TASKS");
    }

    #[test]
    fn test_get_execution_timeout() {
        // These tests must run sequentially to avoid race conditions with env vars
        // Test 1: Default value when env var is not set
        std::env::remove_var("EXECUTION_TIMEOUT_SECS");
        assert_eq!(
            get_execution_timeout(),
            Duration::from_secs(DEFAULT_EXECUTION_TIMEOUT_SECS),
            "Should use default (10 minutes) when env var not set"
        );

        // Test 2: Valid value from env var
        std::env::set_var("EXECUTION_TIMEOUT_SECS", "5");
        assert_eq!(
            get_execution_timeout(),
            Duration::from_secs(5),
            "Should use value from env var"
        );

        // Test 3: Invalid value falls back to default
        std::env::set_var("EXECUTION_TIMEOUT_SECS", "not-a-number");
        assert_eq!(
            get_execution_timeout(),
            Duration::from_secs(DEFAULT_EXECUTION_TIMEOUT_SECS),
            "Should fall back to default for invalid value"
        );

        // Cleanup
        std::env::remove_var("EXECUTION_TIMEOUT_SECS");
    }

    /// The poll loop detects cancellations with `peek_cancellation` rather than by
    /// deserializing, so it must cope with the shape cancels actually arrive in:
    /// every nesting field an explicit `null`. A strict parse would reject these
    /// and the cancel would be treated as an ordinary job.
    /// The classification that decides retry-vs-answer. Getting it wrong either
    /// way is bad: a validation error on the ladder burns ~11 minutes to reach a
    /// conclusion available immediately, and an infrastructure error answered as
    /// a response reports success for a job that produced nothing usable.
    #[test]
    fn infrastructure_errors_are_retryable_and_validation_errors_are_not() {
        let infra: anyhow::Error = anyhow!(RetryableError("S3 upload failed".into()));
        assert!(is_retryable(&infra));

        let validation = anyhow!("'parts' array is empty");
        assert!(!is_retryable(&validation));
    }

    /// The marker has to survive the `.context(...)` wrapping it picks up on the
    /// way out of the handler, or an S3 failure silently becomes a permanent one.
    #[test]
    fn retryable_survives_context_wrapping() {
        let deep: anyhow::Error = anyhow!(RetryableError("S3 timeout".into()));
        let wrapped = deep
            .context("uploading final page 0")
            .context("processing nesting request");

        assert!(
            is_retryable(&wrapped),
            "the marker must be found anywhere in the chain, not just at the root"
        );
        assert!(format!("{wrapped:#}").contains("S3 timeout"));
    }

    #[test]
    fn peek_cancellation_reads_the_production_cancel_shape() {
        let body = r#"{"correlationId":"9abdb358-35fd-4dbf-ba06-1ae9023a4512",
            "binWidth":null,"binHeight":null,"spacing":null,
            "amountOfRotations":null,"cancelled":true,"parts":null}"#;

        assert_eq!(
            peek_cancellation(body).as_deref(),
            Some("9abdb358-35fd-4dbf-ba06-1ae9023a4512")
        );
    }

    /// A normal request must not be mistaken for a cancel — that would silently
    /// drop the job instead of running it.
    #[test]
    fn peek_cancellation_ignores_normal_and_malformed_requests() {
        assert_eq!(
            peek_cancellation(r#"{"correlationId":"c-1","cancelled":false}"#),
            None
        );
        assert_eq!(
            peek_cancellation(r#"{"correlationId":"c-1","binWidth":100}"#),
            None,
            "absent `cancelled` is not a cancellation"
        );
        assert_eq!(peek_cancellation("not json at all"), None);
        assert_eq!(
            peek_cancellation(r#"{"cancelled":true}"#),
            None,
            "a cancel with no correlationId identifies no job"
        );
    }

    #[test]
    fn test_sqs_nesting_request_cancelled_field_default() {
        let request_json = r#"{
            "correlationId": "test-123",
            "svgBase64": "dGVzdA==",
            "binWidth": 100.0,
            "binHeight": 100.0,
            "spacing": 10.0,
            "amountOfParts": 1
        }"#;

        let request: SqsNestingRequest = serde_json::from_str(request_json).unwrap();
        assert!(!request.cancelled, "cancelled should default to false");
    }

    #[test]
    fn test_sqs_nesting_request_cancelled_field_explicit() {
        let request_json = r#"{
            "correlationId": "test-123",
            "svgBase64": "dGVzdA==",
            "binWidth": 100.0,
            "binHeight": 100.0,
            "spacing": 10.0,
            "amountOfParts": 1,
            "cancelled": true
        }"#;

        let request: SqsNestingRequest = serde_json::from_str(request_json).unwrap();
        assert!(request.cancelled, "cancelled should be true when set");
    }

    #[test]
    fn test_sqs_nesting_request_cancelled_with_null_fields() {
        // Reproduces the production error: cancellation requests arrive with all fields
        // set to null rather than omitted. The deserializer must handle explicit nulls.
        let request_json = r#"{"correlationId":"9abdb358-35fd-4dbf-ba06-1ae9023a4512","binWidth":null,"binHeight":null,"spacing":null,"amountOfRotations":null,"cancelled":true,"parts":null}"#;

        let request: SqsNestingRequest = serde_json::from_str(request_json).unwrap();
        assert!(request.cancelled);
        assert_eq!(
            request.correlation_id,
            "9abdb358-35fd-4dbf-ba06-1ae9023a4512"
        );
        assert_eq!(
            request.amount_of_rotations, 8,
            "null amountOfRotations should default to 8"
        );
        assert!(request.bin_width.is_none());
        assert!(request.bin_height.is_none());
        assert!(request.spacing.is_none());
        assert!(request.parts.is_none());
    }

    #[tokio::test]
    async fn test_parallel_cancellation_flag_shared_between_workers() {
        let processor = test_processor().await;

        let correlation_id = "parallel-cancelled".to_string();

        // Register the correlation_id first (simulating an active processing task)
        processor.register_correlation_id(&correlation_id);

        let cancel_processor = processor.clone();
        let cancellation_request = SqsNestingRequest {
            svg_url: None,
            correlation_id: correlation_id.clone(),
            svg_base64: None,
            bin_width: None,
            bin_height: None,
            spacing: None,
            amount_of_parts: None,
            parts: None,
            amount_of_rotations: 8,
            output_queue_url: None,
            cancelled: true,
            max_fit: None,
            bucket: None,
            s3_prefix: None,
            offcut_policy: None,
            max_seconds: None,
        };
        let cancellation_body =
            serde_json::to_string(&cancellation_request).expect("serialize cancellation");

        let processor_for_watcher = processor.clone();
        let correlation_id_clone = correlation_id.clone();
        let watcher = tokio::spawn(async move {
            let timeout = Duration::from_secs(2);
            let start = Instant::now();
            loop {
                if processor_for_watcher.is_cancelled(&correlation_id_clone) {
                    break;
                }

                if start.elapsed() > timeout {
                    panic!("Timed out waiting for cancellation flag to be set");
                }

                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });

        let canceller = tokio::spawn(async move {
            cancel_processor
                .process_message("receipt-handle", &cancellation_body)
                .await
                .expect("Cancellation request should be processed");
        });

        watcher.await.expect("Watcher task failed");
        canceller.await.expect("Canceller task failed");

        assert!(
            processor.is_cancelled(&correlation_id),
            "Cancellation flag should be set to true"
        );
    }

    #[tokio::test]
    async fn test_cancellation_registry_cleanup() {
        let processor = test_processor().await;

        // Register a correlation_id
        processor.register_correlation_id("test-cleanup");
        assert!(!processor.is_cancelled("test-cleanup"));

        // Unregister it
        processor.unregister_correlation_id("test-cleanup");

        // It should be gone (is_cancelled returns false for non-existent entries)
        let registry = processor.cancellation_registry_handle();
        let reg = registry.lock().unwrap();
        assert!(reg.get("test-cleanup").is_none());
    }

    #[tokio::test]
    async fn test_retry_with_backoff_success() {
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = call_count.clone();

        let result: std::result::Result<i32, String> = retry_with_backoff("test_op", || {
            let count = call_count_clone.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(42)
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_eventual_success() {
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = call_count.clone();

        let result: std::result::Result<i32, String> = retry_with_backoff("test_op", || {
            let count = call_count_clone.clone();
            async move {
                let calls = count.fetch_add(1, Ordering::SeqCst) + 1;
                if calls < 3 {
                    Err(format!("Failed attempt {}", calls))
                } else {
                    Ok(42)
                }
            }
        })
        .await;

        assert_eq!(result.unwrap(), 42);
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_all_failures() {
        let call_count = Arc::new(AtomicU64::new(0));
        let call_count_clone = call_count.clone();

        let result: std::result::Result<i32, String> = retry_with_backoff("test_op", || {
            let count = call_count_clone.clone();
            async move {
                let calls = count.fetch_add(1, Ordering::SeqCst) + 1;
                Err(format!("Failed attempt {}", calls))
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(Ordering::SeqCst), MAX_RETRY_ATTEMPTS as u64);
    }

    #[tokio::test]
    async fn test_s3_download() {
        use aws_config::BehaviorVersion;
        use aws_sdk_s3::error::ProvideErrorMetadata;
        use aws_sdk_s3::Client as S3Client;
        use std::env;

        // Initialize logger for test output
        let _ = env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Debug)
            .try_init();

        // Get configuration from environment variables
        let bucket = env::var("S3_BUCKET").unwrap_or_else(|_| "cutl-staging-uploads".to_string());
        let test_key = "22db4d1f-44cb-4c3d-917d-17836ba986ac/projectParts/9720e425-6a18-4a46-aa4c-7a7934ae9f23/project_part_internal_svg.svg";

        println!("Testing S3 download:");
        println!("  Bucket: {}", bucket);
        println!("  Key: {}", test_key);
        println!("  AWS_REGION: {:?}", env::var("AWS_REGION"));
        println!("  AWS_ENDPOINT_URL: {:?}", env::var("AWS_ENDPOINT_URL"));
        println!(
            "  AWS_ACCESS_KEY_ID: {:?}",
            env::var("AWS_ACCESS_KEY_ID").map(|s| format!("{}...", &s[..10.min(s.len())]))
        );

        // Initialize AWS config
        let mut config_loader = aws_config::defaults(BehaviorVersion::latest());

        // Configure LocalStack endpoint if provided
        if let Ok(endpoint_url) = env::var("AWS_ENDPOINT_URL") {
            config_loader = config_loader.endpoint_url(&endpoint_url);
            println!("Using AWS endpoint: {}", endpoint_url);
        }

        let config = config_loader.load().await;
        let s3_client = S3Client::new(&config);

        // Test 1: Try to download the file
        println!("\nTest 1: Downloading file from S3...");
        let result = s3_client
            .get_object()
            .bucket(&bucket)
            .key(test_key)
            .send()
            .await;

        match result {
            Ok(response) => {
                println!("✓ Successfully got object from S3");

                // Try to read the body
                let svg_bytes = match response.body.collect().await {
                    Ok(data) => data.into_bytes().to_vec(),
                    Err(e) => {
                        println!("✗ Error reading body: {}", e);
                        return;
                    }
                };
                println!("✓ Successfully downloaded {} bytes", svg_bytes.len());

                // Try to parse as SVG
                let svg_content = String::from_utf8_lossy(&svg_bytes);
                if svg_content.contains("<svg") {
                    println!("✓ Content appears to be valid SVG");
                } else {
                    println!(
                        "⚠ Content doesn't appear to be SVG (first 100 chars: {})",
                        svg_content.chars().take(100).collect::<String>()
                    );
                }
            }
            Err(e) => {
                println!("✗ Failed to download from S3: {}", e);
                println!("Error details:");

                // Try to get more error information
                if let Some(code) = e.code() {
                    println!("  Error code: {:?}", code);
                }
                if let Some(message) = e.message() {
                    println!("  Error message: {:?}", message);
                }

                // Test 2: Try to list objects in the bucket to verify connectivity
                println!("\nTest 2: Testing bucket connectivity by listing objects...");
                let list_result = s3_client
                    .list_objects_v2()
                    .bucket(&bucket)
                    .max_keys(5)
                    .send()
                    .await;

                match list_result {
                    Ok(list_response) => {
                        println!("✓ Successfully connected to bucket");
                        let contents = list_response.contents();
                        if !contents.is_empty() {
                            println!("  Found {} objects (showing first 5)", contents.len());
                            for (i, obj) in contents.iter().take(5).enumerate() {
                                println!(
                                    "    {}. {}",
                                    i + 1,
                                    obj.key()
                                        .map(|k| k.to_string())
                                        .unwrap_or_else(|| "(no key)".to_string())
                                );
                            }
                        } else {
                            println!("  Bucket is empty");
                        }
                    }
                    Err(e) => {
                        println!("✗ Failed to list objects: {}", e);
                        println!("  This suggests a connectivity or permissions issue");
                    }
                }

                // Test 3: Try to check if bucket exists
                println!("\nTest 3: Checking if bucket exists...");
                let head_result = s3_client.head_bucket().bucket(&bucket).send().await;

                match head_result {
                    Ok(_) => {
                        println!("✓ Bucket exists and is accessible");
                    }
                    Err(e) => {
                        println!("✗ Bucket check failed: {}", e);
                        if let Some(code) = e.code() {
                            println!("  Error code: {:?}", code);
                        }
                    }
                }
            }
        }
    }
}
