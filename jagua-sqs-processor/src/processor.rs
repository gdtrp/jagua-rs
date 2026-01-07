use anyhow::{anyhow, Context, Result};
use aws_sdk_sqs::Client as SqsClient;
use aws_sdk_s3::Client as S3Client;
use base64::{engine::general_purpose, Engine as _};
use jagua_utils::svg_nesting::{NestingStrategy, AdaptiveNestingStrategy, NestingResult};
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError, MutexGuard};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

/// Default maximum concurrent processing tasks
const DEFAULT_MAX_CONCURRENT_TASKS: usize = 20;

/// Maximum retry attempts for AWS operations
const MAX_RETRY_ATTEMPTS: u32 = 3;

/// Base delay for exponential backoff (in milliseconds)
const RETRY_BASE_DELAY_MS: u64 = 100;

/// TTL for orphaned cancellation registry entries (in seconds)
const CANCELLATION_REGISTRY_TTL_SECS: u64 = 300; // 5 minutes

/// Request message structure for SQS queue
/// For cancellation requests, only `correlation_id` and `cancelled: true` are required.
/// All other fields are required only when `cancelled` is false or not present.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct SqsNestingRequest {
    /// Unique identifier for tracking the request
    pub correlation_id: String,
    /// Base64-encoded SVG payload (deprecated, use svg_url instead)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub svg_base64: Option<String>,
    /// S3 URL to the input SVG file (format: s3://bucket/key or https://bucket.s3.region.amazonaws.com/key)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub svg_url: Option<String>,
    /// Bin width for nesting (required if not cancelled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_width: Option<f32>,
    /// Bin height for nesting (required if not cancelled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_height: Option<f32>,
    /// Spacing between parts (required if not cancelled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacing: Option<f32>,
    /// Number of parts to nest (required if not cancelled)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub amount_of_parts: Option<usize>,
    /// Number of rotations to try (default: 8)
    #[serde(default = "default_rotations")]
    pub amount_of_rotations: usize,
    /// Output queue URL for results (falls back to default if omitted)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_queue_url: Option<String>,
    /// Whether this is a cancellation request
    #[serde(default)]
    pub cancelled: bool,
}

/// Generate an empty page SVG (used when all parts are placed)
fn generate_empty_page_svg(bin_width: f32, bin_height: f32) -> Vec<u8> {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">
  <g id="container_0">
    <path d="M 0,0 L {},0 L {},{} L 0,{} z" fill="transparent" stroke="gray" stroke-width="1"/>
  </g>
  <text x="{}" y="{}" font-size="{}" font-family="monospace">Unplaced parts: 0</text>
</svg>"#,
        bin_width,
        bin_height,
        bin_width,
        bin_width,
        bin_height,
        bin_height,
        bin_width * 0.02,
        bin_height * 0.05,
        bin_width * 0.02
    )
    .into_bytes()
}

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

fn default_rotations() -> usize {
    8
}

/// Helper to safely lock a mutex, recovering from poison errors
fn safe_lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
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
            Ok(result) => return Ok(result),
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
                return Err(e);
            }
        }
    }
}

/// Determine the last page SVG bytes based on nesting result
/// Returns empty page SVG if all parts placed, unplaced parts SVG if available,
/// otherwise the last filled page
fn determine_last_page_svg(
    result: &NestingResult,
    first_page_bytes: &[u8],
    bin_width: f32,
    bin_height: f32,
) -> Vec<u8> {
    if result.parts_placed == result.total_parts_requested {
        // All parts placed - generate empty page
        info!(
            "All parts placed ({}), generating empty page",
            result.parts_placed
        );
        generate_empty_page_svg(bin_width, bin_height)
    } else if let Some(ref unplaced_svg) = result.unplaced_parts_svg {
        // Some parts unplaced - use unplaced parts SVG
        info!(
            "Some parts unplaced ({} of {}), using unplaced parts SVG",
            result.parts_placed, result.total_parts_requested
        );
        unplaced_svg.clone()
    } else {
        // No unplaced parts SVG - use last filled page or first page
        info!(
            "No unplaced parts SVG available, using last filled page (parts_placed: {} of {})",
            result.parts_placed, result.total_parts_requested
        );
        result.page_svgs.last().unwrap_or(&first_page_bytes.to_vec()).clone()
    }
}

/// Get the maximum concurrent tasks from environment or use default
fn get_max_concurrent_tasks() -> usize {
    std::env::var("MAX_CONCURRENT_TASKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_TASKS)
}

/// Cancellation registry entry with timestamp for TTL-based cleanup
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CancellationEntry {
    pub(crate) cancelled: bool,
    pub(crate) created_at: Instant,
}

/// Response message structure for SQS queue
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SqsNestingResponse {
    /// Correlation ID from request
    pub correlation_id: String,
    /// S3 URL to the first page SVG (format: s3://bucket/nesting/{requestId}/first-page.svg)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_page_svg_url: Option<String>,
    /// S3 URL to the last page SVG (format: s3://bucket/nesting/{requestId}/last-page.svg)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_page_svg_url: Option<String>,
    /// Number of parts placed
    pub parts_placed: usize,
    /// Whether this is an intermediate improvement (always false for simple strategy)
    #[serde(rename = "improvement")]
    pub is_improvement: bool,
    /// Whether this is the final result (always true for simple strategy)
    #[serde(rename = "final")]
    pub is_final: bool,
    /// Timestamp in seconds since epoch
    pub timestamp: u64,
    /// Error message if processing failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

/// SQS Processor for handling SVG nesting requests
#[derive(Clone)]
pub struct SqsProcessor {
    sqs_client: SqsClient,
    s3_client: S3Client,
    s3_bucket: String,
    aws_region: String,
    input_queue_url: String,
    output_queue_url: String,
    cancellation_registry: Arc<Mutex<HashMap<String, CancellationEntry>>>,
}

impl SqsProcessor {
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

    /// Create a new SQS processor
    pub fn new(
        sqs_client: SqsClient,
        s3_client: S3Client,
        s3_bucket: String,
        aws_region: String,
        input_queue_url: String,
        output_queue_url: String,
    ) -> Self {
        Self {
            sqs_client,
            s3_client,
            s3_bucket,
            aws_region,
            input_queue_url,
            output_queue_url,
            cancellation_registry: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Download SVG from S3 URL
    async fn download_svg_from_s3(&self, s3_url: &str) -> Result<Vec<u8>> {
        // Parse S3 URL (supports both s3://bucket/key and https://bucket.s3.region.amazonaws.com/key)
        let (bucket, key) = parse_s3_url(s3_url)?;
        
        info!("Downloading SVG from S3: url={}, bucket={}, key={}", s3_url, bucket, key);

        let response = match self.s3_client
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
                
                return Err(anyhow::anyhow!(
                    "Failed to download SVG from S3: bucket={}, key={}, error={}",
                    bucket, key, e
                ));
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
    if s3_url.starts_with("s3://") {
        let path = &s3_url[5..];
        if let Some(slash_pos) = path.find('/') {
            let bucket = path[..slash_pos].to_string();
            let key = path[slash_pos + 1..].to_string();
            if bucket.is_empty() || key.is_empty() {
                return Err(anyhow!("Invalid S3 URL: bucket or key is empty: {}", s3_url));
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
        return Err(anyhow!("Unsupported S3 URL format (must start with s3://, http://, or https://): {}", s3_url));
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
                    return Err(anyhow!("Invalid S3 path-style URL: bucket or key is empty: {}", s3_url));
                }
                return Ok((bucket, key));
            }
            return Err(anyhow!("Invalid S3 path-style URL (missing key): {}", s3_url));
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
                return Err(anyhow!("Invalid S3 virtual-hosted URL: bucket or key is empty: {}", s3_url));
            }
            return Ok((bucket, key));
        }
        return Err(anyhow!("Invalid S3 virtual-hosted URL (missing .amazonaws.com): {}", s3_url));
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
                return Err(anyhow!("Invalid S3-compatible URL: bucket or key is empty: {}", s3_url));
            }
            return Ok((bucket, key));
        }
        return Err(anyhow!("Invalid S3-compatible URL (missing key after bucket): {}", s3_url));
    }

    Err(anyhow!("Invalid S3-compatible URL format: {}", s3_url))
}

/// Internal helper function to upload SVG to S3 (used by both improvement and final responses)
async fn upload_svg_to_s3_internal(
    s3_client: &S3Client,
    s3_bucket: &str,
    aws_region: &str,
    svg_bytes: &[u8],
    request_id: &str,
    filename: &str,
) -> Result<String> {
    let s3_key = format!("nesting/{}/{}", request_id, filename);
    let s3_url = format!("https://{}.s3.{}.amazonaws.com/{}", s3_bucket, aws_region, s3_key);
    
    info!("Uploading SVG to S3: bucket={}, key={}, size={} bytes", 
        s3_bucket, s3_key, svg_bytes.len());

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

impl SqsProcessor {
    /// Send message to output queue
    /// Helper function to send a message to SQS (used by both error and improvement responses)
    async fn send_message_to_sqs(
        sqs_client: &SqsClient,
        queue_url: &str,
        response: &SqsNestingResponse,
    ) -> Result<()> {
        let message_body =
            serde_json::to_string(response).context("Failed to serialize response")?;

        // Check message size (SQS limit is 1 MiB = 1,048,576 bytes)
        let message_size_kb = message_body.len() / 1024;
        const SQS_MAX_SIZE: usize = 1024 * 1024; // 1 MiB
        if message_body.len() > SQS_MAX_SIZE {
            return Err(anyhow!(
                "Message size {} KB exceeds SQS limit of {} KB (1 MiB)",
                message_size_kb,
                SQS_MAX_SIZE / 1024
            ));
        }

        debug!(
            "Sending message to output queue: correlation_id={}, is_final={}, size={} KB",
            response.correlation_id, response.is_final, message_size_kb
        );

        sqs_client
            .send_message()
            .queue_url(queue_url)
            .message_body(&message_body)
            .send()
            .await
            .with_context(|| {
                format!(
                    "Failed to send message to queue {}: correlation_id={}, size={} KB",
                    queue_url, response.correlation_id, message_size_kb
                )
            })?;

        Ok(())
    }

    pub async fn send_to_output_queue(
        &self,
        queue_url: &str,
        response: &SqsNestingResponse,
    ) -> Result<()> {
        let sqs_client = self.sqs_client.clone();
        let queue_url_owned = queue_url.to_string();
        let response_clone = response.clone();

        retry_with_backoff("send_to_output_queue", || {
            let client = sqs_client.clone();
            let url = queue_url_owned.clone();
            let resp = response_clone.clone();
            async move { Self::send_message_to_sqs(&client, &url, &resp).await }
        })
        .await?;

        debug!(
            "Emitted response to {}: correlation_id={}, parts_placed={}, is_final={}",
            queue_url, response.correlation_id, response.parts_placed, response.is_final
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
                            .unwrap_or_else(|| self.output_queue_url.clone());

                        let error_response = SqsNestingResponse {
                            correlation_id: corr_id.to_string(),
                            first_page_svg_url: None,
                            last_page_svg_url: None,
                            parts_placed: 0,
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
                    format!("~{} bytes (base64: {} bytes, decode failed)", approx_decoded_size, base64_len)
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
            request.output_queue_url.as_ref().map(|s| s.as_str()).unwrap_or("default")
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
            .unwrap_or_else(|| self.output_queue_url.clone());

        // Validate required fields for non-cancellation requests
        // Either svg_base64 or svg_url must be provided
        if request.svg_base64.is_none() && request.svg_url.is_none() {
            let error_msg = "Missing required field: either svg_base64 or svg_url must be provided";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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
        if request.bin_width.is_none() {
            let error_msg = "Missing required field: bin_width";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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
        if request.bin_height.is_none() {
            let error_msg = "Missing required field: bin_height";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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
        if request.spacing.is_none() {
            let error_msg = "Missing required field: spacing";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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
        if request.amount_of_parts.is_none() {
            let error_msg = "Missing required field: amount_of_parts";
            error!("{}", error_msg);
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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

        // Process the request and handle errors by sending error response
        let result = self
            .process_nesting_request(&request, &output_queue_url)
            .await;

        if let Err(e) = &result {
            let error_msg = format!("{}", e);
            error!("Failed to process message: {}", error_msg);

            // Send error response for internal processing errors
            let error_response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url: None,
                last_page_svg_url: None,
                parts_placed: 0,
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
            let amount_of_parts = request.amount_of_parts.unwrap();

            // Get SVG bytes - either from S3 or from base64
            let decode_start = Instant::now();
            let svg_bytes = if let Some(ref svg_url) = request.svg_url {
                // Download from S3
                info!("Downloading SVG from S3: {}", svg_url);
                self.download_svg_from_s3(svg_url).await?
            } else if let Some(ref svg_base64) = request.svg_base64 {
                // Decode from base64
                decode_svg(svg_base64)?
            } else {
                return Err(anyhow!("Neither svg_url nor svg_base64 provided"));
            };
            info!("SVG payload ready: {} bytes (took {:?})", svg_bytes.len(), decode_start.elapsed());

            // Create cancellation checker closure using the helper method
            let processor_clone = self.clone();
            let correlation_id_clone = request.correlation_id.clone();
            let cancellation_check_count = Arc::new(AtomicU64::new(0));
            let cancellation_check_count_for_log = cancellation_check_count.clone();
            let cancellation_checker = move || {
                let count = cancellation_check_count_for_log.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 1000 == 0 {
                    log::debug!("Cancellation checker called {} times", count);
                }
                processor_clone.is_cancelled(&correlation_id_clone)
            };

            // Create channel for sending improvement results from sync callback to async task
            let (tx, mut rx) = mpsc::unbounded_channel::<NestingResult>();

            // Spawn async task to handle improvement messages
            info!("Spawning async task to handle improvement messages");
            let sqs_client_for_task = self.sqs_client.clone();
            let s3_client_for_task = self.s3_client.clone();
            let s3_bucket_for_task = self.s3_bucket.clone();
            let aws_region_for_task = self.aws_region.clone();
            let output_queue_url_for_task = output_queue_url.to_string();
            let correlation_id_for_task = request.correlation_id.clone();
            let bin_width_for_task = bin_width;
            let bin_height_for_task = bin_height;

            let improvement_task_handle = tokio::spawn(async move {
                info!("Improvement task started, waiting for messages...");
                while let Some(result) = rx.recv().await {
                    info!("Improvement task received message: {} parts placed, {} pages", result.parts_placed, result.page_svgs.len());

                    // Get the first and last page SVGs for uploading to S3
                    let first_page_bytes = result.page_svgs.first()
                        .unwrap_or(&result.combined_svg);

                    // Use shared helper function to determine last page
                    let last_page_bytes = determine_last_page_svg(
                        &result,
                        first_page_bytes,
                        bin_width_for_task,
                        bin_height_for_task,
                    );

                    // Upload first page SVG to S3 with retry
                    let first_page_svg_url = match retry_with_backoff("upload improvement first page", || {
                        let client = s3_client_for_task.clone();
                        let bucket = s3_bucket_for_task.clone();
                        let region = aws_region_for_task.clone();
                        let bytes = first_page_bytes.clone();
                        let corr_id = correlation_id_for_task.clone();
                        async move {
                            upload_svg_to_s3_internal(&client, &bucket, &region, &bytes, &corr_id, "first-page.svg").await
                        }
                    }).await {
                        Ok(url) => {
                            info!("Uploaded improvement first page SVG to S3: {}", url);
                            Some(url)
                        }
                        Err(e) => {
                            error!("Failed to upload improvement first page SVG to S3 after retries: {}", e);
                            None
                        }
                    };

                    // Upload last page SVG to S3 with retry
                    let last_page_svg_url = match retry_with_backoff("upload improvement last page", || {
                        let client = s3_client_for_task.clone();
                        let bucket = s3_bucket_for_task.clone();
                        let region = aws_region_for_task.clone();
                        let bytes = last_page_bytes.clone();
                        let corr_id = correlation_id_for_task.clone();
                        async move {
                            upload_svg_to_s3_internal(&client, &bucket, &region, &bytes, &corr_id, "last-page.svg").await
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
                    };

                    // Create improvement response with S3 URLs
                    let response = SqsNestingResponse {
                        correlation_id: correlation_id_for_task.clone(),
                        first_page_svg_url,
                        last_page_svg_url,
                        parts_placed: result.parts_placed,
                        is_improvement: true,
                        is_final: false,
                        timestamp: current_timestamp(),
                        error_message: None,
                    };

                    info!("Sending improvement response: {} parts placed", response.parts_placed);

                    // Send to SQS with retry
                    if let Err(e) = retry_with_backoff("send improvement to SQS", || {
                        let client = sqs_client_for_task.clone();
                        let url = output_queue_url_for_task.clone();
                        let resp = response.clone();
                        async move { Self::send_message_to_sqs(&client, &url, &resp).await }
                    }).await {
                        error!("Failed to send improvement to queue after retries: {}", e);
                    } else {
                        info!("Successfully sent improvement response to queue");
                    }
                }
                info!("Improvement task finished (channel closed)");
            });

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
            let strategy = AdaptiveNestingStrategy::with_cancellation_checker(Box::new(cancellation_checker));
            info!("Strategy created (took {:?})", strategy_start.elapsed());

            // Clone cancellation_check_count for logging after spawn_blocking
            let cancellation_check_count_for_final_log = cancellation_check_count.clone();

            // Run nest() in a blocking task to avoid blocking the async runtime
            info!("Starting nesting optimization in spawn_blocking task");
            let nest_start = Instant::now();
            let svg_bytes_for_nest = svg_bytes.clone();
            let amount_of_rotations = request.amount_of_rotations;
            let correlation_id_for_error = request.correlation_id.clone();
            let nesting_result = tokio::task::spawn_blocking(move || {
                info!("Inside spawn_blocking: calling strategy.nest()");
                let nest_call_start = Instant::now();
                let result = strategy.nest(
                    bin_width,
                    bin_height,
                    spacing,
                    &svg_bytes_for_nest,
                    amount_of_parts,
                    amount_of_rotations,
                    improvement_callback,
                );
                info!("Inside spawn_blocking: strategy.nest() completed (took {:?})", nest_call_start.elapsed());
                result
            })
            .await
            .context("Failed to spawn blocking task for nesting")?;

            info!("spawn_blocking task completed (took {:?})", nest_start.elapsed());
            let nesting_result = nesting_result.with_context(|| {
                format!(
                    "Failed to process SVG nesting for correlation_id={}",
                    correlation_id_for_error
                )
            })?;
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

            // Prepare final response images using shared helper
            let first_page_bytes = nesting_result.page_svgs.first()
                .unwrap_or(&nesting_result.combined_svg);

            let last_page_bytes = determine_last_page_svg(
                &nesting_result,
                first_page_bytes,
                bin_width,
                bin_height,
            );

            // Upload first page SVG to S3 with retry
            let first_page_svg_url = match retry_with_backoff("upload final first page", || {
                let s3_client = self.s3_client.clone();
                let bucket = self.s3_bucket.clone();
                let region = self.aws_region.clone();
                let bytes = first_page_bytes.clone();
                let corr_id = request.correlation_id.clone();
                async move {
                    upload_svg_to_s3_internal(&s3_client, &bucket, &region, &bytes, &corr_id, "first-page.svg").await
                }
            }).await {
                Ok(url) => {
                    info!("Uploaded final result first page SVG to S3: {}", url);
                    Some(url)
                }
                Err(e) => {
                    error!("Failed to upload final result first page SVG to S3 after retries: {}", e);
                    None
                }
            };

            // Upload last page SVG to S3 with retry
            let last_page_svg_url = match retry_with_backoff("upload final last page", || {
                let s3_client = self.s3_client.clone();
                let bucket = self.s3_bucket.clone();
                let region = self.aws_region.clone();
                let bytes = last_page_bytes.clone();
                let corr_id = request.correlation_id.clone();
                async move {
                    upload_svg_to_s3_internal(&s3_client, &bucket, &region, &bytes, &corr_id, "last-page.svg").await
                }
            }).await {
                Ok(url) => {
                    info!("Uploaded final result last page SVG to S3: {}", url);
                    Some(url)
                }
                Err(e) => {
                    error!("Failed to upload final result last page SVG to S3 after retries: {}", e);
                    None
                }
            };

            // Send final result to queue (with S3 URLs)
            let response = SqsNestingResponse {
                correlation_id: request.correlation_id.clone(),
                first_page_svg_url,
                last_page_svg_url,
                parts_placed: nesting_result.parts_placed,
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

    /// Listen and process messages from the queue (concurrent processing)
    /// Processes messages concurrently using tokio tasks with semaphore-based concurrency control.
    /// The maximum number of concurrent tasks is configurable via MAX_CONCURRENT_TASKS env var (default: 20).
    pub async fn listen_and_process(
        &self,
        mut shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<()> {
        let max_concurrent = get_max_concurrent_tasks();
        info!(
            "Starting concurrent worker on queue: {} (max {} concurrent tasks)",
            self.input_queue_url, max_concurrent
        );

        // Create semaphore to limit concurrent processing
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        // Track all spawned tasks for graceful shutdown
        let mut active_tasks: JoinSet<(String, bool)> = JoinSet::new();

        // Track last cleanup time for cancellation registry
        let mut last_cleanup = Instant::now();
        let cleanup_interval = Duration::from_secs(60); // Cleanup every minute

        loop {
            // Periodic cleanup of expired cancellation registry entries
            if last_cleanup.elapsed() > cleanup_interval {
                self.cleanup_expired_entries();
                last_cleanup = Instant::now();
            }

            tokio::select! {
                _ = shutdown_rx.recv() => {
                    info!("Received shutdown signal, waiting for {} active tasks to complete...", active_tasks.len());
                    break;
                }
                // Handle completed tasks (non-blocking check)
                Some(result) = active_tasks.join_next(), if !active_tasks.is_empty() => {
                    match result {
                        Ok((receipt_handle, success)) => {
                            if success {
                                debug!("Task completed successfully for receipt_handle: {}", receipt_handle);
                            } else {
                                warn!("Task completed with error for receipt_handle: {}", receipt_handle);
                            }
                        }
                        Err(e) => {
                            error!("Task panicked: {}", e);
                        }
                    }
                }
                result = self.sqs_client
                    .receive_message()
                    .queue_url(&self.input_queue_url)
                    .max_number_of_messages(10)
                    .wait_time_seconds(20)
                    .send() => {
                    let response = result.context("Failed to receive messages from queue")?;

                    if let Some(messages) = response.messages {
                        for message in messages {
                            // Check for shutdown before spawning
                            if shutdown_rx.try_recv().is_ok() {
                                info!("Stopping before processing message due to shutdown");
                                break;
                            }

                            let receipt_handle = match message.receipt_handle() {
                                Some(h) => h.to_string(),
                                None => {
                                    error!("Message missing receipt handle, skipping");
                                    continue;
                                }
                            };
                            let body = match message.body() {
                                Some(b) => b.to_string(),
                                None => {
                                    error!("Message missing body, skipping");
                                    continue;
                                }
                            };
                            let message_id = message.message_id().map(|s| s.to_string());

                            if let Some(msg_id) = &message_id {
                                info!("Received message {}, spawning processing task", msg_id);
                            } else {
                                info!("Received message, spawning processing task");
                            }

                            // Clone necessary data for the spawned task
                            let processor = self.clone();
                            let semaphore_clone = semaphore.clone();
                            let sqs_client_clone = self.sqs_client.clone();
                            let input_queue_url_clone = self.input_queue_url.clone();
                            let receipt_handle_clone = receipt_handle.clone();

                            // Spawn concurrent task for processing
                            // Message is deleted AFTER successful processing to prevent data loss
                            active_tasks.spawn(async move {
                                // Acquire semaphore permit (waits if max tasks are already running)
                                let _permit = match semaphore_clone.acquire().await {
                                    Ok(permit) => permit,
                                    Err(e) => {
                                        error!("Failed to acquire semaphore permit: {}", e);
                                        return (receipt_handle_clone, false);
                                    }
                                };

                                // Process the message
                                let process_result = processor.process_message(&receipt_handle, &body).await;
                                let success = process_result.is_ok();

                                if let Err(e) = &process_result {
                                    error!("Error during message processing: {}", e);
                                }

                                // Delete message from queue AFTER processing completes
                                // This ensures messages are not lost if processing fails
                                if let Err(e) = retry_with_backoff("delete_message", || {
                                    let client = sqs_client_clone.clone();
                                    let url = input_queue_url_clone.clone();
                                    let handle = receipt_handle_clone.clone();
                                    async move {
                                        client
                                            .delete_message()
                                            .queue_url(&url)
                                            .receipt_handle(&handle)
                                            .send()
                                            .await
                                            .map_err(|e| anyhow!("SQS delete failed: {}", e))
                                    }
                                }).await {
                                    error!("Failed to delete message after processing: {}", e);
                                    // Message will become visible again after visibility timeout
                                    // and may be reprocessed (at-least-once delivery)
                                }

                                (receipt_handle_clone, success)
                            });
                        }
                    }
                }
            }
        }

        // Graceful shutdown: wait for all active tasks to complete
        info!("Waiting for {} active tasks to complete...", active_tasks.len());
        while let Some(result) = active_tasks.join_next().await {
            match result {
                Ok((receipt_handle, success)) => {
                    if success {
                        info!("Shutdown: task completed successfully for receipt_handle: {}", receipt_handle);
                    } else {
                        warn!("Shutdown: task completed with error for receipt_handle: {}", receipt_handle);
                    }
                }
                Err(e) => {
                    error!("Shutdown: task panicked: {}", e);
                }
            }
        }

        info!("Worker exiting gracefully, all tasks completed");
        Ok(())
    }
}

#[cfg(test)]
impl SqsProcessor {
    pub(crate) fn cancellation_registry_handle(&self) -> Arc<Mutex<HashMap<String, CancellationEntry>>> {
        self.cancellation_registry.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_config::BehaviorVersion;
    use aws_sdk_sqs::Client as SqsClient;
    use tokio::time::Duration;

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
        let (bucket, key) = parse_s3_url("http://localstack:4566/my-bucket/path/to/file.svg").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(key, "path/to/file.svg");
    }

    #[test]
    fn test_parse_s3_url_localstack_https() {
        // LocalStack/Minio path-style URL with HTTPS
        let (bucket, key) = parse_s3_url("https://localhost:4566/my-bucket/path/to/file.svg").unwrap();
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
        assert_eq!(
            request.cancelled, false,
            "cancelled should default to false"
        );
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
        assert_eq!(request.cancelled, true, "cancelled should be true when set");
    }

    #[tokio::test]
    async fn test_parallel_cancellation_flag_shared_between_workers() {
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let sqs_client = SqsClient::new(&config);
        let s3_client = aws_sdk_s3::Client::new(&config);
        let processor = SqsProcessor::new(
            sqs_client,
            s3_client,
            "test-bucket".to_string(),
            "us-east-1".to_string(),
            "test-input-queue".to_string(),
            "test-output-queue".to_string(),
        );

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
            amount_of_rotations: 8,
            output_queue_url: None,
            cancelled: true,
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
        let config = aws_config::defaults(BehaviorVersion::latest()).load().await;
        let sqs_client = SqsClient::new(&config);
        let s3_client = aws_sdk_s3::Client::new(&config);
        let processor = SqsProcessor::new(
            sqs_client,
            s3_client,
            "test-bucket".to_string(),
            "us-east-1".to_string(),
            "test-input-queue".to_string(),
            "test-output-queue".to_string(),
        );

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

        let result: std::result::Result<i32, String> =
            retry_with_backoff("test_op", || {
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

        let result: std::result::Result<i32, String> =
            retry_with_backoff("test_op", || {
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

        let result: std::result::Result<i32, String> =
            retry_with_backoff("test_op", || {
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
        use std::env;
        use aws_config::BehaviorVersion;
        use aws_sdk_s3::Client as S3Client;
        use aws_sdk_s3::error::ProvideErrorMetadata;

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
        println!("  AWS_ACCESS_KEY_ID: {:?}", env::var("AWS_ACCESS_KEY_ID").map(|s| format!("{}...", &s[..10.min(s.len())])));

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
                    println!("⚠ Content doesn't appear to be SVG (first 100 chars: {})", 
                        svg_content.chars().take(100).collect::<String>());
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
                                println!("    {}. {}", i + 1, obj.key().map(|k| k.to_string()).unwrap_or_else(|| "(no key)".to_string()));
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
                let head_result = s3_client
                    .head_bucket()
                    .bucket(&bucket)
                    .send()
                    .await;
                
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
