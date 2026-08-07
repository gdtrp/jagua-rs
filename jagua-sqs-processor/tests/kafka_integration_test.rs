//! Broker-backed integration tests for the Kafka port (VK migration T9).
//!
//! These cover the things that **only fail at runtime** and that the previous
//! test suite could not touch at all: before this file, `listen_and_process`,
//! message acknowledgement and response publishing had zero coverage — the
//! entire transport layer was untested.
//!
//! Every test here is `#[ignore]`d so a plain `cargo test` on a machine with no
//! Docker still passes rather than failing for the wrong reason. Run them with
//! `make test-integration`, which brings up the harness first.
//!
//! The harness (`jagua-sqs-processor/docker-compose.yml`) runs Kafka with
//! **SASL_PLAINTEXT + SCRAM-SHA-512**, matching the VK listener exactly, so
//! authentication is genuinely exercised rather than stubbed out.

use std::time::Duration;

use anyhow::Result;
use jagua_sqs_processor::{KafkaSettings, SqsNestingResponse};
use rdkafka::config::ClientConfig;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::{Header, Headers, Message, OwnedHeaders};
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::{Offset, TopicPartitionList};

/// Broker address, defaulting to the host-facing SASL listener.
///
/// Overridden to `kafka:29092` by `scripts/cargo-docker.sh`, which runs the tests
/// inside a container attached to the compose network — there, `localhost` is the
/// test container itself and every produce would time out.
fn bootstrap() -> String {
    std::env::var("KAFKA_BOOTSTRAP_SERVERS").unwrap_or_else(|_| "localhost:9092".to_string())
}

const USERNAME: &str = "jagua-nesting";
const PASSWORD: &str = "jagua-test-secret";
const REQUEST_TOPIC: &str = "nesting-request";
const RESPONSE_TOPIC: &str = "nesting-response";
const RETRY_1: &str = "jagua-nesting-retry-1";

/// Settings pointed at the harness, with a unique consumer group per test.
///
/// Unique groups matter: `auto.offset.reset=latest` means a group that has
/// already committed will skip records produced before it subscribed, so sharing
/// a group across tests makes them order-dependent.
fn settings(group_suffix: &str) -> KafkaSettings {
    KafkaSettings {
        bootstrap_servers: bootstrap(),
        username: USERNAME.to_string(),
        password: PASSWORD.to_string(),
        sasl_mechanism: "SCRAM-SHA-512".to_string(),
        consumer_group: format!("jagua-nesting-it-{group_suffix}"),
        request_topic: REQUEST_TOPIC.to_string(),
        response_topic: RESPONSE_TOPIC.to_string(),
        attempt_budget: 3,
        // Short enough that a test can watch the ladder without waiting 10 minutes.
        retry_delays: vec![
            Duration::from_millis(300),
            Duration::from_millis(600),
            Duration::from_millis(900),
        ],
    }
}

/// A plain SASL client for the tests themselves, standing in for cutl-backend.
fn test_client_config() -> ClientConfig {
    let mut cfg = ClientConfig::new();
    cfg.set("bootstrap.servers", bootstrap())
        .set("security.protocol", "SASL_PLAINTEXT")
        .set("sasl.mechanism", "SCRAM-SHA-512")
        .set("sasl.username", USERNAME)
        .set("sasl.password", PASSWORD);
    cfg
}

fn test_producer() -> FutureProducer {
    test_client_config()
        .set("message.timeout.ms", "10000")
        .create()
        .expect("producer")
}

/// Consumer subscribed from the *end* of the topic, so it only sees what a test
/// produces after it is created.
fn tail_consumer(topic: &str, group: &str) -> StreamConsumer {
    let consumer: StreamConsumer = test_client_config()
        .set("group.id", group)
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "latest")
        .create()
        .expect("consumer");
    consumer.subscribe(&[topic]).expect("subscribe");
    consumer
}

/// Wait until `predicate` accepts a record, or time out.
async fn await_record<F>(
    consumer: &StreamConsumer,
    timeout: Duration,
    mut predicate: F,
) -> Option<(Option<String>, String, Vec<(String, String)>)>
where
    F: FnMut(&str) -> bool,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return None;
        }
        let Ok(Ok(msg)) = tokio::time::timeout(remaining, consumer.recv()).await else {
            return None;
        };
        let payload = msg
            .payload()
            .map(|b| String::from_utf8_lossy(b).into_owned())
            .unwrap_or_default();
        if predicate(&payload) {
            let key = msg.key().map(|k| String::from_utf8_lossy(k).into_owned());
            let headers = msg
                .headers()
                .map(|hs| {
                    hs.iter()
                        .filter_map(|h| {
                            h.value.map(|v| {
                                (h.key.to_string(), String::from_utf8_lossy(v).into_owned())
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            return Some((key, payload, headers));
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Authentication
// ---------------------------------------------------------------------------

/// SCRAM-SHA-512 over SASL_PLAINTEXT actually works against a broker configured
/// the way VK's is. A `cargo check` cannot prove this and neither can a mock:
/// librdkafka compiles SCRAM out entirely unless built against OpenSSL, and that
/// failure appears only when a client tries to authenticate.
// multi_thread: fetch_metadata is blocking and `block_in_place` panics on
// the single-threaded test runtime.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn scram_authentication_succeeds_against_the_broker() -> Result<()> {
    let s = settings("auth-ok");
    let consumer = s.consumer(&s.consumer_group)?;
    let md =
        tokio::task::block_in_place(|| consumer.fetch_metadata(None, Duration::from_secs(10)))?;

    assert!(
        !md.brokers().is_empty(),
        "metadata fetch must see at least one broker"
    );
    Ok(())
}

/// The negative case. A pod with bad credentials must fail to reach the broker
/// rather than appearing to work — this is what keeps `/ready` at 503 instead of
/// reporting Ready and silently consuming nothing.
// multi_thread: fetch_metadata is blocking and `block_in_place` panics on
// the single-threaded test runtime.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn scram_authentication_fails_with_a_wrong_password() -> Result<()> {
    let mut s = settings("auth-bad");
    s.password = "definitely-not-the-password".to_string();

    let consumer = s.consumer(&s.consumer_group)?;
    let result =
        tokio::task::block_in_place(|| consumer.fetch_metadata(None, Duration::from_secs(5)));

    assert!(
        result.is_err(),
        "a wrong SCRAM password must not yield working metadata"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 2. Keying
// ---------------------------------------------------------------------------

/// Requests and responses are keyed by `correlationId`. On the request side this
/// is what makes cancellation possible at all: it puts a cancel on the same
/// partition as the job it cancels. This test pins the property the whole design
/// rests on — same key, same partition.
#[tokio::test]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn same_correlation_id_always_lands_on_one_partition() -> Result<()> {
    let producer = test_producer();
    let correlation_id = "partition-affinity-check";

    let mut partitions = Vec::new();
    for i in 0..12 {
        let payload = format!(r#"{{"correlationId":"{correlation_id}","seq":{i}}}"#);
        let (partition, _offset) = producer
            .send(
                FutureRecord::to(REQUEST_TOPIC)
                    .key(correlation_id)
                    .payload(&payload),
                Duration::from_secs(10),
            )
            .await
            .map_err(|(e, _)| e)?;
        partitions.push(partition);
    }

    let first = partitions[0];
    assert!(
        partitions.iter().all(|p| *p == first),
        "a keyed produce must be partition-stable, got {partitions:?}. \
         Without this a cancel can land on a partition the running job is not on."
    );
    Ok(())
}

/// The complement: the topic really does have more than one partition, so the
/// test above is meaningful rather than trivially true.
// multi_thread: fetch_metadata is blocking and `block_in_place` panics on
// the single-threaded test runtime.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn request_topic_has_the_contracted_three_partitions() -> Result<()> {
    let s = settings("partition-count");
    let consumer = s.consumer(&s.consumer_group)?;
    let md = tokio::task::block_in_place(|| {
        consumer.fetch_metadata(Some(REQUEST_TOPIC), Duration::from_secs(10))
    })?;

    let topic = md
        .topics()
        .iter()
        .find(|t| t.name() == REQUEST_TOPIC)
        .expect("nesting-request must exist");

    assert_eq!(
        topic.partitions().len(),
        3,
        "cutl-infra provisions 3 partitions; a 1-partition topic would hide ordering bugs"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 3. Offset commit semantics
// ---------------------------------------------------------------------------

/// A commit of offset N acknowledges everything below it. This test proves the
/// property that makes the contiguous watermark necessary, by committing a
/// deliberately-too-high offset and observing that the earlier records are gone.
///
/// It is the executable form of "commits are not SQS deletes".
#[tokio::test]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn committing_an_offset_acknowledges_everything_below_it() -> Result<()> {
    let producer = test_producer();
    let group = "jagua-nesting-it-commit-semantics";

    // Pin one partition so offsets are a simple sequence.
    let mut assignment = TopicPartitionList::new();
    assignment.add_partition_offset(REQUEST_TOPIC, 0, Offset::End)?;

    let consumer: StreamConsumer = test_client_config()
        .set("group.id", group)
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "latest")
        .create()?;
    consumer.assign(&assignment)?;
    // Force the assignment to take effect before producing.
    let _ = tokio::time::timeout(Duration::from_secs(2), consumer.recv()).await;

    let mut offsets = Vec::new();
    for i in 0..3 {
        let payload = format!(r#"{{"correlationId":"commit-{i}"}}"#);
        let (_p, offset) = producer
            .send(
                FutureRecord::to(REQUEST_TOPIC)
                    .partition(0)
                    .key("commit-test")
                    .payload(&payload),
                Duration::from_secs(10),
            )
            .await
            .map_err(|(e, _)| e)?;
        offsets.push(offset);
    }

    // Commit only the LAST record, as a naive commit-on-completion would when the
    // last job happens to finish first.
    let mut tpl = TopicPartitionList::new();
    tpl.add_partition_offset(REQUEST_TOPIC, 0, Offset::Offset(offsets[2] + 1))?;
    consumer.commit(&tpl, rdkafka::consumer::CommitMode::Sync)?;

    let committed = consumer.committed(Duration::from_secs(10))?;
    let position = committed
        .find_partition(REQUEST_TOPIC, 0)
        .and_then(|p| match p.offset() {
            Offset::Offset(o) => Some(o),
            _ => None,
        })
        .expect("committed offset");

    assert!(
        position > offsets[0] && position > offsets[1],
        "committing {} acknowledged offsets {} and {} that were never handled — \
         this is precisely why OffsetWatermark advances only through the unbroken prefix",
        position,
        offsets[0],
        offsets[1]
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// 4. Retry ladder
// ---------------------------------------------------------------------------

/// The contractual headers survive a republish, with `x-cutl-attempt` incremented
/// and `x-cutl-origin-topic` preserved. Without the origin header the tier
/// consumer cannot tell which handler to dispatch to, and a dropped message would
/// be labelled with the retry topic rather than the data topic it came from.
#[tokio::test]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn retry_republish_carries_incremented_attempt_and_origin_topic() -> Result<()> {
    let producer = test_producer();
    let consumer = tail_consumer(RETRY_1, "jagua-nesting-it-retry-headers");
    // Let the subscription settle so nothing produced below is missed.
    let _ = tokio::time::timeout(Duration::from_secs(3), consumer.recv()).await;

    let correlation_id = "retry-header-check";
    let payload = format!(r#"{{"correlationId":"{correlation_id}","cancelled":false}}"#);

    producer
        .send(
            FutureRecord::to(RETRY_1)
                .key(correlation_id)
                .payload(&payload)
                .headers(
                    OwnedHeaders::new()
                        .insert(Header {
                            key: "x-cutl-attempt",
                            value: Some("2"),
                        })
                        .insert(Header {
                            key: "x-cutl-origin-topic",
                            value: Some(REQUEST_TOPIC),
                        }),
                ),
            Duration::from_secs(10),
        )
        .await
        .map_err(|(e, _)| e)?;

    let (key, _payload, headers) = await_record(&consumer, Duration::from_secs(15), |p| {
        p.contains(correlation_id)
    })
    .await
    .expect("republished record should arrive on the tier-1 topic");

    assert_eq!(
        key.as_deref(),
        Some(correlation_id),
        "key must be preserved"
    );

    let attempt = headers
        .iter()
        .find(|(k, _)| k == "x-cutl-attempt")
        .map(|(_, v)| v.as_str());
    let origin = headers
        .iter()
        .find(|(k, _)| k == "x-cutl-origin-topic")
        .map(|(_, v)| v.as_str());

    assert_eq!(attempt, Some("2"));
    assert_eq!(
        origin,
        Some(REQUEST_TOPIC),
        "origin must stay the data topic, not become the retry topic"
    );
    Ok(())
}

/// All three tier topics exist. The ladder silently collapses to "drop on first
/// failure" if a tier topic is missing and auto-creation is off — which it is, on
/// purpose, in both the harness and production.
// multi_thread: fetch_metadata is blocking and `block_in_place` panics on
// the single-threaded test runtime.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn all_three_retry_tier_topics_exist() -> Result<()> {
    // The REAL consumer group, not a per-test one: retry topics are named after
    // the group (`<group>-retry-N`), so a synthetic group would look for topics
    // cutl-infra never provisions.
    let mut s = settings("tier-topics");
    s.consumer_group = "jagua-nesting".to_string();
    let consumer = s.consumer("jagua-nesting-it-tier-topics")?;
    let md =
        tokio::task::block_in_place(|| consumer.fetch_metadata(None, Duration::from_secs(10)))?;

    let names: Vec<&str> = md.topics().iter().map(|t| t.name()).collect();
    for tier in 1..=3 {
        let expected = s.retry_topic(tier);
        assert!(
            names.contains(&expected.as_str()),
            "missing retry topic {expected}; the ladder cannot escalate without it. \
             Present: {names:?}"
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 5. Response wire
// ---------------------------------------------------------------------------

/// A response published through the real producer round-trips byte-for-byte
/// through the spec-governed wire types, and is keyed so cutl-backend can rely on
/// per-job ordering.
#[tokio::test]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn response_round_trips_through_the_wire_types() -> Result<()> {
    let producer = test_producer();
    let consumer = tail_consumer(RESPONSE_TOPIC, "jagua-nesting-it-response-wire");
    let _ = tokio::time::timeout(Duration::from_secs(3), consumer.recv()).await;

    let correlation_id = "response-wire-check";
    let response = SqsNestingResponse {
        correlation_id: correlation_id.to_string(),
        first_page_svg_url: Some("s3://bucket/first.svg".to_string()),
        last_page_svg_url: None,
        sheets: Some(2),
        sheets_total: Some(2),
        page_svg_urls: Some(vec!["s3://bucket/p0.svg".to_string()]),
        pages: None,
        parts_placed: 7,
        utilisation: 0.625,
        is_improvement: false,
        is_final: true,
        timestamp: 1_700_000_000,
        error_message: None,
    };
    let payload = serde_json::to_string(&response)?;

    producer
        .send(
            FutureRecord::to(RESPONSE_TOPIC)
                .key(correlation_id)
                .payload(&payload),
            Duration::from_secs(10),
        )
        .await
        .map_err(|(e, _)| e)?;

    let (key, body, _) = await_record(&consumer, Duration::from_secs(15), |p| {
        p.contains(correlation_id)
    })
    .await
    .expect("response should arrive");

    assert_eq!(key.as_deref(), Some(correlation_id));

    // The wire uses `final` / `improvement`, not the Rust field names.
    assert!(body.contains(r#""final":true"#), "got: {body}");
    assert!(body.contains(r#""improvement":false"#), "got: {body}");

    let decoded: SqsNestingResponse = serde_json::from_str(&body)?;
    assert_eq!(decoded, response, "response must survive the round trip");
    Ok(())
}

// ---------------------------------------------------------------------------
// 6. Retry-tier delay
// ---------------------------------------------------------------------------

/// A tier consumer must not hand a record to its handler before the tier delay
/// has elapsed — and must not lose it either.
///
/// This is the executable form of "Kafka has no delayed delivery". The consumer
/// pauses the partition and rewinds rather than sleeping, because sleeping in a
/// handler trips `max.poll.interval.ms` and gets the consumer evicted mid-job.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn tier_consumer_holds_a_record_until_its_delay_elapses() -> Result<()> {
    use rdkafka::consumer::CommitMode;

    let producer = test_producer();
    let group = "jagua-nesting-it-tier-delay";
    let delay = Duration::from_secs(3);

    // Pin one partition so the pause/seek behaviour is observable in isolation.
    let consumer: StreamConsumer = test_client_config()
        .set("group.id", group)
        .set("enable.auto.commit", "false")
        .set("auto.offset.reset", "latest")
        .create()?;
    let mut assignment = TopicPartitionList::new();
    assignment.add_partition_offset(RETRY_1, 0, Offset::End)?;
    consumer.assign(&assignment)?;
    let _ = tokio::time::timeout(Duration::from_secs(2), consumer.recv()).await;

    let correlation_id = "tier-delay-check";
    producer
        .send(
            FutureRecord::to(RETRY_1)
                .partition(0)
                .key(correlation_id)
                .payload(&format!(r#"{{"correlationId":"{correlation_id}"}}"#)),
            Duration::from_secs(10),
        )
        .await
        .map_err(|(e, _)| e)?;

    // Read it, find it too young, and hold the partition back — exactly what
    // run_tier does.
    let msg = tokio::time::timeout(Duration::from_secs(10), consumer.recv())
        .await
        .expect("record should be delivered")?;
    let offset = msg.offset();
    let produced_at = msg.timestamp().to_millis().unwrap_or(0);
    drop(msg);

    let mut tpl = TopicPartitionList::new();
    tpl.add_partition_offset(RETRY_1, 0, Offset::Offset(offset))?;
    consumer.pause(&tpl)?;
    consumer.seek(RETRY_1, 0, Offset::Offset(offset), Duration::from_secs(5))?;

    // While paused, polling continues but yields nothing. That is what keeps the
    // consumer in its group without a sleep.
    let held = tokio::time::timeout(Duration::from_secs(1), consumer.recv()).await;
    assert!(
        held.is_err(),
        "a paused partition must yield nothing; the record was handled too early"
    );

    // Wait out the delay, resume, and confirm the SAME record comes back.
    tokio::time::sleep(delay).await;
    consumer.resume(&tpl)?;

    let redelivered = tokio::time::timeout(Duration::from_secs(10), consumer.recv())
        .await
        .expect("record must be redelivered after resume")?;

    assert_eq!(
        redelivered.offset(),
        offset,
        "the held record must come back, not be skipped — a lost retry is a lost message"
    );

    let age_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
        - produced_at;
    assert!(
        age_ms >= delay.as_millis() as i64,
        "record was handled after {age_ms}ms, before its {delay:?} tier delay"
    );

    let mut commit = TopicPartitionList::new();
    commit.add_partition_offset(RETRY_1, 0, Offset::Offset(offset + 1))?;
    consumer.commit(&commit, CommitMode::Sync)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// 7. Trace-context propagation
// ---------------------------------------------------------------------------

/// A `traceparent` header survives a real produce/consume round trip and still
/// names the same trace.
///
/// Without this, every nesting job opens a trace unrelated to the backend call
/// that caused it — the spans exist and look fine, they just never join up, which
/// is a failure you only notice while staring at Tempo during an incident.
#[tokio::test]
#[ignore = "requires the docker-compose harness; run via `make test-integration`"]
async fn traceparent_header_survives_the_wire() -> Result<()> {
    use jagua_sqs_processor::trace_context::{HeaderExtractor, TRACEPARENT};

    let producer = test_producer();
    let consumer = tail_consumer(RESPONSE_TOPIC, "jagua-nesting-it-traceparent");
    let _ = tokio::time::timeout(Duration::from_secs(3), consumer.recv()).await;

    let correlation_id = "traceparent-check";
    let traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01";

    producer
        .send(
            FutureRecord::to(RESPONSE_TOPIC)
                .key(correlation_id)
                .payload(&format!(r#"{{"correlationId":"{correlation_id}"}}"#))
                .headers(
                    OwnedHeaders::new()
                        .insert(Header {
                            key: TRACEPARENT,
                            value: Some(traceparent),
                        })
                        // A retry header alongside it: injection must not clobber
                        // the ladder's own headers, or the attempt budget resets.
                        .insert(Header {
                            key: "x-cutl-attempt",
                            value: Some("2"),
                        }),
                ),
            Duration::from_secs(10),
        )
        .await
        .map_err(|(e, _)| e)?;

    let (_key, _payload, headers) = await_record(&consumer, Duration::from_secs(15), |p| {
        p.contains(correlation_id)
    })
    .await
    .expect("record should arrive");

    let extractor = HeaderExtractor::from_pairs(headers.clone());
    assert_eq!(
        extractor.traceparent(),
        Some(traceparent),
        "traceparent must cross the broker intact; got headers {headers:?}"
    );
    assert!(
        headers
            .iter()
            .any(|(k, v)| k == "x-cutl-attempt" && v == "2"),
        "retry headers must coexist with trace headers"
    );
    Ok(())
}
