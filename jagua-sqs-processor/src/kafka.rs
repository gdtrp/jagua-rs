//! Kafka transport for the VK Cloud deployment (migration T9).
//!
//! Replaces AWS SQS. Not a preference: measured from a VK pod, `sqs.eu-north-1`
//! answered 0 of 6 probes while `cutl-kafka-bootstrap.kafka.svc:9092` answered 6
//! of 6.
//!
//! The semantics here are fixed by `cutl-infra/docs/kafka-contract.md`, which
//! exists because **SQS semantics do not survive a naive port**. Three of its
//! rules shape this module, and each fails only at runtime:
//!
//! 1. **Never pause a partition while a nesting job runs.** Both topics are keyed
//!    by `correlationId`, so a cancellation lands on the *same partition* as the
//!    job it cancels, at a later offset. Pausing to throttle would mean the cancel
//!    is only read after the job it cancels has already finished — silently making
//!    cancellation a no-op. Keep fetching while jobs run.
//! 2. **A commit of offset 105 implicitly acknowledges 100..=104.** SQS deletes
//!    were order-independent; offsets are not. With jobs completing out of order
//!    behind a semaphore, committing on completion would acknowledge in-flight
//!    work. Hence [`OffsetWatermark`].
//! 3. **Never `sleep()` in a handler and never `seek()` back to retry.** Sleeping
//!    trips `max.poll.interval.ms` and gets the consumer evicted mid-job;
//!    `seek()`-back head-of-line-blocks the partition on a poison message. Retry is
//!    republish-to-the-next-tier-and-commit.

use std::collections::{BTreeSet, HashMap};
use std::time::Duration;

use anyhow::{Context, Result};
use rdkafka::config::ClientConfig;
use rdkafka::consumer::StreamConsumer;
use rdkafka::message::{Header, Headers, Message, OwnedHeaders};
use rdkafka::producer::FutureProducer;

/// Header carrying how many times a handler has been invoked for this message.
///
/// Replaces SQS's `ApproximateReceiveCount`, which has no Kafka equivalent.
/// Absent means 1 — the first attempt does not set it.
pub const HEADER_ATTEMPT: &str = "x-cutl-attempt";

/// Header carrying the data topic a message originally came from.
///
/// Retry topics are per consumer group rather than per data topic (18 topics
/// instead of 66), so without this the tier consumer cannot tell which handler to
/// dispatch to, or which `origin_topic` to label a dropped message with.
pub const HEADER_ORIGIN_TOPIC: &str = "x-cutl-origin-topic";

/// Default consumer group. Its SCRAM user and per-tier retry topics already exist
/// in every environment.
pub const DEFAULT_CONSUMER_GROUP: &str = "jagua-nesting";

/// Topics carry **no environment suffix**: each environment has its own cluster,
/// so the suffix lives in the cluster boundary rather than the topic name.
pub const DEFAULT_REQUEST_TOPIC: &str = "nesting-request";
pub const DEFAULT_RESPONSE_TOPIC: &str = "nesting-response";

/// Handler invocations allowed before a message is dropped. Matches the contract's
/// default attempt budget and today's SQS `maxReceiveCount` where one exists.
pub const DEFAULT_ATTEMPT_BUDGET: u32 = 3;

/// Tier delays: 5s, 1min, 10min. Overridable so tests do not wait ten minutes.
pub const DEFAULT_RETRY_DELAYS_MS: [u64; 3] = [5_000, 60_000, 600_000];

/// `max.poll.interval.ms` for this group.
///
/// 15 minutes, the contract's value for nesting. It is the replacement for the SQS
/// visibility timeout and must exceed the failsafe execution timeout
/// (`maxSeconds` caps at 600s, plus a 60s failsafe). Exceeding it means Kafka
/// evicts the consumer, rebalances, and hands the same message to another instance
/// while the first is still working — a bit-for-bit reproduction of incident
/// MAJ-29.
const MAX_POLL_INTERVAL_MS: &str = "900000";

/// Connection and topic configuration, all from the environment.
///
/// `KAFKA_USERNAME` / `KAFKA_PASSWORD` / `KAFKA_BOOTSTRAP_SERVERS` /
/// `KAFKA_SASL_MECHANISM` are supplied verbatim by the `kafka-jagua-nesting`
/// Secret, which cutl-infra mounts with `envFrom` — the key names are already
/// env-var shaped precisely so no parsing or file mounting is needed.
#[derive(Debug, Clone)]
pub struct KafkaSettings {
    pub bootstrap_servers: String,
    pub username: String,
    pub password: String,
    pub sasl_mechanism: String,
    pub consumer_group: String,
    pub request_topic: String,
    pub response_topic: String,
    pub attempt_budget: u32,
    pub retry_delays: Vec<Duration>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| default.to_string())
}

impl KafkaSettings {
    pub fn from_env() -> Result<Self> {
        let bootstrap_servers = std::env::var("KAFKA_BOOTSTRAP_SERVERS").context(
            "KAFKA_BOOTSTRAP_SERVERS is required (supplied by the kafka-<group> Secret)",
        )?;
        let username = std::env::var("KAFKA_USERNAME")
            .context("KAFKA_USERNAME is required (supplied by the kafka-<group> Secret)")?;
        let password = std::env::var("KAFKA_PASSWORD")
            .context("KAFKA_PASSWORD is required (supplied by the kafka-<group> Secret)")?;

        let retry_delays = match std::env::var("KAFKA_RETRY_DELAYS_MS") {
            Ok(raw) if !raw.is_empty() => raw
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<u64>()
                        .map(Duration::from_millis)
                        .with_context(|| format!("invalid duration in KAFKA_RETRY_DELAYS_MS: {s}"))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => DEFAULT_RETRY_DELAYS_MS
                .iter()
                .map(|ms| Duration::from_millis(*ms))
                .collect(),
        };

        let attempt_budget = env_or("KAFKA_ATTEMPT_BUDGET", "")
            .parse::<u32>()
            .unwrap_or(DEFAULT_ATTEMPT_BUDGET);

        Ok(Self {
            bootstrap_servers,
            username,
            password,
            sasl_mechanism: env_or("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512"),
            consumer_group: env_or("KAFKA_CONSUMER_GROUP", DEFAULT_CONSUMER_GROUP),
            request_topic: env_or("KAFKA_REQUEST_TOPIC", DEFAULT_REQUEST_TOPIC),
            response_topic: env_or("KAFKA_RESPONSE_TOPIC", DEFAULT_RESPONSE_TOPIC),
            attempt_budget,
            retry_delays,
        })
    }

    /// Topic for the given tier (1-based). `jagua-nesting-retry-2`, etc.
    pub fn retry_topic(&self, tier: u32) -> String {
        format!("{}-retry-{}", self.consumer_group, tier)
    }

    /// Consumer group for a tier's dedicated consumer.
    ///
    /// The KafkaUser ACL grants group **prefix** `jagua-nesting`, so these
    /// authorize under the same SCRAM credential.
    pub fn retry_group(&self, tier: u32) -> String {
        format!("{}-retry-{}", self.consumer_group, tier)
    }

    pub fn delay_for_tier(&self, tier: u32) -> Duration {
        self.retry_delays
            .get((tier as usize).saturating_sub(1))
            .copied()
            .unwrap_or_else(|| Duration::from_millis(DEFAULT_RETRY_DELAYS_MS[2]))
    }

    /// Auth settings shared by producer and consumer.
    ///
    /// `SASL_PLAINTEXT`, not `SASL_SSL`: the VK listener sets `tls = false`
    /// (cutl-infra `modules-vk/kafka/main.tf:147-153`). Setting `ssl.ca.location`
    /// here would break the handshake, not harden it.
    fn apply_auth(&self, cfg: &mut ClientConfig) {
        cfg.set("bootstrap.servers", &self.bootstrap_servers)
            .set("security.protocol", "SASL_PLAINTEXT")
            .set("sasl.mechanism", &self.sasl_mechanism)
            .set("sasl.username", &self.username)
            .set("sasl.password", &self.password);
    }

    /// Consumer for the main request topic, or for a retry tier.
    pub fn consumer(&self, group: &str) -> Result<StreamConsumer> {
        let mut cfg = ClientConfig::new();
        self.apply_auth(&mut cfg);
        cfg.set("group.id", group)
            // Offsets are committed explicitly through the contiguous-completed
            // watermark. Auto-commit would acknowledge whatever the client last
            // handed us, including jobs still running.
            .set("enable.auto.commit", "false")
            // Staging is destroyed and recreated on demand, so a recreated group
            // with `earliest` would replay the entire 96h retention window. SQS
            // semantics were "unconsumed messages eventually expire", which
            // `latest` approximates and `earliest` badly violates.
            .set("auto.offset.reset", "latest")
            .set("max.poll.interval.ms", MAX_POLL_INTERVAL_MS)
            .set("session.timeout.ms", "45000");

        // The contract specifies `max.poll.records = 1`. That property is
        // Java-consumer-only — librdkafka rejects it outright at client creation
        // with "No such configuration property", which is why this is a comment
        // rather than a `.set()`.
        //
        // The requirement is met structurally instead: `StreamConsumer::recv()`
        // yields exactly one message per call, and each is dispatched to its own
        // task, so a handler is never responsible for a batch. That is the
        // property the contract was protecting — at the Java default of 500, the
        // poll interval would have to cover 500 times the handler duration, which
        // for a minutes-long nesting run is arithmetically impossible.

        cfg.create()
            .context("failed to create the Kafka consumer (check SCRAM credentials)")
    }

    pub fn producer(&self) -> Result<FutureProducer> {
        let mut cfg = ClientConfig::new();
        self.apply_auth(&mut cfg);
        cfg.set("message.timeout.ms", "30000")
            // Responses are keyed by correlationId and a retry would otherwise be
            // able to reorder two responses for the same job.
            .set("enable.idempotence", "true");

        cfg.create()
            .context("failed to create the Kafka producer (check SCRAM credentials)")
    }
}

/// The two contractual headers, read off an incoming record.
#[derive(Debug, Clone)]
pub struct RecordHeaders {
    /// How many times a handler has been invoked for this message, 1-based.
    ///
    /// Absent means 1: the original produce from cutl-backend does not set it, and
    /// treating absence as 0 would silently grant every message one extra attempt.
    pub attempt: u32,
    /// The data topic this message started on. `None` when absent, in which case
    /// the caller substitutes the record's own topic.
    pub origin_topic: Option<String>,
}

impl RecordHeaders {
    /// Extract the retry headers from any rdkafka message.
    pub fn from_message<M: Message>(msg: &M) -> Self {
        let mut attempt = 1;
        let mut origin_topic = None;

        if let Some(headers) = msg.headers() {
            for header in headers.iter() {
                let Some(value) = header.value else { continue };
                let Ok(text) = std::str::from_utf8(value) else {
                    continue;
                };
                match header.key {
                    HEADER_ATTEMPT => {
                        if let Ok(n) = text.trim().parse::<u32>() {
                            attempt = n.max(1);
                        }
                    }
                    HEADER_ORIGIN_TOPIC => origin_topic = Some(text.to_string()),
                    _ => {}
                }
            }
        }

        Self {
            attempt,
            origin_topic,
        }
    }

    /// The topic to attribute this message to, falling back to where it was read
    /// from. Used as the `origin_topic` label on `cutl_retries_exhausted_total`.
    pub fn origin_or(&self, fallback: &str) -> String {
        self.origin_topic
            .clone()
            .unwrap_or_else(|| fallback.to_string())
    }
}

/// Build the retry headers for a republished message.
pub fn retry_headers(next_attempt: u32, origin_topic: &str) -> OwnedHeaders {
    OwnedHeaders::new()
        .insert(Header {
            key: HEADER_ATTEMPT,
            value: Some(&next_attempt.to_string()),
        })
        .insert(Header {
            key: HEADER_ORIGIN_TOPIC,
            value: Some(origin_topic),
        })
}

/// Tracks which offsets have been fully handled, and how far it is safe to commit.
///
/// **This is the piece that has no SQS analogue.** SQS deleted each message
/// independently, so completion order did not matter. A Kafka commit of offset 105
/// implicitly acknowledges everything below it, so committing on completion while
/// twenty jobs run concurrently would acknowledge messages still in flight — and a
/// crash would lose them with no DLQ to recover from.
///
/// The rule: advance only through the unbroken prefix of completed offsets. If 100
/// and 102 are done but 101 is still running, the commit point stays at 101.
#[derive(Debug, Default)]
pub struct OffsetWatermark {
    /// Per `(topic, partition)`: offsets completed out of order, and the next
    /// offset we are waiting on.
    partitions: HashMap<(String, i32), PartitionState>,
}

#[derive(Debug, Default)]
struct PartitionState {
    /// Completed offsets at or above `next_expected`, held until the gap below
    /// them fills in.
    completed: BTreeSet<i64>,
    /// The lowest offset not yet known to be complete. `None` until the first
    /// record is seen, because the starting offset depends on where the group
    /// resumed.
    next_expected: Option<i64>,
}

impl OffsetWatermark {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a record was received, before its handler runs.
    ///
    /// Establishes where the partition starts. Without it the first completion
    /// could not tell "offset 500 is the first record we ever saw" from "offsets
    /// 0..499 are still in flight".
    pub fn observe(&mut self, topic: &str, partition: i32, offset: i64) {
        let state = self
            .partitions
            .entry((topic.to_string(), partition))
            .or_default();
        if state.next_expected.is_none() {
            state.next_expected = Some(offset);
        }
    }

    /// Record a handled record. Returns the new commit position — the offset to
    /// commit, which by Kafka convention is *one past* the last handled record —
    /// or `None` when the prefix did not advance.
    pub fn complete(&mut self, topic: &str, partition: i32, offset: i64) -> Option<i64> {
        let state = self
            .partitions
            .entry((topic.to_string(), partition))
            .or_default();

        // A completion for an offset we never observed still establishes the
        // start, so a caller that skips `observe` degrades gracefully.
        let next = *state.next_expected.get_or_insert(offset);

        // Already-committed territory: a duplicate delivery after a rebalance.
        // Ignore rather than rewinding the watermark.
        if offset < next {
            return None;
        }

        state.completed.insert(offset);

        let mut cursor = next;
        while state.completed.remove(&cursor) {
            cursor += 1;
        }

        if cursor == next {
            // The prefix did not move: this completion filled a hole above a gap.
            return None;
        }

        state.next_expected = Some(cursor);
        Some(cursor)
    }

    /// Offsets still in flight for a partition, for diagnostics.
    pub fn pending(&self, topic: &str, partition: i32) -> usize {
        self.partitions
            .get(&(topic.to_string(), partition))
            .map(|s| s.completed.len())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: &str = "nesting-request";

    #[test]
    fn in_order_completion_advances_every_time() {
        let mut w = OffsetWatermark::new();
        for o in 10..15 {
            w.observe(T, 0, o);
        }
        assert_eq!(w.complete(T, 0, 10), Some(11));
        assert_eq!(w.complete(T, 0, 11), Some(12));
        assert_eq!(w.complete(T, 0, 12), Some(13));
    }

    /// The reason this type exists: a later job finishing first must NOT commit
    /// past the earlier one still running.
    #[test]
    fn out_of_order_completion_holds_the_watermark_at_the_gap() {
        let mut w = OffsetWatermark::new();
        for o in 0..4 {
            w.observe(T, 0, o);
        }

        // 1, 2 and 3 finish while 0 is still running. Nothing may be committed:
        // committing 4 would acknowledge offset 0 as well.
        assert_eq!(w.complete(T, 0, 1), None);
        assert_eq!(w.complete(T, 0, 2), None);
        assert_eq!(w.complete(T, 0, 3), None);

        // 0 finishes and the whole run collapses in one commit.
        assert_eq!(w.complete(T, 0, 0), Some(4));
    }

    #[test]
    fn partitions_are_tracked_independently() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 0, 0);
        w.observe(T, 1, 0);

        assert_eq!(w.complete(T, 0, 0), Some(1));
        // Partition 1 is untouched by partition 0's progress.
        assert_eq!(w.complete(T, 1, 0), Some(1));
    }

    #[test]
    fn topics_are_tracked_independently() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 0, 5);
        w.observe("jagua-nesting-retry-1", 0, 100);

        assert_eq!(w.complete(T, 0, 5), Some(6));
        assert_eq!(w.complete("jagua-nesting-retry-1", 0, 100), Some(101));
    }

    /// After a rebalance the same record can be delivered twice. The second
    /// completion must not rewind the committed position.
    #[test]
    fn duplicate_completion_does_not_rewind() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 0, 0);
        assert_eq!(w.complete(T, 0, 0), Some(1));
        assert_eq!(w.complete(T, 0, 0), None, "duplicate must be ignored");
        assert_eq!(w.complete(T, 0, 1), Some(2), "progress still works after");
    }

    /// The group may resume at any offset; the first record seen defines the
    /// origin rather than offset 0.
    #[test]
    fn starts_at_the_first_observed_offset_not_zero() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 0, 5_000);
        assert_eq!(
            w.complete(T, 0, 5_000),
            Some(5_001),
            "must not wait for offsets 0..4999 that this consumer never owned"
        );
    }

    #[test]
    fn pending_counts_completions_held_behind_a_gap() {
        let mut w = OffsetWatermark::new();
        for o in 0..4 {
            w.observe(T, 0, o);
        }
        w.complete(T, 0, 2);
        w.complete(T, 0, 3);
        assert_eq!(w.pending(T, 0), 2);

        w.complete(T, 0, 1);
        assert_eq!(w.pending(T, 0), 3, "still blocked on offset 0");

        w.complete(T, 0, 0);
        assert_eq!(w.pending(T, 0), 0, "gap filled, everything drained");
    }

    #[test]
    fn retry_topic_and_group_follow_the_consumer_group() {
        let s = settings();
        assert_eq!(s.retry_topic(1), "jagua-nesting-retry-1");
        assert_eq!(s.retry_topic(3), "jagua-nesting-retry-3");
        assert_eq!(s.retry_group(2), "jagua-nesting-retry-2");
    }

    #[test]
    fn tier_delays_match_the_contract() {
        let s = settings();
        assert_eq!(s.delay_for_tier(1), Duration::from_secs(5));
        assert_eq!(s.delay_for_tier(2), Duration::from_secs(60));
        assert_eq!(s.delay_for_tier(3), Duration::from_secs(600));
    }

    fn settings() -> KafkaSettings {
        KafkaSettings {
            bootstrap_servers: "localhost:9092".into(),
            username: "jagua-nesting".into(),
            password: "x".into(),
            sasl_mechanism: "SCRAM-SHA-512".into(),
            consumer_group: DEFAULT_CONSUMER_GROUP.into(),
            request_topic: DEFAULT_REQUEST_TOPIC.into(),
            response_topic: DEFAULT_RESPONSE_TOPIC.into(),
            attempt_budget: DEFAULT_ATTEMPT_BUDGET,
            retry_delays: DEFAULT_RETRY_DELAYS_MS
                .iter()
                .map(|ms| Duration::from_millis(*ms))
                .collect(),
        }
    }
}
