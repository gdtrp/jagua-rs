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

use std::collections::{BTreeMap, BTreeSet, HashMap};
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

/// librdkafka's `message.max.bytes`: a round **1,000,000**, not 1 MiB.
///
/// The distinction is load-bearing. A response of 1,024,000 bytes ("1000 KB")
/// clears a `1024 * 1024` check and is then rejected by the producer, which is
/// exactly how two finished production jobs were discarded on 2026-08-31. The
/// rejection happens client-side in `rd_kafka_msg_new0()` — the broker never sees
/// the message, so raising the broker's own limit would not have helped, and
/// neither would compression: the check is against the UNCOMPRESSED wire size.
pub const PRODUCER_MAX_MESSAGE_BYTES: usize = 1_000_000;

/// Worst-case per-message framing librdkafka adds on top of key + value + headers
/// (`RD_KAFKAP_MESSAGE_V2_MAX_OVERHEAD`: five 32-bit varints, one 64-bit varint and
/// the attribute byte). The limit is checked against
/// `overhead + keylen + valuelen + hdrslen`, so a guard that looks only at the
/// payload is measuring the wrong thing.
pub const PRODUCER_MESSAGE_OVERHEAD_BYTES: usize = 36;

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
            .set("enable.idempotence", "true")
            // Set explicitly even though it equals librdkafka's own default, so the
            // producer-side guard in `processor.rs` can be derived from the same
            // number instead of restating it. They were allowed to drift once and it
            // cost two completed nesting jobs — see PRODUCER_MAX_MESSAGE_BYTES.
            .set("message.max.bytes", PRODUCER_MAX_MESSAGE_BYTES.to_string());

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
    /// Jobs whose handler has started and not finished. Written into every commit's
    /// metadata, so after a SIGKILL the next owner knows which records were running.
    started: BTreeMap<i64, CrashMark>,
    /// What the previous owner's last commit said was running. `None` until loaded.
    prior: Option<BTreeMap<i64, CrashMark>>,
}

/// Crash bookkeeping for one in-flight record (cutl-tests#392).
///
/// An OOM kill (SIGKILL) never reaches [`crate::processor::NestingProcessor::escalate`],
/// and a redelivered record is the original from cutl-backend, with no attempt header.
/// Headers on a consumed record cannot be changed, so the count rides in the one place
/// Kafka keeps per-partition state for a consumer group: the offset commit's metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CrashMark {
    /// How many times a process died while this record was running.
    pub crashes: u32,
    /// The record was running alone (it held every permit), so a crash is its own.
    pub exclusive: bool,
}

/// What to do with a record, given what the previous owner's commit said about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrashVerdict {
    /// Not running at the last crash (or no crash): handle normally.
    Run,
    /// Running at a crash, alongside others: run it alone, so the next crash is its own.
    RunAlone { crashes: u32 },
    /// Crashed while running alone, or crashed `attempt_budget` times: answer with an
    /// error and commit. Running it again would kill the worker again.
    GiveUp { crashes: u32 },
}

impl CrashVerdict {
    pub fn decide(prior: Option<CrashMark>, attempt_budget: u32) -> Self {
        let Some(prior) = prior else {
            return CrashVerdict::Run;
        };
        let crashes = prior.crashes + 1;
        if prior.exclusive || crashes >= attempt_budget {
            CrashVerdict::GiveUp { crashes }
        } else {
            CrashVerdict::RunAlone { crashes }
        }
    }

    /// The mark to record when the handler starts.
    pub fn mark(self) -> CrashMark {
        match self {
            CrashVerdict::Run => CrashMark::default(),
            CrashVerdict::RunAlone { crashes } | CrashVerdict::GiveUp { crashes } => CrashMark {
                crashes,
                exclusive: true,
            },
        }
    }
}

/// Version tag of the commit-metadata format. Unknown formats decode as "nothing in flight".
const INFLIGHT_METADATA_PREFIX: &str = "jagua-inflight-v1;";

/// Encode the in-flight set as commit metadata: `jagua-inflight-v1;187:1:x,190:0:-`.
/// At most `MAX_CONCURRENT_TASKS` entries of ~20 bytes, far below the broker's 4 KiB limit.
pub fn encode_inflight(started: &BTreeMap<i64, CrashMark>) -> String {
    let entries: Vec<String> = started
        .iter()
        .map(|(offset, m)| {
            format!(
                "{offset}:{}:{}",
                m.crashes,
                if m.exclusive { "x" } else { "-" }
            )
        })
        .collect();
    format!("{INFLIGHT_METADATA_PREFIX}{}", entries.join(","))
}

/// Decode [`encode_inflight`]. Malformed entries are skipped: losing a crash count costs at
/// most one more crash, while refusing to start would stop the partition.
pub fn decode_inflight(metadata: &str) -> BTreeMap<i64, CrashMark> {
    let Some(body) = metadata.strip_prefix(INFLIGHT_METADATA_PREFIX) else {
        return BTreeMap::new();
    };
    body.split(',')
        .filter_map(|entry| {
            let mut fields = entry.split(':');
            let offset = fields.next()?.parse().ok()?;
            let crashes = fields.next()?.parse().ok()?;
            let exclusive = match fields.next()? {
                "x" => true,
                "-" => false,
                _ => return None,
            };
            Some((offset, CrashMark { crashes, exclusive }))
        })
        .collect()
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

        state.started.remove(&offset);

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

    /// Whether the previous owner's in-flight set is still to be loaded for a partition.
    pub fn needs_prior(&self, topic: &str, partition: i32) -> bool {
        self.partitions
            .get(&(topic.to_string(), partition))
            .is_none_or(|s| s.prior.is_none())
    }

    /// Install what the last commit's metadata said was running.
    pub fn set_prior(&mut self, topic: &str, partition: i32, prior: BTreeMap<i64, CrashMark>) {
        self.partitions
            .entry((topic.to_string(), partition))
            .or_default()
            .prior = Some(prior);
    }

    /// The crash mark the previous owner left for this offset, if it was running then.
    /// Taken, so the same offset delivered twice is not counted twice.
    pub fn take_prior(&mut self, topic: &str, partition: i32, offset: i64) -> Option<CrashMark> {
        self.partitions
            .get_mut(&(topic.to_string(), partition))?
            .prior
            .as_mut()?
            .remove(&offset)
    }

    /// Record that a record's handler started. Commit [`Self::commit_point`] right after.
    pub fn start(&mut self, topic: &str, partition: i32, offset: i64, mark: CrashMark) {
        self.partitions
            .entry((topic.to_string(), partition))
            .or_default()
            .started
            .insert(offset, mark);
    }

    /// The offset and metadata to commit for a partition: the watermark and the in-flight set.
    /// `None` before the partition's first record.
    ///
    /// **Every commit for a partition must carry this metadata.** A commit without it clears
    /// the in-flight set, and the next crash would be invisible again.
    pub fn commit_point(&self, topic: &str, partition: i32) -> Option<(i64, String)> {
        let state = self.partitions.get(&(topic.to_string(), partition))?;
        Some((state.next_expected?, encode_inflight(&state.started)))
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

    const RAN_WITH_OTHERS: CrashMark = CrashMark {
        crashes: 0,
        exclusive: false,
    };

    #[test]
    fn inflight_metadata_round_trips() {
        let started = BTreeMap::from([
            (
                187,
                CrashMark {
                    crashes: 1,
                    exclusive: true,
                },
            ),
            (190, RAN_WITH_OTHERS),
        ]);
        let encoded = encode_inflight(&started);
        assert_eq!(encoded, "jagua-inflight-v1;187:1:x,190:0:-");
        assert_eq!(decode_inflight(&encoded), started);
        assert_eq!(
            decode_inflight(&encode_inflight(&BTreeMap::new())),
            BTreeMap::new()
        );
    }

    /// Metadata written by something else (or nothing) must not stop the partition.
    #[test]
    fn foreign_or_damaged_metadata_decodes_as_nothing_running() {
        assert!(decode_inflight("").is_empty());
        assert!(decode_inflight("some other client's note").is_empty());
        assert_eq!(
            decode_inflight("jagua-inflight-v1;bad,187:2:x,190:1:?"),
            BTreeMap::from([(
                187,
                CrashMark {
                    crashes: 2,
                    exclusive: true
                }
            )]),
            "damaged entries are skipped, intact ones kept"
        );
    }

    /// The decision table behind cutl-tests#392, with the default budget of 3.
    #[test]
    fn crash_verdicts() {
        assert_eq!(CrashVerdict::decide(None, 3), CrashVerdict::Run);
        assert_eq!(
            CrashVerdict::decide(Some(RAN_WITH_OTHERS), 3),
            CrashVerdict::RunAlone { crashes: 1 },
            "first crash among other jobs: run it alone"
        );
        assert_eq!(
            CrashVerdict::decide(
                Some(CrashMark {
                    crashes: 1,
                    exclusive: true
                }),
                3
            ),
            CrashVerdict::GiveUp { crashes: 2 },
            "crashed while alone: it is the cause"
        );
        assert_eq!(
            CrashVerdict::decide(
                Some(CrashMark {
                    crashes: 2,
                    exclusive: false
                }),
                3
            ),
            CrashVerdict::GiveUp { crashes: 3 },
            "the budget caps a record that can never run alone (retry tiers)"
        );
        assert_eq!(CrashVerdict::Run.mark(), RAN_WITH_OTHERS);
        assert_eq!(
            CrashVerdict::RunAlone { crashes: 1 }.mark(),
            CrashMark {
                crashes: 1,
                exclusive: true
            }
        );
    }

    /// Run 4a930f88 on staging: offset 187 started, the pod died, the next owner reads it back.
    #[test]
    fn commit_point_carries_running_jobs_until_they_finish() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 1, 186);
        w.observe(T, 1, 187);
        w.start(T, 1, 186, RAN_WITH_OTHERS);
        w.start(T, 1, 187, RAN_WITH_OTHERS);
        assert_eq!(
            w.commit_point(T, 1),
            Some((186, "jagua-inflight-v1;186:0:-,187:0:-".into()))
        );

        w.complete(T, 1, 186);
        let (offset, metadata) = w.commit_point(T, 1).unwrap();
        assert_eq!(offset, 187);

        // The pod is OOM-killed here. The next owner of partition 1:
        let mut next = OffsetWatermark::new();
        assert!(next.needs_prior(T, 1));
        next.set_prior(T, 1, decode_inflight(&metadata));
        assert!(!next.needs_prior(T, 1));
        assert_eq!(next.take_prior(T, 1, 187), Some(RAN_WITH_OTHERS));
        assert_eq!(
            next.take_prior(T, 1, 187),
            None,
            "a second delivery is not a second crash"
        );
        assert_eq!(next.take_prior(T, 1, 188), None);
    }

    #[test]
    fn a_finished_job_leaves_the_inflight_set_even_behind_a_gap() {
        let mut w = OffsetWatermark::new();
        w.observe(T, 0, 10);
        w.observe(T, 0, 11);
        w.start(T, 0, 10, RAN_WITH_OTHERS);
        w.start(T, 0, 11, RAN_WITH_OTHERS);
        assert_eq!(w.complete(T, 0, 11), None, "10 still holds the watermark");
        assert_eq!(
            w.commit_point(T, 0),
            Some((10, "jagua-inflight-v1;10:0:-".into()))
        );
    }

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
