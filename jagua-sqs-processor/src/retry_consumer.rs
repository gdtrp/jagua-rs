//! Retry-tier consumers (VK migration T9).
//!
//! Kafka has **no delayed delivery**. The tiered-topic ladder in
//! `cutl-infra/docs/kafka-contract.md` §4 works around that: a failed message is
//! republished to `<group>-retry-N`, and a dedicated consumer group per tier
//! applies the delay before re-invoking the same handler.
//!
//! | Tier | Topic | Delay |
//! |---|---|---|
//! | 1 | `jagua-nesting-retry-1` | 5 s |
//! | 2 | `jagua-nesting-retry-2` | 1 min |
//! | 3 | `jagua-nesting-retry-3` | 10 min |
//!
//! **The delay is applied by pausing the partition, not by sleeping.** Two rules
//! from the contract force this, and both fail only at runtime:
//!
//! - `sleep()` inside a handler stops the consumer polling. Exceed
//!   `max.poll.interval.ms` and Kafka evicts the consumer, rebalances, and hands
//!   the same message to another instance while the first is still working — a
//!   reproduction of incident MAJ-29. A 10-minute tier-3 delay against a
//!   15-minute poll interval leaves almost no margin for the handler itself.
//! - `seek()`-back as a retry mechanism head-of-line-blocks the partition on a
//!   poison message for the whole attempt budget. Here a seek is used only to
//!   re-read a record whose delay has not yet elapsed, which is the *intended*
//!   blocking: a retry tier is supposed to hold its records back.
//!
//! While a partition is paused the loop keeps calling `recv()`, which is what
//! keeps the client heartbeating and the group membership alive.

use std::collections::HashMap;
use std::time::Duration;

use anyhow::{Context, Result};
use log::{debug, error, info, warn};
use rdkafka::consumer::{CommitMode, Consumer};
use rdkafka::message::Message;
use rdkafka::{Offset, TopicPartitionList};
use tokio::sync::broadcast;
use tracing::Instrument;

use crate::kafka::{KafkaSettings, RecordHeaders};
use crate::metrics::metrics;
use crate::processor::NestingProcessor;

/// How long to block in `recv()` before looping to re-check paused partitions.
///
/// Short enough that a resume is not noticeably late, long enough that an idle
/// tier is not spinning.
const POLL_TICK: Duration = Duration::from_millis(250);

/// Run one tier's consumer until `shutdown` fires.
///
/// `tier` is 1-based and selects both the topic and the delay.
pub async fn run_tier(
    processor: NestingProcessor,
    settings: KafkaSettings,
    tier: u32,
    mut shutdown: broadcast::Receiver<()>,
) -> Result<()> {
    let topic = settings.retry_topic(tier);
    let group = settings.retry_group(tier);
    let delay = settings.delay_for_tier(tier);

    let consumer = settings
        .consumer(&group)
        .with_context(|| format!("failed to build the tier-{tier} consumer"))?;
    consumer
        .subscribe(&[topic.as_str()])
        .with_context(|| format!("failed to subscribe to {topic}"))?;

    info!("Tier {tier} consumer started on {topic} as group {group}, delay {delay:?}");

    // Partitions currently held back, and when each may resume.
    let mut paused: HashMap<i32, tokio::time::Instant> = HashMap::new();

    loop {
        // Resume anything whose delay has elapsed, before polling again.
        let now = tokio::time::Instant::now();
        let ready: Vec<i32> = paused
            .iter()
            .filter(|(_, until)| **until <= now)
            .map(|(p, _)| *p)
            .collect();
        for partition in ready {
            let mut tpl = TopicPartitionList::new();
            if tpl
                .add_partition_offset(&topic, partition, Offset::Invalid)
                .is_ok()
            {
                if let Err(e) = consumer.resume(&tpl) {
                    warn!("Tier {tier}: failed to resume {topic}-{partition}: {e}");
                    continue;
                }
            }
            paused.remove(&partition);
            debug!("Tier {tier}: resumed {topic}-{partition}, delay elapsed");
        }

        tokio::select! {
            _ = shutdown.recv() => {
                info!("Tier {tier} consumer shutting down");
                break;
            }

            // A paused partition yields nothing, so this times out on every tick
            // while a delay is in force. That is the point: polling continues, so
            // the client keeps its group membership.
            received = tokio::time::timeout(POLL_TICK, consumer.recv()) => {
                let Ok(received) = received else { continue };

                let borrowed = match received {
                    Ok(m) => m,
                    Err(e) => {
                        error!("Tier {tier} receive error: {e}");
                        continue;
                    }
                };

                let partition = borrowed.partition();
                let offset = borrowed.offset();
                let Some(payload) = borrowed
                    .payload()
                    .map(|b| String::from_utf8_lossy(b).into_owned())
                else {
                    error!("Tier {tier}: {topic}-{partition}-{offset} has no payload, skipping");
                    continue;
                };
                let key = borrowed.key().map(|k| String::from_utf8_lossy(k).into_owned());
                let headers = RecordHeaders::from_message(&borrowed);
                // Continue the trace the original failure started, so all three
                // ladder hops appear as one story rather than three orphans.
                let parent = crate::trace_context::extract_context(
                    &crate::trace_context::HeaderExtractor::from_message(&borrowed),
                );

                // Kafka's record timestamp is when the tier producer wrote it, so
                // `timestamp + delay` is when this message becomes eligible.
                let produced_at_ms = borrowed.timestamp().to_millis().unwrap_or(0);
                let elapsed = elapsed_since_ms(produced_at_ms);

                if elapsed < delay {
                    let remaining = delay - elapsed;

                    // Hold the partition back and rewind so this record is
                    // re-delivered once the delay is up. Records behind it are
                    // held too, which is correct: they are newer, so their own
                    // delays have not elapsed either.
                    let mut tpl = TopicPartitionList::new();
                    tpl.add_partition_offset(&topic, partition, Offset::Offset(offset))?;

                    if let Err(e) = consumer.pause(&tpl) {
                        warn!("Tier {tier}: failed to pause {topic}-{partition}: {e}");
                    }
                    if let Err(e) = consumer.seek(&topic, partition, Offset::Offset(offset), Duration::from_secs(5)) {
                        warn!("Tier {tier}: failed to seek {topic}-{partition} to {offset}: {e}");
                    }

                    paused.insert(partition, tokio::time::Instant::now() + remaining);
                    debug!(
                        "Tier {tier}: holding {topic}-{partition}-{offset} for a further {remaining:?}"
                    );
                    continue;
                }

                // Delay satisfied — run the same handler the main loop uses.
                metrics().messages_consumed.with_label_values(&[&topic]).inc();

                let span = tracing::info_span!(
                    "nesting.retry",
                    messaging.system = "kafka",
                    messaging.source.name = %topic,
                    messaging.kafka.partition = partition,
                    messaging.kafka.offset = offset,
                    retry.tier = tier,
                    retry.attempt = headers.attempt,
                );
                {
                    use tracing_opentelemetry::OpenTelemetrySpanExt;
                    span.set_parent(parent);
                }

                let result = async {
                    let r = processor
                        .process_message(key.as_deref().unwrap_or_default(), &payload)
                        .await;

                    if let Err(e) = r {
                        error!("Tier {tier} handler failed for {topic}-{partition}-{offset}: {e}");
                        metrics().messages_failed.with_label_values(&[&topic]).inc();
                        // Inside the span, so the escalation's injected trace
                        // context continues this one.
                        processor
                            .escalate(&payload, key.as_deref(), &headers, &topic, partition, offset)
                            .await;
                    }
                }
                .instrument(span);
                result.await;

                // Commit regardless of outcome. An uncommitted failure would make
                // this tier redeliver forever, which is the head-of-line blocking
                // the ladder exists to avoid — escalation already moved it on.
                let mut tpl = TopicPartitionList::new();
                tpl.add_partition_offset(&topic, partition, Offset::Offset(offset + 1))?;
                if let Err(e) = consumer.commit(&tpl, CommitMode::Async) {
                    warn!("Tier {tier}: commit of {topic}@{} failed: {e}", offset + 1);
                } else {
                    metrics()
                        .committed_offset
                        .with_label_values(&[&topic, &partition.to_string()])
                        .set(offset + 1);
                }
            }
        }
    }

    if let Err(e) = consumer.commit_consumer_state(CommitMode::Sync) {
        warn!("Tier {tier}: final commit failed: {e}");
    }
    Ok(())
}

/// Wall-clock time since a Kafka record timestamp, saturating at zero.
///
/// Saturates because a record produced by a host with a slightly fast clock would
/// otherwise yield a negative age and be held for longer than its tier delay.
fn elapsed_since_ms(produced_at_ms: i64) -> Duration {
    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    Duration::from_millis((now_ms - produced_at_ms).max(0) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elapsed_is_zero_for_a_future_timestamp() {
        let future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            + 60_000;
        assert_eq!(
            elapsed_since_ms(future),
            Duration::ZERO,
            "clock skew must not extend a tier delay"
        );
    }

    #[test]
    fn elapsed_grows_with_age() {
        let past = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
            - 5_000;
        let elapsed = elapsed_since_ms(past);
        assert!(
            elapsed >= Duration::from_secs(4) && elapsed <= Duration::from_secs(7),
            "expected ~5s, got {elapsed:?}"
        );
    }

    /// A tier-1 record older than its 5s delay is eligible; a fresh one is not.
    /// This is the comparison that decides pause-vs-handle.
    #[test]
    fn eligibility_is_decided_by_age_against_the_tier_delay() {
        let tier_1_delay = Duration::from_secs(5);

        let fresh = elapsed_since_ms(now_ms());
        assert!(fresh < tier_1_delay, "a just-produced record must be held");

        let old = elapsed_since_ms(now_ms() - 10_000);
        assert!(old >= tier_1_delay, "a 10s-old record must be eligible");
    }

    fn now_ms() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as i64
    }
}
