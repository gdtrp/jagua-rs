---
name: project_kafka_contract
description: "The binding Kafka consumer-semantics contract for the VK migration lives in cutl-infra, not jagua-rs"
metadata: 
  node_type: memory
  type: project
  originSessionId: e3839775-fdb3-44f7-9432-a11c17aadab0
  modified: 2026-08-07T09:20:21.850Z
---

The VK Cloud Kafka port (T9) is governed by two documents in the **sibling `cutl-infra` repo**,
not by anything in jagua-rs:

- `cutl-infra/docs/kafka-contract.md` — topics, keying, consumer config, retry tiers, auth.
- `cutl-infra/docs/rfc/001-vk-cloud-migration.md:1139-1160` — the T9 directive naming jagua.

Key bindings for jagua (as of 2026-08-07): consumer group `jagua-nesting`; topics
`nesting-request` / `nesting-response` + `jagua-nesting-retry-{1,2,3}`, 3 partitions, already
created by Strimzi. Auth is SASL_PLAINTEXT + SCRAM-SHA-512, **no TLS**, credentials arriving as
four env vars (`KAFKA_USERNAME`/`KAFKA_PASSWORD`/`KAFKA_BOOTSTRAP_SERVERS`/`KAFKA_SASL_MECHANISM`)
from the `kafka-jagua-nesting` Secret via `envFrom`.

Three non-obvious constraints the contract imposes, each of which fails only at runtime:

1. **Never pause a partition while a nesting job runs.** Both topics are keyed by `correlationId`,
   so a cancel lands on the same partition at a later offset; pausing makes cancellation a
   silent no-op. Keep the `tokio::select!` loop fetching while jobs run.
2. **Commits are not SQS deletes** — offset 105 acks 100-104. Concurrent out-of-order completion
   requires a contiguous-completed watermark.
3. **Never `sleep()` in a handler or `seek()` back to retry.** Retry is republish-to-next-tier
   -and-commit; tier delay is applied by pausing the *retry* topic's partition.

**Why:** none of this is derivable from the jagua-rs codebase, and the natural implementation
(serialize per partition, pause while busy) is explicitly wrong here.

**How to apply:** read `cutl-infra/docs/kafka-contract.md` before touching the transport layer.
See [[project_vk_migration_status]].
