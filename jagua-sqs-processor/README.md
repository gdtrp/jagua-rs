# jagua-sqs-processor

Nesting worker for the jagua-rs SVG packing engine. Consumes nesting requests from
Kafka, packs the parts, uploads the resulting page SVGs to S3, and publishes
intermediate "improvement" responses plus a final layout.

> **The crate is still named `jagua-sqs-processor`.** It consumes Kafka, not SQS —
> the name is kept because it is the ECR/VK image name, the Kubernetes Deployment
> name and the CI artifact name. cutl-backend kept `SqsService` through its own
> port for the same reason. The wire DTOs keep their `Sqs` prefix on the same
> grounds; only the transport itself was renamed.

## Transport

| | |
|---|---|
| Consumer group | `jagua-nesting` |
| Request topic | `nesting-request` |
| Response topic | `nesting-response` |
| Retry topics | `jagua-nesting-retry-{1,2,3}` |
| Partitions | 3 |
| Auth | SASL_PLAINTEXT + SCRAM-SHA-512 (no TLS) |

**Topics carry no environment suffix.** Each environment has its own cluster, so
the suffix lives in the cluster boundary rather than the topic name.

**Both topics are keyed by `correlationId`.** This is mandatory, not stylistic: the
cancellation registry is in-process, so a cancel must reach the instance running
the job it cancels. Keying puts it on the same partition.

The binding semantics are specified in `cutl-infra/docs/kafka-contract.md`. Read it
before changing anything in `src/kafka.rs` or the receive loop — several of its
rules exist because the obvious implementation is wrong, and each fails silently.

## Environment variables

Supplied by the `kafka-jagua-nesting` Secret via `envFrom` (cutl-infra
`modules-vk/kafka-credentials`); the key names are already env-var shaped:

- `KAFKA_BOOTSTRAP_SERVERS` — e.g. `cutl-kafka-bootstrap.kafka.svc:9092`
- `KAFKA_USERNAME`, `KAFKA_PASSWORD` — the per-group SCRAM user
- `KAFKA_SASL_MECHANISM` — `SCRAM-SHA-512`

Required, set on the Deployment:

- `S3_BUCKET` — where result SVGs are written. No default: guessing would write
  results somewhere nobody reads.

Optional:

| Variable | Default | Notes |
|---|---|---|
| `KAFKA_CONSUMER_GROUP` | `jagua-nesting` | Also names the retry topics |
| `KAFKA_REQUEST_TOPIC` | `nesting-request` | |
| `KAFKA_RESPONSE_TOPIC` | `nesting-response` | |
| `KAFKA_ATTEMPT_BUDGET` | `3` | Handler invocations before a message is dropped |
| `KAFKA_RETRY_DELAYS_MS` | `5000,60000,600000` | Tier delays; lowered in tests |
| `AWS_REGION` | `eu-north-1` | Region of the S3 bucket |
| `AWS_ENDPOINT_URL` | unset | S3 endpoint override — MinIO locally, and the hook for the eu-west-1 relay |
| `MAX_CONCURRENT_TASKS` | `20` | Concurrent nesting jobs |
| `EXECUTION_TIMEOUT_SECS` | `600` | Per-job cap; `maxSeconds` may lower it |
| `HEALTH_PORT` | `8080` | Serves `/health`, `/ready`, `/metrics` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | unset | Unset means stdout logging only, not an error |
| `RUST_LOG` | `info` | |

**Missing configuration does not exit the process.** It logs the problem and keeps
`/ready` at 503 so Kubernetes withholds traffic. Exiting produced a crash-loop that
held a CPU reservation on a saturated cluster, which is why this workload was
scaled to 0 during the migration.

## Retry and message loss

**There is no dead-letter queue, by design.** A failed message is republished to
the next retry tier — never retried in place, never blocking its partition. After
the attempt budget is spent the message is **dropped**, and the only record of it
is:

- an `ERROR` log carrying the origin topic, key and `topic-partition-offset`, and
- the Prometheus counter `cutl_retries_exhausted_total{group, origin_topic}`.

cutl-infra alerts on any increment of that counter. Treat it as a first-class
requirement rather than instrumentation: if it stops being emitted, message loss
becomes silent.

## Tracing

Trace context propagates through **Kafka headers**, as the standard W3C pair:

- `traceparent` — read off every consumed record and set as the handler span's
  parent; written onto every produced record.
- `tracestate` — passed through untouched.

So a job continues the trace of whatever published the request, and the backend's
consumer of `nesting-response` continues the job's. The retry ladder carries it
too, which means all three tier hops and the eventual drop sit in **one** trace
rather than three orphans — the difference between reading "why was this message
lost" as a story and grepping three topics by `correlationId`.

Headers rather than a body field: the payload stays governed purely by the
AsyncAPI schema, and a republish forwards a message it is not otherwise
inspecting. Spring's Kafka instrumentation reads and writes these same headers,
so traces will join across services once cutl-backend moves to Kafka.

The global propagator is registered in `init_tracing`. Without that registration
inject/extract are silent no-ops — spans still appear, they just never link up.

## Endpoints

`/health` answers 200 while the process lives and is deliberately **not** gated on
readiness — a failing dependency should withhold traffic, not trigger a restart
loop that cannot fix it. `/ready` answers 200 only once the broker has answered a
metadata request, which is what actually exercises the SCRAM handshake.
`/metrics` serves the Prometheus text format and is scraped by the ServiceMonitor
in `deploy/k8s/service-staging.yaml`.

## Building and testing

There may be no local Rust toolchain; `scripts/cargo-docker.sh` runs cargo in the
pinned builder image with the librdkafka system dependencies already installed.

```bash
make build                 # docker build — the authoritative check
make test                  # broker-free: unit, wire-contract, in-process nesting e2e
make test-integration      # brings up Kafka + MinIO, runs the #[ignore]d tests
make compose-down          # tear the harness down
make check                 # fmt + clippy

scripts/cargo-docker.sh test -p jagua-sqs-processor
```

`rdkafka` compiles librdkafka from C source, so the builder needs `cmake`, `g++`,
`libsasl2-dev`, `zlib1g-dev` and `libcurl4-openssl-dev`. None of these are in
`rust:slim-bookworm`, and every one of them fails at `cargo build` rather than at
`cargo check` — verify with `make build`, not with a type check.

## Request format

Governed by the AsyncAPI spec in `cutl-schemas` (`asyncapi/jagua-rs.yaml`), pulled
in by `scripts/sync-schema.sh` and code-generated by `build.rs`. The wire is
**camelCase**. Do not hand-edit the generated types.

```json
{
  "correlationId": "unique-request-id",
  "binWidth": 350.0,
  "binHeight": 350.0,
  "spacing": 5.0,
  "amountOfRotations": 8,
  "parts": [
    { "itemId": "part-A", "svgUrl": "s3://bucket/part-a.svg", "amountOfParts": 4,
      "allowedRotations": [0, 180] }
  ]
}
```

A message with `"cancelled": true` and a matching `correlationId` aborts a running
job. Cancellations arrive with every other field explicitly `null`, and are
handled on the poll thread rather than behind the concurrency semaphore so they
are never queued behind the jobs they are meant to stop.

## Response format

```json
{
  "correlationId": "unique-request-id",
  "firstPageSvgUrl": "s3://bucket/nesting/<id>/page-0.svg",
  "lastPageSvgUrl": "s3://bucket/nesting/<id>/last-page.svg",
  "pageSvgUrls": ["..."],
  "sheets": 2,
  "sheetsTotal": 2,
  "partsPlaced": 7,
  "utilisation": 0.62,
  "improvement": false,
  "final": true,
  "timestamp": 1234567890
}
```

- `improvement: true` marks an intermediate layout that beat the previous best for
  that `correlationId`; several may be emitted per request.
- `final: true` marks the last message for a request. Exactly one is emitted,
  including for failures — where `errorMessage` is populated and `partsPlaced` is 0.
