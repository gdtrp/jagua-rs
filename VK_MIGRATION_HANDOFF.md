# jagua-rs: VK migration handoff

Branch `vk-cloud`. Written 2026-08-07.

## State

**Scaled to 0 on the VK staging cluster.** It cannot start: `main.rs` requires
`S3_BUCKET`, `INPUT_QUEUE_URL` and `OUTPUT_QUEUE_URL`, and exits when one is
missing. It was crash-looping and holding a CPU reservation on a saturated
cluster, so it was scaled down rather than left to restart forever.

Supplying those variables would not fix it. The worker is an SQS consumer
against `eu-north-1`, which this cluster cannot reach at all.

## Egress: blocked per AWS REGION, not AWS-wide

The RFC's original note ("VK cannot reach AWS") was wrong — it had only tested
`eu-north-1`. Measured from a pod, 6 attempts each:

| Destination | |
|---|---|
| `s3.eu-west-1`, `s3.ap-northeast-1`, `s3.ap-east-1` | **6/6** |
| `s3.eu-north-1` (**our buckets and queues**) | **0/6** |
| `s3.eu-central-1`, `s3.us-east-1`, `sqs.eu-north-1`, `sts` | **0/6** |
| Alibaba (`oss-cn-hangzhou`, `ecs.aliyuncs.com`) | **0/6** — not a way round |
| `cutl-kafka-bootstrap.kafka.svc:9092` | 6/6 |

One relay in **`eu-west-1`** unblocks S3 and SQS together. Deferred.

**Probe with repeats.** A single-attempt probe reported AWS "REACHABLE"; eight
attempts on the same hosts returned 0/8.

## Done in T7

- **`/health` and `/ready` on an axum server** (`src/observability.rs`) — this
  binary had no HTTP server at all. Without probes a wedged worker keeps its pod
  Ready forever and silently stops consuming.
  - `/health` answers while the process lives and is **not** gated on readiness:
    a failing dependency should withhold traffic, not trigger a restart loop
    that cannot fix it.
  - The server binds **before** the AWS clients are built, so if that startup
    work hangs the probes still report the truth.
- **Tracing from scratch**: `tracing` + `tracing-subscriber` +
  `opentelemetry-otlp`, `service.name = jagua-nesting`. Existing `log::info!`
  call sites are bridged, not rewritten. Exports when
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set, logs to stdout when it is not, and
  flushes the provider on shutdown (otherwise the spans from a crash — the ones
  worth having — are dropped).
- **Base images pinned.** Runtime was `debian:unstable-slim` (Debian sid: a
  rolling, unreleased distribution in production) → `bookworm-slim`. Builder was
  `rustlang/rust:nightly-slim`, rolling *and* nightly → pinned stable 1.97.
- CI dual-pushes to ECR and the VK registry; `build-vk` deploys from `vk-cloud`.
- `deploy/k8s/` manifests: internal worker, no Ingress, headless Service,
  `terminationGracePeriodSeconds: 600` and `strategy: Recreate`.

## Left to do

### 1. Port SQS to Kafka
Consumer group is **`jagua-nesting`**; its SCRAM credentials are already mounted
as the `kafka-jagua-nesting` Secret by cutl-infra's `modules-vk/kafka-credentials`,
so connectivity can be proven before the code depends on it. Mounting only its
own group is deliberate — jagua cannot read another worker's topics.

**Topics carry no environment suffix.** Each environment has its own cluster, so
topics are namespaced by the cluster rather than the name: `nesting-request`,
`nesting-response`. Drop `INPUT_QUEUE_URL` / `OUTPUT_QUEUE_URL` entirely.

### 2. Emit `cutl_retries_exhausted_total`
There is **no dead-letter queue by design**, so this counter is the only signal
a message was lost. cutl-infra already ships the alert; nothing produces the
metric.

Related: `deploy/k8s/service-staging.yaml` has a ServiceMonitor scraping
`/metrics`, **which this binary does not serve yet**. The target shows as down
in Prometheus, which is honest and visible, rather than the reverse failure
where an endpoint is served and scraped by nobody. Adding a metrics endpoint
closes both at once.

### 3. Decide what `S3_BUCKET` points at
S3 stays on AWS for now, so the value is the AWS bucket
(`cutl-staging-uploads`) reached through the relay. Until the relay exists, no
value works — `eu-north-1` is 0/6.

## Traps

- **Bugs that only running the binary revealed** (it compiled cleanly through
  all three):
  - `with_endpoint` lives on the `WithExportConfig` trait, which was not in
    scope.
  - Pinning the builder to Rust 1.90 fails outright: the `aws-smithy` /
    `aws-types` crates in the lockfile declare `rust-version 1.91`.
  - Calling `LogTracer::init()` before `tracing-subscriber`'s own `init()`
    claims the global logger and **panics at startup** with `SetLoggerError`.
    `tracing-subscriber`'s default `tracing-log` feature already installs the
    bridge. Do not add a direct `tracing-log` dependency back.
  - Verify by building the image and running it, not by `cargo check`.
- **`terminationGracePeriodSeconds: 600` and `Recreate` are load-bearing.** A
  nesting run is minutes long and single-owner. The default 30s would SIGKILL a
  job mid-flight on every rollout, and with no DLQ that is a silent loss.
- **CI stays `linux/amd64` only**, unlike backend and frontend.
  `CutlEnvironmentStack.java:1028` records that jagua stays on x86_64 due to ARM
  build issues, and VK nodes are amd64 too.
- **Nodes are small**: 5.4Gi / 1930m allocatable each. This worker requests
  500m/256Mi and bursts to 1500m/1Gi.

## Verification

1. Pod reaches 1/1 with `/ready` returning 200 (503 while dependencies are down
   is correct behaviour, not a fault).
2. A message on `nesting-request` is consumed and a result published to
   `nesting-response`.
3. `cutl_retries_exhausted_total` appears in Prometheus and the ServiceMonitor
   target goes up.
4. A trace reaches Tempo with `service.name=jagua-nesting`.
5. A rollout during an in-flight nesting job drains rather than losing it.
