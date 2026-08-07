# jagua-rs: VK migration handoff

Branch `vk-cloud`. Written 2026-08-07, updated after T9.

## State

**T9 is implemented: the worker consumes Kafka, not SQS.** `aws-sdk-sqs` and both
`*_QUEUE_URL` variables are gone. See "Done in T9" below for what changed and
"Left to do" for what is still open.

The workload is still **scaled to 0** on the VK staging cluster and needs scaling
back up to be verified — nothing here has run against the real cluster yet.

Missing configuration no longer kills the process. It logs the problem and holds
`/ready` at 503, so a misconfigured pod withholds traffic instead of crash-looping
against a CPU reservation on a saturated cluster — which is what forced the scale
to 0 in the first place.

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

## Done in T9

Read **`cutl-infra/docs/kafka-contract.md`** before touching any of this. It is
the binding contract, and three of its rules exist because the obvious
implementation is wrong in a way that fails silently:

- **The receive loop never pauses a partition while a job runs.** Both topics are
  keyed by `correlationId`, so a cancel lands on the same partition as the job it
  cancels, at a later offset. Pausing to throttle would mean the cancel is only
  read after the job finishes — cancellation becomes a no-op that nothing reports.
- **Offsets advance through a contiguous watermark** (`OffsetWatermark` in
  `src/kafka.rs`), because a commit of 105 implicitly acknowledges 100..=104 and
  jobs finish out of order behind the semaphore.
- **Cancellations are handled inline on the poll thread**, before the semaphore,
  so they are never queued behind the twenty jobs they are meant to stop.

Also:

- **Retry ladder** to `jagua-nesting-retry-{1,2,3}` with the `x-cutl-attempt` /
  `x-cutl-origin-topic` headers, plus a consumer per tier
  (`src/retry_consumer.rs`) that applies its delay by **pausing the partition and
  rewinding**, never by sleeping. Sleeping in a handler stops the consumer polling
  and gets it evicted mid-job; a 10-minute tier-3 delay against a 15-minute poll
  interval leaves no margin.
- **`cutl_retries_exhausted_total{group, origin_topic}`** and a `/metrics`
  endpoint on the existing axum router — the ServiceMonitor has pointed at it
  since T7 with nothing serving it. The counter is **pre-registered at zero** at
  startup: a labelled counter has no series until its child exists, and
  `increase(...[5m]) > 0` cannot fire on a series that was never reported. That is
  how `cutl_log_errors_total` came to report healthy forever.
- **W3C trace context over Kafka headers** (`src/trace_context.rs`):
  `traceparent`/`tracestate` are extracted from every consumed record and set as
  the handler span's parent, and injected into every produced record — responses
  and retry republishes alike. A job therefore continues the trace of whatever
  published the request, and all three ladder hops stay in one trace.
  The global propagator is registered in `init_tracing`; **without that
  registration inject/extract are silent no-ops** and traces simply never join up.
  Note the RFC describes the existing envelope carrying a hand-rolled
  `traceparent` *inside the JSON body* — this uses headers instead, so the payload
  stays governed by the AsyncAPI schema alone.
- Readiness gated on a real broker metadata fetch, so a pod with bad credentials
  no longer reports Ready and then silently consumes nothing.
- `S3_BUCKET` set on the Deployment.
- A **docker-compose harness** running Kafka with SASL_PLAINTEXT + SCRAM-SHA-512,
  matching the VK listener, plus MinIO for S3.
- **The first CI workflow that runs tests at all** (`.github/workflows/ci.yml`) —
  previously nothing gated a merge on any branch.
- `scripts/cargo-docker.sh`, because there may be no local Rust toolchain and
  rdkafka needs `cmake`, `g++`, `libsasl2-dev`, `zlib1g-dev` and
  `libcurl4-openssl-dev` that `rust:slim-bookworm` does not ship.

## Left to do

### 1. Verify on the cluster
Nothing below has run against real VK infrastructure. Scale the Deployment up
from 0 and work through the Verification list at the bottom.

### 2. cutl-backend still publishes to SQS (T8)
It is still `@SqsListener` with `SqsTemplate.send(queue, payload)` and **no
message key**. An end-to-end round trip in staging needs T8 first; jagua's own
side can be proven by hand-producing to `nesting-request` with the key set to the
`correlationId`. Producing without a key breaks cancellation.

### 3. The eu-west-1 S3 relay
`S3_BUCKET` is set to `cutl-staging-uploads`, but `s3.eu-north-1` is still 0/6
from a pod. The relay is set via `AWS_ENDPOINT_URL` — the code path already
exists and is exercised by MinIO in the test harness. Until then S3 calls fail
into the retry ladder and raise `CutlRetriesExhausted`, which is visible rather
than silent, but it is not working.

### 4. Which failures should climb the ladder
Right now **any** `Err` from the handler escalates. In practice `process_message`
returns `Ok(())` for almost everything — validation failures become an error
*response* on `nesting-response` rather than an error — so the ladder mostly sees
genuine infrastructure faults, which is the intent.

Two things are worth tightening once the relay exists and real failures are
observable:

- `download_svg_from_s3` (`processor.rs`) is **not** wrapped in
  `retry_with_backoff`, unlike all four upload sites. With the relay adding a
  network hop it should be.
- A malformed message that can never parse still consumes all three attempts
  before being dropped. Harmless, but it spends ~11 minutes of ladder to reach a
  conclusion available immediately.

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
- **T9 added four more of the same kind.** rdkafka compiles librdkafka from C
  source, and each missing piece failed at `cargo build`, never at `cargo check`:
  - `rust:slim-bookworm` has `cc` but **no `c++`**. librdkafka's top-level
    CMakeLists declares a CXX project, so cmake aborts before compiling anything.
    Needs `g++`.
  - It also needs `libcurl4-openssl-dev` **despite** rdkafka-sys passing
    `-DWITH_CURL=0`: the `ssl` feature enables SASL OAUTHBEARER (SCRAM and
    OAUTHBEARER share the OpenSSL dependency) and `rdkafka_conf.c` includes
    `<curl/curl.h>` on that path. We never use OAUTHBEARER.
  - **`max.poll.records` is a Java-consumer property.** librdkafka rejects it at
    client creation with "No such configuration property", so the contract's
    `max.poll.records=1` is satisfied structurally instead — `recv()` yields one
    message per call. `max.poll.interval.ms` *is* valid; only the records one is not.
  - `fetch_metadata` lives on the `Consumer` trait, not on `StreamConsumer`, and
    is blocking — it needs `block_in_place`, which **panics on a single-threaded
    runtime**. Tests calling it need `#[tokio::test(flavor = "multi_thread")]`.
- **`test_e2e_processing_dr_svg` fails under Docker-on-macOS** with "Test timed
  out after 1 minute". Confirmed pre-existing: it fails identically on the
  unmodified commit before T9. It is CPU-bound nesting against a hard 60s budget,
  so it is an environment artifact, not a regression — but do not read it as a
  green baseline.
- **The compose harness advertises two SASL listeners** (`localhost:9092` for the
  host, `kafka:29092` in-network) because an advertised address can only be
  correct from one vantage point. Advertising only localhost makes every
  in-network produce time out, which looks like a broker fault rather than a
  routing mistake.
- **`scripts/sync-schema.sh` pulls cutl-schemas' default branch.** The Kafka
  `servers:` declaration is on that repo's `vk-cloud` branch, so CI currently
  vendors a spec that still says `aws-sqs`. Harmless — typify only reads
  `components.schemas`, so the generated types are byte-identical — but set
  `CUTL_SCHEMAS_REF=vk-cloud` if you want them to match.
- **`terminationGracePeriodSeconds: 600` and `Recreate` are load-bearing.** A
  nesting run is minutes long and single-owner. The default 30s would SIGKILL a
  job mid-flight on every rollout, and with no DLQ that is a silent loss.
- **CI stays `linux/amd64` only**, unlike backend and frontend.
  `CutlEnvironmentStack.java:1028` records that jagua stays on x86_64 due to ARM
  build issues, and VK nodes are amd64 too.
- **Nodes are small**: 5.4Gi / 1930m allocatable each. This worker requests
  500m/256Mi and bursts to 1500m/1Gi.

## Verification

### Repo-local (done)

```bash
make test              # 39 unit + 25 e2e pass; see the dr_svg trap above
make test-integration  # 9 broker-backed tests against real SCRAM auth
make check             # fmt + clippy -D warnings
make build             # docker image builds with librdkafka
```

The integration suite covers SCRAM auth (positive and negative), partition
affinity under keying, the 3-partition topic layout, the implicit-acknowledgement
property that forces the watermark, retry-header propagation, tier-topic
existence, the tier delay's pause-and-rewind, and the response wire round trip.

### On the cluster — done 2026-08-07

Driven by `.github/workflows/validate-vk.yml`, which authenticates the same way
the deploy job does. Use it rather than a local kubeconfig: the VK bearer token
is a project-scoped Keystone token that expires within hours.

Confirmed on VK staging:

- **Pod 1/1 Running, 0 restarts**, up from the scaled-to-0 state. The two prior
  deploys had failed with `BackOff restarting failed container jagua`.
- `/health` **200**, `/ready` **200**, `/metrics` **200** — the ServiceMonitor
  target is no longer dead, and `cutl_retries_exhausted_total` is present at 0
  with both `group` and `origin_topic` labels.
- **All four consumer groups joined** (`jagua-nesting` plus the three tiers),
  3 partitions each, over SASL_PLAINTEXT + SCRAM-SHA-512.
- A keyed request carrying a `traceparent` was consumed on `nesting-request`
  partition 2, handled inside a `nesting.handle` span parented to it.
- **The retry ladder climbs**: S3 upload failed (expected — no relay), the
  handler escalated, `Republished to jagua-nesting-retry-1 (attempt 2 of 3)`,
  the tier-1 consumer applied its delay and re-invoked, then
  `Republished to jagua-nesting-retry-2 (attempt 3 of 3)`. Tier spans carry
  `retry.tier` and `retry.attempt`.

Exhaustion itself landed ~4s after the validation snapshot (tier 2's delay is
60s), so the counter still read 0 in that run. **Check Prometheus for
`cutl_retries_exhausted_total` and whether `CutlRetriesExhausted` fired** — the
counter is in-process and resets on every redeploy, so Prometheus is the only
place the increment survives. End-to-end exhaustion was proven locally against
the harness with S3 pointed at a dead host and the tier delays shortened.

### Still to verify on the cluster

1. A **cancellation** honoured mid-job — the property the whole keying and
   no-pause design exists for. Needs a job long enough to cancel, which needs S3
   working, so it is blocked on the relay.
2. A **successful** round trip producing a real result on `nesting-response`.
   Every cluster run so far dies at the S3 upload, so the success path is proven
   only locally. Blocked on the relay.
3. A trace actually arriving in **Tempo** with `service.name=jagua-nesting`, and
   the job's span appearing as a *child* of the producer's `traceparent`.
   Propagation is verified on the wire; what has not been checked is that Tempo
   received and joined it.
4. `CutlRetriesExhausted` firing in Alertmanager. The counter increments (proven
   locally), but nobody has watched the alert route.
5. A rollout during an in-flight nesting job draining rather than losing it —
   the `terminationGracePeriodSeconds: 600` behaviour. Also needs a job long
   enough to interrupt, so also blocked on the relay.

Note the existing envelope reportedly carries a hand-rolled `traceparent` as a
JSON body field. This implementation ignores that and uses headers. If a producer
is only setting the body field, traces will not join — worth checking against
cutl-backend during T8 rather than after.
