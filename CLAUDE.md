# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Constraints (read first)

- **Only modify `jagua-utils/` and `jagua-sqs-processor/`.** `jagua-rs/` and `lbf/` are upstream library crates (synced from `JeroenGar/jagua-rs` via the `upstream` remote) — don't make standalone edits to them. To change packing quality or algorithm behavior, tune parameters, adjust strategies, or add pre/post-processing in the two wrapper crates. (A few load-bearing local patches do exist in the library — e.g. `lbf` cancellation/callback API, a disabled over-eager `debug_assert!`, a NaN-safe density assert, and **size-gated CDE consistency asserts** (search `LOCAL PATCH (jagua-utils)` in `jagua-rs/src`: `layout.rs`, `cd_engine.rs`, `qt_hazard_vec.rs`) — preserve them across upstream syncs.)
  - The size-gated asserts skip a handful of O(n) internal-consistency `debug_assert!`s (`layout_qt_matches_fresh_qt`, `qt_contains_no_dangling_hazards`, `assert_caches_correct`, duplicate-entity scan) once a layout/CDE grows past 32 placed parts. Unconditionally they are O(n²) over a layout's lifetime, so bulk **periodic rendering** (thousands of identical parts on one sheet, the deterministic fast paths) takes minutes in test/debug builds; release strips them entirely. The invariants are size-independent, so 32 inserts still exercises them for jagua-rs's own tests.
- **`jagua-sqs-processor` is edition 2021** (no let-chains — use nested `if let`). `jagua-rs`, `lbf`, and `jagua-utils` are edition 2024.
- **The AsyncAPI spec is git-ignored and must be synced before anything builds.** `scripts/sync-schema.sh` (needs `gh` auth) writes `jagua-sqs-processor/asyncapi/jagua-rs.yaml`; without it every `cargo` invocation fails inside `build.rs`, not with a compile error.
- **Branch divergence**: `main` is the AWS SQS/ECS worker; `vk-cloud` is the Kafka + Kubernetes (VK Cloud) port. The processor sections below describe `vk-cloud`. `VK_MIGRATION_HANDOFF.md` records what is done, what is still unverified on the cluster, and a list of traps worth reading before touching the transport.
- The crate is still **named** `jagua-sqs-processor` and its DTOs keep the `Sqs` prefix (`SqsNestingRequest`/`SqsNestingResponse`) on `vk-cloud` — only the transport was ported. Don't "fix" the names.

## Build & Test Commands

```bash
# Build entire workspace
cargo build
cargo build --release          # with optimizations (LTO=fat)

# Run all tests (tests run with opt-level=3 + debug-assertions)
cargo test
cargo test -p jagua-rs         # core library only
cargo test -p lbf              # LBF optimizer only
cargo test -p jagua-utils      # utils only
cargo test -p jagua-sqs-processor  # SQS processor only

# Run a single test
cargo test -p jagua-rs test_name
cargo test -p lbf test_strip_packing::albano
cargo test -p jagua-sqs-processor test_parse_s3_url_s3_scheme

# Benchmarks (criterion-based)
cargo bench --bench ci_bench -p lbf
cargo bench --bench quadtree_bench -p lbf
cargo bench --bench fast_fail_bench -p lbf

# Run LBF binary
cargo run --release -p lbf --bin lbf -- -i assets/swim.json -p spp -c assets/config_lbf.json -s solutions

# Format and lint
cargo fmt --check
cargo clippy

# Build docs locally
cargo doc --open

# Makefile targets (operate only on the crates we own: jagua-utils + jagua-sqs-processor)
make fmt          # cargo fmt the two wrapper crates
make check        # fmt-check + clippy (-D warnings, --no-deps)
make lint         # clippy only
make lint-fix     # clippy --fix
make test         # broker-free suite: unit + wire-contract goldens + in-process nesting e2e
make compose-up   # Kafka (SASL_PLAINTEXT/SCRAM-SHA-512) + MinIO harness, then provision user & topics
make test-integration  # compose-up, then the #[ignore]d broker-backed tests (--test-threads=1)
make compose-down # tear the harness down (-v)
make sync-spec    # Makefile copy of the spec from a local $CUTL_BACKEND checkout (default ../cutl-backend)
make codegen      # sync-spec + touch the spec + cargo build (forces build.rs to re-run typify)
make build        # docker build of jagua-sqs-processor (validates the container codegen path)

# Current canonical spec source (preferred over `make sync-spec`): pull from the cutl-schemas repo
scripts/sync-schema.sh   # gh api gdtrp/cutl-schemas .../jagua-rs.yaml -> jagua-sqs-processor/asyncapi/jagua-rs.yaml (needs gh auth)
CUTL_SCHEMAS_REF=vk-cloud scripts/sync-schema.sh   # the branch whose `servers:` block declares Kafka

# Build/test inside the pinned builder image — needed when there is no local Rust toolchain, and the
# only way to reproduce the librdkafka C build (cmake/g++/libsasl2/zlib/libcurl) the Dockerfile does.
scripts/cargo-docker.sh test -p jagua-sqs-processor
```

`cargo check` is **not** a sufficient signal for the processor: rdkafka compiles librdkafka from C
source, and every missing system dependency so far has failed at `cargo build`/link or at runtime,
never at `cargo check`. Verify transport changes with `make build` (docker) or `scripts/cargo-docker.sh`.

## Architecture

### Workspace Crates (dependency order)

```
jagua-sqs-processor → jagua-utils → lbf → jagua-rs
```

- **`jagua-rs`** — Core collision detection engine library (published to crates.io, edition 2024). Feature-gated problem variants: `spp` (strip packing), `bpp` (bin packing), `mspp` (multi strip packing). No default features — downstream crates enable what they need.
- **`lbf`** — Left-Bottom-Fill reference optimizer (edition 2024). Both a library (`rlib` + `cdylib` for WASM) and a CLI binary. Enables `spp` + `bpp` features on jagua-rs. Not for production use — solution quality is chaotic by nature.
- **`jagua-utils`** — SVG nesting utilities wrapping lbf (edition 2024). Provides the `NestingStrategy` trait (`SimpleNestingStrategy`, `AdaptiveNestingStrategy`) plus the classifier/fast-path router that production actually calls. Enables `bpp` feature on jagua-rs (via lbf).
- **`jagua-sqs-processor`** — the nesting worker (edition 2021, Tokio async). Consumes nesting requests off Kafka, runs the nesting router, uploads result SVGs to S3, publishes responses. Deployed as a Kubernetes Deployment on VK Cloud (`deploy/k8s/`); `.github/workflows/deploy.yml` dual-pushes to ECR and the VK registry, `validate-vk.yml` drives cluster verification (use it rather than a local kubeconfig — the VK bearer token expires within hours). **API-first**: wire types are code-generated from an AsyncAPI spec (see below), not hand-written.

### Core jagua-rs Architecture

**Problem-Instance-Solution pattern**: `Instance` (immutable problem definition) → `Problem` (mutable working state with snapshot/restore) → `Solution` (immutable snapshot). Each problem variant (SPP, BPP, MSPP) implements this independently under `jagua-rs/src/probs/`. Problem variants are gated by cargo features; each variant has its own `entities/` and `io/` submodules.

**Collision Detection Engine (CDEngine)** in `jagua-rs/src/collision_detection/`: Quadtree-based spatial indexing with configurable depth. All spatial constraints are unified as **Hazards** (placed items, container exterior, holes, quality zones). `HazardFilter` trait enables per-query selective checking.

**Fail-fast surrogates** in `jagua-rs/src/geometry/fail_fast/`: Items have simplified shape representations (`SPSurrogate` with poles/piers) for quick collision rejection before expensive polygon checks.

**Two-level shape representation**: `OriginalShape` (exact input geometry) vs `SPolygon` (simplified collision detection shape). Polygon simplification is controlled by `poly_simpl_tolerance`.

**Key geometry traits** in `jagua-rs/src/geometry/geo_traits.rs`: `CollidesWith<T>`, `AlmostCollidesWith<T>`, `Transformable`, `TransformableFrom` (in-place transform to avoid allocation).

**SlotMap arena allocation**: `PItemKey` for placed items, `HazKey` for hazards, `LayKey` for layouts. Provides O(1) insert/remove with stable keys.

**Error handling**: Uses `anyhow::Result` throughout all crates. No custom error types.

### jagua-utils: classifier and fast paths

Production does **not** call a strategy directly — `processor.rs` calls `nest_auto` / `nest_max_fit_auto`
(`svg_nesting/classify.rs`), which buckets a request by cheap per-part geometry (`PackingClass`) and routes
it to the cheapest correct packer, falling back to `AdaptiveNestingStrategy` (LBF) for anything the fast
paths don't handle:

| Class | Trigger | Packer |
|---|---|---|
| `SingleHighFill` / `SingleRectangle` | 1 part type, bbox ≈ fills sheet, or rectangularity ≥ 0.98 | `periodic.rs` (deterministic grid) |
| `SinglePairable` | 1 part type, area ≈ ½ bbox, ≤ 5 vertices, no holes | `pairing.rs` |
| `MixedFewTypes` | 2–4 rectangular part types | `mixed.rs` (per-type sheets + shelf remainder) |
| `SingleIrregular` / `General` | everything else | `AdaptiveNestingStrategy` (byte-for-byte unchanged) |

`lattice.rs` backs the max-fit variants. `PackingMode::{Grid,Periodic,General}` exists only so tests can
force a path; production is always `Auto`. Classification never fails the caller — a measurement error
falls back to `General` so the strategy surfaces the real parse error. Design rationale and phasing live in
`docs/rfcs/CUTL-160-nesting-optimization.md`.

Rotation semantics are subtle and contract-fixed (`strategy.rs`): per-part `allowed_rotations` is in
**degrees**; `None` **and** an empty list both mean unconstrained; a single `[0]` means 0°-only.

### Processor flow and Kafka transport (`vk-cloud`)

Record on `nesting-request` (keyed by `correlationId`) → SVG downloaded from S3 or decoded from base64 →
nesting runs in `spawn_blocking` → intermediate improvements published as they are found → result SVGs
uploaded to S3 → response published on `nesting-response`. Cancellation goes through a `correlation_id`
registry; a per-request `maxSeconds` (600s ceiling) overrides the default time budget.

Module map: `kafka.rs` (settings, consumer/producer, `OffsetWatermark`, retry headers) · `retry_consumer.rs`
(the three delay tiers) · `processor.rs` (the whole job pipeline) · `metrics.rs` · `observability.rs`
(axum `/health`, `/ready`, `/metrics`) · `trace_context.rs` (W3C context over Kafka headers) · `wire.rs`.

Semantics are fixed by **`cutl-infra/docs/kafka-contract.md`**; read it before touching the transport. Three
of its rules exist because the obvious port of SQS semantics is wrong in a way that only fails at runtime:

1. **Never pause a partition while a job runs.** Both topics are keyed by `correlationId`, so a cancel lands
   on the same partition as the job it cancels, at a later offset — pausing to throttle silently turns
   cancellation into a no-op. Cancellations are handled inline on the poll thread, before the semaphore.
2. **A commit of offset 105 implicitly acknowledges 100..=104.** Jobs finish out of order behind the
   semaphore, so offsets advance through a contiguous watermark (`OffsetWatermark`), never on completion.
3. **Never `sleep()` in a handler and never `seek()` back.** Sleeping trips `max.poll.interval.ms` and gets
   the consumer evicted mid-job. Retry is republish-to-the-next-tier-and-commit; the tier consumer applies
   its delay by pausing the partition and rewinding.

There is **no DLQ by design**, so `cutl_retries_exhausted_total` (served on `/metrics`) is the only signal a
message was lost. It is pre-registered at zero at startup — a labelled counter has no series until its child
exists, and the alert's `increase(...[5m]) > 0` cannot fire on a series that was never reported.

Two operational invariants worth preserving: missing configuration **does not exit** (it logs and holds
`/ready` at 503, because a crash-loop held a CPU reservation on a saturated cluster and forced the workload
to be scaled to 0), and the health server binds **before** the AWS/Kafka clients are built so probes stay
truthful if startup hangs. In `deploy/k8s/`, `terminationGracePeriodSeconds: 600` and `strategy: Recreate`
are load-bearing: a nesting run is minutes long and single-owner, and the default 30s would SIGKILL a job
mid-flight on every rollout.

### AsyncAPI Wire Codegen (jagua-sqs-processor)

The wire contract is **spec-governed**, not hand-written. The AsyncAPI spec (`jagua-sqs-processor/asyncapi/jagua-rs.yaml`) is the single source of truth, pulled from the `gdtrp/cutl-schemas` repo via `scripts/sync-schema.sh` (the file is **git-ignored** and must be synced before building the processor; CI does this automatically).

- `jagua-sqs-processor/build.rs` reads that spec, lifts `components.schemas` into a draft-07 JSON Schema, and runs **typify** to emit `generated.rs` into `OUT_DIR` (included via `src/generated.rs`). typify pins `schemars 0.8` (not 1.x). It re-runs on spec change (`cargo:rerun-if-changed`).
- `build.rs` uses typify `with_replacement` so generated types **reuse the jagua-utils serde types** (`OffcutPolicy`/`Offcut`, `NestingResponsePage`→`PageResult`, `NestingPlacement`→`PlacedPartInfo`) — keeping the wire byte-identical to the tested library serde.
- `src/wire.rs` maps generated wire types ⇄ the ergonomic `SqsNestingRequest`/`SqsNestingResponse`/`SvgPartSpec` used in `processor.rs`, so `processor.rs` is untouched but the wire stays spec-governed.
- **Dockerfile must `COPY build.rs` and `asyncapi/`** before the dep-build phase (they sit at the crate root, not under `src/`) — the dummy-source `cargo build` runs `build.rs` and needs the spec present, else the container build fails with `OUT_DIR not defined` / missing `NestingRequest`. A host `cargo build` hides this; always verify with `make build` (docker).

### Key Configuration

- `CDEConfig`: `quadtree_depth` (default 5), `cd_threshold` (default 16), `item_surrogate_config`
- LBF config: `n_samples`, `ls_frac`, `poly_simpl_tolerance`, `min_item_separation`, `prng_seed`
- Processor env vars — Kafka: `KAFKA_BOOTSTRAP_SERVERS`, `KAFKA_USERNAME`, `KAFKA_PASSWORD` (all required; supplied verbatim by the `kafka-jagua-nesting` Secret via `envFrom`), `KAFKA_SASL_MECHANISM` (SCRAM-SHA-512), `KAFKA_CONSUMER_GROUP`, `KAFKA_REQUEST_TOPIC`, `KAFKA_RESPONSE_TOPIC`, `KAFKA_ATTEMPT_BUDGET` (3), `KAFKA_RETRY_DELAYS_MS` (5s,60s,600s — shorten in tests)
- Processor env vars — rest: `S3_BUCKET` (required, no default on purpose), `AWS_REGION` (eu-north-1), `AWS_ENDPOINT_URL` (S3 relay / MinIO), `MAX_CONCURRENT_TASKS` (20), `EXECUTION_TIMEOUT_SECS` (600), `NEST_RUN_PARALLELISM`, `HEALTH_PORT` (8080), `OTEL_EXPORTER_OTLP_ENDPOINT` (unset ⇒ stdout only)

### Important Conventions

- Extensive `debug_assert!` checks verify engine correctness in test/debug builds but are stripped in release for performance.
- Tests run with `opt-level = 3` to match production-like performance while keeping debug assertions enabled.
- Integration tests in `lbf/tests/tests.rs` use `#[test_case]` for parameterized testing across 13 SPP and 6 BPP instances, each tested at quadtree depths [0, 3, 10]. Tests exercise solve → remove items → save → solve → restore → solve cycle.
- Test layout for the crates we own: `jagua-utils/tests/cutl160_*.rs` cover one fast path each (`_grid`, `_periodic`, `_pairing`, `_mixed`, `_maxfit_*`, `_progress`); `jagua-sqs-processor/tests/` holds `e2e_test.rs` (in-process, no broker), `wire_contract_test.rs` (spec goldens), and `kafka_integration_test.rs` (every test `#[ignore]`d — `make test-integration` is the only thing that runs them, and they need `#[tokio::test(flavor = "multi_thread")]` because `fetch_metadata` blocks).
- `cutl160_prod.rs` / `cutl160_maxfit_prod.rs` replay **real production requests** from `jagua-sqs-processor/tests/testdata/prod-tests/case-NNN/` (generated by `scripts/gen_prod_cases.py` from a CSV export; fixtures committed, `out/` and `REPORT.*` git-ignored). They never abort on a single case — an LBF/rejected outcome is recorded, only a genuine fast-path bug (placed < requested, or non-deterministic output) fails.
- **Wall-clock tests are `#[ignore]`d, not deleted.** `test_e2e_processing_dr_svg` (processor) and four tests in `jagua-utils/tests/adaptive_strategy_test.rs` assert throughput, not correctness — they fail on a slow or loaded machine and pass on a developer box. The dr one kept the CI `test` job red on every `vk-cloud` push from the day CI was introduced (26 passed, 1 failed) because a GitHub runner cannot hit its 60s budget. Run them with `-- --ignored`; don't let one gate a merge.
- Test logging defaults to `Warn` via `init_test_logging()` in `e2e_test.rs`. Don't reintroduce a per-test `env_logger` at `Debug`: `env_logger` writes past libtest's capture, the logger is global and first-init wins, and lbf logs per improving sample — that combination produced 700k+ lines per CI run and truncated the log before the failure summary. Use `RUST_LOG=debug` when you want it back.
- The `NestingResult` struct returns `combined_svg`, per-page `page_svgs`, `parts_placed`, `total_parts_requested`, `unplaced_parts_svg`, and `utilisation`.
- SVG post-processing (holes, colors) is done via regex in `jagua-utils/src/svg_nesting/svg_generation.rs`.
- WASM target (`wasm32-unknown-unknown`) is configured in `rust-toolchain.toml` and `lbf/.cargo/config.toml` (enables atomics, bulk-memory, SIMD128).
- Hosted documentation: [jagua-rs docs](https://jeroengar.github.io/jagua-rs/jagua_rs/), [lbf docs](https://jeroengar.github.io/jagua-rs/lbf/).
