# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Constraints (read first)

- **Only modify `jagua-utils/` and `jagua-sqs-processor/`.** `jagua-rs/` and `lbf/` are upstream library crates (synced from `JeroenGar/jagua-rs` via the `upstream` remote) — don't make standalone edits to them. To change packing quality or algorithm behavior, tune parameters, adjust strategies, or add pre/post-processing in the two wrapper crates. (A few load-bearing local patches do exist in the library — e.g. `lbf` cancellation/callback API, a disabled over-eager `debug_assert!`, a NaN-safe density assert — preserve them across upstream syncs.)
- **`jagua-sqs-processor` is edition 2021** (no let-chains — use nested `if let`). `jagua-rs`, `lbf`, and `jagua-utils` are edition 2024.

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
make sync-spec    # Makefile copy of the spec from a local $CUTL_BACKEND checkout (default ../cutl-backend)
make codegen      # sync-spec + touch the spec + cargo build (forces build.rs to re-run typify)
make build        # docker build of jagua-sqs-processor (validates the container codegen path)

# Current canonical spec source (preferred over `make sync-spec`): pull from the cutl-schemas repo
scripts/sync-schema.sh   # gh api gdtrp/cutl-schemas .../jagua-rs.yaml -> jagua-sqs-processor/asyncapi/jagua-rs.yaml (needs gh auth)
```

## Architecture

### Workspace Crates (dependency order)

```
jagua-sqs-processor → jagua-utils → lbf → jagua-rs
```

- **`jagua-rs`** — Core collision detection engine library (published to crates.io, edition 2024). Feature-gated problem variants: `spp` (strip packing), `bpp` (bin packing), `mspp` (multi strip packing). No default features — downstream crates enable what they need.
- **`lbf`** — Left-Bottom-Fill reference optimizer (edition 2024). Both a library (`rlib` + `cdylib` for WASM) and a CLI binary. Enables `spp` + `bpp` features on jagua-rs. Not for production use — solution quality is chaotic by nature.
- **`jagua-utils`** — SVG nesting utilities wrapping lbf (edition 2024). Provides `NestingStrategy` trait with `SimpleNestingStrategy` and `AdaptiveNestingStrategy` implementations. Enables `bpp` feature on jagua-rs (via lbf).
- **`jagua-sqs-processor`** — AWS SQS/S3 microservice (edition 2021, Tokio async) that receives nesting requests, runs AdaptiveNestingStrategy, uploads result SVGs to S3, and sends responses to output queue. Deployed via Docker to ECS. Deploy workflow in `.github/workflows/deploy.yml` (pushes to ECR, deploys to ECS in eu-north-1). **API-first**: SQS wire types are code-generated from an AsyncAPI spec (see below), not hand-written.

### Core jagua-rs Architecture

**Problem-Instance-Solution pattern**: `Instance` (immutable problem definition) → `Problem` (mutable working state with snapshot/restore) → `Solution` (immutable snapshot). Each problem variant (SPP, BPP, MSPP) implements this independently under `jagua-rs/src/probs/`. Problem variants are gated by cargo features; each variant has its own `entities/` and `io/` submodules.

**Collision Detection Engine (CDEngine)** in `jagua-rs/src/collision_detection/`: Quadtree-based spatial indexing with configurable depth. All spatial constraints are unified as **Hazards** (placed items, container exterior, holes, quality zones). `HazardFilter` trait enables per-query selective checking.

**Fail-fast surrogates** in `jagua-rs/src/geometry/fail_fast/`: Items have simplified shape representations (`SPSurrogate` with poles/piers) for quick collision rejection before expensive polygon checks.

**Two-level shape representation**: `OriginalShape` (exact input geometry) vs `SPolygon` (simplified collision detection shape). Polygon simplification is controlled by `poly_simpl_tolerance`.

**Key geometry traits** in `jagua-rs/src/geometry/geo_traits.rs`: `CollidesWith<T>`, `AlmostCollidesWith<T>`, `Transformable`, `TransformableFrom` (in-place transform to avoid allocation).

**SlotMap arena allocation**: `PItemKey` for placed items, `HazKey` for hazards, `LayKey` for layouts. Provides O(1) insert/remove with stable keys.

**Error handling**: Uses `anyhow::Result` throughout all crates. No custom error types.

### SQS Processor Flow

Request (`SqsNestingRequest`) arrives via SQS → SVG downloaded from S3 (or decoded from base64) → `AdaptiveNestingStrategy.nest()` runs in `spawn_blocking` → intermediate improvements sent to output queue → final result SVGs uploaded to S3 → response sent to output queue. Supports cancellation via `correlation_id` registry and cooperative timeout checking.

### AsyncAPI Wire Codegen (jagua-sqs-processor)

The SQS wire contract is **spec-governed**, not hand-written. The AsyncAPI spec (`jagua-sqs-processor/asyncapi/jagua-rs.yaml`) is the single source of truth, pulled from the `gdtrp/cutl-schemas` repo via `scripts/sync-schema.sh` (the file is **git-ignored** and must be synced before building the processor; CI does this automatically).

- `jagua-sqs-processor/build.rs` reads that spec, lifts `components.schemas` into a draft-07 JSON Schema, and runs **typify** to emit `generated.rs` into `OUT_DIR` (included via `src/generated.rs`). typify pins `schemars 0.8` (not 1.x). It re-runs on spec change (`cargo:rerun-if-changed`).
- `build.rs` uses typify `with_replacement` so generated types **reuse the jagua-utils serde types** (`OffcutPolicy`/`Offcut`, `NestingResponsePage`→`PageResult`, `NestingPlacement`→`PlacedPartInfo`) — keeping the wire byte-identical to the tested library serde.
- `src/wire.rs` maps generated wire types ⇄ the ergonomic `SqsNestingRequest`/`SqsNestingResponse`/`SvgPartSpec` used in `processor.rs`, so `processor.rs` is untouched but the wire stays spec-governed.
- **Dockerfile must `COPY build.rs` and `asyncapi/`** before the dep-build phase (they sit at the crate root, not under `src/`) — the dummy-source `cargo build` runs `build.rs` and needs the spec present, else the container build fails with `OUT_DIR not defined` / missing `NestingRequest`. A host `cargo build` hides this; always verify with `make build` (docker).

### Key Configuration

- `CDEConfig`: `quadtree_depth` (default 5), `cd_threshold` (default 16), `item_surrogate_config`
- LBF config: `n_samples`, `ls_frac`, `poly_simpl_tolerance`, `min_item_separation`, `prng_seed`
- SQS processor env vars: `INPUT_QUEUE_URL`, `OUTPUT_QUEUE_URL`, `S3_BUCKET`, `AWS_REGION`, `MAX_CONCURRENT_TASKS`, `EXECUTION_TIMEOUT_SECS`

### Important Conventions

- Extensive `debug_assert!` checks verify engine correctness in test/debug builds but are stripped in release for performance.
- Tests run with `opt-level = 3` to match production-like performance while keeping debug assertions enabled.
- Integration tests in `lbf/tests/tests.rs` use `#[test_case]` for parameterized testing across 13 SPP and 6 BPP instances, each tested at quadtree depths [0, 3, 10]. Tests exercise solve → remove items → save → solve → restore → solve cycle.
- The `NestingResult` struct returns `combined_svg`, per-page `page_svgs`, `parts_placed`, `total_parts_requested`, `unplaced_parts_svg`, and `utilisation`.
- SVG post-processing (holes, colors) is done via regex in `jagua-utils/src/svg_nesting/svg_generation.rs`.
- WASM target (`wasm32-unknown-unknown`) is configured in `rust-toolchain.toml` and `lbf/.cargo/config.toml` (enables atomics, bulk-memory, SIMD128).
- Hosted documentation: [jagua-rs docs](https://jeroengar.github.io/jagua-rs/jagua_rs/), [lbf docs](https://jeroengar.github.io/jagua-rs/lbf/).
