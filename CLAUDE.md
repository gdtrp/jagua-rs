# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
```

## Architecture

### Workspace Crates (dependency order)

```
jagua-sqs-processor → jagua-utils → lbf → jagua-rs
```

- **`jagua-rs`** — Core collision detection engine library (published to crates.io, edition 2024). Feature-gated problem variants: `spp` (strip packing), `bpp` (bin packing), `mspp` (multi strip packing). No default features — downstream crates enable what they need.
- **`lbf`** — Left-Bottom-Fill reference optimizer (edition 2024). Both a library (`rlib` + `cdylib` for WASM) and a CLI binary. Enables `spp` + `bpp` features on jagua-rs. Not for production use — solution quality is chaotic by nature.
- **`jagua-utils`** — SVG nesting utilities wrapping lbf (edition 2024). Provides `NestingStrategy` trait with `SimpleNestingStrategy` and `AdaptiveNestingStrategy` implementations. Enables `bpp` feature on jagua-rs (via lbf).
- **`jagua-sqs-processor`** — AWS SQS/S3 microservice (edition 2021, Tokio async) that receives nesting requests, runs AdaptiveNestingStrategy, uploads result SVGs to S3, and sends responses to output queue. Deployed via Docker to ECS. Deploy workflow in `.github/workflows/deploy.yml` (pushes to ECR, deploys to ECS in eu-north-1).

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
