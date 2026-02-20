# Build stage
FROM rustlang/rust:nightly-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace configuration and all Cargo.toml files for dependency resolution
COPY Cargo.toml Cargo.lock ./
COPY jagua-rs/Cargo.toml ./jagua-rs/
COPY jagua-utils/Cargo.toml ./jagua-utils/
COPY lbf/Cargo.toml ./lbf/
COPY jagua-sqs-processor/Cargo.toml ./jagua-sqs-processor/

# Create dummy source files so cargo can resolve and build dependencies only
RUN mkdir -p jagua-rs/src jagua-utils/src lbf/src lbf/benches jagua-sqs-processor/src && \
    echo "pub fn _dummy() {}" > jagua-rs/src/lib.rs && \
    echo "pub fn _dummy() {}" > jagua-utils/src/lib.rs && \
    echo "pub fn _dummy() {}" > lbf/src/lib.rs && \
    echo "fn main() {}" > lbf/benches/ci_bench.rs && \
    echo "fn main() {}" > lbf/benches/fast_fail_bench.rs && \
    echo "fn main() {}" > lbf/benches/quadtree_bench.rs && \
    echo "fn main() {}" > lbf/benches/edge_sensitivity_bench.rs && \
    echo "fn main() {}" > jagua-sqs-processor/src/main.rs && \
    echo "pub fn _dummy() {}" > jagua-sqs-processor/src/lib.rs

# Build dependencies only (cached unless Cargo.toml/Cargo.lock change)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --release --package jagua-sqs-processor && \
    rm -rf jagua-rs/src jagua-utils/src lbf/src jagua-sqs-processor/src

# Copy actual source code
COPY jagua-rs/src ./jagua-rs/src
COPY jagua-utils/src ./jagua-utils/src
COPY lbf/src ./lbf/src
COPY lbf/benches ./lbf/benches
COPY jagua-sqs-processor/src ./jagua-sqs-processor/src

# Build the application (only recompiles workspace crates, deps are cached)
# Touch source files to invalidate cargo's fingerprint cache for workspace crates
# (the cached /app/target may have stale fingerprints from the dummy-source build above)
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    find jagua-rs/src jagua-utils/src lbf/src jagua-sqs-processor/src -name '*.rs' -exec touch {} + && \
    cargo build --release --package jagua-sqs-processor && \
    cp /app/target/release/jagua-sqs-processor /app/jagua-sqs-processor-bin

# Runtime stage
FROM debian:unstable-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/jagua-sqs-processor-bin /app/jagua-sqs-processor

# Set environment variables (can be overridden)
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
ENV INPUT_QUEUE_URL=nesting-request
ENV OUTPUT_QUEUE_URL=nesting-response

# Run the application
ENTRYPOINT ["/app/jagua-sqs-processor"]
