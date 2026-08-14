# Build stage
#
# PINNED STABLE, not rustlang/rust:nightly-slim (VK migration T7). The old tag
# was rolling AND nightly, so the build could break on any given day through no
# change of ours. `cargo check` across the whole workspace — jagua-rs,
# jagua-utils, lbf and this crate — passes on stable, so nothing here needs
# nightly features.
#
# The floor is 1.91, not "any stable": the aws-smithy / aws-types crates in the
# lockfile declare rust-version 1.91, and an older toolchain fails resolution
# with "requires rustc 1.91" rather than a compile error. Verified by building
# this image — 1.90 was rejected outright.
FROM mirror.gcr.io/library/rust:1.97-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies.
#
# cmake / g++ / libsasl2-dev / zlib1g-dev are for rdkafka (T9): rdkafka-sys compiles
# the vendored librdkafka with the `cmake-build` feature, and none of the four ships
# in rust:slim-bookworm. Omitting them fails at `cargo build`, not at `cargo check`
# — this is the "verify by building the image" trap from T7.
#
# g++ specifically: the image has `cc` (the Rust linker needs it) but no `c++`, and
# librdkafka's top-level CMakeLists declares a CXX project, so cmake aborts with
# "CMAKE_CXX_COMPILER: c++ is not a full path and was not found in the PATH"
# before compiling a single C file.
#
# libcurl4-openssl-dev despite rdkafka-sys passing -DWITH_CURL=0: SASL OAUTHBEARER
# is enabled by the `ssl` feature (SCRAM and OAUTHBEARER share the OpenSSL
# dependency), and rdkafka_conf.c includes <curl/curl.h> unconditionally on that
# path. We never use OAUTHBEARER — the header is needed to compile, not to run.
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    libsasl2-dev \
    zlib1g-dev \
    libcurl4-openssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace configuration and all Cargo.toml files for dependency resolution
COPY Cargo.toml Cargo.lock ./
COPY jagua-rs/Cargo.toml ./jagua-rs/
COPY jagua-utils/Cargo.toml ./jagua-utils/
COPY lbf/Cargo.toml ./lbf/
COPY jagua-sqs-processor/Cargo.toml ./jagua-sqs-processor/

# build.rs generates the SQS wire types from the vendored AsyncAPI spec at compile time, so both
# must be present in every build phase (the dummy dep-build below runs build.rs too — it compiles
# the heavy build-deps like typify/syn into the cached layer). They live at the crate root /
# asyncapi dir, NOT under src/, so they need their own COPY.
COPY jagua-sqs-processor/build.rs ./jagua-sqs-processor/build.rs
COPY jagua-sqs-processor/asyncapi ./jagua-sqs-processor/asyncapi

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
#
# bookworm-slim, NOT debian:unstable-slim. `unstable` is Debian sid: a rolling
# tag whose contents change continuously, which meant production was tracking an
# unreleased distribution and a rebuild could pull a different libssl on any
# day. This is the one base-image change in T7 that fixes a real bug rather than
# removing an AWS dependency.
FROM mirror.gcr.io/library/debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies. libsasl2-2 pairs with the libsasl2-dev the builder
# links librdkafka against; without it the binary fails to start with a loader
# error rather than anything that mentions Kafka.
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libsasl2-2 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/jagua-sqs-processor-bin /app/jagua-sqs-processor

# Set environment variables (can be overridden)
ENV RUST_LOG=info
ENV RUST_BACKTRACE=1
# Topics carry NO environment suffix: each environment has its own Kafka cluster,
# so the suffix lives in the cluster boundary rather than the topic name. These are
# the defaults; the bootstrap address and SCRAM credentials arrive at runtime from
# the kafka-jagua-nesting Secret via envFrom.
ENV KAFKA_REQUEST_TOPIC=nesting-request
ENV KAFKA_RESPONSE_TOPIC=nesting-response
ENV KAFKA_CONSUMER_GROUP=jagua-nesting
# Kubernetes /health and /ready (VK migration T7). Overridable via HEALTH_PORT.
ENV HEALTH_PORT=8080
EXPOSE 8080

# Run the application
ENTRYPOINT ["/app/jagua-sqs-processor"]
