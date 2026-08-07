#!/usr/bin/env bash
# Run a cargo command against this workspace inside the pinned builder image.
#
# WHY: the crate now depends on rdkafka, which compiles librdkafka from C source
# and needs cmake/g++/libsasl2-dev/zlib1g-dev — none of which ship in
# rust:slim-bookworm. This script installs exactly what the Dockerfile builder
# stage installs, so a green run here means the image build will also be green.
# It is also the only way to build on a machine with no local Rust toolchain.
#
# The cargo registry and target dir live in named volumes, so repeated runs are
# incremental rather than rebuilding librdkafka every time.
#
# Usage:
#   scripts/cargo-docker.sh build -p jagua-sqs-processor
#   scripts/cargo-docker.sh test -p jagua-sqs-processor
#   scripts/cargo-docker.sh clippy -p jagua-utils -p jagua-sqs-processor --all-targets --no-deps -- -D warnings
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="mirror.gcr.io/library/rust:1.97-slim-bookworm"

# Kept in lockstep with jagua-sqs-processor/Dockerfile's builder stage. g++ is
# required because librdkafka's top-level CMakeLists declares a CXX project even
# though the library itself is C.
BUILD_DEPS="pkg-config libssl-dev cmake g++ libsasl2-dev zlib1g-dev libcurl4-openssl-dev"

# When the integration harness is up, join its network and point the tests at the
# in-network listeners. `localhost` inside this container is the container itself,
# so without this every produce times out and it looks like a broker fault.
#
# The harness advertises two SASL listeners for exactly this reason:
# kafka:29092 in-network, localhost:9092 from the host.
NETWORK_ARGS=()
if docker network inspect jagua-test_default >/dev/null 2>&1; then
  NETWORK_ARGS=(
    --network jagua-test_default
    -e KAFKA_BOOTSTRAP_SERVERS=kafka:29092
    -e AWS_ENDPOINT_URL=http://minio:9000
  )
fi

# `bash -c`, never `bash -lc`: a login shell resets PATH and loses the image's
# /usr/local/cargo/bin, making cargo "command not found".
exec docker run --rm \
  -v "$ROOT":/app -w /app \
  -v jagua-cargo-registry:/usr/local/cargo/registry \
  -v jagua-cargo-git:/usr/local/cargo/git \
  -v jagua-target:/app/target \
  -e CARGO_TERM_COLOR=never \
  "${NETWORK_ARGS[@]}" \
  "$IMAGE" \
  bash -c "apt-get update -qq >/dev/null && apt-get install -y -qq $BUILD_DEPS >/dev/null 2>&1 && cargo $(printf '%q ' "$@")"
