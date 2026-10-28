# The jagua-rs oracle (RFC-030 N0, dxf-processor): runs jagua-utils' oracle harness over the
# production corpus in a pinned toolchain and C library, so every recorded trace can be
# reproduced bit for bit. Same base as the production image (jagua-sqs-processor/Dockerfile).
#
#   docker build -f docker/jagua-oracle.Dockerfile -t jagua-oracle .
#   docker run --rm -v "$PWD:/src" -v jagua-cargo:/usr/local/cargo/registry \
#     -v <corpus>:/corpus:ro -v <out>:/out jagua-oracle \
#     cargo test -p jagua-utils --features oracle --test oracle_prod -- --nocapture
FROM --platform=linux/amd64 mirror.gcr.io/library/rust:1.97-slim-bookworm
ENV CARGO_TARGET_DIR=/target ORACLE_CORPUS=/corpus ORACLE_OUT=/out
# rust-toolchain.toml asks for the wasm target and these components; install them once here.
RUN rustup component add rust-src rustfmt clippy && rustup target add wasm32-unknown-unknown
WORKDIR /src
