.PHONY: build fmt fmt-check lint lint-fix check sync-spec codegen \
        test test-integration compose-up compose-down

LINT_CRATES := -p jagua-utils -p jagua-sqs-processor

COMPOSE := docker compose -f jagua-sqs-processor/docker-compose.yml

# Path to the cutl-backend checkout that owns the AsyncAPI contracts (override if elsewhere).
CUTL_BACKEND ?= ../cutl-backend
# Vendored copy of the jagua-rs AsyncAPI spec. build.rs generates the wire types from THIS file
# (the Docker build context can't reach cutl-backend), so re-run `make sync-spec` after the
# contract changes upstream. The committed YAML is the sole codegen source; nothing derived is committed.
sync-spec:
	cp $(CUTL_BACKEND)/docs/asyncapi/jagua-rs.yaml jagua-sqs-processor/asyncapi/jagua-rs.yaml
	@echo "Synced jagua-rs.yaml from $(CUTL_BACKEND). Rebuild to regenerate wire types."

# Regenerate the SQS wire types from the AsyncAPI contract: re-sync the vendored spec, then build
# (build.rs runs typify on asyncapi/jagua-rs.yaml -> $OUT_DIR/generated.rs, reused via src/generated.rs).
# `touch` forces build.rs to re-run even if the synced file is byte-identical. The Java models live
# in cutl-backend — regenerate those separately there with `npm run gen:async`.
codegen: sync-spec
	touch jagua-sqs-processor/asyncapi/jagua-rs.yaml
	cargo build -p jagua-sqs-processor
	@echo "Wire types regenerated from asyncapi/jagua-rs.yaml."

build:
	docker build -t jagua-sqs-processor -f jagua-sqs-processor/Dockerfile .

# Format jagua-utils + jagua-sqs-processor (jagua-rs and lbf are upstream).
fmt:
	cargo fmt $(LINT_CRATES)

fmt-check:
	cargo fmt $(LINT_CRATES) -- --check

# Run clippy on the crates we own. -D warnings turns warnings into errors.
# --no-deps skips linting upstream crates pulled in via path dependencies.
lint:
	cargo clippy $(LINT_CRATES) --all-targets --no-deps -- -D warnings

lint-fix:
	cargo clippy $(LINT_CRATES) --all-targets --no-deps --fix --allow-dirty --allow-staged

# Run all checks (format + lint).
check: fmt-check lint

# ── Tests ──
# `test` is the broker-free suite: unit tests, the wire contract goldens and the
# in-process nesting e2e tests. Safe to run anywhere, no Docker needed.
test:
	cargo test $(LINT_CRATES)

# Brings up Kafka (SASL_PLAINTEXT + SCRAM-SHA-512) and MinIO, waits for both to be
# healthy, then provisions the SCRAM user and the five topics. Idempotent.
#
# The broker advertises its SASL listener as localhost:9092 because the tests run
# on the host via `cargo test`, not inside the compose network.
compose-up:
	$(COMPOSE) up -d --wait kafka minio
	$(COMPOSE) up kafka-init minio-init

compose-down:
	$(COMPOSE) down -v

# The broker-backed tests are #[ignore]d so a plain `cargo test` on a machine with
# no Docker still passes rather than failing for the wrong reason. This target is
# the only thing that runs them.
test-integration: compose-up
	cargo test $(LINT_CRATES) -- --ignored --test-threads=1
