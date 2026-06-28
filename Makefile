.PHONY: build fmt fmt-check lint lint-fix check sync-spec codegen

LINT_CRATES := -p jagua-utils -p jagua-sqs-processor

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
