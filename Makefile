.PHONY: build fmt fmt-check lint lint-fix check

LINT_CRATES := -p jagua-utils -p jagua-sqs-processor

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
