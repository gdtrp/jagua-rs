#!/usr/bin/env bash
# Pull the canonical jagua-rs AsyncAPI spec from cutl-schemas (the single source of truth) into
# the path build.rs/typify reads. The spec is NOT committed here — run this before building
# (CI does it automatically). Needs gh auth with read access to the private cutl-schemas repo.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/jagua-sqs-processor/asyncapi/jagua-rs.yaml"
mkdir -p "$(dirname "$DEST")"
gh api repos/gdtrp/cutl-schemas/contents/asyncapi/jagua-rs.yaml \
  -H 'Accept: application/vnd.github.raw' > "$DEST"
echo "synced jagua-rs.yaml <- cutl-schemas ($(wc -l < "$DEST") lines)"
