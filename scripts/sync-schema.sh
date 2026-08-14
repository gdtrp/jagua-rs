#!/usr/bin/env bash
# Pull the canonical jagua-rs AsyncAPI spec from cutl-schemas (the single source of truth) into
# the path build.rs/typify reads. The spec is NOT committed here — run this before building
# (CI does it automatically). Needs gh auth with read access to the private cutl-schemas repo.
#
# CUTL_SCHEMAS_REF selects the branch/tag to pull from; unset means the repo's
# default branch. It exists because the Kafka transport declaration landed on
# cutl-schemas' `vk-cloud` branch first — pulling the default branch during the
# migration gets a spec whose `servers:` block still says `aws-sqs`.
#
# That mismatch is cosmetic rather than dangerous: build.rs/typify only reads
# `components.schemas`, so the generated wire types are byte-identical either way.
# Set the ref when you want the vendored copy to match what is actually deployed.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/jagua-sqs-processor/asyncapi/jagua-rs.yaml"
mkdir -p "$(dirname "$DEST")"

REF_QUERY=""
if [[ -n "${CUTL_SCHEMAS_REF:-}" ]]; then
  REF_QUERY="?ref=${CUTL_SCHEMAS_REF}"
fi

gh api "repos/gdtrp/cutl-schemas/contents/asyncapi/jagua-rs.yaml${REF_QUERY}" \
  -H 'Accept: application/vnd.github.raw' > "$DEST"
echo "synced jagua-rs.yaml <- cutl-schemas${CUTL_SCHEMAS_REF:+@$CUTL_SCHEMAS_REF} ($(wc -l < "$DEST") lines)"
