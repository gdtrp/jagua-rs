# Trigger file for .github/workflows/validate-vk.yml.
#
# Touching this file on the vk-cloud branch runs the VK staging validation.
# It exists because GitHub only offers workflow_dispatch for workflows present on
# the DEFAULT branch, and that workflow lives on vk-cloud until the migration
# merges. Once it reaches main, use `gh workflow run validate-vk.yml` instead and
# this file can go.
#
# mode=readonly  inspect only — pod state, logs, /ready, /metrics, consumer groups
# mode=cancel    also produce a cancellation: exercises consume, keying, the
#                inline cancel path and the offset commit, and never touches S3
# mode=full      also produce a real nesting request. WARNING: S3 eu-north-1 is
#                unreachable from VK until the relay exists, so this WILL fail its
#                upload, walk all three retry tiers (~11 min) and fire
#                CutlRetriesExhausted at severity: critical. Someone gets paged.

mode=full
