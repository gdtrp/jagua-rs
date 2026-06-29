---
name: Do not modify library crates
description: Only modify jagua-utils and jagua-sqs-processor - never modify jagua-rs or lbf crates
type: feedback
---

NEVER modify code in `jagua-rs/` or `lbf/` crates — these are upstream library code that Boris wraps.

Only modify code in:
- `jagua-utils/` — SVG nesting utilities wrapping lbf
- `jagua-sqs-processor/` — AWS SQS/S3 microservice

**Why:** jagua-rs and lbf are external library crates. Boris only controls the wrapper layer.

**How to apply:** When asked to improve packing quality, algorithm behavior, or fix bugs — only change code in jagua-utils or jagua-sqs-processor. Tune parameters, adjust strategies, add pre/post-processing, but never touch the library.
