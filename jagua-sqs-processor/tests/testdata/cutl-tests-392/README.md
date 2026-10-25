# cutl-tests#392 / #393 / #396 — run 4a930f88

The staging request that OOM-killed jagua on 2026-10-25 (`nesting-request` partition 1, offset 187,
test NMJ-N05), followed 1 s later by its cancel.

- `hive_with_holes.svg`, `hive_plain.svg`: cutl-backend's conversions of the two shapes of
  `batch/hive.dxf` (450 × 1800 mm each; the first carries 2,215 Ø15 holes, a 2.5 MB SVG).
- `request.json`: the request in wire shape. Bin, spacing, rotations and counts are from jagua's
  log line for offset 187 and the test setup. The item ids and `svgUrl`s were not logged, so they
  are placeholders that point at the two files above.
- `cancel.json`: the cancel record for the same `correlationId`, in the production cancel shape.

Covered by `jagua-utils/tests/cutl_tests392_hive_memory.rs` (peak heap; 1404 MiB before the fix,
24 MiB after) and by the cancel-before-start tests in `jagua-sqs-processor`.
