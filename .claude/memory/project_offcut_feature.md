---
name: project_offcut_feature
description: CUTL business-offcuts feature (JG-OFF-1/2/3) — free-space detection after nesting
metadata: 
  node_type: memory
  type: project
  originSessionId: 54aa2ad0-40ef-4dfd-99b5-dc85a52a996b
---

Business-offcuts feature from `docs/cutl_business_offcuts_jagua_handoff.md`: after nesting,
report reusable free-space ("offcuts") per page.

**JG-OFF-1 (done)** — detection lives in `jagua-utils/src/svg_nesting/offcut.rs` (NOT
`jagua-rs` per the no-touch-library rule). `Offcut` (Rect | Poly{vertices, holes}),
`OffcutPolicy{min_offcut_width_mm,min_offcut_height_mm,shape:RECTANGLE|QUADRILATERAL,kerf_mm}`,
`.with_offcut_policy()` builder on both strategies, per-page `offcuts` on `PageResult`, and a
teal overlay drawn onto the page SVGs (`overlay_offcuts_svg` in svg_generation.rs). Computed
**final layout only** via a finalize pass at `nest_inner`'s return sites; intermediates stay
empty. Per user: RECTANGLE tiles ALL free space into non-overlapping rects (slab sweep);
QUADRILATERAL = `bin − union(part outlines)` with holes. Runtime-gated (no cargo feature).

**JG-OFF-2 (done)** — `jagua-sqs-processor`: added `offcut_policy: Option<jagua_utils::OffcutPolicy>`
to `SqsNestingRequest` (reused the library type directly — serde already matches the wire),
applied to the strategy **only on the non-max_fit path** (user: max_fit must NOT run
detection — perf + little free space). Response already carried offcuts via `PageResult` clone.
**Gotcha found:** the "Bucket" commit left ~23 `SqsNestingRequest{..}` test literals missing
`bucket`/`s3_prefix`, so processor tests didn't compile — fixed by appending those + the new
`offcut_policy` field. Deploy is **user-triggered** (push to `bucket` branch → ECR/ECS staging).

**Offcut refinements (2026-06-02, after user's 3-part report)** — three fixes in `offcut.rs`:
1. *No spacing gap*: the detector ran on the engine's COLLISION geometry (sheet deflated by
   `spacing/2`, parts inflated by `spacing/2` — `PlacedItem.shape` = `item.shape_cd`, the
   inflated CD shape; `layout.container.outer_cd.bbox` = deflated sheet). Now `spacing` is
   threaded through `apply_offcuts → write_page_offcuts → detect_*` and the offset is undone:
   bin bbox is GROWN by `spacing/2` (true sheet edges), rect obstacles SHRUNK by `spacing/2`,
   poly parts DEFLATED by `spacing/2` (`geo_buffer::buffer_polygon(p, -half)`). Offcuts now
   touch real part outlines + real walls.
2. *More rectangles*: replaced the vertical-slab sweep with greedy **maximal-rectangle**
   decomposition (`largest_empty_rect` over obstacle/bin candidate edges, emit, re-add as
   obstacle, repeat). Big areas (e.g. block below a packed column) survive instead of being
   sliced into sub-min-width strips.
3. *Kerf band*: `kerf_mm` NO LONGER insets the reported offcut (offcut = full free material).
   Instead `kerf_band_paths()` builds a shaded ring (`overlay_offcuts_svg` draws `<g
   id="offcut_kerf">` `#FF7043` 0.35) = offcut boundary minus the kerf-inset usable interior
   (even-odd). Decisions: shaded band (not inner line), driven by `offcutPolicy.kerfMm`.
4. *Merged single shape* (user: "combine all 3 rectangles into a single shape... makes more
   sense"): new `OffcutShape::RectangleMerged` (`"RECTANGLE_MERGED"`) unions the maximal rects
   (`merge_rect_offcuts`, geo `BooleanOps::union` + RDP) into ONE rectilinear `Offcut::Poly`,
   so the kerf band wraps only the true perimeter (no kerf on internal split lines). `Rectangle`
   (multi-rect) and `Quadrilateral` (hugs outlines) still available. NEW wire enum value —
   backend must opt in to send it.
E2e repro: `jagua-utils/tests/offcut_detection_test.rs::report_repro_three_parts_1500x3000_rectangle`
(+ `..._rectangle_merged`)
(fixtures `offcut_part_{a,b,c}.svg` in jagua-sqs-processor/tests/testdata), writes
`jagua-utils/test_output/offcuts/report_repro_*` for visual check.

**JG-OFF-3 (not started)** — optional `extraBins` variable-size bins + tag each placement
with originating `binId`.

**Why:** backend forwards a per-material min reusable size; jagua scans the final layout.
**How to apply:** edition mismatch — `jagua-sqs-processor` is edition 2021 (NO let-chains;
use nested `if let`), while jagua-rs/lbf/jagua-utils are edition 2024. See
[[feedback_no_modify_library]] and [[project_maxfit_parallel_budget]].
