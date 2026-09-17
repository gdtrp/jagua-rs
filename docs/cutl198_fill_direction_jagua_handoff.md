# CUTL-198: fill direction, start corner, one rectangular remnant — handoff to jagua-rs

- **Date:** 2026-09-17 · **From:** cutl-backend · **Contract:** cutl-schemas PR #85, released as **`v1.66.0`**
  → `asyncapi/jagua-rs.yaml` **1.3.0**
- **Backend:** branch `staging-nesting-layout`, already forwards both fields (`NestingLayoutIT` green)
- **Product decision (Nikita, 2026-09-17):** one nesting parameter *direction* with three values; the start corner
  as a minor control; **no** slats, micro-joints, cut order, offcut accounting or machine selection

## TL;DR

Two optional fields on `NestingRequest`, both absent today:

```jsonc
{ "correlationId": "…", "binWidth": 3000, "binHeight": 1500, "spacing": 0.2, "parts": [ … ],
  "fillDirection": "HORIZONTAL",   // NEW: HORIZONTAL | VERTICAL | STAIRCASE   (absent/null ⇒ STAIRCASE)
  "startCorner":   "TOP_LEFT",     // NEW: TOP_LEFT | TOP_RIGHT | BOTTOM_LEFT | BOTTOM_RIGHT (absent/null ⇒ TOP_LEFT)
  "offcutPolicy":  { … } }
```

- `STAIRCASE` + `TOP_LEFT` (the defaults) must reproduce **today's output byte for byte** — every existing golden
  and prod replay stays green with no request change.
- `HORIZONTAL`: rows along `binWidth` from the start corner; `VERTICAL`: columns along `binHeight` from the start
  corner. Either way the leftover is **one rectangle**, returned as **exactly one `RECT` in `pages[].offcuts`** on
  the final message (that is the remnant the UI shows — no new response field; see the ADR in cutl-backend
  `docs/adr/2026-09-17-nesting-remnant-as-first-offcut.md`).
- `startCorner` changes *where the parts sit*, never mirrors a part's outline.

Sync the spec (`scripts/sync-schema.sh` with `CUTL_SCHEMAS_REF=v1.66.0`), add the two fields to
`SqsNestingRequest` and both `From` impls in `wire.rs` (the `offcutPolicy` precedent), thread them into
`nest_auto` **and** `nest_max_fit_auto`, update the `wire_contract_test.rs` goldens.

## 1. Semantics — pinned in rendered terms

The SVG output is **y-down and unflipped** (`layout_to_svg.rs` emits the engine's raw numbers; `combine_svg_documents`
stacks pages downward). So engine `(0, 0)` — where today's grid stencil anchors — **renders top-left**. The corner
enum is named as the viewer sees it:

| `startCorner` | engine anchor | today? |
|---|---|---|
| `TOP_LEFT` | `(0, 0)` | yes — the current anchor |
| `TOP_RIGHT` | `(binWidth, 0)` | |
| `BOTTOM_LEFT` | `(0, binHeight)` | |
| `BOTTOM_RIGHT` | `(binWidth, binHeight)` | |

`docs/cutl_business_offcuts_jagua_handoff.md:105` says "bin origin bottom-left" for placements; that describes the
engine's own vocabulary (`grid.rs` calls high-y the "bottom strip"), not the picture. Please add a one-line note there
or in this file's successor so the two do not get reconciled the wrong way round.

| `fillDirection` | placement | remnant |
|---|---|---|
| `HORIZONTAL` | rows parallel to the x axis, first row along the start corner's horizontal edge, rows stacking away from it; within a row, parts advance away from the corner | the strip on the far side **along x**: `x ∈ [used_w + spacing, binWidth]` (mirrored for the right-hand corners), full height |
| `VERTICAL` | columns parallel to the y axis, first column along the start corner's vertical edge; within a column, parts advance away from the corner | the strip on the far side **along y**, full width |
| `STAIRCASE` | today's `PackingMode::Auto` classify-and-route, untouched | today's offcut behaviour (LBF path: `apply_offcuts`; fast paths: none) |

Rotation policy is unchanged: `amountOfRotations` / per-part `allowedRotations` still apply; a row packer may
choose, per part type, the orientation that packs more per row exactly as `grid_single_sheet` does today.

## 2. Where it plugs in

- **Request:** `jagua-sqs-processor/src/wire.rs:51-81` (`generated::NestingRequest` → `SqsNestingRequest`) and
  `:98-123` (reverse); `processor.rs:80-121` (`SqsNestingRequest` fields). Both enums come out of typify as Rust enums;
  map absent/null to the defaults next to `amount_of_rotations` (`wire.rs:67-70`). Remember `strip_nulls`: a cancel
  message sends everything as null.
- **Nest call:** `processor.rs:1538-1581` — pass the two values into `nest_auto` **and** `nest_max_fit_auto`
  (`classify.rs:290`). Max-fit shares the stencil builder, so "max per sheet" must be computed with the same
  direction or the WS-7/G4 invariant (max ≥ every actual sheet) breaks.
- **Strategy layer:** do **not** widen `NestingStrategy::nest` (`strategy.rs:139-147`) — it touches every impl and
  test. Prefer a builder on `AdaptiveNestingStrategy` (`with_layout(direction, corner)` beside `with_offcut_policy`,
  `adaptive.rs:132`) plus an extra argument on the free functions `nest_auto` / `nest_max_fit_auto`.
- **Packers:** `single_sheet_stencil` (`classify.rs:243-285`) is the shared entry for periodic, max-fit and mixed —
  the natural injection point. The existing rectangles-only next-fit shelf packer
  (`mixed.rs:273-326 shelf_pack_leftovers`: rows left→right, shelves stacking in +y, tallest-first) is the closest
  code to `HORIZONTAL`; a general row/column packer over part **bboxes** (any shape, using `cx_off/cy_off` from
  `prepared`) that serves every class is the recommended shape. `VERTICAL` is the same packer with the axes swapped.
- **Start corner as a post-transform of cell anchors**, applied once to the finished `Vec<Placement>`: reflect the
  anchor grid about `binWidth` and/or `binHeight`, then re-place each part inside its reflected cell with the part's
  **own orientation kept** (`render.rs:159-168` anchoring). A plain mirror of positions is not enough for
  non-symmetric parts, and a mirror of outlines is a different part — neither is acceptable. This one function then
  covers every `StencilKind`, including lattice (`lattice.rs:401` already enumerates the four corners internally, as a
  density trick — leave that alone).
- **Remnant:** closed form from the packer's `used_w` / `used_h` (`grid.rs:101-102` already computes them):
  `HORIZONTAL` ⇒ `RECT { x: used_w + spacing, y: 0, width: binWidth − used_w − spacing, height: binHeight }` at
  `TOP_LEFT`, mirrored per corner; `VERTICAL` ⇒ the analogue along y. Emit it into `PageResult.offcuts` from
  `render_periodic` / `render_page_list` (`render.rs:349, 366, 433` are where fast paths currently write
  `offcuts: Vec::new()`). Apply the policy's minimum sizes; do not run `maximal_free_rects` for these two directions.
  Each page gets its own remnant (a full sheet may have none).
- **Keep it through the offload.** `offload_placements` (`processor.rs:848`) clears `page.offcuts` at `:917` when a
  response exceeds `PRODUCER_MAX_MESSAGE_BYTES`. The remnant must survive: keep `offcuts` on the inline page (it is one
  object) or move it into the `pagesUrl` manifest with the placements — either is fine for the backend, which reads
  both (`NestingPagesResolver`).
- **Utilisation:** unchanged definition per packer family (fast paths: bare `density`; LBF: halo-inclusive
  `sheet_utilisation`). The backend shows the number as is. Worth noting in the RFC that the two families differ.

## 3. Determinism and defaults

- `cutl160_prod.rs` / `cutl160_maxfit_prod.rs` replay production cases and fail on any output change. Adding the
  fields must leave every replay identical when the fields are absent.
- The row/column packer must be deterministic (stable part order, no PRNG) — QA compares SVGs.

## 4. Tests to add

- `jagua-utils/tests/cutl198_fill_direction.rs`: for each of a rectangle set, an L-shape set and a mixed 3-type
  set — HORIZONTAL × 4 corners and VERTICAL × 4 corners: no overlaps, parts inside the bin, the remnant rectangle
  contains no part bbox and touches the far edge, every part's outline is a rotation (never a mirror) of its
  input, and `TOP_LEFT` ↔ `BOTTOM_LEFT` layouts are related by the anchor reflection.
- `cutl198_defaults.rs`: absent fields ⇒ output identical to `nest_auto` without them (byte-compare the SVG).
- `cutl198_maxfit.rs`: max-fit with `HORIZONTAL` ≥ every actual sheet's count with the same direction.
- `jagua-sqs-processor/tests/wire_contract_test.rs`: goldens for a request with both fields, with one, with none, and
  with nulls (cancel shape); a response page with exactly one RECT offcut.
- `e2e_test.rs`: one HORIZONTAL job end to end, `offcuts.len() == 1` on the final page, and the same with a forced
  offload (`PRODUCER_MAX_MESSAGE_BYTES` lowered) — the remnant still arrives.
- Gate anything over ~10 s behind `slow-tests`; never assert on wall-clock.

## 5. Task breakdown

| Task | Content | Depends |
|---|---|---|
| **JG-198-1** (~0.5 d) | Spec sync to `v1.66.0`; `SqsNestingRequest` + `wire.rs` both directions; defaults; forward to `nest_auto` / `nest_max_fit_auto`; wire-contract goldens. No behaviour change yet. | — |
| **JG-198-2** (~2 d) | Row/column packer over bboxes for all classes; start-corner anchor reflection keeping part orientation; STAIRCASE untouched; `cutl198_fill_direction.rs`, `cutl198_defaults.rs`, `cutl198_maxfit.rs`. | JG-198-1 |
| **JG-198-3** (~0.5 d) | Remnant as one RECT per page from `used_w/used_h`, policy minimums applied, kept through the offload; e2e tests. | JG-198-2 |

`make check`, `make test`, `make test-slow` over both crates before committing (`CLAUDE.md`); `jagua-sqs-processor` is
edition 2021 (no let-chains); do not touch `jagua-rs/` or `lbf/`.

## 6. Deploy order

1. cutl-schemas `v1.66.0` is released (done). Backend `staging-nesting-layout` forwards the fields (done).
2. Worker: JG-198-1..3 on staging.
3. Frontend on `v1.66.0`: the three-value direction and the corner buttons in the nesting form.
4. Demo on staging: HORIZONTAL from each corner, VERTICAL, the remnant drawn and sized.

## 7. Open questions (answer in the PR, not blocking JG-198-1)

1. **Pairing and lattice classes under HORIZONTAL/VERTICAL.** Proposal: they fall back to the bbox row/column
   packer (simpler, always rectangular remnant) rather than being refused; utilisation may drop, which the product
   accepts (the editor shows the number).
2. **Remnant below the policy minimum.** Proposal: omit it (no offcut) rather than report a sliver — the UI then
   shows no remnant size. Confirm.
3. **Spacing at the remnant edge.** Proposal: the remnant starts `spacing` after the last part, i.e. one kerf/gap
   is left between parts and remnant, none between remnant and sheet edge. Confirm.
