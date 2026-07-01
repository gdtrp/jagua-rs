---
name: project_cutl160_nesting_rfc
description: "CUTL-160 nesting quality/UX overhaul — RFC at docs/rfcs/, special-case fast paths + periodic packing"
metadata: 
  node_type: memory
  type: project
  originSessionId: f9f8c1ab-e4e3-409d-a223-c91e9caefe6e
---

CUTL-160 (tracker: "Оптимизация модуля раскроя") — QA (Никита Зайцев) filed a detailed
report of nesting defects. RFC written 2026-06-29 at `docs/rfcs/CUTL-160-nesting-optimization.md`;
QA screenshots in `docs/bugfix/image (3..11).png` (map by file size to tracker attach 100–109; video=110).

**Root cause:** one general nester (`AdaptiveNestingStrategy` → jagua-rs BPP + LBF, stochastic)
is applied to ALL inputs, incl. easy bulk rectangles. QA's own thesis: carve out special cases
with fast deterministic packers. No per-bin "repeat stencil" concept exists in the worker.

**RFC plan (workstreams, all in jagua-utils + jagua-sqs-processor per [[feedback_no_modify_library]]):**
- WS-1 classifier+router (`classify.rs`, `nest_auto`); optional `packingMode` request hint.
- WS-2 grid fast path (`grid.rs`) — closed-form axis-aligned rect grid, two-orientation guillotine, no LBF.
- WS-3 **periodic packing** (`periodic.rs`) — compute stencil once, emit K identical sheets + 1 remainder.
  Core fix: kills "K different sheets", collapses runtime, and makes bug #4 (max/sheet 44 but sheet has 45)
  structurally impossible. Generalises existing `skip_middle` page-clone in adaptive.rs:347.
- WS-4 pairing fast path for half-bbox parts (right-triangle «косынка» → pair into rectangle → grid).
- WS-5 mixed-types rectangle pre-analysis + grouping (last phase).
- WS-6 determinate progress block on SqsNestingResponse (sheetsTotalEst known after stencil).
- WS-7 bug fix: unify max-per-sheet with stencil cap; identical pages share one S3 URL (blank-preview/load).
- WS-8 rotation default {0°,90°} for rect parts (vs amountOfRotations=8 in wire.rs:70).

**Key QA cases → screenshots:** frames 980×2000/200×295 (img5/6, 66% density, 178 sheets), tall-thin
29-different-sheets (img7), triangle 300×100 area=½bbox (img8/11, 66.7%), max/sheet bug (img10),
indeterminate progress bar (img3).

**SPACING / RENDER DECISIONS (2026-06-30, from prod-case review):**
- `allowedRotations: []` = **any rotation** (contract: empty array = unconstrained, NOT 0°-lock).
  Fixed in `strategy.rs::effective_allowed` (empty→treated as None); `resolve_rotation_range`/
  `fit_orientations` use it. `[0]` (explicit) still = 0°-lock. Pairing/mixed routing & swap now use it.
- Spacing model = **single 2mm kerf between every pair of adjacent parts** (one gap per cut, NOT
  2mm-per-side and NOT abut/0). Grid pitch = bbox + spacing already gives this. **No border gap** —
  parts sit flush to the sheet edge (user: "remove gaps from borders, not needed").
- Pairing: the two paired triangles are **separate parts** → held a full 2mm apart on the shared
  hypotenuse too (perpendicular push by spacing/2 each along the hypotenuse normal; pair-cell grown
  by spacing*|normal| so neighbour kerf stays 2mm). Measured: all triangle gaps uniformly 2.0mm.
- **Render overlay was the "looks doubled / 4mm" confusion:** `SvgDrawOptions::default()` drew the
  inflated collision-shape outline + fail-fast surrogate poles (a 2nd dashed outline + green dots per
  part), reading as phantom spacing. `render_one_page` now sets quadtree/surrogate/highlight_*/
  draw_cd_shapes = false → only the real part outline. Actual gaps were always exactly 2mm.
- Abut (0-gap shared cut) was tried then REVERTED: it undersizes parts by kerf/2; the 2mm gap is
  required for exact-size parts.

**PRODUCTION E2E HARNESS (2026-06-30):** `jagua-utils/tests/cutl160_prod.rs` is data-driven over
`jagua-sqs-processor/tests/testdata/prod-tests/<case>/` (each: `data.json` =
{binWidth,binHeight,spacing,amountOfRotations,items:[{itemId,count,allowedRotations,...}]} + one
`<itemId>.svg` per item). SVGs come from the request's PRIVATE S3 url — fetch with
`aws s3 cp s3://cutl-data-production/<key> ...` (anon curl 403s; aws creds present, acct 597312200625).
Cases load at RUNTIME (no include_bytes → drop a new folder, no recompile). Outputs written to
**`<case>/out/`** (user's chosen location, next to the data): `combined.svg`, `sheets/run*.svg`
(one per run of identical sheets, self-describing names), `index.md` (run table), `summary.json`.
User has 10 prod cases, drip-feeding; I download SVGs per case. **case-01** = QA "НПЭ 45": 3 rect
frames 310²/265²/295×200, counts 2000/600/4000=6600, grain-locked `allowedRotations:[]` →
**6600/6600 placed, 288 sheets (111+28+148 identical + 1 remainder), 83% util, ~170ms, deterministic**
(was slow LBF / 178 mixed sheets). **BUG FOUND VIA PROD DATA + FIXED:** mixed router gated on
`allowed_rotations.is_none()` so grain-locked parts (`[]`) wrongly fell to LBF; made `nest_mixed`
grain-aware (per-part allow_swap via `fit_orientations`), gate now `parts.iter().all(single_part_allow_original)`.
OPEN Q for backend: is `allowedRotations:[]` truly 0°-lock (current grain contract) or "use global
rotations"? If the latter these frames could rotate 90° and pack denser.

NOTE: repo remounted `/Volumes/USB`→`/Volumes/Projects` mid-session; memory lives in-repo at
`.claude/memory/` (the `~/.claude/projects/<key>/memory` symlink to the old USB path is dead).

**STATUS 2026-06-30: IMPLEMENTED on branch `chore/consume-cutl-schemas`** (plan approved via
`/plan`, file `~/.claude/plans/floofy-hatching-pretzel.md`). All WS done, all tests green, `make check`
clean. New jagua-utils modules: `render.rs` (shared deterministic-render infra: `prepare`,
`render_periodic`, `render_page_list`, `Placement`, `measure_part`/`PartMetrics`), `classify.rs`
(`classify`→`PackingClass`, `nest_auto`/`nest_max_fit_auto`, `PackingMode`), `grid.rs`
(`grid_single_sheet` two-orientation guillotine), `periodic.rs` (`nest_periodic_grid`,
`nest_max_fit_grid`), `pairing.rs` (`nest_pairing` — kerf via centroid-separation vector), `mixed.rs`
(`nest_mixed` — per-type identical sheets + shelf-packed shared remainder = QA scheme #3.1).
Deterministic packers DON'T run LBF; they build `Layout`→`save`→`s_layout_to_svg` (no inflation:
importer `min_item_separation=None`, grid adds `spacing` to pitch). General/SingleIrregular still →
`AdaptiveNestingStrategy` unchanged.

Processor: `nest()`→`jagua_utils::nest_auto`, max_fit→`nest_max_fit_auto` (rectangles use grid stencil
so max-per-sheet == periodic full-sheet ⇒ #4 impossible). **The packer is chosen by jagua-rs from the
incoming part shapes (the classifier) — NOT a caller field.** (An earlier `packingMode` request field
was added then REMOVED per user: "packing mode should be determined by jagua-rs based on incoming
shapes". `PackingMode` enum {Auto,Grid,Periodic,General} stays internal to classify.rs; processor
always passes `PackingMode::Auto`; the Grid/Periodic/General variants are test/diagnostic only.) WS-6:
`NestingResult.sheets_total_estimate` → `SqsNestingResponse.sheets_total` → wire `sheetsTotal`
(int32, optional, on NestingResponse only) + typify regen + wire.rs both-way map.

**SPEC / CROSS-REPO DONE (2026-06-30):** the ONLY wire change is `sheetsTotal` on NestingResponse
(additive). Pushed to `gdtrp/cutl-schemas` branch `feat/cutl-160-sheets-total`, version bumped
1.0.0→1.1.0 in BOTH `asyncapi/jagua-rs.yaml` and `package.json`, **PR #1**
(github.com/gdtrp/cutl-schemas/pull/1, asyncapi validate 0 errors). cutl-schemas model = "nothing
committed; CI publishes on tag `vX.Y.Z`" (Java→GitHub Packages, Rust/Go→release tarballs) per its
CONSUMING.md — so issuing the version = merge PR #1 + tag `v1.1.0`. Backend/frontend handoff written:
`docs/cutl160_backend_handoff.md` (backend: bump dep 1.0.0→1.1.0, forward sheetsTotal; frontend:
determinate bar percent≈sheets/sheetsTotal).

Tests (all write SVGs to `jagua-utils/test_output/cutl160/<case>/` for visual check): `cutl160_grid`
(frame 200×295→27/sheet vs LBF 22, cardinal only), `cutl160_periodic` (byte-identical K sheets +
remainder), `cutl160_pairing` (triangle 0.872 density vs 0.667), `cutl160_mixed` (deterministic, 3
types), `cutl160_maxfit_consistency` (no page > cap), `cutl160_progress` (sheets_total_estimate set);
classify unit tests in classify.rs; `wire_contract_test::sheets_total_round_trips`. Synthetic fixtures
in `jagua-sqs-processor/tests/testdata/cutl160/` (rect/frame/triangle/highfill/mixed_a/b/c). Measured
speedup: e2e `test_single_page...` 0.09s (was LBF). Note `cutl160_mixed` ~29s (renders many pages ×2).

**CROSS-REPO FOLLOW-UPS (not done here):** (1) mirror spec additions (`sheetsTotal` on NestingResponse,
optional `packingMode` on NestingRequest) into `gdtrp/cutl-schemas` + regen Java (`npm run gen:async`)
— the vendored `asyncapi/jagua-rs.yaml` is git-ignored so my edit is local-only. (2) frontend: render
determinate bar from `sheets`/`sheetsTotal`; consume the now-consistent max-per-sheet. (3) confirm
identical-page S3-URL reuse for the blank-preview (#3 «проблемы с прогрузкой») — not yet verified in
processor upload path. (4) WS-5 "1+3" co-pack of different types on dominant sheets left as future
(current mixed only co-packs the remainder). (5) SingleIrregular periodic-LBF stencil not done
(irregular single parts still full LBF).

Related: [[project_maxfit_parallel_budget]] (current adaptive strategy internals), [[project_offcut_feature]],
[[project_grain_direction]], [[project_asyncapi_codegen]] (wire/spec codegen for WS-6).
