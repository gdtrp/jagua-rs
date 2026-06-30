# RFC: CUTL-160 — Nesting Quality & UX Overhaul

| | |
|---|---|
| **Ticket** | [CUTL-160 — Оптимизация модуля раскроя](https://tracker.yandex.ru/CUTL-160) |
| **Status** | Draft — for review |
| **Author** | Boris Varshavsky (jagua-rs) |
| **Affected crates** | `jagua-utils`, `jagua-sqs-processor` (per repo constraint, `jagua-rs`/`lbf` stay untouched) |
| **Cross-repo** | `gdtrp/cutl-schemas` (AsyncAPI), `cutl-backend`, frontend (progress UI) |
| **Evidence** | QA report by Никита Зайцев + screenshots in [`docs/bugfix/`](../bugfix/) |

---

## 1. Problem statement

QA exercised the nesting module across a range of real customer inputs and found that **result quality, runtime, and feedback are all poor on the common, easy cases** — large quantities of identical or near-rectangular parts on rectangular sheets. The reporter's closing remark is the thesis of this RFC:

> «Основная суть в том, что одним алгоритмом мы качественного результата не добьёмся. Нам нужно как можно больше выделить частных случаев и сделать для них быстрые расчёты, не доводя до сложного анализа.»
> *(One general algorithm will not give quality results. We need to carve out as many special cases as possible and do fast computations for them instead of always falling into the expensive general analysis.)*

That is exactly right, and it matches the architecture. Today **every** request — one rectangle or a hundred irregular parts — is routed through the same general-purpose irregular nester (`AdaptiveNestingStrategy` → jagua-rs BPP + LBF, a stochastic single-pass heuristic that is explicitly *"not for production use — solution quality is chaotic by nature"*). The general nester is the right tool for genuinely irregular mixed parts and the wrong tool for the bread-and-butter cases that dominate production traffic.

---

## 2. Root-cause analysis

Almost every complaint traces back to **one tool applied to all inputs**, plus **no determinate progress signal**:

| # | QA observation (screenshot) | Root cause in current code |
|---|---|---|
| 1 | Indeterminate progress bar; «не известно когда закончится», user gives up after ~2 min (`image (3)`, `image (4)`) | Worker streams `is_improvement` messages but emits **no progress %, no ETA, no sheets-total estimate**. With global LBF the total sheet count is unknown until the run ends, so the frontend *cannot* draw a determinate bar. |
| 2 | «Результат никуда не годится» — rectangular frames scattered at random small angles, 66% density (`image (5)` 178 sheets/32% waste, `image (6)` d=66%) | Default `amountOfRotations = 8` lets near-rectangles tilt; LBF places them stochastically. No grid fast path for axis-aligned parts. A trivial grid gives 24/sheet where LBF gives 22. |
| 3 / 6 | «Каждый лист считается отдельно… 13 разных раскроев вместо 12 одинаковых + остаток»; 29 sheets all different (`image (7)`); 6 different sheets though sheet 1 (32 pcs) appeared instantly (video `110`) | BPP/LBF packs all bins as one global stochastic layout. **Nothing forces a repeating sheet pattern**; there is no per-bin grouping or "repeat stencil" concept anywhere in the worker (confirmed in `processor.rs`/`wire.rs`). Each bin gets an independent, visually different packing, and the optimizer keeps grinding sheets that can't beat sheet 1. |
| 4 | **Bug:** «Максимальное количество деталей на листе: 44», but «Лист 3 из 4 → Количество 45»; preview blank (`image (10)`) | The "max per sheet" number comes from a **separate `maxFit` single-sheet computation** (deterministic → 44), while the actual nest is a **different code path** (LBF BPP) that packed 45 on one bin. The two paths disagree → an impossible label (`actual > max`). Blank preview = the image-loading problem noted in #3 (178 large distinct SVGs, S3 latency). |
| 5 | Half-bbox parts (right-triangle «косынка» 300×100, area = ½ bbox, `image (8)`) pack at 66.7% with a misaligned bottom row wasting space (`image (11)`) | No pairing fast path. Two right triangles tile a rectangle at ~100%; LBF reaches only 2/3 and shifts the last row. |
| 7 | «Нет смысла крутить деталь на каждой итерации — разложить в одном положении по максимуму, потом повернуть» | LBF samples rotations per-placement; there is no deterministic "fill main orientation, then fill the leftover strip rotated" guillotine pass. |

**Conclusion:** the fix is not "tune LBF harder." It is to **classify the request and route the easy cases to fast, deterministic special-case packers**, reserving LBF for the irregular remainder — and to **report determinate progress** so the wait is legible.

---

## 3. Goals / non-goals

**Goals**
- G1. Identical/near-rectangular bulk inputs produce **K identical full sheets + 1 remainder sheet**, not K different sheets (#2, #3, #6).
- G2. Axis-aligned rectangles and pairable shapes pack at **≈ grid density** (#2, #5), deterministically.
- G3. **Determinate** progress: % done, sheets done / total estimate, elapsed / budget (#1).
- G4. The "max parts per sheet" label can **never** be exceeded by an actual sheet (#4).
- G5. Big runs finish in **seconds** when the answer is a repeated stencil (#6), not minutes.
- G6. **Regression test per QA case** (SVG fixtures supplied by reporter).

**Non-goals**
- Touching `jagua-rs`/`lbf` source (constraint). All new packers live in `jagua-utils`.
- Replacing LBF — it remains the fallback for genuinely irregular mixed parts and for the remainder sheet.
- Frontend implementation (this RFC specifies the wire contract; UI is cross-repo).

---

## 4. Proposed architecture

A thin **classifier + router** in `jagua-utils` in front of the existing strategy. The router picks the cheapest packer that fits the input and falls back to today's `AdaptiveNestingStrategy` for the general case.

```
                         ┌─────────────────────────────────────────────┐
 request ──► classify ──►│  SingleHighFill   → grid (1–2/sheet, instant)│
 (parts, bin, spacing)   │  SingleRectangle  → grid fast path           │──► nest_periodic
                         │  SinglePairable   → pair→rectangle→grid      │   (stencil ×K + remainder)
                         │  SingleIrregular  → LBF stencil ×K + remainder│
                         │  MixedFewTypes    → rect pre-analysis + group │──► per-group periodic
                         │  General          → AdaptiveNestingStrategy   │   (today's behaviour)
                         └─────────────────────────────────────────────┘
```

Two reusable primitives underpin everything:

- **Grid packer** (`grid.rs`): closed-form placement of an axis-aligned rectangle on a sheet, two-orientation guillotine (fill main orientation to the max, then fill the leftover right/bottom strips with the 90°-rotated part — QA #1, #7). Pure geometry, no engine, microseconds, deterministic.
- **Periodic packer** (`periodic.rs`): given a single-sheet *stencil* (placements that fit one sheet) and a requested quantity `Q` with `cap` parts/stencil, emit `K = ⌊Q/cap⌋` **byte-identical** sheets + one remainder sheet packed once. Page-SVG cloning already exists (`skip_middle` in `adaptive.rs:347`); this generalises it.

---

## 5. Workstreams

Each is an independently shippable CUTL-160 sub-task. Files are where the work lands (all `jagua-utils` unless noted).

### WS-1 — Request classifier + router skeleton
- **What:** `svg_nesting/classify.rs` → `classify(parts, bin_w, bin_h, spacing) -> PackingClass`. Cheap geometry only: per-part `area / bbox_area` (rectangularity), `part_bbox_area / bin_area` (fill), part-type count, pairability test.
- **Route:** new entry `nest_auto(...)` wrapping the existing `NestingStrategy::nest`; processor calls `nest_auto` instead of branching on `max_fit`. Optional request hint `packingMode: AUTO|GRID|PERIODIC|GENERAL` (default `AUTO`) for override/debugging.
- **Acceptance:** classification unit tests over the QA fixtures land in the expected bucket; `General` path is byte-identical to today.

### WS-2 — Rectangular grid fast path *(fixes #2)*
- **What:** `svg_nesting/grid.rs`. For rect `w×h`, sheet `W×H`, spacing `s`:
  `cols = ⌊(W+s)/(w+s)⌋`, `rows = ⌊(H+s)/(h+s)⌋`, plus the 90°-swapped grid; keep the better, then a guillotine pass fills the leftover right strip `W−cols·(w+s)` and bottom strip with the rotated part. Emits `PlacedPartInfo` directly — **no LBF**.
- **Why it matters:** frames in `image (5)/(6)` (980×2000, part 200×295) → `⌊980/200⌋·⌊2000/295⌋ = 4·6 = 24`/sheet vs LBF's 22, at ~95% density vs 66%, instantly, every sheet identical.
- **Acceptance:** grid count ≥ LBF count on rectangle fixtures; density ≥ 0.9 (modulo spacing); 0 rotations other than {0°,90°}.

### WS-3 — Pattern-and-repeat / periodic packing *(fixes #3, #6, structurally fixes #4)* — **core deliverable**
- **What:** `svg_nesting/periodic.rs` → `nest_periodic(stencil_fn, qty)`. Compute the stencil **once** (grid for rectangles, bounded LBF `max_fit` for irregular), measure `cap`, then materialise `K` identical sheets (clone SVG + placements — cheap) + one remainder sheet packed once.
- **Why it matters:** turns 29/178/6 different sheets into "`K` identical + 1 remainder". Runtime collapses (stencil is computed once; the rest is replication — exactly the reporter's "первый лист получился почти сразу, остальные не имели смысла"). And **"max per sheet" = stencil `cap`**, which every full sheet equals and the remainder never exceeds → bug #4 is impossible by construction.
- **Trade-off:** gives up the marginal efficiency of a globally-optimised *last* sheet in exchange for determinism, speed, and manufacturability (identical sheets are easier to cut and track) — which is precisely what QA asks for.
- **Acceptance:** for a single part type at qty `Q`, output = `⌊Q/cap⌋` identical pages (assert SVG equality) + 1 remainder; total placed = `Q`; wall-clock dominated by one stencil computation.

### WS-4 — Pairing fast path for half-bbox parts *(fixes #5)*
- **What:** detect `area ≈ ½·bbox_area` parts whose union with their 180°-rotation ≈ the bbox rectangle (right triangles, some trapezoids). Treat the **pair** as a composite rectangle, grid-pack the composite (WS-2), then split each cell back into two oppositely-oriented parts. Handle odd counts (last part placed alone).
- **Why it matters:** triangles in `image (8)/(11)` go from 66.7% to ~95%+, no misaligned last row.
- **Acceptance:** triangle fixture density ≥ 0.9; placement count ≈ `2·grid(composite)`.

### WS-5 — Mixed-parts rectangle pre-analysis & grouping *(fixes #3 variants, #4 layout intent)*
- **What:** reduce each part type to its bbox; run a fast deterministic rectangle pack (skyline/maxrects, ms-scale) purely as a **decision heuristic** to choose grouping: either (a) nest each type periodically then combine leftovers on a final mixed sheet, or (b) build a multi-type stencil and repeat it for the min-count type, recompute remainder (the reporter's "1+3" scheme). Actual placement still goes through grid/LBF for fidelity.
- **Note:** highest complexity → last phase. Ships after WS-1..4 prove the routing.
- **Acceptance:** on a 2–3 type fixture, fewer sheets and ≥ today's density; grouping decision is deterministic.

### WS-6 — Determinate progress / ETA contract *(fixes #1)*
- **What:** add an additive `progress` block to `SqsNestingResponse`: `{ phase, partsDone, partsTotalEst, sheetsDone, sheetsTotalEst, percent, elapsedSecs, budgetSecs }`. Emit it periodically (not only on improvement). With WS-3, `sheetsTotalEst` is **known right after the stencil**, so the bar is determinate almost immediately; `elapsedSecs/budgetSecs` (from `maxSeconds`) is the time-based fallback for the `General` path.
- **Where:** `processor.rs` improvement/heartbeat path (`processor.rs:1106`, `1141`); spec amendment in `cutl-schemas` (keep schemas **flat**, regenerate via `build.rs`/typify and `npm run gen:async`).
- **Cross-repo:** backend forwards the block; frontend renders a determinate bar. Flagged, not in scope here.
- **Acceptance:** progress messages carry monotonic `percent`; `sheetsTotalEst` set before the first full sheet on periodic paths.

### WS-7 — Bug: "max per sheet" vs actual + blank preview *(fixes #4)*
- **What:** (a) **structural** — once "max per sheet" and the real nest both come from the periodic stencil (WS-3), `actual ≤ max` always holds; remove the independent `maxFit` estimate as the source of the label. (b) **blank preview / «проблемы с прогрузкой»** — periodic emits ≤ 2 *distinct* SVGs (stencil + remainder), so identical pages should **share one S3 object/URL** instead of uploading 178 copies; verify upload/caching in `processor.rs` page-upload path.
- **Open question:** confirm exactly where the frontend/backend reads "max per sheet" (cross-repo) to ensure the unified value is consumed.
- **Acceptance:** no sheet reports a count > the stated max on any fixture; identical pages reuse one URL.

### WS-8 — Rotation policy default *(supports #2, #1, #7)*
- **What:** for rectangle-like / fast-path parts, default the rotation set to cardinal `{0°, 90°}` instead of `amountOfRotations = 8` (`wire.rs:70`). Free-angle rotation stays available for irregular parts and via per-part `allowedRotations` (grain feature already present). This removes the random tilt visible in `image (6)`.
- **Acceptance:** rectangle/grid outputs contain only {0°,90°} placements.

---

## 6. Wire / API changes (all additive, backward-compatible)

- **Request (optional):** `packingMode: AUTO|GRID|PERIODIC|GENERAL` (default `AUTO`).
- **Response (optional):** `progress` block (WS-6).
- Both modelled as **flat** schemas in `cutl-schemas/asyncapi/jagua-rs.yaml` (no `oneOf`/discriminator — see codegen constraints), regenerated through typify (`build.rs`) and Java (`npm run gen:async`). Existing requests behave exactly as today (`AUTO` + absent progress).

---

## 7. Sequencing

| Phase | Workstreams | Outcome |
|---|---|---|
| **0 — quick wins** | WS-8, WS-6 (scaffolding), WS-7 investigation | Determinate-ish bar, no more random tilt, bug scoped |
| **1 — rectangles** | WS-1, WS-2 | Frames/rectangles pack at grid density instantly (#2) |
| **2 — periodicity** | WS-3 | Identical sheets + remainder; runtime collapses; #4 gone (#3, #6) |
| **3 — pairing** | WS-4 | Triangles/half-bbox at ~95% (#5) |
| **4 — mixed** | WS-5 | Multi-type grouping (#3 variants, #4 intent) |

Phases 1–2 deliver the bulk of the visible quality win and can ship before 3–4.

---

## 8. Test coverage (per QA case — fixtures supplied by reporter)

Reporter will provide the SVGs + full request data; **each case gets a named fixture and assertions** in `jagua-utils/tests/cutl160_*.rs` (fixtures under `jagua-sqs-processor/tests/testdata/`). Matrix:

| Case | Source | Fixture (TBD from reporter) | Key assertions |
|---|---|---|---|
| C1 frames bulk | `image (5)/(6)` 980×2000, СЭ-0018 200×295, qty 4000 | `cutl160_frames_980x2000.svg` | grid path chosen; ≥ 24/sheet; only {0°,90°}; all full sheets SVG-identical |
| C2 tall-thin bulk | `image (7)` 29 sheets | `cutl160_tall_thin.svg` | periodic; `⌊Q/cap⌋` identical sheets + 1 remainder; total placed = Q |
| C3 high-fill | suggestion #1 (part > 0.95 sheet) | `cutl160_highfill.svg` | classified `SingleHighFill`; instant; both orientations checked |
| C4 triangle/косынка | `image (8)/(11)` 300×100, area ½ bbox, qty 24 | `cutl160_triangle_300x100.svg` | pairing path; density ≥ 0.9; no misaligned last row |
| C5 max-per-sheet bug | `image (10)` 44 vs 45 | reuse C1/C2 data | no sheet count > stated max; label = stencil cap |
| C6 mixed types | `image (5)` "3 детали, 6600 шт" | `cutl160_mixed_*.svg` | grouping deterministic; sheets ≤ today; density ≥ today |
| C7 runtime/early-stop | video `110` (32/sheet instantly) | reuse C2 | wall-clock ≈ one stencil computation; not minutes |
| C8 progress | `image (3)` | any bulk case | progress messages monotonic; `sheetsTotalEst` set before first full sheet |

Existing regression tests (`tests/adaptive_strategy_test.rs`, `tests/offcut_detection_test.rs`) must stay green — the `General` path is unchanged.

---

## 9. Risks & open questions

- **R1.** Periodic trades last-sheet optimality for determinism. *Mitigation:* the remainder sheet is still packed properly; net waste change is sub-percent and the manufacturability/UX win is large. Confirm with reporter this trade-off is acceptable (it matches the ticket).
- **R2.** Rectangularity / pairability thresholds need tuning to avoid mis-routing concave parts into the grid path. *Mitigation:* conservative thresholds + always-correct fallback to LBF; the classifier never *worsens* a case it misroutes because the fast paths self-validate (a part that doesn't actually grid-fit falls through).
- **R3.** "Max per sheet" and blank preview live partly in cutl-backend/frontend. *Open:* locate the exact read sites (WS-7) before claiming the bug closed end-to-end.
- **R4.** Spec changes touch three repos (cutl-schemas → jagua-rs worker + cutl-backend Java). Keep schemas flat; regenerate and commit per-repo (documented in the codegen runbook).

---

## 10. Appendix — screenshot index

Stored in [`docs/bugfix/`](../bugfix/), ordered as in the ticket:

| File | Tracker attach | Shows |
|---|---|---|
| `image (3).png` | 100 | indeterminate progress bar |
| `image (4).png` | 101/102 | "started, now waiting" screen |
| `image (5).png` | 103 | 6600 pcs → 178 sheets / 32% waste, 22/sheet (frames) |
| `image (6).png` | 104 | rectangles scattered at random angles, d=66% |
| `image (7).png` | 105 | 29 sheets, all different layouts |
| `image (8).png` | 106 | right-triangle part, area = ½ bbox |
| `image (9)/(10).png` | 107/108 | **bug:** max/sheet 44 but a sheet has 45; blank preview |
| `image (11).png` | 109 | triangles at 66.7%, misaligned bottom row |
| (video) | 110 | 6 different sheets; sheet 1 (32 pcs) instant |
