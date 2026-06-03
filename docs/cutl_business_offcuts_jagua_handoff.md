# Business Offcuts — jagua-rs Handoff

| Field      | Value                                                                  |
|------------|------------------------------------------------------------------------|
| Status     | Contract frozen — **backend lands first** (fields optional), then jagua |
| Repo       | `jagua-rs` (Rust SQS worker; sibling checkout)                          |
| Source     | `cutl_business_offcuts_rfc.md`, `cutl_business_offcuts_tasks.md` (CUTL-112) |
| Components | JG-OFF-1 … JG-OFF-3                                                     |

Self-contained. The chosen architecture (RFC §4 "Option C") is: **offcut
detection runs here**, as a side-output of nesting — jagua already holds the
placements, the bin geometry and the `CDEngine` quadtree, so free-space
scanning is a few hundred lines, no new SQS topic, no extra hop.

---

## 1. Context

After parts are nested, the unused sheet area is reusable stock if a piece is
big enough. Each material declares a minimum reusable size on the backend; the
backend forwards that as an **`offcutPolicy`** on the existing nesting request.
jagua scans the final layout for free rectangles (and, when asked, polygons),
respecting an optional kerf inset, and returns them per page on the existing
nesting response. Phase 3 of the feature also asks jagua to accept extra
variable-size bins so a later nesting can pack into stored offcuts.

**Hard requirement: full backwards compatibility.** Every field below is
optional. When `offcutPolicy` is absent (every caller today, and the backend
on day one), jagua must produce a **byte-identical** response to the current
build. This lets the backend deploy before the new jagua image with no flag
day.

Wire format note: the backend (Jackson, default naming) serializes
**camelCase** JSON — `correlationId`, `binWidth`, `amountOfRotations`. Use
`#[serde(rename_all = "camelCase")]` (or explicit renames) on the new structs
so `minOffcutWidthMm`, `extraBins`, `binId`, `pageIndex` match the wire exactly.

---

## 2. SQS contract changes

Queues are unchanged (`nesting-queue-name` in, `nesting-response-queue-name`
out). Only the request/response payloads gain optional fields.

### 2.1 Request — backend → jagua (adds `offcutPolicy`, `extraBins`)

Existing fields unchanged (`correlationId`, `binWidth`, `binHeight`,
`spacing`, `amountOfRotations`, `cancelled`, `maxFit`, `parts[]`).

```jsonc
{
  "correlationId": "…",
  "binWidth": 2000.0, "binHeight": 1000.0,
  "spacing": 5.0, "amountOfRotations": 4,
  "parts": [ { "itemId": "…", "svgUrl": "s3://…", "amountOfParts": 20 } ],

  // NEW — JG-OFF-2. Absent ⇒ skip detection, identical to today.
  "offcutPolicy": {
    "minOffcutWidthMm": 200,
    "minOffcutHeightMm": 200,
    "shape": "RECTANGLE",          // "RECTANGLE" | "QUADRILATERAL"
    "kerfMm": 0.0                   // optional; 0/absent ⇒ no inset
  },

  // NEW — JG-OFF-3. Absent ⇒ no extra bins (today's behaviour).
  "extraBins": [
    { "binId": "offcut-uuid-1", "width": 800.0, "height": 1000.0 }
  ]
}
```

### 2.2 Response — jagua → backend (adds `pages[].offcuts`, placement `binId`)

Existing fields unchanged (`correlationId`, `firstPageSvgUrl`,
`lastPageSvgUrl`, `sheets`, `pageSvgUrls[]`, `partsPlaced`, `utilisation`,
`improvement`, `final`, `errorMessage`, `timestamp`, and each
`pages[]`'s `pageIndex`/`utilisation`/`svgUrl`/`partsPlaced`/`placements[]`).

```jsonc
{
  "correlationId": "…",
  "pages": [
    {
      "pageIndex": 0,
      "utilisation": 0.42,
      "svgUrl": "s3://…/page-0.svg",
      "placements": [
        { "itemId": "…", "partIndex": 0, "x": 302.4, "y": 459.2,
          "rotation": 90.0, "centroidX": 0, "centroidY": 0,
          "binId": null }              // NEW (JG-OFF-3): which bin this landed
                                       // in; null/"main" = standard sheet,
                                       // else the extraBins[].binId
      ],
      // NEW (JG-OFF-2). Empty [] = scan ran, nothing ≥ threshold.
      "offcuts": [
        { "kind": "RECT", "x": 1200.0, "y": 0.0, "width": 800.0, "height": 1000.0 },
        { "kind": "POLY", "vertices": [ {"x":0,"y":0},{"x":300,"y":0},{"x":150,"y":250} ] }
      ]
    }
  ],
  "improvement": false,
  "final": true
}
```

- `offcuts` is populated **only on the final layout** (`improvement=false`).
  For `improvement=true` intermediate messages it must be `[]` — don't burn
  CPU on layouts that get superseded.
- Coordinates: mm, bin origin bottom-left, same frame as `placements`.
- `kind=RECT` ⇒ `x,y,width,height`; `kind=QUADRILATERAL`/`POLY` ⇒ `vertices`
  (closed, CCW).

---

## 3. Algorithm spec (RFC §7)

**Rectangle path (`shape=RECTANGLE`):**
1. Union bounding box of `Layout.placed_items` (each item's `SPolygon` +
   applied transform).
2. Bin is `[0,binWidth]×[0,binHeight]`; subtract the union bbox → ≤4 candidate
   strips (top/bottom/left/right).
3. Reject a strip a straggler part pokes into via `CDEngine.collides(rect)`.
4. Inset every surviving candidate by `kerfMm` on each side (if set).
5. Drop candidates with `width < minOffcutWidthMm` or
   `height < minOffcutHeightMm`. The rest are the offcuts.

**Polygon path (`shape=QUADRILATERAL`):**
1. Convex hull of placed-part polygons — `geo::ConvexHull`.
2. Simplify with `geo::Simplify` (Ramer–Douglas–Peucker) to drop staircase
   steps from small parts.
3. `geo::BooleanOps::difference` of the bin polygon minus the simplified hull
   → arbitrary quadrilateral offcuts; apply kerf inset; threshold by min size
   (bounding-box of each piece ≥ min dims).

**Extra bins (`extraBins`):** treat each as an additional variable-size bin in
the same request (the engine already supports per-request variable bins). Tag
every placement with the `binId` of the bin it landed in so the backend can
detect an offcut bin was consumed.

---

## 4. Task breakdown

### JG-OFF-1 — core free-space detection (~4d). Depends: none.
**Files:** new `jagua-rs/src/free_space/`; `jagua-rs/Cargo.toml`
(`offcut_detection` feature); `jagua-utils/src/svg_nesting/svg_generation.rs`
(`PageResult`).
All geometry above (rect + polygon + kerf) behind the `offcut_detection`
cargo feature so the core crate pays nothing when off. Expose
`offcuts: Vec<Offcut>` (`Rect` | `Poly`) on `PageResult`, populated **only on
the final layout**.
**Acceptance:** unit tests — corner placement → strips; full sheet → 0;
straggler rejection; threshold discard; RDP collapses staircase edges; kerf
shrinks each side. Crate builds with and without the feature; intermediate
`PageResult` has empty `offcuts`.

### JG-OFF-2 — request/response contract + deploy (~1d). Depends: JG-OFF-1.
**Files:** `jagua-sqs-processor/src/processor.rs` (`SqsNestingRequest`,
`SqsNestingResponse`); ECR image.
Add optional `offcutPolicy` to the request, plumb into the strategy call;
serialize per-page `offcuts` onto the response. Absent policy ⇒ scan skipped ⇒
byte-identical response. Build + push the ECR image; deploy to
`CUTLStagingJaguaSqsProcessorService`.
**Acceptance:** request without `offcutPolicy` ⇒ identical response vs. today
(diff a recorded fixture); with a policy ⇒ `offcuts` per page; staging image
deployed and smoke-tested with one real nesting.

### JG-OFF-3 — extra variable-size bins (~1d). Depends: JG-OFF-2.
**Files:** `jagua-sqs-processor/src/processor.rs`.
Accept optional `extraBins`; nest into them when beneficial; tag each response
placement with its originating `binId`. Absent ⇒ unchanged behaviour.
**Acceptance:** request with extra bins packs into them when it helps;
placements carry `binId`; absent field ⇒ byte-identical to today; staging
deployed.

---

## 5. Deployment

Backend ships first with all fields optional. Then push the jagua ECR image
(JG-OFF-2 → JG-OFF-3) to `CUTLStagingJaguaSqsProcessorService`. No flag day:
the contract is additive and absent fields reproduce current behaviour. Smoke
test: one real nesting with `offcutPolicy` set, confirm `offcuts` on the final
page and that an `offcutPolicy`-less request is byte-identical to the recorded
baseline.
