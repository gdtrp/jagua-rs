# SVG generator defect — top-left fillet folds back onto the top edge

**Severity:** breaks nesting (self-intersecting outer polygon, rejected by jagua-rs ≥ v0.7.2)
**Scope:** the DXF → SVG generator that produces `result.svg`. **Not** a nesting-engine bug.

## Affected request

| Field | Value |
|---|---|
| `itemId` | `00d77007-adea-4322-95e8-a158ed1e6f71` |
| `dxfUrl` | `…/project_part_dxf/00d77007-adea-4322-95e8-a158ed1e6f71/project_part_dxf.dxf` |
| generated SVG | `…/calculation/2f182c22-95b9-4103-9248-2fc00306bb48/result.svg` |
| correlation_id | `b9f7a886-4b61-40f1-b197-1aed958bb9d7` |
| params | width 1500, height 3000, rotations 4, spacing 2.0, maxFit true |

## Symptom

The nesting engine fails with:

```
Simple polygon contains intersecting edges 0 and 2:
[Point(117.0, 70.5), Point(-117.5746, 70.5), Point(-119.35766, 69.35511),
 Point(-117.0, 70.5), Point(-120.0, -67.5), ...]
```

## Root cause

The part is a ~240 × 141 rounded rectangle (fillet radius ≈ 3) with a bottom tab.
**Three of the four outer corners are correct; only the top-left fillet is broken.**

The top edge runs along `y = 70.5`. The top-left fillet's end point is emitted **back
on the top edge, to the right of where the fillet started** — instead of on the left
edge. This makes the fillet a zero-width fold: edge `pt2→pt3` crosses edge `pt0→pt1`.

### The generated SVG (full file)

```svg
<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-147.10379090998543 -91.26160007367899 267.10379090998543 161.761600073679">
  <path d="M116.99999999999827,70.49999999999501 L-117.57460033234875,70.50000000000001 L-119.35766178233175,69.35510940921101 L-117.0000000000001,70.49999999999993 L-120.00000000000172,-67.49999999999997 L-120.00000000000175,-68.074600332348 L-119.91581370408993,-68.2057127199091 L-117.00000000000175,-70.49999999999999 L-57.039314822834754,-91.26160007367899 L117,-70.50000000000003 L119.35766178233027,-69.35510940920899 L119.99999999999827,-68.94266878271098 L120,67.4999999999935 L119.91581370408827,68.20571271991001 L119.75595842979627,69.54573961138101 L118.85510940920915,69.85766178233088 L116.99999999999827,70.50000000000001 L116.99999999999827,70.49999999999501 Z M-68.35,50.75 L-78.85,50.75 L-78.85,53.25 L-68.35,53.25 L-68.35,50.75 Z M74.28,-53.25 L63.78,-53.25 L63.78,-50.75 L74.28,-50.75 L74.28,-53.25 Z M74.28,50.75 L63.78,50.75 L63.78,53.25 L74.28,53.25 L74.28,50.75 Z M-78.85,-50.75 L-68.35,-50.75 L-68.35,-53.25 L-78.85,-53.25 L-78.85,-50.75 Z M81.5,40 L101,40 L101,-40 L81.5,-40 L81.5,40 Z M-81.5,-40 L-101,-40 L-101,40 L-81.5,40 L-81.5,-40 Z ... (circular holes, correctly tessellated, omitted) ..." fill="#000" stroke="none" fill-rule="nonzero"/>
</svg>
```

### The broken segment (outer subpath, top-left corner)

```
M  117.0,      70.5       ← top-right corner          (pt0)
L -117.5746,   70.5       ← top edge end              (pt1)
L -119.358,    69.355     ← single fillet point       (pt2)
L -117.0,      70.5       ← BACK ON THE TOP EDGE  ❌   (pt3)
L -120.0,     -67.5       ← jumps straight to bottom-left (pt4)
...
```

`pt3 = (-117.0, 70.5)` lies on the line `y = 70.5` between `pt0` and `pt1`, so edge
`pt2→pt3` intersects edge `pt0→pt1`. Also note the "left edge" `pt3→pt4` is the slanted
segment `(-117, 70.5) → (-120, -67.5)` instead of the vertical line `x = -120`.

### Per-corner comparison (the bug is localized)

| Corner | Edge tangent points (entry → exit) | Status |
|---|---|---|
| Bottom-left  | left `(-120, -67.5)` → bottom `(-117, -70.5)` | ✅ correct |
| Bottom-right | bottom `(117, -70.5)` → right `(120, -68.9…)` | ✅ correct |
| Top-right    | right `(120, 67.5)` → top `(117, 70.5)`       | ✅ correct |
| **Top-left** | top `(-117.57, 70.5)` → **`(-117.0, 70.5)`**  | ❌ ends back on top edge; left-edge tangent point `≈ (-120, 67.5)` is never emitted |

(The circular holes later in the same path are tessellated with ~64 points each and are
fine — so the generator's general arc handling works; this is specific to the top-left
outer fillet.)

## Expected geometry

The top-left fillet should be the mirror of the (correct) top-right corner:

```
L -117.0,   70.5     ← leave the top edge
L -118.86,  69.86    ← fillet arc (mirror of top-right)
L -119.76,  69.55
L -119.92,  68.21
L -120.0,   67.5     ← land on the LEFT edge
L -120.0,  -67.5     ← straight down the vertical left edge
```

## Likely fix location

The top-left (second-quadrant) fillet's two tangent/endpoint coordinates appear
swapped, or its arc sweep direction / quadrant sign is wrong — a corner-specific
error, since the other three quadrants are correct. Check the corresponding
fillet / `ARC` entity in the source DXF and the arc-endpoint / sweep-direction math
for the top-left case. The fillet is also under-tessellated there (1 intermediate
point vs the proper arc elsewhere).

## Verification checklist after the generator fix

- [ ] Outer path has no self-intersections (no vertex lands on a non-adjacent edge).
- [ ] Left edge is vertical at `x ≈ -120`, from `y ≈ 67.5` down to `y ≈ -67.5`.
- [ ] Top-left corner is the geometric mirror of the top-right corner.

## Note on the nesting side

Older jagua-rs silently tolerated this fold. The strict self-intersection check was
introduced in the **v0.7.2 upstream sync** (`SPolygon::new`), which is why the same
SVG worked before that sync and fails after. As defense-in-depth, jagua-utils now
runs `sanitize_polygon()` to repair such rings before handing them to jagua-rs, but
the correct fix is upstream in the SVG generator.
