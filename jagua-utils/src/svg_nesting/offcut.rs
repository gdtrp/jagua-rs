//! Free-space ("offcut") detection.
//!
//! After parts are nested, the unused sheet area is reusable stock if a piece is big
//! enough. Given an [`OffcutPolicy`] (minimum reusable size, shape, optional kerf inset),
//! these functions scan a final [`LayoutSnapshot`] for free rectangles
//! (`shape = Rectangle`) or free polygons (`shape = Quadrilateral`) and return them per
//! page.
//!
//! Detection is opt-in: it only runs when a strategy has an [`OffcutPolicy`] set, so the
//! default (no policy) path does zero work and produces empty `offcuts`.
//!
//! See `docs/cutl_business_offcuts_jagua_handoff.md` for the wire contract and algorithm.

use geo::{BooleanOps, BoundingRect, Coord, LineString, MultiPolygon, Polygon, Simplify};
use jagua_rs::entities::LayoutSnapshot;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::probs::bpp::entities::BPSolution;
use serde::{Deserialize, Serialize};

use crate::svg_nesting::svg_generation::{
    NestingResult, PageResult, combine_svg_documents, overlay_offcuts_svg,
};

/// Ramer–Douglas–Peucker tolerance (mm) used to collapse the staircase edges that small
/// parts leave on the convex hull. The effective epsilon is `max(kerf, this)`.
const RDP_EPSILON_MM: f32 = 1.0;

/// Tolerance (mm) for treating coordinates as coincident during rectangle decomposition.
const COORD_EPS: f32 = 1e-3;

/// A 2D vertex on the offcut wire format (`{ "x": .., "y": .. }`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OffcutVertex {
    pub x: f32,
    pub y: f32,
}

/// A reusable free-space region detected after nesting.
///
/// Serializes with a `kind` tag per the handoff §2.2 wire contract:
/// `{"kind":"RECT","x":..,"y":..,"width":..,"height":..}` or
/// `{"kind":"POLY","vertices":[{"x":..,"y":..}], "holes":[[..]]}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Offcut {
    #[serde(rename = "RECT")]
    Rect {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
    #[serde(rename = "POLY")]
    Poly {
        /// Exterior ring, CCW, open (no duplicated closing vertex).
        vertices: Vec<OffcutVertex>,
        /// Interior rings (holes where placed parts sit), CW, open. Omitted from JSON when
        /// empty so a hole-free polygon matches the original `{vertices}` wire shape.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        holes: Vec<Vec<OffcutVertex>>,
    },
}

/// Which detection geometry to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OffcutShape {
    /// Axis-aligned rectangle strips (`"RECTANGLE"`).
    Rectangle,
    /// The rectangle decomposition merged into a single connected outline
    /// (`"RECTANGLE_MERGED"`): same coverage as `Rectangle` but reported as one rectilinear
    /// polygon (plus holes), so the kerf band wraps only the true material perimeter.
    RectangleMerged,
    /// Arbitrary quadrilateral / polygon offcuts (`"QUADRILATERAL"`).
    Quadrilateral,
}

/// Policy controlling offcut detection. Mirrors the request's `offcutPolicy`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OffcutPolicy {
    /// Minimum reusable width (mm); offcuts narrower than this are discarded.
    pub min_offcut_width_mm: f32,
    /// Minimum reusable height (mm); offcuts shorter than this are discarded.
    pub min_offcut_height_mm: f32,
    /// Detection geometry to run.
    pub shape: OffcutShape,
    /// Optional kerf inset (mm) applied to every side of each offcut. `0` ⇒ no inset.
    #[serde(default)]
    pub kerf_mm: f32,
}

/// Finalize a result with offcuts: compute them per page, draw them onto each page SVG,
/// and rebuild the combined SVG. Called only on the final result so intermediates stay
/// untouched.
pub(crate) fn apply_offcuts(
    result: &mut NestingResult,
    solution: &BPSolution,
    policy: &OffcutPolicy,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
) {
    write_page_offcuts(&mut result.pages, solution, policy, spacing);

    // Stroke sized to the sheet so the overlay is visible at any scale.
    let stroke = (bin_width.min(bin_height) * 0.002).max(0.5);
    let kerf = policy.kerf_mm;
    let patched: Vec<String> = result
        .page_svgs
        .iter()
        .enumerate()
        .map(|(i, bytes)| {
            let svg = String::from_utf8_lossy(bytes).into_owned();
            let offcuts = result.pages.get(i).map(|p| &p.offcuts[..]).unwrap_or(&[]);
            let bands = kerf_band_paths(offcuts, kerf);
            overlay_offcuts_svg(&svg, offcuts, &bands, stroke)
        })
        .collect();

    result.page_svgs = patched.iter().map(|s| s.clone().into_bytes()).collect();
    result.combined_svg = combine_svg_documents(&patched, bin_width, bin_height).into_bytes();
}

/// Populate `offcuts` on every page of a final result from its solution.
///
/// Pages and layout snapshots are both ordered by `container.id`, matching the order the
/// strategies build `PageResult`s, so they align 1:1.
pub(crate) fn write_page_offcuts(
    pages: &mut [PageResult],
    solution: &BPSolution,
    policy: &OffcutPolicy,
    spacing: f32,
) {
    let mut layouts: Vec<&LayoutSnapshot> = solution.layout_snapshots.values().collect();
    layouts.sort_by_key(|ls| ls.container.id);
    debug_assert_eq!(
        pages.len(),
        layouts.len(),
        "offcut finalize: page/layout count mismatch"
    );
    for (page, layout) in pages.iter_mut().zip(layouts.iter()) {
        page.offcuts = detect_offcuts_for_layout(layout, policy, spacing);
    }
}

/// Extract the bin bbox and placed-item geometry from one page and run detection.
pub(crate) fn detect_offcuts_for_layout(
    layout: &LayoutSnapshot,
    policy: &OffcutPolicy,
    spacing: f32,
) -> Vec<Offcut> {
    let bin_bbox = layout.container.outer_cd.bbox;
    // `PlacedItem.shape` is already world-space transformed.
    let polys: Vec<&SPolygon> = layout
        .placed_items
        .values()
        .map(|pi| pi.shape.as_ref())
        .collect();
    let bboxes: Vec<Rect> = polys.iter().map(|p| p.bbox).collect();
    detect_offcuts(bin_bbox, &bboxes, &polys, policy, spacing)
}

/// Dispatch on `policy.shape`.
///
/// `spacing` is the nesting separation (`min_item_separation`). The engine bakes half of it
/// into the geometry the detector sees — the sheet's collision box is deflated by `spacing/2`
/// and every part's collision shape is inflated by `spacing/2`. The detectors undo both so
/// offcuts span the true reusable material (touching the real part outlines and sheet edges)
/// rather than respecting the nesting spacing.
pub(crate) fn detect_offcuts(
    bin_bbox: Rect,
    placed_bboxes: &[Rect],
    placed_polys: &[&SPolygon],
    policy: &OffcutPolicy,
    spacing: f32,
) -> Vec<Offcut> {
    match policy.shape {
        OffcutShape::Rectangle => detect_rect_offcuts(bin_bbox, placed_bboxes, policy, spacing),
        OffcutShape::RectangleMerged => {
            merge_rect_offcuts(detect_rect_offcuts(bin_bbox, placed_bboxes, policy, spacing))
        }
        OffcutShape::Quadrilateral => detect_poly_offcuts(bin_bbox, placed_polys, policy, spacing),
    }
}

/// Union axis-aligned rectangle offcuts into connected rectilinear polygons (typically one).
/// The shared internal edges between adjacent rectangles dissolve, leaving a single outline
/// whose border is the real material boundary — so the kerf band wraps it once.
fn merge_rect_offcuts(rects: Vec<Offcut>) -> Vec<Offcut> {
    let mut acc: Option<MultiPolygon<f64>> = None;
    for o in &rects {
        if let Offcut::Rect {
            x,
            y,
            width,
            height,
        } = o
            && let Ok(r) = Rect::try_new(*x, *y, x + width, y + height)
        {
            let mp = MultiPolygon::new(vec![rect_to_geo_polygon(r)]);
            acc = Some(match acc {
                None => mp,
                Some(a) => a.union(&mp),
            });
        }
    }
    match acc {
        None => Vec::new(),
        // Simplify to drop the redundant colinear vertices the union leaves on straight runs.
        Some(mp) => mp
            .into_iter()
            .map(|p| p.simplify(&(COORD_EPS as f64)))
            .map(|p| geo_polygon_to_offcut(&p))
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// RECTANGLE path
// ---------------------------------------------------------------------------

/// Detect axis-aligned rectangular offcuts capturing the free space as a few large,
/// non-overlapping rectangles.
///
/// Free space = `bin − union(part bounding boxes)`, measured against the **true** sheet edges
/// and part outlines (the `spacing/2` collision offset is undone first). The free area is
/// decomposed greedily largest-rectangle-first so big reusable areas (e.g. the block below a
/// packed column) survive as whole rectangles instead of being sliced into thin sub-minimum
/// strips. Rectangles below the minimum size are dropped.
pub(crate) fn detect_rect_offcuts(
    bin_bbox: Rect,
    placed_bboxes: &[Rect],
    policy: &OffcutPolicy,
    spacing: f32,
) -> Vec<Offcut> {
    let half = (spacing / 2.0).max(0.0);
    // Undo the collision offset: grow the sheet back to its true edges and shrink each part
    // bbox back to its true outline so offcuts touch the real parts and walls.
    let bin = grow_rect(bin_bbox, half).unwrap_or(bin_bbox);
    let obstacles: Vec<Rect> = placed_bboxes
        .iter()
        .filter_map(|b| inset_rect(*b, half))
        .filter_map(|b| Rect::intersection(bin, b))
        .collect();

    maximal_free_rects(
        bin,
        &obstacles,
        policy.min_offcut_width_mm,
        policy.min_offcut_height_mm,
    )
    .into_iter()
    .map(|r| Offcut::Rect {
        x: r.x_min,
        y: r.y_min,
        width: r.width(),
        height: r.height(),
    })
    .collect()
}

/// Greedily extract non-overlapping maximal free rectangles. Repeatedly takes the
/// largest-area empty rectangle, emits it if it meets the minimum size, then marks it
/// occupied and repeats — yielding a small set of large rectangles rather than many slivers.
fn maximal_free_rects(bin: Rect, obstacles: &[Rect], min_w: f32, min_h: f32) -> Vec<Rect> {
    let mut obs = obstacles.to_vec();
    let mut out = Vec::new();
    while let Some(r) = largest_empty_rect(bin, &obs) {
        if r.width() < min_w || r.height() < min_h {
            break;
        }
        out.push(r);
        obs.push(r);
    }
    out
}

/// Find the maximum-area axis-aligned rectangle inside `bin` that overlaps no obstacle. A
/// maximal empty rectangle's sides always lie on obstacle/bin edges, so candidate edges are
/// the bin edges plus every obstacle edge; enumerate edge pairs and keep the largest empty one.
fn largest_empty_rect(bin: Rect, obstacles: &[Rect]) -> Option<Rect> {
    let mut xs = vec![bin.x_min, bin.x_max];
    let mut ys = vec![bin.y_min, bin.y_max];
    for o in obstacles {
        xs.push(o.x_min);
        xs.push(o.x_max);
        ys.push(o.y_min);
        ys.push(o.y_max);
    }
    sort_dedup(&mut xs);
    sort_dedup(&mut ys);

    let mut best: Option<Rect> = None;
    let mut best_area = COORD_EPS;
    for i in 0..xs.len() {
        for j in (i + 1)..xs.len() {
            let (x0, x1) = (xs[i], xs[j]);
            let w = x1 - x0;
            for a in 0..ys.len() {
                for b in (a + 1)..ys.len() {
                    let (y0, y1) = (ys[a], ys[b]);
                    let area = w * (y1 - y0);
                    if area <= best_area {
                        continue;
                    }
                    if rect_is_empty(x0, y0, x1, y1, obstacles)
                        && let Ok(r) = Rect::try_new(x0, y0, x1, y1)
                    {
                        best = Some(r);
                        best_area = area;
                    }
                }
            }
        }
    }
    best
}

/// `true` if no obstacle overlaps the open rectangle `(x0,y0)-(x1,y1)` with positive area.
fn rect_is_empty(x0: f32, y0: f32, x1: f32, y1: f32, obstacles: &[Rect]) -> bool {
    !obstacles.iter().any(|o| {
        o.x_min < x1 - COORD_EPS
            && o.x_max > x0 + COORD_EPS
            && o.y_min < y1 - COORD_EPS
            && o.y_max > y0 + COORD_EPS
    })
}

/// Sort ascending and drop near-duplicate coordinates.
fn sort_dedup(xs: &mut Vec<f32>) {
    xs.sort_by(|a, b| a.total_cmp(b));
    xs.dedup_by(|a, b| (*a - *b).abs() <= COORD_EPS);
}

/// Shrink a rect by `m` on every side. `m <= 0` is a no-op; `None` if it would collapse.
fn inset_rect(rect: Rect, m: f32) -> Option<Rect> {
    if m <= 0.0 {
        return Some(rect);
    }
    rect.resize_by(-m, -m)
}

/// Grow a rect by `m` on every side. `m <= 0` is a no-op.
fn grow_rect(rect: Rect, m: f32) -> Option<Rect> {
    if m <= 0.0 {
        return Some(rect);
    }
    rect.resize_by(m, m)
}

// ---------------------------------------------------------------------------
// QUADRILATERAL path
// ---------------------------------------------------------------------------

/// Detect polygonal offcuts capturing ALL free space.
///
/// Free space = `bin − union(part outlines)`, measured against the true sheet edges and part
/// outlines: the sheet is grown back by `spacing/2` and each part is deflated by `spacing/2`
/// to undo the collision offset. The boolean difference yields free regions that follow the
/// real, possibly concave, part edges and may contain holes where parts sit. Each region's
/// rings are simplified (RDP) and returned as an [`Offcut::Poly`] (exterior + holes).
pub(crate) fn detect_poly_offcuts(
    bin_bbox: Rect,
    placed_polys: &[&SPolygon],
    policy: &OffcutPolicy,
    spacing: f32,
) -> Vec<Offcut> {
    let half = (spacing / 2.0).max(0.0);
    let bin = grow_rect(bin_bbox, half).unwrap_or(bin_bbox);
    let bin_mp = MultiPolygon::new(vec![rect_to_geo_polygon(bin)]);

    // Union of all part outlines, each deflated by spacing/2 to recover its true (pre-inflate)
    // outline. Parts thinner than the separation deflate to nothing and are simply skipped.
    let occupied: Option<MultiPolygon<f64>> = placed_polys
        .iter()
        .map(|sp| {
            let poly = part_to_geo_polygon(sp);
            if half > 0.0 {
                geo_buffer::buffer_polygon(&poly, -(half as f64))
            } else {
                MultiPolygon::new(vec![poly])
            }
        })
        .fold(None, |acc, mp| {
            Some(match acc {
                None => mp,
                Some(a) => a.union(&mp),
            })
        });

    let free = match occupied {
        None => bin_mp,
        Some(occ) => bin_mp.difference(&occ),
    };

    let eps = RDP_EPSILON_MM as f64;
    free.into_iter()
        .map(|p| p.simplify(&eps))
        .filter(|p| piece_meets_threshold(p, policy))
        .map(|p| geo_polygon_to_offcut(&p))
        .collect()
}

/// Build a closed `geo` polygon from a placed part's world-space outline.
fn part_to_geo_polygon(sp: &SPolygon) -> Polygon<f64> {
    let mut coords: Vec<Coord<f64>> = sp
        .vertices
        .iter()
        .map(|v| Coord {
            x: v.0 as f64,
            y: v.1 as f64,
        })
        .collect();
    if let Some(first) = coords.first().copied() {
        coords.push(first); // close the ring
    }
    Polygon::new(LineString::from(coords), vec![])
}

/// Build a CCW `geo` polygon from a [`Rect`].
fn rect_to_geo_polygon(r: Rect) -> Polygon<f64> {
    let [bl, br, tr, tl] = r.corners();
    // `Rect::corners` yields [bottom-left, bottom-right, top-right, top-left] — CCW.
    let ring = LineString::from(vec![
        (bl.0 as f64, bl.1 as f64),
        (br.0 as f64, br.1 as f64),
        (tr.0 as f64, tr.1 as f64),
        (tl.0 as f64, tl.1 as f64),
    ]);
    Polygon::new(ring, vec![])
}

/// `true` if the polygon's bounding box meets the policy's minimum dimensions.
fn piece_meets_threshold(poly: &Polygon<f64>, policy: &OffcutPolicy) -> bool {
    match poly.bounding_rect() {
        Some(bb) => {
            bb.width() as f32 >= policy.min_offcut_width_mm
                && bb.height() as f32 >= policy.min_offcut_height_mm
        }
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Kerf band (visual only)
// ---------------------------------------------------------------------------

/// Build SVG path `d` strings — one per offcut — for the kerf band: the ring of width `kerf`
/// just inside each offcut, drawn as a shaded overlay so the cut allowance is visible while
/// the reported offcut still spans the full reusable material. Each path is the offcut
/// boundary with the kerf-inset usable interior punched out (even-odd). Empty when
/// `kerf <= 0`.
pub(crate) fn kerf_band_paths(offcuts: &[Offcut], kerf: f32) -> Vec<String> {
    if kerf <= 0.0 {
        return Vec::new();
    }
    offcuts
        .iter()
        .map(|o| kerf_band_path(o, kerf))
        .collect()
}

fn kerf_band_path(offcut: &Offcut, kerf: f32) -> String {
    match offcut {
        Offcut::Rect {
            x,
            y,
            width,
            height,
        } => {
            let outer = rect_ring_d(*x, *y, *width, *height);
            let (iw, ih) = (width - 2.0 * kerf, height - 2.0 * kerf);
            if iw > COORD_EPS && ih > COORD_EPS {
                // Outer rect + inset rect (hole) ⇒ even-odd renders just the band.
                format!("{outer} {}", rect_ring_d(x + kerf, y + kerf, iw, ih))
            } else {
                // Offcut is no wider than the kerf on a side: the whole thing is band.
                outer
            }
        }
        Offcut::Poly { vertices, holes } => {
            let mut d = verts_ring_d(vertices);
            for h in holes {
                d.push(' ');
                d.push_str(&verts_ring_d(h));
            }
            // Deflate the offcut by kerf to get the usable interior, punch it out as holes.
            let poly = offcut_poly_to_geo(vertices, holes);
            for ip in geo_buffer::buffer_polygon(&poly, -(kerf as f64)).iter() {
                d.push(' ');
                d.push_str(&geo_ring_d(ip.exterior()));
                for ih in ip.interiors() {
                    d.push(' ');
                    d.push_str(&geo_ring_d(ih));
                }
            }
            d
        }
    }
}

/// Closed rectangle ring as SVG path data.
fn rect_ring_d(x: f32, y: f32, w: f32, h: f32) -> String {
    format!(
        "M {x},{y} L {x1},{y} L {x1},{y1} L {x},{y1} Z",
        x1 = x + w,
        y1 = y + h
    )
}

/// Closed ring (from offcut vertices) as SVG path data.
fn verts_ring_d(verts: &[OffcutVertex]) -> String {
    let mut d = String::new();
    for (i, v) in verts.iter().enumerate() {
        d.push_str(if i == 0 { "M " } else { " L " });
        d.push_str(&format!("{},{}", v.x, v.y));
    }
    d.push_str(" Z");
    d
}

/// Closed ring (from a `geo` LineString) as SVG path data.
fn geo_ring_d(ring: &LineString<f64>) -> String {
    let mut d = String::new();
    for (i, c) in ring.coords().enumerate() {
        d.push_str(if i == 0 { "M " } else { " L " });
        d.push_str(&format!("{},{}", c.x as f32, c.y as f32));
    }
    d.push_str(" Z");
    d
}

/// Rebuild a `geo` polygon (exterior + holes) from an [`Offcut::Poly`]'s rings.
fn offcut_poly_to_geo(vertices: &[OffcutVertex], holes: &[Vec<OffcutVertex>]) -> Polygon<f64> {
    let to_ring = |vs: &[OffcutVertex]| {
        let mut c: Vec<Coord<f64>> = vs
            .iter()
            .map(|v| Coord {
                x: v.x as f64,
                y: v.y as f64,
            })
            .collect();
        if let Some(f) = c.first().copied() {
            c.push(f);
        }
        LineString::from(c)
    };
    Polygon::new(to_ring(vertices), holes.iter().map(|h| to_ring(h)).collect())
}

/// Convert a `geo` polygon to an [`Offcut::Poly`]: exterior ring (CCW) plus interior rings
/// (CW holes), each open (no duplicated closing vertex).
fn geo_polygon_to_offcut(poly: &Polygon<f64>) -> Offcut {
    let vertices = ring_to_vertices(poly.exterior(), true);
    let holes = poly
        .interiors()
        .iter()
        .map(|r| ring_to_vertices(r, false))
        .filter(|r| r.len() >= 3)
        .collect();
    Offcut::Poly { vertices, holes }
}

/// Convert a `geo` ring to an open vertex list with the requested winding (CCW for an
/// exterior ring, CW for a hole).
fn ring_to_vertices(ring: &LineString<f64>, want_ccw: bool) -> Vec<OffcutVertex> {
    let mut v: Vec<OffcutVertex> = ring
        .coords()
        .map(|c| OffcutVertex {
            x: c.x as f32,
            y: c.y as f32,
        })
        .collect();
    // `geo` repeats the first coord to close the ring; drop it for the open wire list.
    if v.len() > 1 && v.first() == v.last() {
        v.pop();
    }
    let is_ccw = signed_area(&v) >= 0.0;
    if is_ccw != want_ccw {
        v.reverse();
    }
    v
}

/// Shoelace signed area; positive ⇒ CCW.
fn signed_area(verts: &[OffcutVertex]) -> f32 {
    let n = verts.len();
    if n < 3 {
        return 0.0;
    }
    let mut acc = 0.0f32;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        acc += a.x * b.y - b.x * a.y;
    }
    acc / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use jagua_rs::geometry::primitives::Point;

    const BIN: fn() -> Rect = || Rect::try_new(0.0, 0.0, 2000.0, 1000.0).unwrap();

    fn rect_policy(min_w: f32, min_h: f32, kerf: f32) -> OffcutPolicy {
        OffcutPolicy {
            min_offcut_width_mm: min_w,
            min_offcut_height_mm: min_h,
            shape: OffcutShape::Rectangle,
            kerf_mm: kerf,
        }
    }

    fn poly_policy(min_w: f32, min_h: f32, kerf: f32) -> OffcutPolicy {
        OffcutPolicy {
            shape: OffcutShape::Quadrilateral,
            ..rect_policy(min_w, min_h, kerf)
        }
    }

    /// Build a CCW rectangular SPolygon from min/max corners.
    fn rect_spoly(x0: f32, y0: f32, x1: f32, y1: f32) -> SPolygon {
        SPolygon::new(vec![
            Point(x0, y0),
            Point(x1, y0),
            Point(x1, y1),
            Point(x0, y1),
        ])
        .unwrap()
    }

    fn approx(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-3, "expected {b}, got {a}");
    }

    #[test]
    fn rect_corner_placement_yields_strips() {
        // One item in the bottom-left corner (0,0)-(500,400). Free space tiles into two
        // non-overlapping rectangles: the full-height right column and the top-left block.
        let item = Rect::try_new(0.0, 0.0, 500.0, 400.0).unwrap();
        let offcuts = detect_rect_offcuts(BIN(), &[item], &rect_policy(100.0, 100.0, 0.0), 0.0);

        assert_eq!(offcuts.len(), 2, "{offcuts:?}");
        let right = Offcut::Rect {
            x: 500.0,
            y: 0.0,
            width: 1500.0,
            height: 1000.0,
        };
        let top_left = Offcut::Rect {
            x: 0.0,
            y: 400.0,
            width: 500.0,
            height: 600.0,
        };
        assert!(offcuts.contains(&right), "missing right column: {offcuts:?}");
        assert!(offcuts.contains(&top_left), "missing top-left block: {offcuts:?}");
        // The two offcuts must not overlap.
        assert!(!rects_overlap(&offcuts));
    }

    /// Total area of all RECT offcuts (panics on a POLY).
    fn rect_area(offcuts: &[Offcut]) -> f32 {
        offcuts
            .iter()
            .map(|o| match o {
                Offcut::Rect { width, height, .. } => width * height,
                other => panic!("expected RECT, got {other:?}"),
            })
            .sum()
    }

    /// `true` if any two RECT offcuts overlap with positive area.
    fn rects_overlap(offcuts: &[Offcut]) -> bool {
        let rects: Vec<Rect> = offcuts
            .iter()
            .filter_map(|o| match o {
                Offcut::Rect {
                    x,
                    y,
                    width,
                    height,
                } => Rect::try_new(*x, *y, x + width, y + height).ok(),
                _ => None,
            })
            .collect();
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                if Rect::intersection(rects[i], rects[j]).is_some_and(|r| r.area() > COORD_EPS) {
                    return true;
                }
            }
        }
        false
    }

    #[test]
    fn rect_full_sheet_zero_offcuts() {
        let full = Rect::try_new(0.0, 0.0, 2000.0, 1000.0).unwrap();
        let offcuts = detect_rect_offcuts(BIN(), &[full], &rect_policy(10.0, 10.0, 0.0), 0.0);
        assert!(offcuts.is_empty(), "full sheet should yield no offcuts: {offcuts:?}");
    }

    #[test]
    fn rect_decomposition_around_straggler() {
        // A single part floating in the middle of the sheet. Free space tiles into 4
        // non-overlapping rectangles (left column, below, above, right column) that exactly
        // cover the sheet minus the part.
        let item = Rect::try_new(800.0, 400.0, 1200.0, 600.0).unwrap();
        let offcuts = detect_rect_offcuts(BIN(), &[item], &rect_policy(50.0, 50.0, 0.0), 0.0);

        assert_eq!(offcuts.len(), 4, "{offcuts:?}");
        assert!(!rects_overlap(&offcuts), "offcuts overlap: {offcuts:?}");
        let expected = 2000.0 * 1000.0 - 400.0 * 200.0;
        let got = rect_area(&offcuts);
        assert!((got - expected).abs() < 1.0, "area {got} != {expected}");
    }

    #[test]
    fn rect_threshold_discard() {
        // Item leaves a 50mm-wide right strip; min width 100 discards it. Top strip (600mm
        // tall) survives.
        let item = Rect::try_new(0.0, 0.0, 1950.0, 400.0).unwrap();
        let offcuts = detect_rect_offcuts(BIN(), &[item], &rect_policy(100.0, 100.0, 0.0), 0.0);
        assert!(
            offcuts.iter().all(|o| match o {
                Offcut::Rect { width, height, .. } => *width >= 100.0 && *height >= 100.0,
                _ => false,
            }),
            "thin strip not discarded: {offcuts:?}"
        );
        // Only the top strip qualifies.
        assert_eq!(offcuts.len(), 1);
    }

    #[test]
    fn rect_kerf_does_not_shrink_offcut() {
        // Kerf no longer insets the offcut — it only drives the visual band. A corner item
        // leaves the full right strip (500,0)-(2000,1000) regardless of kerf.
        let item = Rect::try_new(0.0, 0.0, 500.0, 1000.0).unwrap();
        let no_kerf = detect_rect_offcuts(BIN(), &[item], &rect_policy(100.0, 100.0, 0.0), 0.0);
        let kerf = detect_rect_offcuts(BIN(), &[item], &rect_policy(100.0, 100.0, 10.0), 0.0);
        assert_eq!(no_kerf, kerf, "kerf must not change offcut geometry");
        assert_eq!(
            kerf,
            vec![Offcut::Rect {
                x: 500.0,
                y: 0.0,
                width: 1500.0,
                height: 1000.0,
            }]
        );
    }

    #[test]
    fn rect_spacing_reaches_true_edges() {
        // With spacing 10 the engine deflates the sheet by 5 and inflates the part by 5, so
        // the detector receives bin [5,5]-[1995,995] and a part bbox of [5,5]-[505,995] for a
        // true corner part (0,0)-(500,990). Undoing the offset, the offcut must reach the true
        // sheet edges (x_max 2000, y 0..1000) and the true part edge (x 500).
        let bin = Rect::try_new(5.0, 5.0, 1995.0, 995.0).unwrap();
        let part = Rect::try_new(5.0, 5.0, 505.0, 995.0).unwrap();
        let offcuts = detect_rect_offcuts(bin, &[part], &rect_policy(100.0, 100.0, 0.0), 10.0);
        assert_eq!(
            offcuts,
            vec![Offcut::Rect {
                x: 500.0,
                y: 0.0,
                width: 1500.0,
                height: 1000.0,
            }],
            "offcut should reach true edges: {offcuts:?}"
        );
    }

    #[test]
    fn kerf_band_punches_usable_interior() {
        // The band path for a RECT offcut is the outer rect plus the kerf-inset rect (hole),
        // so an even-odd fill shades only the cut allowance ring.
        let offcut = Offcut::Rect {
            x: 500.0,
            y: 0.0,
            width: 1500.0,
            height: 1000.0,
        };
        let bands = kerf_band_paths(std::slice::from_ref(&offcut), 10.0);
        assert_eq!(bands.len(), 1);
        // Outer ring + inset ring (two subpaths).
        assert_eq!(bands[0].matches('M').count(), 2, "{}", bands[0]);
        assert!(bands[0].contains("510"), "inset edge missing: {}", bands[0]);
        // No band when kerf is zero.
        assert!(kerf_band_paths(std::slice::from_ref(&offcut), 0.0).is_empty());
    }

    #[test]
    fn rect_empty_layout_whole_bin() {
        // No items + a policy ⇒ the whole (kerf-inset) bin is one offcut.
        let offcuts = detect_rect_offcuts(BIN(), &[], &rect_policy(100.0, 100.0, 0.0), 0.0);
        assert_eq!(
            offcuts,
            vec![Offcut::Rect {
                x: 0.0,
                y: 0.0,
                width: 2000.0,
                height: 1000.0,
            }]
        );
    }

    #[test]
    fn poly_difference_basic() {
        // A single triangular item near the bottom-left; bin minus its hull is non-empty.
        let tri = SPolygon::new(vec![
            Point(0.0, 0.0),
            Point(800.0, 0.0),
            Point(0.0, 600.0),
        ])
        .unwrap();
        let offcuts = detect_poly_offcuts(BIN(), &[&tri], &poly_policy(50.0, 50.0, 0.0), 0.0);
        assert!(!offcuts.is_empty(), "expected polygon offcuts");
        for o in &offcuts {
            match o {
                Offcut::Poly { vertices, .. } => {
                    assert!(vertices.len() >= 3);
                    // CCW.
                    assert!(signed_area(vertices) > 0.0, "ring not CCW: {vertices:?}");
                    // Inside the bin.
                    for v in vertices {
                        assert!((-1e-3..=2000.0 + 1e-3).contains(&v.x));
                        assert!((-1e-3..=1000.0 + 1e-3).contains(&v.y));
                    }
                }
                other => panic!("expected POLY, got {other:?}"),
            }
        }
    }

    #[test]
    fn poly_rdp_collapses_staircase() {
        // A near-rectangle [0,1000]x[0,500] whose top edge is a shallow dome made of many
        // sub-millimetre steps. Each dome vertex is strictly convex (so the convex hull
        // keeps all of them), but lies < 1mm above the flat chord, so RDP collapses the
        // dome away — leaving a low-vertex offcut. Without simplification the resulting
        // ring would carry 20+ top vertices.
        let mut pts = vec![Point(0.0, 0.0), Point(1000.0, 0.0), Point(1000.0, 500.0)];
        for k in (1..=19).rev() {
            let x = k as f32 * 50.0;
            let bulge = 0.5 * (std::f32::consts::PI * x / 1000.0).sin(); // < 0.5mm
            pts.push(Point(x, 500.0 + bulge));
        }
        pts.push(Point(0.0, 500.0));
        let domed = SPolygon::new(pts).unwrap();
        assert!(domed.n_vertices() >= 23, "fixture should be vertex-rich");

        let offcuts = detect_poly_offcuts(BIN(), &[&domed], &poly_policy(50.0, 50.0, 0.0), 0.0);
        assert!(!offcuts.is_empty());
        for o in &offcuts {
            if let Offcut::Poly { vertices, .. } = o {
                assert!(
                    vertices.len() <= 8,
                    "RDP did not collapse the dome: {} vertices",
                    vertices.len()
                );
            }
        }
    }

    #[test]
    fn poly_full_sheet_zero_offcuts() {
        // An item covering the whole bin ⇒ difference is empty.
        let full = rect_spoly(0.0, 0.0, 2000.0, 1000.0);
        let offcuts = detect_poly_offcuts(BIN(), &[&full], &poly_policy(10.0, 10.0, 0.0), 0.0);
        assert!(offcuts.is_empty(), "full sheet poly offcuts: {offcuts:?}");
    }

    #[test]
    fn poly_kerf_does_not_shrink_offcut() {
        // Kerf no longer affects the polygon offcut geometry (only the visual band), so the
        // reported area is identical with and without kerf.
        let item = rect_spoly(0.0, 0.0, 500.0, 1000.0);
        let no_kerf = detect_poly_offcuts(BIN(), &[&item], &poly_policy(50.0, 50.0, 0.0), 0.0);
        let kerf = detect_poly_offcuts(BIN(), &[&item], &poly_policy(50.0, 50.0, 20.0), 0.0);
        assert_eq!(no_kerf, kerf, "kerf must not change offcut geometry");
    }

    #[test]
    fn poly_spacing_deflates_part() {
        // With spacing 10, a part whose collision outline is (5,0)-(505,1000) really occupies
        // (10,0)-(500,1000). Undoing the inflate, the free region's right portion must extend
        // to the part's true right edge (~500), not the inflated 505.
        let bin = Rect::try_new(0.0, 0.0, 2000.0, 1000.0).unwrap();
        let item = rect_spoly(5.0, 0.0, 505.0, 1000.0);
        let offcuts = detect_poly_offcuts(bin, &[&item], &poly_policy(50.0, 50.0, 0.0), 10.0);
        assert!(!offcuts.is_empty());
        let min_x = offcuts
            .iter()
            .flat_map(|o| match o {
                Offcut::Poly { vertices, .. } => vertices.clone(),
                _ => vec![],
            })
            .map(|v| v.x)
            .fold(f32::INFINITY, f32::min);
        // Free region reaches the deflated part edge (~500), proving the inflate was undone.
        assert!(min_x <= 501.0, "offcut left edge {min_x} did not reach true part edge");
    }

    #[test]
    fn offcut_serde_roundtrip() {
        let rect = Offcut::Rect {
            x: 1200.0,
            y: 0.0,
            width: 800.0,
            height: 1000.0,
        };
        let json = serde_json::to_string(&rect).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"RECT","x":1200.0,"y":0.0,"width":800.0,"height":1000.0}"#
        );
        assert_eq!(serde_json::from_str::<Offcut>(&json).unwrap(), rect);

        // Hole-free polygon: `holes` is omitted (matches the original wire shape).
        let poly = Offcut::Poly {
            vertices: vec![
                OffcutVertex { x: 0.0, y: 0.0 },
                OffcutVertex { x: 300.0, y: 0.0 },
                OffcutVertex { x: 150.0, y: 250.0 },
            ],
            holes: vec![],
        };
        let json = serde_json::to_string(&poly).unwrap();
        assert_eq!(
            json,
            r#"{"kind":"POLY","vertices":[{"x":0.0,"y":0.0},{"x":300.0,"y":0.0},{"x":150.0,"y":250.0}]}"#
        );
        assert_eq!(serde_json::from_str::<Offcut>(&json).unwrap(), poly);

        // Polygon with a hole round-trips and includes the `holes` array.
        let holed = Offcut::Poly {
            vertices: vec![
                OffcutVertex { x: 0.0, y: 0.0 },
                OffcutVertex { x: 100.0, y: 0.0 },
                OffcutVertex { x: 100.0, y: 100.0 },
                OffcutVertex { x: 0.0, y: 100.0 },
            ],
            holes: vec![vec![
                OffcutVertex { x: 40.0, y: 40.0 },
                OffcutVertex { x: 40.0, y: 60.0 },
                OffcutVertex { x: 60.0, y: 60.0 },
            ]],
        };
        let json = serde_json::to_string(&holed).unwrap();
        assert!(json.contains(r#""holes":[["#), "{json}");
        assert_eq!(serde_json::from_str::<Offcut>(&json).unwrap(), holed);
    }

    #[test]
    fn policy_serde_camelcase() {
        let policy = OffcutPolicy {
            min_offcut_width_mm: 200.0,
            min_offcut_height_mm: 200.0,
            shape: OffcutShape::Quadrilateral,
            kerf_mm: 0.0,
        };
        let json = serde_json::to_string(&policy).unwrap();
        assert!(json.contains(r#""minOffcutWidthMm":200.0"#), "{json}");
        assert!(json.contains(r#""minOffcutHeightMm":200.0"#), "{json}");
        assert!(json.contains(r#""shape":"QUADRILATERAL""#), "{json}");
        assert!(json.contains(r#""kerfMm":0.0"#), "{json}");

        // kerfMm defaults to 0 when absent.
        let parsed: OffcutPolicy = serde_json::from_str(
            r#"{"minOffcutWidthMm":100.0,"minOffcutHeightMm":100.0,"shape":"RECTANGLE"}"#,
        )
        .unwrap();
        assert_eq!(parsed.shape, OffcutShape::Rectangle);
        approx(parsed.kerf_mm, 0.0);
    }
}
