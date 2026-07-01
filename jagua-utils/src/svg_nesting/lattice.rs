//! Lattice packer — densest *periodic* packing of many copies of ONE polygon.
//!
//! For identical parts the optimal packing is a repeating unit cell (two lattice vectors, optionally
//! a 180°-rotated "double lattice"). We find that cell with a collision-driven contact search
//! (reusing jagua-rs `SPolygon`/`Edge`/surrogate poles — no Minkowski/NFP reinvented), tile the bin,
//! then repeat the stencil. This turns slow/inconsistent/void-y LBF results for trapezoids,
//! parallelograms and concave parts into fast, deterministic, identical-sheet packings.
//!
//! All geometry runs in **centroid-centred space**: a lattice point is therefore a `Placement`
//! centroid directly. Holes are ignored — only the outer ring matters for inter-part collision.

use crate::svg_nesting::render::default_cde_config;
use anyhow::Result;
use jagua_rs::geometry::Transformation;
use jagua_rs::geometry::geo_traits::{CollidesWith, Transformable};
use jagua_rs::geometry::primitives::{Point, SPolygon};

/// Build a centroid-centred `SPolygon` from an outer ring, with its fail-fast surrogate generated
/// (so `ff_poles()` is available for fast overlap rejection). Consecutive and closing duplicate
/// vertices are dropped (`geo` closes its rings; `SPolygon::new` rejects duplicates).
pub(crate) fn centered_spolygon(ring: &[Point]) -> Result<SPolygon> {
    let same = |a: &Point, b: &Point| (a.x() - b.x()).abs() < 1e-5 && (a.y() - b.y()).abs() < 1e-5;
    let mut pts: Vec<Point> = Vec::with_capacity(ring.len());
    for &p in ring {
        if pts.last().is_none_or(|q| !same(q, &p)) {
            pts.push(p);
        }
    }
    if pts.len() >= 2 && same(&pts[0], pts.last().unwrap()) {
        pts.pop();
    }
    let mut poly = SPolygon::new(pts)?;
    let c = poly.centroid();
    poly.transform(&Transformation::from_translation((-c.x(), -c.y())));
    poly.generate_surrogate(default_cde_config().item_surrogate_config)?;
    Ok(poly)
}

/// Translate a (surrogate-bearing) polygon by `(dx, dy)`, keeping its surrogate/bbox consistent.
fn translated(poly: &SPolygon, dx: f32, dy: f32) -> SPolygon {
    poly.transform_clone(&Transformation::from_translation((dx, dy)))
}

/// Orientation sign of `(a,b,c)`: +1 CCW, −1 CW, 0 collinear (with a small tolerance).
fn orient(a: Point, b: Point, c: Point) -> i32 {
    let v = (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x());
    if v > 1e-4 {
        1
    } else if v < -1e-4 {
        -1
    } else {
        0
    }
}

/// PROPER crossing of segments `a1a2` and `b1b2` (both straddle). Collinear/endpoint touches → false.
fn proper_cross(a1: Point, a2: Point, b1: Point, b2: Point) -> bool {
    let (o1, o2) = (orient(a1, a2, b1), orient(a1, a2, b2));
    let (o3, o4) = (orient(b1, b2, a1), orient(b1, b2, a2));
    o1 != 0 && o1 == -o2 && o3 != 0 && o3 == -o4
}

/// Fast, **zero-allocation** interior overlap test for `a` vs `b` translated by `(dx, dy)` — the
/// search workhorse. Instead of cloning/translating `b` (which would re-allocate its vertices and
/// surrogate poles on every call, the dominant cost), the offset is applied inline: query points are
/// shifted by `∓delta` into the other's frame. Touching boundaries ⇒ NOT overlapping. Order: bbox
/// reject → interior-pole disk overlap (fast accept) → vertex/edge-midpoint of one strictly inside
/// the other (the midpoint catches sliver overlaps bounded by collinear edges) → proper edge crossing.
fn overlaps_delta(a: &SPolygon, b: &SPolygon, dx: f32, dy: f32) -> bool {
    // bbox reject (b's bbox shifted by delta)
    if a.bbox.x_min > b.bbox.x_max + dx
        || a.bbox.x_max < b.bbox.x_min + dx
        || a.bbox.y_min > b.bbox.y_max + dy
        || a.bbox.y_max < b.bbox.y_min + dy
    {
        return false;
    }
    for ca in a.surrogate().ff_poles() {
        for cb in b.surrogate().ff_poles() {
            let ddx = ca.center.x() - (cb.center.x() + dx);
            let ddy = ca.center.y() - (cb.center.y() + dy);
            let rs = ca.radius + cb.radius;
            if ddx * ddx + ddy * ddy < rs * rs {
                return true;
            }
        }
    }
    // a's vertices/edge-midpoints inside (b+delta) ⟺ (point − delta) inside b
    for ea in a.edge_iter() {
        let s = Point(ea.start.x() - dx, ea.start.y() - dy);
        let m = Point(
            0.5 * (ea.start.x() + ea.end.x()) - dx,
            0.5 * (ea.start.y() + ea.end.y()) - dy,
        );
        if b.collides_with(&s) || b.collides_with(&m) {
            return true;
        }
    }
    // (b+delta)'s vertices/edge-midpoints inside a
    for eb in b.edge_iter() {
        let s = Point(eb.start.x() + dx, eb.start.y() + dy);
        let m = Point(
            0.5 * (eb.start.x() + eb.end.x()) + dx,
            0.5 * (eb.start.y() + eb.end.y()) + dy,
        );
        if a.collides_with(&s) || a.collides_with(&m) {
            return true;
        }
    }
    // proper edge crossing: a's edges vs (b+delta)'s edges
    for ea in a.edge_iter() {
        for eb in b.edge_iter() {
            let bs = Point(eb.start.x() + dx, eb.start.y() + dy);
            let be = Point(eb.end.x() + dx, eb.end.y() + dy);
            if proper_cross(ea.start, ea.end, bs, be) {
                return true;
            }
        }
    }
    false
}

/// Interior overlap test (touching ⇒ not overlapping). Thin wrapper over [`overlaps_delta`].
pub(crate) fn polys_overlap(a: &SPolygon, b: &SPolygon) -> bool {
    overlaps_delta(a, b, 0.0, 0.0)
}

/// Smallest distance to translate `mover` along unit `dir` so it no longer overlaps `fixed` (the
/// first-contact distance — the no-fit boundary in that direction). A coarse linear pre-scan exits
/// the (possibly non-convex) overlap region monotonically in `t` so the bracket is the *first* exit
/// (robust for L-shapes/notches), then a binary search refines it.
fn first_contact_between(
    fixed: &SPolygon,
    mover: &SPolygon,
    dir: (f32, f32),
    max_dist: f32,
) -> f32 {
    let step = (fixed.diameter.max(mover.diameter) / 48.0).max(1e-4);
    let overlaps = |t: f32| overlaps_delta(fixed, mover, dir.0 * t, dir.1 * t);
    let mut t = step;
    while t < max_dist && overlaps(t) {
        t += step;
    }
    if t >= max_dist {
        return max_dist;
    }
    let (mut lo, mut hi) = (t - step, t);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if overlaps(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

// ---------------------------------------------------------------------------------------------
// Lattice search
// ---------------------------------------------------------------------------------------------

type V2 = (f32, f32);
fn cross(a: V2, b: V2) -> f32 {
    a.0 * b.1 - a.1 * b.0
}

/// A solved lattice: two cell vectors + the motif (1 part = single lattice, 2 = double lattice).
pub(crate) struct Lattice {
    pub v1: V2,
    pub v2: V2,
    /// `(rotation, offset)` per motif member.
    pub members: Vec<(f32, V2)>,
    pub density: f32,
}

const K_DIRS: usize = 72;
const K_NESTLE: usize = 16;
/// Cap on candidate cells examined per motif (smallest-area first) — bounds the ring-validation cost.
/// The fast `overlaps_delta` ring check is cheap, so this can be generous: a near-rectangular part's
/// only valid single-lattice cell sits just above the part area, behind many invalid tighter oblique
/// cells, so too small a cap loses it.
const MAX_CELLS: usize = 400;

/// True if any member shape at the origin overlaps any member translated by `delta`.
fn motif_overlap(m: &[SPolygon], delta: V2) -> bool {
    m.iter()
        .any(|a| m.iter().any(|b| overlaps_delta(a, b, delta.0, delta.1)))
}

/// First-contact distance of the whole motif against a `dir`-translated copy of itself.
fn motif_first_contact(m: &[SPolygon], dir: V2, max_dist: f32, step: f32) -> f32 {
    let mut t = step;
    while t < max_dist && motif_overlap(m, (dir.0 * t, dir.1 * t)) {
        t += step;
    }
    if t >= max_dist {
        return max_dist;
    }
    let (mut lo, mut hi) = (t - step, t);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        if motif_overlap(m, (dir.0 * mid, dir.1 * mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi
}

/// Densest valid lattice for a `motif` (members mutually non-overlapping). The motif's self-no-fit
/// region is centrally symmetric (the motif is, for the double lattice, symmetric about its centre),
/// so contact directions are sampled over [0, π). Returns `(v1, v2, density)`.
fn best_lattice_for_motif(m: &[SPolygon], motif_area: f32, diameter: f32) -> Option<(V2, V2, f32)> {
    let step = (diameter / 64.0).max(1e-4);
    let max_dist = 3.0 * diameter;
    let mut gens: Vec<V2> = Vec::with_capacity(K_DIRS);
    for k in 0..K_DIRS {
        let phi = std::f32::consts::PI * k as f32 / K_DIRS as f32;
        let dir = (phi.cos(), phi.sin());
        let t = motif_first_contact(m, dir, max_dist, step);
        if t < max_dist {
            gens.push((dir.0 * t, dir.1 * t));
        }
    }
    if gens.len() < 2 {
        return None;
    }
    // A unit cell holds the whole (inflated) motif, so its area can never be below the members'
    // actual total area. Use the *inflated* member areas — not the raw `motif_area` used for the
    // density score — so the threshold matches the geometry the cells are built from; otherwise the
    // valid (slightly larger) cell is mis-ranked behind impossible sub-area "sliver" cells and lost
    // to the candidate cap. Rejecting smaller cells also kills degenerate near-parallel slivers whose
    // phantom density ≫ 1 would win the search and blow up the bin fill.
    let min_cell = m.iter().map(|p| p.area).sum::<f32>() * 0.999;
    let mut cells: Vec<(f32, usize, usize)> = Vec::new();
    for i in 0..gens.len() {
        for j in (i + 1)..gens.len() {
            let a = cross(gens[i], gens[j]).abs();
            if a >= min_cell {
                cells.push((a, i, j));
            }
        }
    }
    cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    cells.truncate(MAX_CELLS);
    for (area, i, j) in cells {
        let (v1, v2) = (gens[i], gens[j]);
        let ring_ok = (-2..=2).all(|mm| {
            (-2..=2).all(|nn| {
                if mm == 0 && nn == 0 {
                    return true;
                }
                !motif_overlap(
                    m,
                    (
                        mm as f32 * v1.0 + nn as f32 * v2.0,
                        mm as f32 * v1.1 + nn as f32 * v2.1,
                    ),
                )
            })
        });
        if ring_ok {
            return Some((v1, v2, motif_area / area));
        }
    }
    None
}

/// Validate a solved lattice on an explicit, full-resolution `motif` (members already rotated and
/// offset, inflated by the kerf half-gap): the members must not overlap each other, and no `[-2,2]²`
/// lattice-neighbour copy may overlap the origin motif. The search runs on a *decimated* outline for
/// speed, which can miss a fine concavity (a notch) that makes the real parts collide — so the winner
/// is re-checked here at full resolution before it is trusted over the always-valid grid floor.
fn lattice_valid_fullres(v1: V2, v2: V2, motif: &[SPolygon]) -> bool {
    for i in 0..motif.len() {
        for j in (i + 1)..motif.len() {
            if polys_overlap(&motif[i], &motif[j]) {
                return false;
            }
        }
    }
    (-2..=2).all(|mm| {
        (-2..=2).all(|nn| {
            if mm == 0 && nn == 0 {
                return true;
            }
            !motif_overlap(
                motif,
                (
                    mm as f32 * v1.0 + nn as f32 * v2.0,
                    mm as f32 * v1.1 + nn as f32 * v2.1,
                ),
            )
        })
    })
}

/// Densest lattice for a base (centred, inflated, rotated-to-θ) polygon: the single lattice, plus —
/// if `allow_double` — the double lattice over 180°-partner offsets generated by sliding the partner
/// to first contact ("nestle") in `K_NESTLE` directions (so the partner tucks against the part, e.g.
/// a trapezoid's vertical stack or a triangle's bbox pairing). `placement_rot` = θ.
fn best_lattice_for_base(
    base: &SPolygon,
    part_area: f32,
    placement_rot: f32,
    allow_double: bool,
) -> Option<Lattice> {
    let diameter = base.diameter;
    let mut best: Option<Lattice> = None;
    let mut consider = |density: f32, v1: V2, v2: V2, members: Vec<(f32, V2)>| {
        if best.as_ref().is_none_or(|b| density > b.density) {
            best = Some(Lattice {
                v1,
                v2,
                members,
                density,
            });
        }
    };

    // single lattice
    let single = [base.clone()];
    if let Some((v1, v2, d)) = best_lattice_for_motif(&single, part_area, diameter) {
        consider(d, v1, v2, vec![(placement_rot, (0.0, 0.0))]);
    }

    // double lattice: partner = base rotated 180°, slid to nestle against the part.
    if allow_double {
        let b180 = base.transform_clone(&Transformation::from_rotation(std::f32::consts::PI));
        let max_dist = 3.0 * diameter;
        let mut offsets: Vec<V2> = Vec::with_capacity(K_NESTLE);
        for k in 0..K_NESTLE {
            let phi = std::f32::consts::TAU * k as f32 / K_NESTLE as f32;
            let dir = (phi.cos(), phi.sin());
            let t = first_contact_between(base, &b180, dir, max_dist);
            if t < max_dist {
                offsets.push((dir.0 * t, dir.1 * t));
            }
        }
        for off in offsets {
            let b_at = translated(&b180, off.0, off.1);
            if polys_overlap(base, &b_at) {
                continue;
            }
            let motif = [base.clone(), b_at];
            if let Some((v1, v2, d)) = best_lattice_for_motif(&motif, 2.0 * part_area, diameter) {
                consider(
                    d,
                    v1,
                    v2,
                    vec![
                        (placement_rot, (0.0, 0.0)),
                        (placement_rot + std::f32::consts::PI, off),
                    ],
                );
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------
// Bin filling
// ---------------------------------------------------------------------------------------------

use crate::svg_nesting::render::Placement;

/// Tile `lattice` across the bin and emit one `Placement` per copy whose RAW part bbox fits fully
/// inside `[0,W]×[0,H]` (touching the edge is allowed — no border gap). `raw_centered` is the part
/// centred at its centroid (non-inflated), used only to size each member's rotated bbox. A 3×3 phase
/// search over the lattice origin keeps the placement count maximal. O(occupied cells).
pub(crate) fn fill_lattice(
    lattice: &Lattice,
    raw_centered: &SPolygon,
    bin_w: f32,
    bin_h: f32,
) -> Vec<Placement> {
    let (v1, v2) = (lattice.v1, lattice.v2);
    let det = cross(v1, v2);
    if det.abs() < 1e-6 {
        return Vec::new();
    }
    // Each member's rotated bbox extents *relative to the centroid* (`raw_centered` is centroid-
    // centred, so its rotated bbox min/max are the true asymmetric reach in each direction). The
    // renderer places by centroid, so a part at centroid (cx, cy) spans [cx+xmin, cx+xmax] ×
    // [cy+ymin, cy+ymax]. Using these (not symmetric half-extents) keeps off-centre-centroid parts —
    // a notched rectangle whose centroid sits below its bbox centre — fully inside the bin.
    let extent: Vec<(f32, f32, f32, f32)> = lattice
        .members
        .iter()
        .map(|&(rot, _)| {
            let r = raw_centered.transform_clone(&Transformation::from_rotation(rot));
            (r.bbox.x_min, r.bbox.x_max, r.bbox.y_min, r.bbox.y_max)
        })
        .collect();

    // integer (m,n) range covering the bin (map the four corners through inv[v1 v2]).
    let to_mn = |x: f32, y: f32| ((v2.1 * x - v2.0 * y) / det, (-v1.1 * x + v1.0 * y) / det);
    let corners = [(0.0, 0.0), (bin_w, 0.0), (0.0, bin_h), (bin_w, bin_h)];
    let mns: Vec<(f32, f32)> = corners.iter().map(|&(x, y)| to_mn(x, y)).collect();
    let m_lo = mns
        .iter()
        .map(|p| p.0)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let m_hi = mns
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;
    let n_lo = mns
        .iter()
        .map(|p| p.1)
        .fold(f32::INFINITY, f32::min)
        .floor() as i32
        - 1;
    let n_hi = mns
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32
        + 1;
    // Defensive bound: a sane lattice covers the bin in far fewer cells than this. If the cell is
    // degenerate (near-parallel vectors slipped through), refuse rather than enumerate billions —
    // the caller falls back to the bbox grid floor.
    if (m_hi - m_lo) as i64 * (n_hi - n_lo) as i64 > 4_000_000 {
        return Vec::new();
    }

    let emit = |phase: V2| -> Vec<Placement> {
        let mut out = Vec::new();
        for mm in m_lo..=m_hi {
            for nn in n_lo..=n_hi {
                let base = (
                    mm as f32 * v1.0 + nn as f32 * v2.0 + phase.0,
                    mm as f32 * v1.1 + nn as f32 * v2.1 + phase.1,
                );
                for (idx, &(rot, off)) in lattice.members.iter().enumerate() {
                    let (cx, cy) = (base.0 + off.0, base.1 + off.1);
                    let (xmin, xmax, ymin, ymax) = extent[idx];
                    if cx + xmin >= -1e-3
                        && cx + xmax <= bin_w + 1e-3
                        && cy + ymin >= -1e-3
                        && cy + ymax <= bin_h + 1e-3
                    {
                        out.push(Placement {
                            part_idx: 0,
                            rotation: rot,
                            x: cx,
                            y: cy,
                        });
                    }
                }
            }
        }
        out
    };

    // phase search: shift the lattice origin to recover boundary parts.
    let mut best = emit((0.0, 0.0));
    for pi in 0..3 {
        for pj in 0..3 {
            if pi == 0 && pj == 0 {
                continue;
            }
            let (a, b) = (pi as f32 / 3.0, pj as f32 / 3.0);
            let cand = emit((a * v1.0 + b * v2.0, a * v1.1 + b * v2.1));
            if cand.len() > best.len() {
                best = cand;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------------------------

use crate::svg_nesting::grid::grid_single_sheet;
use crate::svg_nesting::parsing::{
    calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path, reverse_winding,
    sanitize_polygon,
};
use crate::svg_nesting::render::{prepare, render_periodic};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::NestingResult;
use jagua_rs::geometry::geo_enums::RotationRange;

/// Parse a part's SVG into a sanitized, CCW outer ring (matching `render::parse_part`).
fn outer_ring(part: &PartInput) -> Result<Vec<Point>> {
    let path = extract_path_from_svg_bytes(&part.svg_bytes)?;
    let (pts, _holes) = parse_svg_path(&path)?;
    let pts = if calculate_signed_area(&pts) < 0.0 {
        reverse_winding(&pts)
    } else {
        pts
    };
    Ok(sanitize_polygon(pts))
}

/// Inflate a ring outward by `dist` (the kerf half-gap) using `geo_buffer`; returns the largest
/// resulting polygon's exterior. Falls back to the raw ring on failure.
fn inflate_ring(ring: &[Point], dist: f32) -> Vec<Point> {
    if dist <= 1e-4 {
        return ring.to_vec();
    }
    let coords: Vec<geo::Coord<f64>> = ring
        .iter()
        .map(|p| geo::Coord {
            x: p.x() as f64,
            y: p.y() as f64,
        })
        .collect();
    let poly = geo::Polygon::new(geo::LineString::new(coords), vec![]);
    let mp = geo_buffer::buffer_polygon(&poly, dist as f64);
    use geo::Area;
    mp.0.into_iter()
        .max_by(|a, b| {
            a.unsigned_area()
                .partial_cmp(&b.unsigned_area())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| {
            p.exterior()
                .points()
                .map(|pt| Point(pt.x() as f32, pt.y() as f32))
                .collect()
        })
        .filter(|v: &Vec<Point>| v.len() >= 3)
        .unwrap_or_else(|| ring.to_vec())
}

/// Decimate a ring to keep the lattice overlap tests cheap (search only; the final placements use the
/// full-resolution part). Douglas–Peucker via `geo::Simplify`.
fn decimate(ring: &[Point], tol: f32) -> Vec<Point> {
    use geo::Simplify;
    let ls = geo::LineString::new(
        ring.iter()
            .map(|p| geo::Coord {
                x: p.x() as f64,
                y: p.y() as f64,
            })
            .collect(),
    );
    let simp = ls.simplify(&(tol as f64));
    let out: Vec<Point> = simp
        .0
        .iter()
        .map(|c| Point(c.x as f32, c.y as f32))
        .collect();
    if out.len() >= 3 { out } else { ring.to_vec() }
}

/// Solve the densest single-sheet stencil for one part type: the densest lattice fill found over
/// `base_rotations` (`allow_double` enables the 180° double lattice), or the bbox grid floor when the
/// lattice is missing or thinner (so the result is never worse than the trivial grid and a valid
/// packing always exists). Returns the render context + the stencil placements. Shared by the
/// periodic nest and the max-fit path so their per-sheet capacity is, by construction, identical.
fn lattice_single_sheet(
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    part: &PartInput,
    base_rotations: &[f32],
    allow_double: bool,
) -> Result<(crate::svg_nesting::render::RenderContext, Vec<Placement>)> {
    use std::f32::consts::{FRAC_PI_2, PI};
    let rot_range = RotationRange::Discrete(vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
    let parts = std::slice::from_ref(part);
    let (prepared, ctx) = prepare(parts, std::slice::from_ref(&rot_range), bin_w, bin_h, 1)?;

    let raw_ring = outer_ring(part)?;
    let raw_centered = centered_spolygon(&raw_ring)?;
    let part_area = raw_centered.area;
    // search base: inflated by the kerf half-gap, decimated for speed.
    let infl = inflate_ring(&raw_ring, spacing / 2.0);
    // Decimate the search outline to keep each O(n²) overlap test cheap: raise the tolerance until the
    // ring is under MAX_SEARCH_VERTS (the final placements still use the full-resolution part).
    const MAX_SEARCH_VERTS: usize = 20;
    let mut tol = (raw_centered.diameter * 0.01).max(0.05);
    let mut search_ring = decimate(&infl, tol);
    for _ in 0..8 {
        if search_ring.len() <= MAX_SEARCH_VERTS {
            break;
        }
        tol *= 1.7;
        search_ring = decimate(&infl, tol);
    }
    let base0 = centered_spolygon(&search_ring)?;

    let mut best: Option<Lattice> = None;
    for &theta in base_rotations {
        let base = if theta.abs() < 1e-6 {
            base0.clone()
        } else {
            base0.transform_clone(&Transformation::from_rotation(theta))
        };
        if let Some(lat) = best_lattice_for_base(&base, part_area, theta, allow_double)
            && best.as_ref().is_none_or(|b| lat.density > b.density)
        {
            best = Some(lat);
        }
    }

    // Re-validate the winner on the FULL-resolution inflated outline (the search used a decimated one;
    // a fine notch it dropped could make the real parts overlap). Reject the lattice on failure — or
    // if the full-resolution inflated ring can't even be rebuilt (`geo_buffer` can emit a degenerate
    // outline `SPolygon::new` refuses) — so the caller falls back to the always-valid grid floor
    // rather than erroring the whole nest.
    if let Some(l) = &best {
        let validated = centered_spolygon(&infl).is_ok_and(|full_infl| {
            let motif: Vec<SPolygon> = l
                .members
                .iter()
                .map(|&(rot, off)| {
                    translated(
                        &full_infl.transform_clone(&Transformation::from_rotation(rot)),
                        off.0,
                        off.1,
                    )
                })
                .collect();
            lattice_valid_fullres(l.v1, l.v2, &motif)
        });
        if !validated {
            best = None;
        }
    }

    let lattice_stencil = best
        .as_ref()
        .map(|l| fill_lattice(l, &raw_centered, bin_w, bin_h))
        .unwrap_or_default();
    // bbox grid floor — guarantees a valid packing and never worse than the trivial grid.
    // `grid_single_sheet` emits the part's *centroid* at each cell centre, which is correct only when
    // the centroid is the bbox centre (a rectangle). For an off-centre centroid the bbox would slide
    // out of its cell (notched parts poke past the sheet edge), so shift each placement by the
    // rotated centroid→bbox-centre offset to seat the bbox flush in its cell.
    let p = &prepared[0];
    let cgap = (p.bbox_w / 2.0 - p.cx_off, p.bbox_h / 2.0 - p.cy_off);
    let floor: Vec<Placement> =
        grid_single_sheet(p.bbox_w, p.bbox_h, bin_w, bin_h, spacing, 0, allow_double)
            .into_iter()
            .map(|pl| {
                let (s, c) = pl.rotation.sin_cos();
                Placement {
                    x: pl.x - (cgap.0 * c - cgap.1 * s),
                    y: pl.y - (cgap.0 * s + cgap.1 * c),
                    ..pl
                }
            })
            .collect();
    let stencil = if lattice_stencil.len() >= floor.len() {
        lattice_stencil
    } else {
        floor
    };
    if stencil.is_empty() {
        anyhow::bail!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2})",
            prepared[0].bbox_w,
            prepared[0].bbox_h,
            bin_w,
            bin_h
        );
    }
    Ok((ctx, stencil))
}

/// Pack a single part type with the densest lattice found, repeated periodically. `base_rotations`
/// are the orientations (radians) to try; `allow_double` enables the 180° double lattice. Falls back
/// to the bbox grid floor when the lattice is missing or thinner (so all parts are always placed).
pub(crate) fn nest_lattice(
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    part: &PartInput,
    base_rotations: &[f32],
    allow_double: bool,
) -> Result<NestingResult> {
    let (ctx, stencil) =
        lattice_single_sheet(bin_w, bin_h, spacing, part, base_rotations, allow_double)?;
    let cap = stencil.len();
    let qty = part.count;
    let full = qty / cap;
    let rem = qty % cap;
    let remainder: Vec<_> = stencil.iter().take(rem).copied().collect();
    Ok(render_periodic(&ctx, &stencil, full, &remainder, qty))
}

/// Max copies of a single irregular part on ONE sheet (the lattice/grid-floor stencil), rendered as a
/// single full page. The reported "max parts per sheet" therefore equals the periodic full-sheet count
/// from [`nest_lattice`] — they consume the identical stencil — so max-fit can never disagree with the
/// real nest for irregular parts (the irregular-part analogue of the rectangle WS-7 fix).
pub(crate) fn nest_max_fit_lattice(
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    part: &PartInput,
    base_rotations: &[f32],
    allow_double: bool,
) -> Result<NestingResult> {
    let (ctx, stencil) =
        lattice_single_sheet(bin_w, bin_h, spacing, part, base_rotations, allow_double)?;
    let cap = stencil.len();
    Ok(render_periodic(&ctx, &stencil, 1, &[], cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn ring(pts: &[(f32, f32)]) -> Vec<Point> {
        pts.iter().map(|&(x, y)| Point(x, y)).collect()
    }

    fn unit_square() -> SPolygon {
        centered_spolygon(&ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])).unwrap()
    }

    /// Wrap a CCW ring in a minimal SVG path so it can drive the full `nest_lattice` pipeline.
    fn svg_from_ring(pts: &[(f32, f32)]) -> Vec<u8> {
        let mut d = String::new();
        for (i, (x, y)) in pts.iter().enumerate() {
            d.push_str(&format!("{} {x},{y} ", if i == 0 { "M" } else { "L" }));
        }
        d.push('Z');
        format!(r#"<svg xmlns="http://www.w3.org/2000/svg"><path d="{d}" fill="black"/></svg>"#)
            .into_bytes()
    }

    /// Assert a `stencil` is a *valid* packing: every placement seats the full (rotated) part inside
    /// `[0,bin_w]×[0,bin_h]`, and no two placed parts overlap (the real, full-resolution outlines —
    /// the guard against an over-eager interlock lattice). `raw_centered` is the part centred on its
    /// centroid, so a placement centroid `(x,y)` puts the part at `raw_centered` rotated + translated.
    fn assert_stencil_valid(stencil: &[Placement], raw_centered: &SPolygon, bw: f32, bh: f32) {
        const EPS: f32 = 0.5;
        let placed: Vec<SPolygon> = stencil
            .iter()
            .map(|pl| {
                let r = raw_centered.transform_clone(&Transformation::from_rotation(pl.rotation));
                translated(&r, pl.x, pl.y)
            })
            .collect();
        for (pl, p) in stencil.iter().zip(&placed) {
            assert!(
                p.bbox.x_min >= -EPS
                    && p.bbox.x_max <= bw + EPS
                    && p.bbox.y_min >= -EPS
                    && p.bbox.y_max <= bh + EPS,
                "part out of bounds: centroid ({:.1},{:.1}) rot {:.2} → x[{:.1},{:.1}] y[{:.1},{:.1}] vs bin {bw}x{bh}",
                pl.x,
                pl.y,
                pl.rotation,
                p.bbox.x_min,
                p.bbox.x_max,
                p.bbox.y_min,
                p.bbox.y_max,
            );
        }
        for i in 0..placed.len() {
            for j in (i + 1)..placed.len() {
                assert!(
                    !polys_overlap(&placed[i], &placed[j]),
                    "parts {i} and {j} overlap (centroids {:?} / {:?})",
                    (stencil[i].x, stencil[i].y),
                    (stencil[j].x, stencil[j].y),
                );
            }
        }
    }

    #[test]
    fn overlap_basic() {
        let s = unit_square();
        // shifted half a side → overlap; shifted two sides → clear.
        assert!(polys_overlap(&s, &translated(&s, 5.0, 0.0)));
        assert!(!polys_overlap(&s, &translated(&s, 20.0, 0.0)));
        assert!(!polys_overlap(&s, &translated(&s, 11.0, 0.0)));
    }

    #[test]
    fn contact_square_axis() {
        let s = unit_square();
        // a 10×10 square just clears its copy at a 10-unit axis shift.
        let cx = first_contact_between(&s, &s, (1.0, 0.0), 40.0);
        assert!((cx - 10.0).abs() < 0.2, "axis contact ~10, got {cx}");
        let cy = first_contact_between(&s, &s, (0.0, 1.0), 40.0);
        assert!((cy - 10.0).abs() < 0.2, "axis contact ~10, got {cy}");
    }

    #[test]
    fn contact_square_diagonal() {
        let s = unit_square();
        // diagonal: copies touch corner-to-corner at distance sqrt(2)*10 ≈ 14.14.
        let d = first_contact_between(&s, &s, (0.70710677, 0.70710677), 60.0);
        assert!((d - 14.142).abs() < 0.3, "diagonal contact ~14.14, got {d}");
    }

    #[test]
    fn contact_triangle() {
        // right triangle bbox 10x10: along +x the copy clears at 10 (the flat right leg).
        let t = centered_spolygon(&ring(&[(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)])).unwrap();
        let cx = first_contact_between(&t, &t, (1.0, 0.0), 40.0);
        assert!((cx - 10.0).abs() < 0.3, "triangle +x contact ~10, got {cx}");
    }

    fn density_of(pts: &[(f32, f32)]) -> Lattice {
        let s = centered_spolygon(&ring(pts)).unwrap();
        let area = s.area;
        best_lattice_for_base(&s, area, 0.0, true).expect("a lattice")
    }

    #[test]
    fn lattice_rectangle_tiles() {
        let lat = density_of(&[(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)]);
        assert!(
            lat.density > 0.98,
            "rectangle tiles ~1.0, got {}",
            lat.density
        );
    }

    #[test]
    fn lattice_parallelogram_tiles() {
        let lat = density_of(&[(0.0, 0.0), (10.0, 0.0), (13.0, 6.0), (3.0, 6.0)]);
        assert!(
            lat.density > 0.97,
            "parallelogram tiles, got {}",
            lat.density
        );
    }

    #[test]
    fn lattice_triangle_double() {
        let lat = density_of(&[(0.0, 0.0), (10.0, 0.0), (0.0, 6.0)]);
        assert!(
            lat.density > 0.95,
            "two triangles ~ rectangle, got {}",
            lat.density
        );
        assert_eq!(
            lat.members.len(),
            2,
            "triangle should use the double lattice"
        );
    }

    #[test]
    fn lattice_trapezoid_double() {
        let lat = density_of(&[(0.0, 0.0), (10.0, 0.0), (10.0, 4.0), (0.0, 8.0)]);
        assert!(
            lat.density > 0.90,
            "trapezoid pairs into rectangle, got {}",
            lat.density
        );
    }

    #[test]
    fn fill_rectangle_bin() {
        // 10×6 rectangle on a 1000×600 bin: ≈100×100 = 10000, minus small boundary loss.
        let raw =
            centered_spolygon(&ring(&[(0.0, 0.0), (10.0, 0.0), (10.0, 6.0), (0.0, 6.0)])).unwrap();
        let lat = best_lattice_for_base(&raw, raw.area, 0.0, true).unwrap();
        let placements = fill_lattice(&lat, &raw, 1000.0, 600.0);
        assert!(
            placements.len() >= 9500,
            "dense rectangle fill, got {}",
            placements.len()
        );
        for p in &placements {
            assert!(
                p.x >= -0.1 && p.x <= 1000.1 && p.y >= -0.1 && p.y <= 600.1,
                "in bin: {p:?}"
            );
        }
    }

    #[test]
    fn lattice_case104_concave_bulk() {
        // Real production part: a 42-vertex concave notched shape ×2000 that hung LBF for ~26s and
        // (before the size-gated debug_assert patch) hung bulk rendering for minutes. Now the whole
        // pipeline must finish quickly and place every part.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../jagua-sqs-processor/tests/testdata/prod-tests/case-104/3c9d2a67-1996-45b4-b66b-da9ae6034e55.svg"
        );
        let svg = std::fs::read(path).unwrap();
        let part = PartInput {
            svg_bytes: svg,
            count: 2000,
            item_id: None,
            allowed_rotations: None,
        };
        let t = std::time::Instant::now();
        let r = nest_lattice(
            1250.0,
            2500.0,
            2.0,
            &part,
            &[0.0, std::f32::consts::FRAC_PI_2],
            true,
        )
        .expect("nest_lattice");
        let elapsed = t.elapsed();
        eprintln!(
            "case104: {}/2000 on {} sheets, {:.1}% util ({:?})",
            r.parts_placed,
            r.pages.len(),
            r.utilisation * 100.0,
            elapsed
        );
        assert_eq!(r.parts_placed, 2000, "all parts placed");
        assert!(elapsed.as_secs() < 10, "must be fast, took {elapsed:?}");
    }

    #[test]
    fn lattice_offcenter_centroid_stays_in_bounds() {
        // Rectangle 100×60 with a notch cut from the top-centre: the removed top area pulls the
        // centroid below the bbox centre. Placing the centroid at a cell centre (the old grid-floor
        // bug) would slide the bbox up and out of the sheet — exactly what broke case-017's render.
        let part = PartInput {
            svg_bytes: svg_from_ring(&[
                (0.0, 0.0),
                (100.0, 0.0),
                (100.0, 60.0),
                (60.0, 60.0),
                (60.0, 40.0),
                (40.0, 40.0),
                (40.0, 60.0),
                (0.0, 60.0),
            ]),
            count: 60,
            item_id: None,
            allowed_rotations: None,
        };
        let (bw, bh) = (520.0, 320.0);
        let (_ctx, stencil) =
            lattice_single_sheet(bw, bh, 2.0, &part, &[0.0, FRAC_PI_2], true).unwrap();
        assert!(!stencil.is_empty(), "notched part should pack");
        let raw = centered_spolygon(&outer_ring(&part).unwrap()).unwrap();
        assert_stencil_valid(&stencil, &raw, bw, bh);
    }

    #[test]
    fn lattice_case017_hardcore_in_bounds() {
        // The 328-vertex production outline whose off-centre centroid pushed parts past the top edge.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../jagua-sqs-processor/tests/testdata/prod-tests/case-017/a19a049d-2bc5-496f-ac53-d6b90607aa21.svg"
        );
        let svg = std::fs::read(path).unwrap();
        let part = PartInput {
            svg_bytes: svg,
            count: 100,
            item_id: None,
            allowed_rotations: None,
        };
        let (bw, bh) = (1500.0, 3000.0);
        let (_ctx, stencil) =
            lattice_single_sheet(bw, bh, 2.0, &part, &[0.0, FRAC_PI_2], true).unwrap();
        assert!(!stencil.is_empty(), "case-017 should pack");
        let raw = centered_spolygon(&outer_ring(&part).unwrap()).unwrap();
        assert_stencil_valid(&stencil, &raw, bw, bh);
    }

    #[test]
    fn fill_triangle_bin_double() {
        // right triangle 10×6 → double lattice; area floor 1000*600/30 = 20000.
        let raw = centered_spolygon(&ring(&[(0.0, 0.0), (10.0, 0.0), (0.0, 6.0)])).unwrap();
        let lat = best_lattice_for_base(&raw, raw.area, 0.0, true).unwrap();
        assert_eq!(lat.members.len(), 2);
        let placements = fill_lattice(&lat, &raw, 1000.0, 600.0);
        assert!(
            placements.len() >= 18500,
            "dense triangle fill, got {}",
            placements.len()
        );
    }
}
