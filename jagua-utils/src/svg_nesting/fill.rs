//! Fill direction, start corner and the single rectangular remnant (CUTL-198).
//!
//! One nesting parameter, *direction*, with three values: `HORIZONTAL` and `VERTICAL` are
//! deterministic row/column fills that leave **one rectangular remnant** per page; `STAIRCASE`
//! is today's classify-and-route packing, untouched. The start corner says where the fill begins.
//!
//! # Coordinates
//!
//! The SVG output is y-down and unflipped (`layout_to_svg` emits the engine's raw numbers and
//! `combine_svg_documents` stacks pages downward), so engine `(0, 0)` renders **top-left**. The
//! corner enum is named as the viewer sees it: `TOP_LEFT` is `(0, 0)` — today's anchor —,
//! `TOP_RIGHT` is `(binWidth, 0)`, `BOTTOM_LEFT` is `(0, binHeight)`, `BOTTOM_RIGHT` is
//! `(binWidth, binHeight)`. Older notes that call high-y the "bottom strip" describe the engine's
//! own vocabulary, not the picture.

use crate::svg_nesting::grid::grid_dims;
use crate::svg_nesting::offcut::{Offcut, OffcutPolicy, overlay_result_offcuts};
use crate::svg_nesting::render::{Placement, PreparedPart, prepare, render_page_list};
use crate::svg_nesting::strategy::{PartInput, effective_allowed, fit_orientations};
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use serde::{Deserialize, Serialize};
use std::f32::consts::FRAC_PI_2;

/// How each sheet is filled (`NestingRequest.fillDirection`). Absent on the wire ⇒ [`Staircase`].
///
/// [`Staircase`]: FillDirection::Staircase
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FillDirection {
    /// The packed block grows along `binWidth` from the start corner's vertical edge: parts stack
    /// in vertical strips, strips advance along x. The remnant is one full-height strip on the far
    /// x side.
    Horizontal,
    /// The packed block grows along `binHeight` from the start corner's horizontal edge: parts
    /// advance in horizontal strips, strips stack along y. The remnant is one full-width strip on
    /// the far y side.
    Vertical,
    /// The packer's own densest fill — exactly today's classify-and-route behaviour, with today's
    /// offcut behaviour. The start corner is ignored (see [`StartCorner`]).
    #[default]
    Staircase,
}

/// The sheet corner the fill starts from (`NestingRequest.startCorner`), named as the SVG renders
/// it. Absent on the wire ⇒ [`TopLeft`], which reproduces today's output byte for byte.
///
/// Honoured only for [`FillDirection::Horizontal`] / [`FillDirection::Vertical`], where every part
/// owns a disjoint bounding-box cell that can be reflected while the part keeps its own
/// orientation. Under [`FillDirection::Staircase`] it is accepted and ignored: pairing places two
/// parts in one cell and the lattice / LBF packers interlock outlines, so reflecting cells there
/// would create overlaps, and mirroring outlines would produce a different part.
///
/// [`TopLeft`]: StartCorner::TopLeft
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StartCorner {
    /// Engine `(0, 0)` — today's anchor.
    #[default]
    TopLeft,
    /// Engine `(binWidth, 0)`.
    TopRight,
    /// Engine `(0, binHeight)`.
    BottomLeft,
    /// Engine `(binWidth, binHeight)`.
    BottomRight,
}

impl StartCorner {
    /// Whether the fill starts at the right-hand edge (x is reflected).
    pub fn is_right(self) -> bool {
        matches!(self, StartCorner::TopRight | StartCorner::BottomRight)
    }

    /// Whether the fill starts at the bottom edge (y is reflected).
    pub fn is_bottom(self) -> bool {
        matches!(self, StartCorner::BottomLeft | StartCorner::BottomRight)
    }
}

/// The request's layout choice: direction + start corner. `Default` is today's behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SheetFill {
    pub direction: FillDirection,
    pub corner: StartCorner,
}

impl SheetFill {
    pub fn new(direction: FillDirection, corner: StartCorner) -> Self {
        Self { direction, corner }
    }

    /// Today's packing: the classifier routes, the corner is ignored.
    pub fn is_staircase(&self) -> bool {
        self.direction == FillDirection::Staircase
    }
}

// ---------------------------------------------------------------------------
// The row/column packer
// ---------------------------------------------------------------------------

/// Tolerance (units) for "fits" comparisons, so an f32 residue never opens a new strip or page.
const EPS: f32 = 1e-3;

/// One part type in its chosen cardinal orientation: the rotated bounding box's extents relative
/// to the centroid (the placement reference point), so a cell corner maps to a centroid directly.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FillType {
    part_idx: usize,
    /// 0 or π/2 (radians, CCW about the centroid).
    rotation: f32,
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
}

impl FillType {
    fn w(&self) -> f32 {
        self.xmax - self.xmin
    }

    fn h(&self) -> f32 {
        self.ymax - self.ymin
    }

    /// Size along the growth axis of `direction` (x for HORIZONTAL, y for VERTICAL).
    fn along(&self, direction: FillDirection) -> f32 {
        match direction {
            FillDirection::Vertical => self.h(),
            _ => self.w(),
        }
    }

    /// Size across the growth axis (the axis parts stack along inside a strip).
    fn across(&self, direction: FillDirection) -> f32 {
        match direction {
            FillDirection::Vertical => self.w(),
            _ => self.h(),
        }
    }
}

/// Extents `(xmin, xmax, ymin, ymax)` of a part's bounding box rotated by `rotation` about its
/// centroid, relative to the centroid. Enumerating the four corners keeps the cardinal cases
/// exact up to the f32 residue of `sin_cos` and stays a safe over-estimate otherwise.
pub(crate) fn extent(p: &PreparedPart, rotation: f32) -> (f32, f32, f32, f32) {
    let (s, c) = rotation.sin_cos();
    let corners = [
        (-p.cx_off, -p.cy_off),
        (p.bbox_w - p.cx_off, -p.cy_off),
        (p.bbox_w - p.cx_off, p.bbox_h - p.cy_off),
        (-p.cx_off, p.bbox_h - p.cy_off),
    ];
    let mut ext = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    for (u, v) in corners {
        // Standard CCW rotation matrix, as jagua-rs applies it.
        let x = u * c - v * s;
        let y = u * s + v * c;
        ext.0 = ext.0.min(x);
        ext.1 = ext.1.max(x);
        ext.2 = ext.2.min(y);
        ext.3 = ext.3.max(y);
    }
    ext
}

fn fill_type(p: &PreparedPart, part_idx: usize, rotation: f32) -> FillType {
    let (xmin, xmax, ymin, ymax) = extent(p, rotation);
    FillType {
        part_idx,
        rotation,
        xmin,
        xmax,
        ymin,
        ymax,
    }
}

/// Which cardinal orientations a part may use. Same rules as the grid path, except that an
/// explicit per-part `allowedRotations` list wins over a global `amountOfRotations == 0` (the
/// documented `PartInput::allowed_rotations` contract; the LBF path already behaves this way).
fn orientations_allowed(part: &PartInput, amount_of_rotations: usize) -> (bool, bool) {
    let (allow_original, allow_swapped) = fit_orientations(&part.allowed_rotations);
    let explicit = effective_allowed(&part.allowed_rotations).is_some();
    (
        allow_original,
        allow_swapped && (amount_of_rotations != 0 || explicit),
    )
}

/// Pick the orientation that fits more copies on a full sheet (`grid_dims` cols × rows), 0° on a
/// tie; `None` when the part fits in no permitted orientation. The count is symmetric in the
/// direction, so the choice does not depend on it — only the traversal order does.
fn choose_orientation(
    p: &PreparedPart,
    part_idx: usize,
    allow_original: bool,
    allow_swapped: bool,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
) -> Option<(FillType, usize)> {
    let count = |t: &FillType| {
        let (cols, rows) = grid_dims(bin_w, bin_h, t.w(), t.h(), spacing);
        cols * rows
    };
    let original = allow_original.then(|| fill_type(p, part_idx, 0.0));
    let swapped = allow_swapped.then(|| fill_type(p, part_idx, FRAC_PI_2));
    let count_a = original.as_ref().map_or(0, count);
    let count_b = swapped.as_ref().map_or(0, count);
    if count_a == 0 && count_b == 0 {
        return None;
    }
    if count_a >= count_b {
        original.map(|t| (t, count_a))
    } else {
        swapped.map(|t| (t, count_b))
    }
}

/// One packed page: its placements (in fill order, so any prefix is compact along the growth
/// axis) and how far the block reaches along that axis, measured from the `TOP_LEFT` frame.
#[derive(Clone, Debug, Default, PartialEq)]
struct FillPage {
    placements: Vec<Placement>,
    used_along: f32,
}

/// Deterministic strip packer in the `TOP_LEFT` frame. Strips advance along the growth axis of
/// `direction`; parts stack across it from 0. A strip is as wide as its first part (callers pass
/// items sorted by descending along-dimension, so later parts never widen it); a part that does
/// not fit across opens the next strip, a strip that does not fit along opens the next page.
/// Items that fit on no page at all are skipped. No PRNG, no ties left to iteration order.
fn pack(
    items: &[FillType],
    direction: FillDirection,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
) -> Vec<FillPage> {
    let (bin_along, bin_across) = match direction {
        FillDirection::Vertical => (bin_h, bin_w),
        _ => (bin_w, bin_h),
    };
    let mut pages: Vec<FillPage> = Vec::new();
    let mut cur: Vec<Placement> = Vec::new();
    let (mut strip_start, mut strip_w, mut across, mut used) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    for it in items {
        let (a, c) = (it.along(direction), it.across(direction));
        if a > bin_along + EPS || c > bin_across + EPS {
            continue;
        }
        if across > 0.0 && (across + c > bin_across + EPS || a > strip_w + EPS) {
            strip_start += strip_w + spacing;
            strip_w = 0.0;
            across = 0.0;
        }
        if strip_start + a > bin_along + EPS {
            if !cur.is_empty() {
                pages.push(FillPage {
                    placements: std::mem::take(&mut cur),
                    used_along: used,
                });
            }
            strip_start = 0.0;
            strip_w = 0.0;
            across = 0.0;
        }
        let (cell_x, cell_y) = match direction {
            FillDirection::Vertical => (across, strip_start),
            _ => (strip_start, across),
        };
        cur.push(Placement {
            part_idx: it.part_idx,
            rotation: it.rotation,
            x: cell_x - it.xmin,
            y: cell_y - it.ymin,
        });
        across += c + spacing;
        strip_w = strip_w.max(a);
        used = strip_start + strip_w;
    }
    if !cur.is_empty() {
        pages.push(FillPage {
            placements: cur,
            used_along: used,
        });
    }
    pages
}

/// Fill the rectangle `(x0, y0, w, h)` with up to `max_count` bounding-box cells of one part type,
/// in the cardinal orientation that fits the rectangle best (same grain rules as the row/column
/// fill). The block is compact from the rectangle's left edge. Returns the placements in sheet
/// coordinates and how far the block reaches along x inside the rectangle. Used by the mixed-types
/// packer to use up rectangular gaps on shared leftover sheets (cutl-tests#77).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_rect(
    p: &PreparedPart,
    part_idx: usize,
    part: &PartInput,
    amount_of_rotations: usize,
    (x0, y0, w, h): (f32, f32, f32, f32),
    spacing: f32,
    max_count: usize,
) -> (Vec<Placement>, f32) {
    if w <= 0.0 || h <= 0.0 || max_count == 0 {
        return (Vec::new(), 0.0);
    }
    let (allow_original, allow_swapped) = orientations_allowed(part, amount_of_rotations);
    let Some((t, cap)) =
        choose_orientation(p, part_idx, allow_original, allow_swapped, w, h, spacing)
    else {
        return (Vec::new(), 0.0);
    };
    let n = cap.min(max_count);
    let Some(page) = pack(&vec![t; n], FillDirection::Horizontal, w, h, spacing)
        .into_iter()
        .next()
    else {
        return (Vec::new(), 0.0);
    };
    let placements = page
        .placements
        .into_iter()
        .map(|pl| Placement {
            x: pl.x + x0,
            y: pl.y + y0,
            ..pl
        })
        .collect();
    (placements, page.used_along)
}

/// Move a page packed in the `TOP_LEFT` frame to `corner`: every part's bounding-box cell is
/// reflected about the sheet's mid-line(s) and the part is re-seated inside its reflected cell
/// with its own orientation kept. A plain mirror of positions would misplace non-symmetric parts;
/// a mirror of outlines would be a different part — this is neither.
fn reflect_to_corner(
    placements: &mut [Placement],
    prepared: &[PreparedPart],
    corner: StartCorner,
    bin_w: f32,
    bin_h: f32,
) {
    if !corner.is_right() && !corner.is_bottom() {
        return;
    }
    for pl in placements.iter_mut() {
        let (xmin, xmax, ymin, ymax) = extent(&prepared[pl.part_idx], pl.rotation);
        if corner.is_right() {
            pl.x = bin_w - pl.x - (xmin + xmax);
        }
        if corner.is_bottom() {
            pl.y = bin_h - pl.y - (ymin + ymax);
        }
    }
}

/// Sort key for co-packing leftovers of several types: widest strip first, request order on
/// ties — deterministic.
fn leftover_order(a: &FillType, b: &FillType, direction: FillDirection) -> std::cmp::Ordering {
    b.along(direction)
        .partial_cmp(&a.along(direction))
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.part_idx.cmp(&b.part_idx))
}

// ---------------------------------------------------------------------------
// The remnant
// ---------------------------------------------------------------------------

/// The one rectangular remnant of a page: the strip beyond the packed block along the growth
/// axis, `spacing` after the last part and touching the sheet edge, full size across. Reflected
/// for the right / bottom corners. `None` when it has no area or, with a policy, when it is
/// thinner than the policy's minimum width or height (then the UI shows no remnant size).
fn remnant(
    used_along: f32,
    fill: SheetFill,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    policy: Option<&OffcutPolicy>,
) -> Option<Offcut> {
    let (x, y, width, height) = match fill.direction {
        FillDirection::Vertical => (
            0.0,
            used_along + spacing,
            bin_w,
            bin_h - used_along - spacing,
        ),
        _ => (
            used_along + spacing,
            0.0,
            bin_w - used_along - spacing,
            bin_h,
        ),
    };
    if width <= EPS || height <= EPS {
        return None;
    }
    if let Some(p) = policy
        && (width < p.min_offcut_width_mm || height < p.min_offcut_height_mm)
    {
        return None;
    }
    let x = if fill.corner.is_right() {
        bin_w - (x + width)
    } else {
        x
    };
    let y = if fill.corner.is_bottom() {
        bin_h - (y + height)
    } else {
        y
    };
    Some(Offcut::Rect {
        x,
        y,
        width,
        height,
    })
}

/// Write each page's remnant (0 or 1 RECT) into the rendered result and draw it on the page
/// SVGs, the way the LBF path draws detected offcuts.
fn apply_remnants(
    result: &mut NestingResult,
    used: &[f32],
    fill: SheetFill,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    policy: Option<&OffcutPolicy>,
) {
    for (page, &used_along) in result.pages.iter_mut().zip(used) {
        page.offcuts = remnant(used_along, fill, bin_w, bin_h, spacing, policy)
            .into_iter()
            .collect();
    }
    if result.pages.iter().any(|p| !p.offcuts.is_empty()) {
        let kerf = policy.map_or(0.0, |p| p.kerf_mm);
        overlay_result_offcuts(result, kerf, bin_w, bin_h);
    }
}

/// The row/column nest (`HORIZONTAL` / `VERTICAL`) for any number of part types of any shape.
///
/// Every type gets its own run of identical full sheets — its single-type stencil, `cap` parts —
/// and the leftovers of all types share the remainder sheets (the CUTL-195 contract). A type that
/// fits in no permitted orientation is left unplaced; the request errors only when nothing at all
/// can be placed (the caller then falls back to the general strategy, which reports it).
pub(crate) fn nest_fill(
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    parts: &[PartInput],
    amount_of_rotations: usize,
    fill: SheetFill,
    policy: Option<OffcutPolicy>,
) -> Result<NestingResult> {
    let rot_range = RotationRange::Discrete(vec![0.0, FRAC_PI_2]);
    let ranges: Vec<RotationRange> = vec![rot_range; parts.len()];
    let (prepared, ctx) = prepare(parts, &ranges, bin_w, bin_h, 1)?;
    let total: usize = parts.iter().map(|p| p.count).sum();

    let mut pages: Vec<FillPage> = Vec::new();
    let mut leftovers: Vec<FillType> = Vec::new();
    for (idx, part) in parts.iter().enumerate() {
        let (allow_original, allow_swapped) = orientations_allowed(part, amount_of_rotations);
        let Some((t, cap)) = choose_orientation(
            &prepared[idx],
            idx,
            allow_original,
            allow_swapped,
            bin_w,
            bin_h,
            spacing,
        ) else {
            log::warn!(
                "fill packer: part #{idx} (bbox {:.2}x{:.2}) fits the sheet {bin_w:.2}x{bin_h:.2} in no permitted orientation; leaving its {} copies unplaced",
                prepared[idx].bbox_w,
                prepared[idx].bbox_h,
                part.count
            );
            continue;
        };
        let stencil = pack(&vec![t; cap], fill.direction, bin_w, bin_h, spacing)
            .into_iter()
            .next()
            .unwrap_or_default();
        let cap = stencil.placements.len().max(1);
        let full = part.count / cap;
        let rem = part.count % cap;
        for _ in 0..full {
            pages.push(stencil.clone());
        }
        leftovers.extend(std::iter::repeat_n(t, rem));
    }
    leftovers.sort_by(|a, b| leftover_order(a, b, fill.direction));
    pages.extend(pack(&leftovers, fill.direction, bin_w, bin_h, spacing));
    if pages.is_empty() {
        anyhow::bail!("no part fits the sheet ({bin_w:.2}x{bin_h:.2}) in a permitted orientation");
    }
    let used: Vec<f32> = pages.iter().map(|p| p.used_along).collect();
    let mut pages: Vec<Vec<Placement>> = pages.into_iter().map(|p| p.placements).collect();
    for page in pages.iter_mut() {
        reflect_to_corner(page, &prepared, fill.corner, bin_w, bin_h);
    }

    let mut result = render_page_list(&ctx, &pages, total);
    apply_remnants(
        &mut result,
        &used,
        fill,
        bin_w,
        bin_h,
        spacing,
        policy.as_ref(),
    );
    Ok(result)
}

/// max_fit for the row/column fill: the single-type stencil (one full sheet) rendered as one page,
/// so "max parts per sheet" is exactly the count [`nest_fill`] repeats for that type.
pub(crate) fn nest_max_fit_fill(
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    part: &PartInput,
    amount_of_rotations: usize,
    fill: SheetFill,
) -> Result<NestingResult> {
    let rot_range = RotationRange::Discrete(vec![0.0, FRAC_PI_2]);
    let (prepared, ctx) = prepare(
        std::slice::from_ref(part),
        std::slice::from_ref(&rot_range),
        bin_w,
        bin_h,
        1,
    )?;
    let (allow_original, allow_swapped) = orientations_allowed(part, amount_of_rotations);
    let (t, cap) = choose_orientation(
        &prepared[0],
        0,
        allow_original,
        allow_swapped,
        bin_w,
        bin_h,
        spacing,
    )
    .ok_or_else(|| {
        anyhow::anyhow!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({bin_w:.2}x{bin_h:.2}) with spacing {spacing:.2}",
            prepared[0].bbox_w,
            prepared[0].bbox_h
        )
    })?;
    let stencil = pack(&vec![t; cap], fill.direction, bin_w, bin_h, spacing)
        .into_iter()
        .next()
        .unwrap_or_default();
    if stencil.placements.is_empty() {
        anyhow::bail!("Part does not fit in the bin ({bin_w:.2}x{bin_h:.2})");
    }
    let mut placements = stencil.placements;
    reflect_to_corner(&mut placements, &prepared, fill.corner, bin_w, bin_h);
    let cap = placements.len();
    let mut result = render_page_list(&ctx, &[placements], cap);
    apply_remnants(
        &mut result,
        &[stencil.used_along],
        fill,
        bin_w,
        bin_h,
        spacing,
        None,
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_are_screaming_snake() {
        assert_eq!(
            serde_json::to_string(&FillDirection::Horizontal).unwrap(),
            "\"HORIZONTAL\""
        );
        assert_eq!(
            serde_json::to_string(&StartCorner::BottomRight).unwrap(),
            "\"BOTTOM_RIGHT\""
        );
        let d: FillDirection = serde_json::from_str("\"STAIRCASE\"").unwrap();
        assert_eq!(d, FillDirection::Staircase);
        let c: StartCorner = serde_json::from_str("\"TOP_LEFT\"").unwrap();
        assert_eq!(c, StartCorner::TopLeft);
    }

    fn prep(bbox_w: f32, bbox_h: f32, cx_off: f32, cy_off: f32) -> PreparedPart {
        PreparedPart {
            bbox_w,
            bbox_h,
            cx_off,
            cy_off,
        }
    }

    #[test]
    fn extents_are_exact_for_the_cardinal_orientations() {
        let p = prep(100.0, 40.0, 30.0, 10.0);
        let (xmin, xmax, ymin, ymax) = extent(&p, 0.0);
        assert!((xmin + 30.0).abs() < 1e-4 && (xmax - 70.0).abs() < 1e-4);
        assert!((ymin + 10.0).abs() < 1e-4 && (ymax - 30.0).abs() < 1e-4);
        // 90° CCW: (u, v) → (-v, u), so x-extents come from -v, y-extents from u.
        let (xmin, xmax, ymin, ymax) = extent(&p, FRAC_PI_2);
        assert!(
            (xmin + 30.0).abs() < 1e-3 && (xmax - 10.0).abs() < 1e-3,
            "{xmin} {xmax}"
        );
        assert!(
            (ymin + 30.0).abs() < 1e-3 && (ymax - 70.0).abs() < 1e-3,
            "{ymin} {ymax}"
        );
    }

    #[test]
    fn horizontal_strips_grow_along_x_and_a_prefix_is_compact() {
        // 100x40 parts on a 350x100 sheet, spacing 10: 2 per strip (40+10+40 ≤ 100), strips at
        // x = 0, 110, 220 (330 ≤ 350) ⇒ cap 6.
        let p = prep(100.0, 40.0, 50.0, 20.0);
        let t = fill_type(&p, 0, 0.0);
        let pages = pack(&[t; 6], FillDirection::Horizontal, 350.0, 100.0, 10.0);
        assert_eq!(pages.len(), 1);
        let cells: Vec<(f32, f32)> = pages[0]
            .placements
            .iter()
            .map(|pl| (pl.x + t.xmin, pl.y + t.ymin))
            .collect();
        assert_eq!(
            cells,
            vec![
                (0.0, 0.0),
                (0.0, 50.0),
                (110.0, 0.0),
                (110.0, 50.0),
                (220.0, 0.0),
                (220.0, 50.0)
            ]
        );
        assert!((pages[0].used_along - 320.0).abs() < 1e-4);
        // Three parts: one full strip + one, block reaches x = 210.
        let pages = pack(&[t; 3], FillDirection::Horizontal, 350.0, 100.0, 10.0);
        assert_eq!(pages[0].placements.len(), 3);
        assert!((pages[0].used_along - 210.0).abs() < 1e-4);
        // Seven parts overflow to a second page holding one.
        let pages = pack(&[t; 7], FillDirection::Horizontal, 350.0, 100.0, 10.0);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[1].placements.len(), 1);
    }

    #[test]
    fn vertical_is_the_transpose() {
        let p = prep(40.0, 100.0, 20.0, 50.0);
        let t = fill_type(&p, 0, 0.0);
        let pages = pack(&[t; 3], FillDirection::Vertical, 100.0, 350.0, 10.0);
        let cells: Vec<(f32, f32)> = pages[0]
            .placements
            .iter()
            .map(|pl| (pl.x + t.xmin, pl.y + t.ymin))
            .collect();
        assert_eq!(cells, vec![(0.0, 0.0), (50.0, 0.0), (0.0, 110.0)]);
        assert!((pages[0].used_along - 210.0).abs() < 1e-4);
    }

    #[test]
    fn reflection_keeps_the_cell_inside_the_sheet_and_the_rotation() {
        let prepared = vec![prep(100.0, 40.0, 30.0, 10.0)]; // asymmetric centroid
        let p = &prepared[0];
        let t = fill_type(p, 0, FRAC_PI_2);
        let mut pls = pack(&[t], FillDirection::Horizontal, 300.0, 200.0, 5.0)[0]
            .placements
            .clone();
        reflect_to_corner(&mut pls, &prepared, StartCorner::BottomRight, 300.0, 200.0);
        let pl = pls[0];
        assert_eq!(pl.rotation, FRAC_PI_2);
        let (xmin, xmax, ymin, ymax) = extent(p, pl.rotation);
        // The rotated bbox is 40 wide, 100 tall and now hugs the bottom-right corner.
        assert!((pl.x + xmax - 300.0).abs() < 1e-3, "{}", pl.x + xmax);
        assert!((pl.x + xmin - 260.0).abs() < 1e-3);
        assert!((pl.y + ymax - 200.0).abs() < 1e-3);
        assert!((pl.y + ymin - 100.0).abs() < 1e-3);
    }

    #[test]
    fn orientation_with_more_copies_wins_and_zero_is_the_tiebreak() {
        // 100x40 on 350x100 with spacing 10: 0° ⇒ 3×2 = 6; 90° (40x100) ⇒ 7×1 = 7.
        let p = prep(100.0, 40.0, 50.0, 20.0);
        let (t, cap) = choose_orientation(&p, 0, true, true, 350.0, 100.0, 10.0).unwrap();
        assert_eq!((t.rotation, cap), (FRAC_PI_2, 7));
        let (t, cap) = choose_orientation(&p, 0, true, false, 350.0, 100.0, 10.0).unwrap();
        assert_eq!((t.rotation, cap), (0.0, 6));
        // Square: tie ⇒ 0°.
        let sq = prep(50.0, 50.0, 25.0, 25.0);
        let (t, _) = choose_orientation(&sq, 0, true, true, 350.0, 100.0, 10.0).unwrap();
        assert_eq!(t.rotation, 0.0);
        // Nothing fits ⇒ None.
        assert!(choose_orientation(&p, 0, true, true, 90.0, 30.0, 0.0).is_none());
    }

    fn policy(min_w: f32, min_h: f32) -> OffcutPolicy {
        OffcutPolicy {
            min_offcut_width_mm: min_w,
            min_offcut_height_mm: min_h,
            shape: crate::svg_nesting::offcut::OffcutShape::Rectangle,
            kerf_mm: 0.0,
        }
    }

    #[test]
    fn remnant_is_the_far_strip_spacing_after_the_block() {
        let h = |c| SheetFill::new(FillDirection::Horizontal, c);
        let v = |c| SheetFill::new(FillDirection::Vertical, c);
        let rect = |x, y, width, height| {
            Some(Offcut::Rect {
                x,
                y,
                width,
                height,
            })
        };
        let tl = StartCorner::TopLeft;
        assert_eq!(
            remnant(320.0, h(tl), 1000.0, 500.0, 2.0, None),
            rect(322.0, 0.0, 678.0, 500.0)
        );
        assert_eq!(
            remnant(320.0, h(StartCorner::BottomRight), 1000.0, 500.0, 2.0, None),
            rect(0.0, 0.0, 678.0, 500.0)
        );
        assert_eq!(
            remnant(150.0, v(tl), 1000.0, 500.0, 2.0, None),
            rect(0.0, 152.0, 1000.0, 348.0)
        );
        assert_eq!(
            remnant(150.0, v(StartCorner::BottomLeft), 1000.0, 500.0, 2.0, None),
            rect(0.0, 0.0, 1000.0, 348.0)
        );
        // A block that reaches the edge (or within spacing of it) leaves no remnant.
        assert_eq!(remnant(998.0, h(tl), 1000.0, 500.0, 2.0, None), None);
        assert_eq!(remnant(1000.0, h(tl), 1000.0, 500.0, 2.0, None), None);
        // Policy minimums: below either ⇒ omitted; at the minimum ⇒ kept.
        let p = policy(700.0, 100.0);
        assert_eq!(remnant(320.0, h(tl), 1000.0, 500.0, 2.0, Some(&p)), None);
        let p = policy(678.0, 500.0);
        assert!(remnant(320.0, h(tl), 1000.0, 500.0, 2.0, Some(&p)).is_some());
    }

    #[test]
    fn defaults_are_todays_behaviour() {
        let fill = SheetFill::default();
        assert!(fill.is_staircase());
        assert_eq!(fill.corner, StartCorner::TopLeft);
        assert!(!fill.corner.is_right() && !fill.corner.is_bottom());
        assert!(StartCorner::BottomRight.is_right() && StartCorner::BottomRight.is_bottom());
    }
}
