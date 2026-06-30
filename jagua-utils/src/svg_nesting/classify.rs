//! Request classifier + router (WS-1).
//!
//! `classify` buckets a nesting request by cheap geometry; `nest_auto` routes each bucket to the
//! cheapest correct packer, falling back to the general LBF strategy (`AdaptiveNestingStrategy`)
//! for anything the fast paths don't (yet) handle. The General path is byte-for-byte unchanged.

use crate::svg_nesting::lattice::nest_max_fit_lattice;
use crate::svg_nesting::mixed::nest_mixed;
use crate::svg_nesting::pairing::{nest_max_fit_pairing, nest_pairing};
use crate::svg_nesting::periodic::{nest_max_fit_grid, nest_periodic_grid};
use crate::svg_nesting::render::measure_part;
use crate::svg_nesting::strategy::{
    AdaptiveNestingStrategy, ImprovementCallback, NestingStrategy, PartInput, effective_allowed,
    fit_orientations,
};
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;

/// Which packer to run. Production always uses `Auto` — jagua-rs picks the packer from the incoming
/// part shapes (see [`classify`]). The explicit variants exist only for tests/diagnostics that want
/// to force a path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PackingMode {
    /// Decide from the part geometry (the classifier). The only mode used in production.
    #[default]
    Auto,
    /// Force the deterministic grid/periodic path (single part type only; else falls to General).
    Grid,
    /// Alias for `Grid`.
    Periodic,
    /// Force the general LBF strategy.
    General,
}

/// The geometric bucket a request falls into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PackingClass {
    /// One part type whose bbox nearly fills the sheet (≈1 per sheet) → periodic grid.
    SingleHighFill,
    /// One (near-)rectangular part type in bulk → periodic grid.
    SingleRectangle,
    /// One part type whose area ≈ ½ its bbox (right triangle etc.) → pairing (Phase 3).
    SinglePairable,
    /// One irregular part type → General (periodic-LBF is a later phase).
    SingleIrregular,
    /// 2–4 rectangular part types → mixed grouping (per-type identical sheets + shelf remainder).
    MixedFewTypes,
    /// Anything else → General LBF.
    General,
}

const HIGH_FILL_RATIO: f32 = 0.95;
const RECT_RATIO: f32 = 0.98;
const PAIR_LO: f32 = 0.40;
const PAIR_HI: f32 = 0.62;
const MIXED_MAX_TYPES: usize = 4;
const MIXED_RECT_RATIO: f32 = 0.90;

/// Classify a request by cheap per-part geometry. Never fails the caller — measurement errors fall
/// back to `General` (the strategy then surfaces the real parse error).
pub(crate) fn classify(parts: &[PartInput], bin_w: f32, bin_h: f32) -> PackingClass {
    let bin_area = (bin_w * bin_h).max(f32::MIN_POSITIVE);

    if parts.len() == 1 {
        let m = match measure_part(&parts[0]) {
            Ok(m) => m,
            Err(_) => return PackingClass::General,
        };
        let fill = (m.bbox_w * m.bbox_h) / bin_area;
        let rect = m.rectangularity();
        if fill > HIGH_FILL_RATIO {
            PackingClass::SingleHighFill
        } else if rect >= RECT_RATIO {
            PackingClass::SingleRectangle
        } else if (PAIR_LO..=PAIR_HI).contains(&rect) && m.n_vertices <= 5 && m.n_holes == 0 {
            PackingClass::SinglePairable
        } else {
            PackingClass::SingleIrregular
        }
    } else if (2..=MIXED_MAX_TYPES).contains(&parts.len()) {
        let all_rect = parts.iter().all(|p| {
            measure_part(p)
                .map(|m| m.rectangularity() >= MIXED_RECT_RATIO)
                .unwrap_or(false)
        });
        if all_rect {
            PackingClass::MixedFewTypes
        } else {
            PackingClass::General
        }
    } else {
        PackingClass::General
    }
}

/// Whether the single part may use the 90° orientation (cardinal grid), given the global rotation
/// count and any per-part grain constraint.
fn single_part_allow_swap(part: &PartInput, amount_of_rotations: usize) -> bool {
    if amount_of_rotations == 0 {
        return false;
    }
    let (_allow_original, allow_swapped) = fit_orientations(&part.allowed_rotations);
    allow_swapped
}

/// Whether the single part may be placed at 0° (grid assumes the un-rotated orientation is legal).
fn single_part_allow_original(part: &PartInput) -> bool {
    let (allow_original, _allow_swapped) = fit_orientations(&part.allowed_rotations);
    allow_original
}

/// Classify the request and run the cheapest correct packer; fall back to the general strategy.
#[allow(clippy::too_many_arguments)]
pub fn nest_auto(
    strategy: &AdaptiveNestingStrategy,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    parts: &[PartInput],
    amount_of_rotations: usize,
    mode: PackingMode,
    improvement_callback: Option<ImprovementCallback>,
) -> Result<NestingResult> {
    let general = |cb| {
        strategy.nest(
            bin_width,
            bin_height,
            spacing,
            parts,
            amount_of_rotations,
            cb,
        )
    };

    match mode {
        PackingMode::General => return general(improvement_callback),
        PackingMode::Grid | PackingMode::Periodic => {
            // Forced grid: only valid for a single part type that can sit at 0°.
            if parts.len() == 1 && single_part_allow_original(&parts[0]) {
                let allow_swap = single_part_allow_swap(&parts[0], amount_of_rotations);
                return nest_periodic_grid(bin_width, bin_height, spacing, &parts[0], allow_swap);
            }
            return general(improvement_callback);
        }
        PackingMode::Auto => {}
    }

    let class = classify(parts, bin_width, bin_height);
    match class {
        PackingClass::SingleHighFill | PackingClass::SingleRectangle
            if single_part_allow_original(&parts[0]) =>
        {
            let allow_swap = single_part_allow_swap(&parts[0], amount_of_rotations);
            nest_periodic_grid(bin_width, bin_height, spacing, &parts[0], allow_swap)
        }
        // Pairing needs the grain-preserving 0°+180° flip; allowed when the part is unconstrained
        // (`None` or `[]` ⇒ any rotation). `allow_swap` lets the pair-rectangle also use the 90°
        // orientation when the part permits it (denser on some bins).
        PackingClass::SinglePairable
            if amount_of_rotations != 0
                && effective_allowed(&parts[0].allowed_rotations).is_none() =>
        {
            let allow_swap = fit_orientations(&parts[0].allowed_rotations).1;
            nest_pairing(bin_width, bin_height, spacing, &parts[0], allow_swap)
        }
        // Mixed rectangular types: per-type identical sheets + shelf-packed shared remainder.
        // nest_mixed is grain-aware (per-part 90° permission), so grain-locked parts
        // (`allowedRotations: []` ⇒ 0° only) still grid-pack. Requires every part to be placeable
        // at 0° (the grid's primary orientation); otherwise fall through to General, as does a
        // part that doesn't fit (nest_mixed errors).
        PackingClass::MixedFewTypes if parts.iter().all(single_part_allow_original) => {
            match nest_mixed(bin_width, bin_height, spacing, parts, amount_of_rotations) {
                Ok(r) => Ok(r),
                Err(_) => general(improvement_callback),
            }
        }
        // Single irregular (trapezoid / parallelogram / L-shape / concave / notched …) → the lattice
        // packer: densest periodic packing via the no-fit-polygon double lattice, with a bbox-grid
        // floor inside (so all parts are always placed). Replaces the slow/inconsistent LBF for
        // single irregular parts; falls back to General only if the part can't fit at all.
        PackingClass::SingleIrregular if parts.len() == 1 => {
            let (rots, allow_double) = lattice_rotations(&parts[0], amount_of_rotations);
            match crate::svg_nesting::lattice::nest_lattice(
                bin_width,
                bin_height,
                spacing,
                &parts[0],
                &rots,
                allow_double,
            ) {
                Ok(r) => Ok(r),
                Err(_) => general(improvement_callback),
            }
        }
        // Not yet specialised — use the proven general path.
        PackingClass::SingleHighFill
        | PackingClass::SingleRectangle
        | PackingClass::SinglePairable
        | PackingClass::SingleIrregular
        | PackingClass::MixedFewTypes
        | PackingClass::General => general(improvement_callback),
    }
}

/// Base orientations (radians) + whether the 180° double lattice is allowed, for the lattice packer.
/// Unconstrained parts try {0°, 90°} (the double lattice covers 180°/270° and the interlock); a
/// grain list uses its angles, enabling the double lattice only when 180° is permitted.
fn lattice_rotations(part: &PartInput, amount_of_rotations: usize) -> (Vec<f32>, bool) {
    use std::f32::consts::FRAC_PI_2;
    match effective_allowed(&part.allowed_rotations) {
        None => {
            if amount_of_rotations == 0 {
                (vec![0.0], false)
            } else {
                (vec![0.0, FRAC_PI_2], true)
            }
        }
        Some(angles) => {
            let rots: Vec<f32> = angles.iter().map(|d| d.to_radians()).collect();
            let allow_double = angles
                .iter()
                .any(|&d| (d.rem_euclid(360.0) - 180.0).abs() < 1.0);
            (if rots.is_empty() { vec![0.0] } else { rots }, allow_double)
        }
    }
}

/// max_fit ("max copies on one sheet") with auto routing. Mirrors [`nest_auto`]'s per-class routing
/// so the reported "max parts per sheet" is the **same deterministic single-sheet stencil** the real
/// nest repeats — grid (rectangles/high-fill), pairing (half-bbox), or the lattice (everything else
/// irregular). They share each path's stencil builder, so max-fit can never disagree with the
/// periodic full-sheet count for any deterministic class (the general WS-7 guarantee). The LBF
/// `nest_max_fit` is only a last resort for parts no deterministic path accepts (or `General` mode).
#[allow(clippy::too_many_arguments)]
pub fn nest_max_fit_auto(
    strategy: &AdaptiveNestingStrategy,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    amount_of_rotations: usize,
    mode: PackingMode,
    improvement_callback: Option<ImprovementCallback>,
) -> Result<NestingResult> {
    if mode != PackingMode::General {
        // Forced grid/periodic: deterministic grid stencil (single part that can sit at 0°).
        if matches!(mode, PackingMode::Grid | PackingMode::Periodic) {
            if single_part_allow_original(part) {
                let allow_swap = single_part_allow_swap(part, amount_of_rotations);
                return nest_max_fit_grid(bin_width, bin_height, spacing, part, allow_swap);
            }
            // forced mode but part can't sit at 0° → fall through to LBF below
        } else {
            // Auto: route by class exactly like `nest_auto`, each via the matching stencil builder.
            let class = classify(std::slice::from_ref(part), bin_width, bin_height);
            let det: Option<Result<NestingResult>> = match class {
                PackingClass::SingleHighFill | PackingClass::SingleRectangle
                    if single_part_allow_original(part) =>
                {
                    let allow_swap = single_part_allow_swap(part, amount_of_rotations);
                    Some(nest_max_fit_grid(
                        bin_width, bin_height, spacing, part, allow_swap,
                    ))
                }
                PackingClass::SinglePairable
                    if amount_of_rotations != 0
                        && effective_allowed(&part.allowed_rotations).is_none() =>
                {
                    let allow_swap = fit_orientations(&part.allowed_rotations).1;
                    Some(nest_max_fit_pairing(
                        bin_width, bin_height, spacing, part, allow_swap,
                    ))
                }
                PackingClass::SingleHighFill
                | PackingClass::SingleRectangle
                | PackingClass::SinglePairable
                | PackingClass::SingleIrregular => {
                    let (rots, allow_double) = lattice_rotations(part, amount_of_rotations);
                    Some(nest_max_fit_lattice(
                        bin_width,
                        bin_height,
                        spacing,
                        part,
                        &rots,
                        allow_double,
                    ))
                }
                PackingClass::MixedFewTypes | PackingClass::General => None,
            };
            // A deterministic path that succeeds wins; if it errors (e.g. part doesn't fit) fall
            // through to LBF, which reports the same impossibility.
            if let Some(Ok(r)) = det {
                return Ok(r);
            }
        }
    }
    strategy.nest_max_fit(
        bin_width,
        bin_height,
        spacing,
        part,
        amount_of_rotations,
        improvement_callback,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(svg: &str, count: usize) -> PartInput {
        PartInput {
            svg_bytes: svg.as_bytes().to_vec(),
            count,
            item_id: None,
            allowed_rotations: None,
        }
    }

    fn rect_svg(w: f32, h: f32) -> String {
        format!(
            r#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="M 0,0 L {w},0 L {w},{h} L 0,{h} Z" fill="black"/></svg>"#
        )
    }

    fn frame_svg(w: f32, h: f32) -> String {
        // outer rect + centred rectangular hole
        let (hx0, hy0, hx1, hy1) = (w * 0.25, h * 0.25, w * 0.75, h * 0.75);
        format!(
            r#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="M 0,0 L {w},0 L {w},{h} L 0,{h} Z M {hx0},{hy0} L {hx1},{hy0} L {hx1},{hy1} L {hx0},{hy1} Z" fill="black"/></svg>"#
        )
    }

    fn triangle_svg(w: f32, h: f32) -> String {
        format!(
            r#"<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}"><path d="M 0,0 L {w},0 L 0,{h} Z" fill="black"/></svg>"#
        )
    }

    #[test]
    fn rectangle_classifies_as_single_rectangle() {
        let p = part(&rect_svg(200.0, 295.0), 4000);
        assert_eq!(classify(&[p], 980.0, 2000.0), PackingClass::SingleRectangle);
    }

    #[test]
    fn frame_with_hole_classifies_as_single_rectangle() {
        // A frame's outer ring is a rectangle (area ratio ~1), so it grid-packs.
        let p = part(&frame_svg(200.0, 295.0), 4000);
        assert_eq!(classify(&[p], 980.0, 2000.0), PackingClass::SingleRectangle);
    }

    #[test]
    fn right_triangle_classifies_as_pairable() {
        let p = part(&triangle_svg(300.0, 100.0), 24);
        assert_eq!(classify(&[p], 980.0, 2000.0), PackingClass::SinglePairable);
    }

    #[test]
    fn near_sheet_rectangle_classifies_as_high_fill() {
        let p = part(&rect_svg(970.0, 1980.0), 10);
        assert_eq!(classify(&[p], 980.0, 2000.0), PackingClass::SingleHighFill);
    }

    #[test]
    fn many_rect_types_classify_as_mixed() {
        let parts = vec![
            part(&rect_svg(100.0, 100.0), 10),
            part(&rect_svg(120.0, 80.0), 10),
            part(&rect_svg(200.0, 50.0), 10),
        ];
        assert_eq!(classify(&parts, 980.0, 2000.0), PackingClass::MixedFewTypes);
    }
}
