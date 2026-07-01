//! Pairing fast path (WS-4) for half-bbox parts such as right triangles («косынка»).
//!
//! Two such parts — one rotated 180° — tile their common bounding rectangle. We grid-pack that
//! *composite rectangle* (reusing [`grid_single_sheet`], including its two-orientation guillotine
//! when rotation is allowed) and drop the two oppositely-oriented parts into each cell. Every part
//! is cut separately, so the two triangles are held a full `spacing` kerf apart on their shared
//! hypotenuse (not abutting), and the pair-rectangles keep `spacing` from each other too. This
//! turns the QA triangle case (`docs/bugfix/image (8)/(11)`, ~66% / 20s with LBF) into a dense,
//! regular layout deterministically and instantly.

use crate::svg_nesting::grid::grid_single_sheet;
use crate::svg_nesting::render::{Placement, prepare, render_periodic};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use std::f32::consts::{FRAC_PI_2, PI};

/// Rotate `(a, b)` by `theta` radians (CCW).
fn rotate(a: f32, b: f32, theta: f32) -> (f32, f32) {
    let (s, c) = theta.sin_cos();
    (a * c - b * s, a * s + b * c)
}

/// Build the single-sheet pairing stencil (render context + placements) for one half-bbox part type.
/// Shared by the periodic nest and the max-fit path so their per-sheet capacity is, by construction,
/// identical (the irregular/pairable analogue of the rectangle WS-7 fix).
fn pairing_stencil(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    allow_swap: bool,
) -> Result<(crate::svg_nesting::render::RenderContext, Vec<Placement>)> {
    // Rotation range only feeds `Item` construction; placements below are explicit (0/90/180/270).
    let rot_range = RotationRange::Discrete(vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
    let parts = std::slice::from_ref(part);
    let (prepared, ctx) = prepare(
        parts,
        std::slice::from_ref(&rot_range),
        bin_width,
        bin_height,
        1,
    )?;
    let p = &prepared[0];
    let (bw, bh) = (p.bbox_w, p.bbox_h);

    // The two triangles are cut as separate parts, so they must be `spacing` apart on their shared
    // hypotenuse too (not abutting). Push each half away from the diagonal by spacing/2 along the
    // hypotenuse normal. The right-angle corner is the bbox corner the centroid sits toward; the
    // hypotenuse joins the two corners adjacent to it.
    let rac_x = if p.cx_off < bw / 2.0 { 0.0 } else { bw };
    let rac_y = if p.cy_off < bh / 2.0 { 0.0 } else { bh };
    let hyp_dir = (bw - 2.0 * rac_x, 2.0 * rac_y - bh); // adj2 - adj1 along the bbox diagonal
    let mut nrm = (-hyp_dir.1, hyp_dir.0); // perpendicular to the hypotenuse
    let nlen = (nrm.0 * nrm.0 + nrm.1 * nrm.1)
        .sqrt()
        .max(f32::MIN_POSITIVE);
    nrm = (nrm.0 / nlen, nrm.1 / nlen);
    // Orient the normal toward part A's half (the right-angle-corner side).
    let to_rac = (rac_x - bw / 2.0, rac_y - bh / 2.0);
    if to_rac.0 * nrm.0 + to_rac.1 * nrm.1 < 0.0 {
        nrm = (-nrm.0, -nrm.1);
    }
    let half = spacing / 2.0;
    // A's offset from the pair centre = its centroid offset + the half-kerf push off the diagonal.
    let (arx, ary) = (
        (p.cx_off - bw / 2.0) + nrm.0 * half,
        (p.cy_off - bh / 2.0) + nrm.1 * half,
    );
    // The push spreads the pair beyond the bbox; grow the pair-rectangle so the kerf to neighbouring
    // pairs stays a full `spacing`.
    let (cell_w, cell_h) = (bw + spacing * nrm.0.abs(), bh + spacing * nrm.1.abs());

    // Grid the (grown) pair-rectangle with the real kerf between pairs. Two-orientation when allowed.
    let composite = grid_single_sheet(
        cell_w, cell_h, bin_width, bin_height, spacing, 0, allow_swap,
    );
    if composite.is_empty() {
        anyhow::bail!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2}) with spacing {:.2}",
            bw,
            bh,
            bin_width,
            bin_height,
            spacing
        );
    }

    let mut stencil: Vec<Placement> = Vec::with_capacity(composite.len() * 2);
    for cell in &composite {
        // The composite may be axis-aligned (rotation 0) or 90°-rotated; rotate the part offsets and
        // the part orientations by the composite rotation so the pair still tiles the (rotated) cell.
        let (dx, dy) = rotate(arx, ary, cell.rotation);
        stencil.push(Placement {
            part_idx: 0,
            rotation: cell.rotation,
            x: cell.x + dx,
            y: cell.y + dy,
        });
        stencil.push(Placement {
            part_idx: 0,
            rotation: cell.rotation + PI,
            x: cell.x - dx,
            y: cell.y - dy,
        });
    }
    Ok((ctx, stencil))
}

/// Periodic pairing pack of a single half-bbox part type. `allow_swap` lets the pair-rectangle use
/// the 90° orientation (denser on some bins) when the part's grain permits it.
pub(crate) fn nest_pairing(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    allow_swap: bool,
) -> Result<NestingResult> {
    let (ctx, stencil) = pairing_stencil(bin_width, bin_height, spacing, part, allow_swap)?;
    let cap = stencil.len();
    let qty = part.count;
    let full_sheets = qty / cap;
    let rem = qty % cap;
    let remainder: Vec<_> = stencil.iter().take(rem).copied().collect();

    Ok(render_periodic(
        &ctx,
        &stencil,
        full_sheets,
        &remainder,
        qty,
    ))
}

/// Max copies of a single half-bbox part on ONE sheet (the pairing stencil), rendered as a single
/// full page — so max-fit's "max parts per sheet" equals [`nest_pairing`]'s periodic full-sheet count.
pub(crate) fn nest_max_fit_pairing(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    allow_swap: bool,
) -> Result<NestingResult> {
    let (ctx, stencil) = pairing_stencil(bin_width, bin_height, spacing, part, allow_swap)?;
    let cap = stencil.len();
    Ok(render_periodic(&ctx, &stencil, 1, &[], cap))
}
