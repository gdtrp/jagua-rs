//! Periodic packing (WS-3): one stencil sheet repeated K times + a remainder sheet.
//!
//! This is the fix for "K different sheets". For a single part type at quantity `Q` with `cap`
//! parts per stencil sheet, we emit `K = ⌊Q/cap⌋` byte-identical full sheets plus one remainder
//! sheet of the leftover `Q mod cap` parts. The stencil is rendered once and cloned (see
//! `render::render_periodic`), so bulk runs are fast and every full sheet is visually identical.

use crate::svg_nesting::grid::grid_single_sheet;
use crate::svg_nesting::render::{prepare, render_periodic};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use std::f32::consts::FRAC_PI_2;

/// Periodic packing of a single **rectangular** part type using a grid stencil.
///
/// `allow_swap` permits the 90° orientation (cardinal grid). Assumes `part` fits in the bin in the
/// 0° orientation (the classifier guarantees this before routing here).
pub(crate) fn nest_periodic_grid(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    allow_swap: bool,
) -> Result<NestingResult> {
    let rot_range = if allow_swap {
        RotationRange::Discrete(vec![0.0, FRAC_PI_2])
    } else {
        RotationRange::Discrete(vec![0.0])
    };
    let parts = std::slice::from_ref(part);
    let (prepared, ctx) = prepare(
        parts,
        std::slice::from_ref(&rot_range),
        bin_width,
        bin_height,
        1,
    )?;
    let p = &prepared[0];

    let stencil = grid_single_sheet(
        p.bbox_w, p.bbox_h, bin_width, bin_height, spacing, 0, allow_swap,
    );
    let cap = stencil.len();
    if cap == 0 {
        anyhow::bail!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2}) with spacing {:.2}",
            p.bbox_w,
            p.bbox_h,
            bin_width,
            bin_height,
            spacing
        );
    }

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

/// Max copies of a single **rectangular** part on ONE sheet (the stencil), rendered as a single
/// full page. Used by the max_fit path so its "max parts per sheet" equals the periodic full-sheet
/// count — the deterministic value that makes the #4 bug (44 vs 45) impossible.
pub(crate) fn nest_max_fit_grid(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    allow_swap: bool,
) -> Result<NestingResult> {
    let rot_range = if allow_swap {
        RotationRange::Discrete(vec![0.0, FRAC_PI_2])
    } else {
        RotationRange::Discrete(vec![0.0])
    };
    let parts = std::slice::from_ref(part);
    let (prepared, ctx) = prepare(
        parts,
        std::slice::from_ref(&rot_range),
        bin_width,
        bin_height,
        1,
    )?;
    let p = &prepared[0];
    let stencil = grid_single_sheet(
        p.bbox_w, p.bbox_h, bin_width, bin_height, spacing, 0, allow_swap,
    );
    if stencil.is_empty() {
        anyhow::bail!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2}) with spacing {:.2}",
            p.bbox_w,
            p.bbox_h,
            bin_width,
            bin_height,
            spacing
        );
    }
    let cap = stencil.len();
    Ok(render_periodic(&ctx, &stencil, 1, &[], cap))
}
