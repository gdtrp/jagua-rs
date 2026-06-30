//! Pairing fast path (WS-4) for half-bbox parts such as right triangles («косынка»).
//!
//! Two such parts — one rotated 180° — tile their common bounding rectangle. We grid-pack that
//! *composite rectangle* (reusing [`grid_single_sheet`]) and then drop the two oppositely-oriented
//! parts into each cell, separated by a `spacing` kerf. This turns the QA triangle case
//! (`docs/bugfix/image (8)/(11)`, ~66% density, misaligned last row) into a dense, regular layout.

use crate::svg_nesting::grid::grid_single_sheet;
use crate::svg_nesting::render::{Placement, prepare, render_periodic};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use std::f32::consts::PI;

/// Periodic pairing pack of a single half-bbox part type.
pub(crate) fn nest_pairing(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
) -> Result<NestingResult> {
    // The part is placed at 0° and 180° (the pair).
    let rot_range = RotationRange::Discrete(vec![0.0, PI]);
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

    // Grid the composite (= the part's bbox). Swap is disabled: rotating the whole pair compounds
    // the per-part rotation/offset maths; the un-swapped grid already packs these densely.
    let composite = grid_single_sheet(bw, bh, bin_width, bin_height, spacing, 0, false);
    if composite.is_empty() {
        anyhow::bail!(
            "Part (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2})",
            bw,
            bh,
            bin_width,
            bin_height
        );
    }

    // Kerf direction = the line separating the two centroids; shifting them apart by spacing/2 each
    // opens a real gap along the shared diagonal so the two parts don't touch.
    let sep_x = bw - 2.0 * p.cx_off;
    let sep_y = bh - 2.0 * p.cy_off;
    let sep_norm = (sep_x * sep_x + sep_y * sep_y)
        .sqrt()
        .max(f32::MIN_POSITIVE);
    let (ux, uy) = (sep_x / sep_norm, sep_y / sep_norm);
    let half_kerf = spacing / 2.0;

    let mut stencil: Vec<Placement> = Vec::with_capacity(composite.len() * 2);
    for cell in &composite {
        // Composite cell centre → lower-left origin of the bbox.
        let ox = cell.x - bw / 2.0;
        let oy = cell.y - bh / 2.0;
        // Part A (0°): bbox aligned to the cell origin, nudged away from the diagonal.
        stencil.push(Placement {
            part_idx: 0,
            rotation: 0.0,
            x: ox + p.cx_off - ux * half_kerf,
            y: oy + p.cy_off - uy * half_kerf,
        });
        // Part B (180°): its bbox-min maps to the cell's far corner, nudged the other way.
        stencil.push(Placement {
            part_idx: 0,
            rotation: PI,
            x: ox + (bw - p.cx_off) + ux * half_kerf,
            y: oy + (bh - p.cy_off) + uy * half_kerf,
        });
    }

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
