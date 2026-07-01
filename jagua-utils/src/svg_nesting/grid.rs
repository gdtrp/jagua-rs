//! Closed-form rectangular grid packer (WS-2).
//!
//! For an axis-aligned rectangle on a rectangular sheet this computes the densest single-sheet
//! tiling directly — no LBF. It uses a **two-orientation guillotine**: lay the part out in the
//! orientation that fits the most copies, then fill the leftover right and bottom strips with the
//! 90°-rotated orientation (QA suggestion #1/#7: "lay one orientation to the max, then rotate the
//! rest in"). Output is a `Vec<Placement>` for one sheet — the stencil the periodic packer repeats.

use crate::svg_nesting::render::Placement;
use std::f32::consts::FRAC_PI_2;

/// How many `cell_w × cell_h` cells (with `spacing` gaps) fit along each axis of a `rw × rh` region.
fn grid_dims(rw: f32, rh: f32, cell_w: f32, cell_h: f32, spacing: f32) -> (usize, usize) {
    if cell_w <= 0.0 || cell_h <= 0.0 {
        return (0, 0);
    }
    let cols = ((rw + spacing) / (cell_w + spacing)).floor().max(0.0) as usize;
    let rows = ((rh + spacing) / (cell_h + spacing)).floor().max(0.0) as usize;
    (cols, rows)
}

/// Tile region `(rx, ry, rw, rh)` with `cell_w × cell_h` cells, placing `part_idx` at `rotation`
/// (radians, about the centred shape origin). Centroids are emitted in bin coordinates. The grid
/// is anchored at the region's lower-left corner so leftover space is a clean right/bottom strip.
fn fill_region(
    rx: f32,
    ry: f32,
    rw: f32,
    rh: f32,
    cell_w: f32,
    cell_h: f32,
    spacing: f32,
    rotation: f32,
    part_idx: usize,
) -> Vec<Placement> {
    let (cols, rows) = grid_dims(rw, rh, cell_w, cell_h, spacing);
    let mut out = Vec::with_capacity(cols * rows);
    for c in 0..cols {
        for r in 0..rows {
            let x = rx + c as f32 * (cell_w + spacing) + cell_w / 2.0;
            let y = ry + r as f32 * (cell_h + spacing) + cell_h / 2.0;
            out.push(Placement {
                part_idx,
                rotation,
                x,
                y,
            });
        }
    }
    out
}

/// Compute the densest single-sheet placement of one rectangular part type.
///
/// `part_w × part_h` is the part bounding box. `allow_swap` enables the 90° orientation (disabled
/// when a grain constraint forbids it). Returns placements for one sheet (possibly empty if the
/// part does not fit in any permitted orientation).
pub(crate) fn grid_single_sheet(
    part_w: f32,
    part_h: f32,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    part_idx: usize,
    allow_swap: bool,
) -> Vec<Placement> {
    // Orientation A: as-is (rotation 0). Orientation B: 90°-swapped.
    let (cols_a, rows_a) = grid_dims(bin_w, bin_h, part_w, part_h, spacing);
    let count_a = cols_a * rows_a;
    let (cols_b, rows_b) = if allow_swap {
        grid_dims(bin_w, bin_h, part_h, part_w, spacing)
    } else {
        (0, 0)
    };
    let count_b = cols_b * rows_b;

    if count_a == 0 && count_b == 0 {
        return Vec::new();
    }

    // Primary = orientation that fits more copies (A wins ties). `prim_*` is the primary cell,
    // `alt_*` the 90°-rotated cell used to fill leftover strips.
    let primary_is_a = count_a >= count_b;
    let (prim_w, prim_h, prim_rot) = if primary_is_a {
        (part_w, part_h, 0.0)
    } else {
        (part_h, part_w, FRAC_PI_2)
    };
    let (alt_w, alt_h, alt_rot, alt_allowed) = if primary_is_a {
        (part_h, part_w, FRAC_PI_2, allow_swap)
    } else {
        (part_w, part_h, 0.0, true)
    };

    let (prim_cols, prim_rows) = grid_dims(bin_w, bin_h, prim_w, prim_h, spacing);
    let mut placements = fill_region(
        0.0, 0.0, bin_w, bin_h, prim_w, prim_h, spacing, prim_rot, part_idx,
    );

    if alt_allowed && prim_cols > 0 && prim_rows > 0 {
        let used_w = prim_cols as f32 * prim_w + (prim_cols.saturating_sub(1)) as f32 * spacing;
        let used_h = prim_rows as f32 * prim_h + (prim_rows.saturating_sub(1)) as f32 * spacing;

        // Right strip: full height, x past the primary grid. The +spacing keeps the guillotine cut.
        let right_x = used_w + spacing;
        let right_w = bin_w - right_x;
        if right_w > 0.0 {
            placements.extend(fill_region(
                right_x, 0.0, right_w, bin_h, alt_w, alt_h, spacing, alt_rot, part_idx,
            ));
        }

        // Bottom strip: only under the primary grid (x in [0, used_w]) so it never overlaps the
        // full-height right strip.
        let bottom_y = used_h + spacing;
        let bottom_h = bin_h - bottom_y;
        if bottom_h > 0.0 {
            placements.extend(fill_region(
                0.0, bottom_y, used_w, bottom_h, alt_w, alt_h, spacing, alt_rot, part_idx,
            ));
        }
    }

    placements
}
