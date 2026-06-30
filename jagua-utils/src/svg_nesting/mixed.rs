//! Mixed-parts grouping (WS-5) for 2–4 rectangular part types.
//!
//! Implements the QA "cut each part separately, then fill the remnants on one sheet" scheme
//! (комментарий #3, variant 1): each type gets its own run of byte-identical full sheets (periodic
//! grid), and the leftover parts of every type are co-packed onto shared remainder sheets via a
//! deterministic shelf packer. This keeps the dominant sheets identical while avoiding a wasted
//! partial sheet per type.
//!
//! Co-packing *different* types onto the dominant sheets (the "1+3" scheme) is a further
//! optimisation left for later; this version is deterministic and strictly denser than today's
//! per-bin LBF for the leftovers.

use crate::svg_nesting::grid::grid_single_sheet;
use crate::svg_nesting::render::{Placement, prepare, render_page_list};
use crate::svg_nesting::strategy::{PartInput, fit_orientations};
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use std::f32::consts::FRAC_PI_2;

/// One leftover part instance to be shelf-packed onto the remainder sheets.
struct Leftover {
    part_idx: usize,
    w: f32,
    h: f32,
    cx_off: f32,
    cy_off: f32,
}

pub(crate) fn nest_mixed(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    parts: &[PartInput],
    amount_of_rotations: usize,
) -> Result<NestingResult> {
    // Per-part 90° permission: honour each part's grain constraint (`allowedRotations`), not just
    // the global rotation count — production frames arrive grain-locked (`allowedRotations: []` ⇒
    // 0° only) and must still grid-pack rather than fall back to LBF.
    let allow_swaps: Vec<bool> = parts
        .iter()
        .map(|p| amount_of_rotations != 0 && fit_orientations(&p.allowed_rotations).1)
        .collect();
    let ranges: Vec<RotationRange> = allow_swaps
        .iter()
        .map(|&swap| {
            if swap {
                RotationRange::Discrete(vec![0.0, FRAC_PI_2])
            } else {
                RotationRange::Discrete(vec![0.0])
            }
        })
        .collect();
    let (prepared, ctx) = prepare(parts, &ranges, bin_width, bin_height, 1)?;

    let total: usize = parts.iter().map(|p| p.count).sum();
    let mut pages: Vec<Vec<Placement>> = Vec::new();
    let mut leftovers: Vec<Leftover> = Vec::new();

    for (idx, (part, p)) in parts.iter().zip(prepared.iter()).enumerate() {
        let stencil = grid_single_sheet(
            p.bbox_w,
            p.bbox_h,
            bin_width,
            bin_height,
            spacing,
            idx,
            allow_swaps[idx],
        );
        let cap = stencil.len();
        if cap == 0 {
            anyhow::bail!(
                "Part #{idx} (bbox {:.2}x{:.2}) does not fit in the bin ({:.2}x{:.2})",
                p.bbox_w,
                p.bbox_h,
                bin_width,
                bin_height
            );
        }
        let full = part.count / cap;
        let rem = part.count % cap;
        for _ in 0..full {
            pages.push(stencil.clone());
        }
        for _ in 0..rem {
            leftovers.push(Leftover {
                part_idx: idx,
                w: p.bbox_w,
                h: p.bbox_h,
                cx_off: p.cx_off,
                cy_off: p.cy_off,
            });
        }
    }

    pages.extend(shelf_pack_leftovers(
        &leftovers, bin_width, bin_height, spacing,
    ));

    Ok(render_page_list(&ctx, &pages, total))
}

/// Pack leftover rectangles onto shared sheets with a deterministic next-fit shelf algorithm
/// (tallest-first within each type group, types in request order). Rotation is fixed at 0°.
fn shelf_pack_leftovers(
    leftovers: &[Leftover],
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
) -> Vec<Vec<Placement>> {
    if leftovers.is_empty() {
        return Vec::new();
    }
    // Tallest-first packs shelves more tightly while staying deterministic.
    let mut order: Vec<usize> = (0..leftovers.len()).collect();
    order.sort_by(|&a, &b| {
        leftovers[b]
            .h
            .partial_cmp(&leftovers[a].h)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(leftovers[a].part_idx.cmp(&leftovers[b].part_idx))
    });

    let mut pages: Vec<Vec<Placement>> = Vec::new();
    let mut cur: Vec<Placement> = Vec::new();
    let (mut x, mut shelf_y, mut shelf_h) = (0.0f32, 0.0f32, 0.0f32);

    for &i in &order {
        let lo = &leftovers[i];
        if x + lo.w > bin_width {
            // Next shelf.
            x = 0.0;
            shelf_y += shelf_h + spacing;
            shelf_h = 0.0;
        }
        if shelf_y + lo.h > bin_height {
            // Next page.
            if !cur.is_empty() {
                pages.push(std::mem::take(&mut cur));
            }
            x = 0.0;
            shelf_y = 0.0;
            shelf_h = 0.0;
        }
        cur.push(Placement {
            part_idx: lo.part_idx,
            rotation: 0.0,
            x: x + lo.cx_off,
            y: shelf_y + lo.cy_off,
        });
        x += lo.w + spacing;
        shelf_h = shelf_h.max(lo.h);
    }
    if !cur.is_empty() {
        pages.push(cur);
    }
    pages
}
