//! Mixed-parts grouping (WS-5, generalised for CUTL-195) for 2–4 part types of any shape.
//!
//! Implements the QA "cut each part separately, then fill the remnants on one sheet" scheme
//! (комментарий #3, variant 1): each type gets its own run of byte-identical full sheets — the
//! **same single-type stencil** it would receive when nested alone (grid for rectangles, pairing for
//! half-bbox parts, the lattice for everything else; see `classify::single_sheet_stencil`) — and the
//! leftover parts of every type are co-packed onto shared remainder sheets deterministically.
//!
//! Before CUTL-195 this path accepted rectangular types only, so a rectangle + a circle fell through
//! to the general LBF optimiser, which nests every sheet independently: 10 different sheets in
//! ~107 s where the two single-type runs had given 2 + 7 identical sheets in under a second.
//!
//! Remainder sheets:
//! * every type rectangular (the pre-CUTL-195 domain) → the original next-fit **shelf** packer
//!   (rotation 0°, tallest first), unchanged, so rectangular-only requests render as before;
//! * otherwise → **band** packing: each type's leftovers are the lowest `rem` placements of its own
//!   stencil (a dense, already-valid strip), and the strips are stacked vertically with `spacing`
//!   between them, moving to a new sheet when one does not fit.
//!
//! If no type fills even one full sheet and a type is irregular, `nest_mixed` refuses so the caller
//! falls back to LBF, whose interlocking co-pack is the better tool for a small one-off mix.
//! Co-packing *different* types onto the dominant sheets (the "1+3" scheme) is a further
//! optimisation left for later.

use crate::svg_nesting::classify::{StencilKind, single_sheet_stencil};
use crate::svg_nesting::render::{Placement, PreparedPart, prepare, render_page_list};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;
use std::f32::consts::{FRAC_PI_2, PI};

/// One leftover part instance to be shelf-packed onto the remainder sheets.
struct Leftover {
    part_idx: usize,
    w: f32,
    h: f32,
    cx_off: f32,
    cy_off: f32,
}

/// The leftovers of one type after its full sheets: `count` parts to place from `stencil`.
struct LeftoverGroup {
    part_idx: usize,
    count: usize,
}

pub(crate) fn nest_mixed(
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    parts: &[PartInput],
    amount_of_rotations: usize,
) -> Result<NestingResult> {
    // Per-type single-sheet stencils, routed exactly like a single-type request (grain-aware:
    // `allowedRotations` is honoured per part, so grain-locked frames still grid-pack).
    let mut stencils: Vec<(StencilKind, Vec<Placement>)> = Vec::with_capacity(parts.len());
    for (idx, part) in parts.iter().enumerate() {
        let (kind, mut stencil) =
            single_sheet_stencil(bin_width, bin_height, spacing, part, amount_of_rotations)?;
        if stencil.is_empty() {
            anyhow::bail!(
                "Part #{idx} does not fit in the bin ({:.2}x{:.2})",
                bin_width,
                bin_height
            );
        }
        for pl in &mut stencil {
            pl.part_idx = idx;
        }
        stencils.push((kind, stencil));
    }

    // The rotation range only feeds `Item` construction; placements below are explicit and
    // cardinal (0/90/180/270), whatever the stencil kind.
    let rot_range = RotationRange::Discrete(vec![0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2]);
    let ranges: Vec<RotationRange> = vec![rot_range; parts.len()];
    let (prepared, ctx) = prepare(parts, &ranges, bin_width, bin_height, 1)?;

    let total: usize = parts.iter().map(|p| p.count).sum();
    let mut pages: Vec<Vec<Placement>> = Vec::new();
    let mut groups: Vec<LeftoverGroup> = Vec::new();
    let mut full_sheets = 0usize;

    for (idx, part) in parts.iter().enumerate() {
        let stencil = &stencils[idx].1;
        let cap = stencil.len();
        let full = part.count / cap;
        let rem = part.count % cap;
        full_sheets += full;
        for _ in 0..full {
            pages.push(stencil.clone());
        }
        if rem > 0 {
            groups.push(LeftoverGroup {
                part_idx: idx,
                count: rem,
            });
        }
    }

    let all_rect = stencils.iter().all(|(k, _)| *k == StencilKind::Grid);
    if !all_rect && full_sheets == 0 {
        // Nothing to repeat: a small irregular mix is LBF's job (its interlocking co-pack beats
        // stacked stencil strips). Refusing here keeps today's behaviour for such requests.
        anyhow::bail!("no type fills a full sheet; leaving the mix to the general optimiser");
    }

    if all_rect {
        let leftovers: Vec<Leftover> = groups
            .iter()
            .flat_map(|g| {
                let p = &prepared[g.part_idx];
                std::iter::repeat_with(move || Leftover {
                    part_idx: g.part_idx,
                    w: p.bbox_w,
                    h: p.bbox_h,
                    cx_off: p.cx_off,
                    cy_off: p.cy_off,
                })
                .take(g.count)
            })
            .collect();
        pages.extend(shelf_pack_leftovers(
            &leftovers, bin_width, bin_height, spacing,
        ));
    } else {
        pages.extend(band_pack_leftovers(
            &groups, &stencils, &prepared, bin_height, spacing,
        ));
    }

    Ok(render_page_list(&ctx, &pages, total))
}

/// Vertical extent `(y_min, y_max)` in bin coordinates of a placed part's rotated bounding box.
/// Exact for the cardinal rotations every stencil uses (a rotated rectangle's bbox is the rotated
/// rectangle); a safe over-estimate otherwise.
fn placement_y_extent(p: &PreparedPart, pl: &Placement) -> (f32, f32) {
    let (s, c) = pl.rotation.sin_cos();
    // bbox corners relative to the centroid (the placement's reference point).
    let corners = [
        (-p.cx_off, -p.cy_off),
        (p.bbox_w - p.cx_off, -p.cy_off),
        (p.bbox_w - p.cx_off, p.bbox_h - p.cy_off),
        (-p.cx_off, p.bbox_h - p.cy_off),
    ];
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for (x, y) in corners {
        // jagua-rs rotation is the standard CCW matrix: y' = x·sinθ + y·cosθ.
        let ry = pl.y + x * s + y * c;
        lo = lo.min(ry);
        hi = hi.max(ry);
    }
    (lo, hi)
}

/// Stack each type's leftover strip onto shared remainder sheets.
///
/// A type's strip is the `count` placements of its own stencil with the lowest top edge (so the strip
/// is as short as possible and complete rows come first); the stencil is already a valid, dense
/// single-sheet packing, so the strip is valid as-is and stays valid when shifted up. Strips are
/// laid tallest-first with `spacing` between them. When a strip does not fit in the height left on
/// the current sheet, the rows that do fit are placed there and the rest continue on a new sheet
/// (a strip is only ever cut between placements, never through one). Deterministic, O(n log n).
fn band_pack_leftovers(
    groups: &[LeftoverGroup],
    stencils: &[(StencilKind, Vec<Placement>)],
    prepared: &[PreparedPart],
    bin_height: f32,
    spacing: f32,
) -> Vec<Vec<Placement>> {
    /// A placement with the vertical extent of its rotated bbox, in stencil coordinates.
    type Extent = (Placement, f32, f32);

    struct Band {
        part_idx: usize,
        /// Sorted by top edge, then x.
        placements: Vec<Extent>,
    }

    fn height(band: &[Extent]) -> f32 {
        let y_min = band.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
        let y_max = band.iter().map(|t| t.2).fold(f32::NEG_INFINITY, f32::max);
        y_max - y_min
    }

    let mut bands: Vec<Band> = Vec::with_capacity(groups.len());
    for g in groups {
        let p = &prepared[g.part_idx];
        let mut pls: Vec<Extent> = stencils[g.part_idx]
            .1
            .iter()
            .map(|pl| {
                let (lo, hi) = placement_y_extent(p, pl);
                (*pl, lo, hi)
            })
            .collect();
        pls.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.0.x
                        .partial_cmp(&b.0.x)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        pls.truncate(g.count);
        bands.push(Band {
            part_idx: g.part_idx,
            placements: pls,
        });
    }
    // Tallest first (first-fit decreasing), request order on ties — deterministic.
    bands.sort_by(|a, b| {
        height(&b.placements)
            .partial_cmp(&height(&a.placements))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.part_idx.cmp(&b.part_idx))
    });

    const EPS: f32 = 1e-3;
    let mut pages: Vec<Vec<Placement>> = Vec::new();
    let mut cur: Vec<Placement> = Vec::new();
    let mut cursor = 0.0f32; // next free y on the current sheet
    let mut queue: std::collections::VecDeque<Vec<Extent>> =
        bands.into_iter().map(|b| b.placements).collect();
    while let Some(strip) = queue.pop_front() {
        if strip.is_empty() {
            continue;
        }
        let y_min = strip.iter().map(|t| t.1).fold(f32::INFINITY, f32::min);
        let dy = cursor - y_min;
        // The strip is sorted by top edge, so the placements that fit form a prefix.
        let n_fit = strip
            .iter()
            .take_while(|t| t.2 + dy <= bin_height + EPS)
            .count();
        if n_fit == 0 {
            // Nothing fits above the cursor: start a new sheet (a strip always fits an empty one).
            if cur.is_empty() {
                // Degenerate (a part taller than the sheet); the stencil builder would have
                // rejected it, so this is unreachable — but never spin.
                break;
            }
            pages.push(std::mem::take(&mut cur));
            cursor = 0.0;
            queue.push_front(strip);
            continue;
        }
        let (fits, rest) = strip.split_at(n_fit);
        let top = fits.iter().map(|t| t.2).fold(f32::NEG_INFINITY, f32::max);
        cur.extend(fits.iter().map(|t| Placement {
            y: t.0.y + dy,
            ..t.0
        }));
        cursor = top + dy + spacing;
        if !rest.is_empty() {
            // Continue the same type on the next sheet before any other strip.
            pages.push(std::mem::take(&mut cur));
            cursor = 0.0;
            queue.push_front(rest.to_vec());
        }
    }
    if !cur.is_empty() {
        pages.push(cur);
    }
    pages
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
