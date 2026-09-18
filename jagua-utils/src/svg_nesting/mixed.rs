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
//! * otherwise → **band** packing: each type's leftovers are the *shortest window* of its own stencil
//!   that holds them (a dense, already-valid strip — never the stencil's ragged top fringe, which for
//!   a double lattice is every other part: cutl-tests#77), and the strips are stacked vertically with
//!   `spacing` between them. A strip that does not fit puts what a window of the remaining height
//!   holds on the sheet and continues on the next; every waiting type gets a chance before a sheet is
//!   closed, and the rectangular holes beside a partly filled last row and under the last band are
//!   used up with bounding-box cells of the waiting types.
//!
//! If no type fills even one full sheet and a type is irregular, `nest_mixed` refuses so the caller
//! falls back to LBF, whose interlocking co-pack is the better tool for a small one-off mix.
//! Co-packing *different* types onto the dominant sheets (the "1+3" scheme) is a further
//! optimisation left for later.

use crate::svg_nesting::classify::{StencilKind, single_sheet_stencil};
use crate::svg_nesting::fill::{extent, fill_rect};
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
            &groups,
            &stencils,
            &prepared,
            parts,
            amount_of_rotations,
            bin_width,
            bin_height,
            spacing,
        ));
    }

    Ok(render_page_list(&ctx, &pages, total))
}

/// A stencil placement with the extents `(x0, y0, x1, y1)` of its rotated bounding box, in the
/// stencil's own sheet coordinates. Exact for the cardinal rotations, a safe over-estimate otherwise.
#[derive(Clone, Copy)]
struct Boxed {
    pl: Placement,
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

fn boxed(p: &PreparedPart, pl: &Placement) -> Boxed {
    let (xmin, xmax, ymin, ymax) = extent(p, pl.rotation);
    Boxed {
        pl: *pl,
        x0: pl.x + xmin,
        y0: pl.y + ymin,
        x1: pl.x + xmax,
        y1: pl.y + ymax,
    }
}

const EPS: f32 = 1e-3;
/// Upper bound on the window starts tried per selection, so huge stencils stay cheap.
const MAX_WINDOW_STARTS: usize = 256;
/// Above this many parts in a band the O(k²) gap search beside its last row is skipped.
const MAX_GAP_SEARCH: usize = 2000;

/// The `n` parts of a stencil (sorted by top edge `y0`) that span the **least height**: the
/// shortest y-window `[y0_i, y_hi]` holding `n` whole parts. Returns `(height, parts)`.
///
/// Why a window and not "the `n` lowest": a lattice stencil's top edge is ragged — the first rows
/// of a double lattice keep only the members that still fit on the sheet (every other part), so
/// the lowest-`n` slice of a short remainder was exactly that sparse fringe, strewn ~one part
/// apart along the sheet (cutl-tests#77). The densest rows span the least height for a given
/// count, so the shortest window lands in the stencil's dense interior. Any subset of a valid
/// packing is valid, and it stays valid when shifted up.
fn shortest_window(sorted: &[Boxed], n: usize) -> Option<(f32, Vec<Boxed>)> {
    if n == 0 || n > sorted.len() {
        return None;
    }
    let last_start = sorted.len() - n;
    let step = (last_start / MAX_WINDOW_STARTS).max(1);
    let mut best: Option<(f32, usize, f32)> = None; // (height, start, y_hi)
    let mut tops: Vec<f32> = Vec::with_capacity(sorted.len());
    let mut start = 0;
    while start <= last_start {
        tops.clear();
        tops.extend(sorted[start..].iter().map(|b| b.y1));
        let (_, y_hi, _) = tops.select_nth_unstable_by(n - 1, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        let height = *y_hi - sorted[start].y0;
        if best.is_none_or(|(h, _, _)| height < h - EPS) {
            best = Some((height, start, *y_hi));
        }
        start += step;
    }
    let (height, start, y_hi) = best?;
    let mut chosen: Vec<Boxed> = sorted[start..]
        .iter()
        .filter(|b| b.y1 <= y_hi + EPS)
        .copied()
        .collect();
    // Bottom rows first, left to right, so a partly used last row is one contiguous run.
    chosen.sort_by(|a, b| {
        a.y1.partial_cmp(&b.y1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal))
    });
    chosen.truncate(n);
    Some((height, chosen))
}

/// The most parts (≤ `count`) of a stencil whose shortest window fits in `max_height`.
fn most_that_fit(sorted: &[Boxed], count: usize, max_height: f32) -> usize {
    let fits = |n: usize| shortest_window(sorted, n).is_some_and(|(h, _)| h <= max_height + EPS);
    let count = count.min(sorted.len());
    if count == 0 || !fits(1) {
        return 0;
    }
    // The window height is monotone in the count, so bisect.
    let (mut lo, mut hi) = (1, count);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// The largest empty rectangle to the right of a just-placed band, inside the band's own height:
/// `x ∈ [x0, bin_width]`, from below every band part that reaches past `x0` down to the band's
/// bottom. This is the hole beside a partly filled last row.
fn gap_beside(
    band: &[Boxed],
    band_top: f32,
    band_bottom: f32,
    bin_width: f32,
    spacing: f32,
) -> Option<(f32, f32, f32, f32)> {
    if band.len() > MAX_GAP_SEARCH {
        return None;
    }
    let mut best: Option<(f32, (f32, f32, f32, f32))> = None;
    for p in band {
        let x0 = p.x1 + spacing;
        let y0 = band
            .iter()
            .filter(|q| q.x1 > p.x1 + EPS)
            .map(|q| q.y1 + spacing)
            .fold(band_top, f32::max);
        let (w, h) = (bin_width - x0, band_bottom - y0);
        if w <= EPS || h <= EPS {
            continue;
        }
        if best.is_none_or(|(area, _)| w * h > area + EPS) {
            best = Some((w * h, (x0, y0, w, h)));
        }
    }
    best.map(|(_, rect)| rect)
}

/// Stack each type's leftovers onto shared remainder sheets.
///
/// A type's strip is the shortest window of its own stencil that holds the parts still to place
/// (see [`shortest_window`]) — a dense, already-valid piece of its single-type packing — shifted up
/// under the previous strip with `spacing` between them. Strips go tallest-first. When a strip does
/// not fit in the height left, as many of its parts as a window of that height holds go there and
/// the rest continue on a new sheet; before a sheet is closed every other waiting type gets the
/// same chance. Two rectangular holes are then used up with bounding-box cells of the waiting
/// types: the gap beside a band's partly filled last row, and the sliver under the last band.
/// Deterministic.
#[allow(clippy::too_many_arguments)]
fn band_pack_leftovers(
    groups: &[LeftoverGroup],
    stencils: &[(StencilKind, Vec<Placement>)],
    prepared: &[PreparedPart],
    parts: &[PartInput],
    amount_of_rotations: usize,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
) -> Vec<Vec<Placement>> {
    // Per type: its stencil with extents, sorted by top edge (then x) for the window search.
    let sorted: Vec<Vec<Boxed>> = stencils
        .iter()
        .enumerate()
        .map(|(idx, (_, stencil))| {
            let mut v: Vec<Boxed> = stencil.iter().map(|pl| boxed(&prepared[idx], pl)).collect();
            v.sort_by(|a, b| {
                a.y0.partial_cmp(&b.y0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.x0.partial_cmp(&b.x0).unwrap_or(std::cmp::Ordering::Equal))
            });
            v
        })
        .collect();

    // Tallest strip first (first-fit decreasing), request order on ties.
    let mut order: Vec<(f32, usize, usize)> = groups
        .iter()
        .map(|g| {
            let height = shortest_window(&sorted[g.part_idx], g.count).map_or(0.0, |(h, _)| h);
            (height, g.part_idx, g.count)
        })
        .collect();
    order.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    // (part_idx, parts still to place)
    let mut queue: std::collections::VecDeque<(usize, usize)> = order
        .into_iter()
        .map(|(_, idx, count)| (idx, count))
        .collect();

    // Use up a rectangle with bbox cells of the waiting types, front of the queue first.
    let fill_gap = |cur: &mut Vec<Placement>,
                    queue: &mut std::collections::VecDeque<(usize, usize)>,
                    (mut x0, y0, mut w, h): (f32, f32, f32, f32)| {
        for slot in queue.iter_mut() {
            let (idx, remaining) = *slot;
            let (cells, used) = fill_rect(
                &prepared[idx],
                idx,
                &parts[idx],
                amount_of_rotations,
                (x0, y0, w, h),
                spacing,
                remaining,
            );
            if cells.is_empty() {
                continue;
            }
            slot.1 -= cells.len();
            cur.extend(cells);
            x0 += used + spacing;
            w -= used + spacing;
            if w <= EPS {
                break;
            }
        }
        queue.retain(|&(_, remaining)| remaining > 0);
    };

    let mut pages: Vec<Vec<Placement>> = Vec::new();
    let mut cur: Vec<Placement> = Vec::new();
    let mut cursor = 0.0f32; // next free y on the current sheet

    while !queue.is_empty() {
        // The first waiting type that can put anything into the height left.
        let room = bin_height - cursor;
        let pick = queue
            .iter()
            .enumerate()
            .find_map(|(pos, &(idx, remaining))| {
                let n = most_that_fit(&sorted[idx], remaining, room);
                (n > 0).then_some((pos, idx, remaining, n))
            });
        let Some((pos, idx, remaining, n)) = pick else {
            if cur.is_empty() {
                // Degenerate (a part taller than the sheet); the stencil builder would have
                // rejected it, so this is unreachable — but never spin.
                break;
            }
            // Nothing fits as a strip: use the sliver under the last band, then a new sheet.
            fill_gap(&mut cur, &mut queue, (0.0, cursor, bin_width, room));
            pages.push(std::mem::take(&mut cur));
            cursor = 0.0;
            continue;
        };
        queue.remove(pos);

        let Some((_, window)) = shortest_window(&sorted[idx], n) else {
            break;
        };
        let y_min = window.iter().map(|b| b.y0).fold(f32::INFINITY, f32::min);
        let dy = cursor - y_min;
        let band: Vec<Boxed> = window
            .iter()
            .map(|b| Boxed {
                pl: Placement {
                    y: b.pl.y + dy,
                    ..b.pl
                },
                y0: b.y0 + dy,
                y1: b.y1 + dy,
                ..*b
            })
            .collect();
        let band_top = cursor;
        let band_bottom = band.iter().map(|b| b.y1).fold(f32::NEG_INFINITY, f32::max);
        cur.extend(band.iter().map(|b| b.pl));
        cursor = band_bottom + spacing;

        if remaining > n {
            // The sheet is full for this type: its remaining parts open the next sheet, before any
            // other strip. Whatever else is waiting may still use the sliver left under the band.
            fill_gap(
                &mut cur,
                &mut queue,
                (0.0, cursor, bin_width, bin_height - cursor),
            );
            pages.push(std::mem::take(&mut cur));
            cursor = 0.0;
            queue.push_front((idx, remaining - n));
        } else if let Some(rect) = gap_beside(&band, band_top, band_bottom, bin_width, spacing) {
            fill_gap(&mut cur, &mut queue, rect);
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
