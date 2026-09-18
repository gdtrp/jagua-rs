//! cutl-tests#77: three part types in one material group, STAIRCASE. The leftovers of the
//! chamfered plate were strewn ~100 mm apart along the bottom edge of the shared sheet: the band
//! packer cut a lattice strip at a y-limit and kept only the members of a slanted lattice row whose
//! top edge happened to clear it (every other part — the 180° ones).
//!
//! Replays QA's request (project 7d7027ea): 100 × 60 rounded plate × 120, Ø120 circle × 120,
//! 100 × 65 chamfered plate × 600 on 1200 × 800, spacing 2, 4 rotations.

use jagua_utils::{
    AdaptiveNestingStrategy, NestingResult, PackingMode, PageResult, PartInput, PlacedPartInfo,
    nest_auto,
};
use std::path::PathBuf;

const PLATE: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/rrect_100x60.svg");
const CIRCLE: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/circle_d120.svg");
const CHAMFER: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl-tests-77/chamfer_100x65.svg");

const BIN_W: f32 = 1200.0;
const BIN_H: f32 = 800.0;
const SPACING: f32 = 2.0;

/// Bbox `(w, h, min_x, min_y)` of each fixture in its own input coordinates.
const BBOX: [(f32, f32, f32, f32); 3] = [
    (100.0, 60.0, 0.1, -60.1),
    (119.858, 119.857, 24.656, -51.607),
    (100.116, 65.027, 0.059, -65.127),
];

fn part(svg: &[u8], count: usize, id: &str) -> PartInput {
    PartInput {
        svg_bytes: svg.to_vec(),
        count,
        item_id: Some(id.to_string()),
        allowed_rotations: None,
    }
}

fn qa_request() -> Vec<PartInput> {
    vec![
        part(PLATE, 120, "plate"),
        part(CIRCLE, 120, "circle"),
        part(CHAMFER, 600, "chamfer"),
    ]
}

fn nest(parts: &[PartInput]) -> NestingResult {
    nest_auto(
        &AdaptiveNestingStrategy::new(),
        BIN_W,
        BIN_H,
        SPACING,
        parts,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest")
}

fn write_output(case: &str, result: &NestingResult) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("cutl_tests77")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (i, svg) in result.page_svgs.iter().enumerate() {
        std::fs::write(dir.join(format!("page-{i}.svg")), svg).unwrap();
    }
    std::fs::write(dir.join("combined.svg"), &result.combined_svg).unwrap();
}

/// Axis-aligned bbox `(x0, y0, x1, y1)` of a placed part in sheet coordinates.
fn cell(p: &PlacedPartInfo) -> (f32, f32, f32, f32) {
    let (w, h, min_x, min_y) = BBOX[p.part_index];
    let (cx, cy) = (p.centroid_x - min_x, p.centroid_y - min_y);
    let (s, c) = p.rotation.to_radians().sin_cos();
    let mut r = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for (u, v) in [(-cx, -cy), (w - cx, -cy), (w - cx, h - cy), (-cx, h - cy)] {
        let (x, y) = (p.x + u * c - v * s, p.y + u * s + v * c);
        r = (r.0.min(x), r.1.min(y), r.2.max(x), r.3.max(y));
    }
    r
}

/// The loneliest part of one type on a page: the largest bbox gap from a part to its nearest
/// neighbour of the same type, as a multiple of the part's width. Interlocked or row-packed parts
/// sit at ~0; the #77 sheet had 1.2 (parts 100 wide, each 120 from the next).
fn loneliest(page: &PageResult, part_index: usize) -> f32 {
    let cells: Vec<_> = page
        .placements
        .iter()
        .filter(|p| p.part_index == part_index)
        .map(cell)
        .collect();
    if cells.len() < 2 {
        return 0.0;
    }
    let mut worst = 0.0f32;
    for (i, a) in cells.iter().enumerate() {
        let nearest = cells
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, b)| {
                let gap_x = (b.0 - a.2).max(a.0 - b.2).max(0.0);
                let gap_y = (b.1 - a.3).max(a.1 - b.3).max(0.0);
                gap_x.max(gap_y)
            })
            .fold(f32::INFINITY, f32::min);
        worst = worst.max(nearest / (a.2 - a.0));
    }
    worst
}

#[test]
fn leftovers_are_never_strewn_along_a_shared_sheet() {
    let result = nest(&qa_request());
    write_output("qa_7d7027ea", &result);

    assert_eq!(result.parts_placed, 840);
    assert!(
        result.sheets_total_estimate.is_some(),
        "must stay on the deterministic path"
    );
    assert!(result.pages.len() <= 7, "{} sheets", result.pages.len());

    // Every part stays on the sheet; parts of different types never share space.
    for (i, page) in result.pages.iter().enumerate() {
        let cells: Vec<_> = page
            .placements
            .iter()
            .map(|p| (p.part_index, cell(p)))
            .collect();
        for (idx, c) in &cells {
            assert!(
                c.0 >= -0.01 && c.1 >= -0.01 && c.2 <= BIN_W + 0.01 && c.3 <= BIN_H + 0.01,
                "page {i}: part type {idx} at {c:?} leaves the sheet"
            );
        }
        for a in 0..cells.len() {
            for b in (a + 1)..cells.len() {
                if cells[a].0 == cells[b].0 {
                    continue;
                }
                let (p, q) = (cells[a].1, cells[b].1);
                let gap_x = (q.0 - p.2).max(p.0 - q.2);
                let gap_y = (q.1 - p.3).max(p.1 - q.3);
                assert!(
                    gap_x >= SPACING - 0.01 || gap_y >= SPACING - 0.01,
                    "page {i}: types {} and {} overlap: {p:?} vs {q:?}",
                    cells[a].0,
                    cells[b].0
                );
            }
        }
    }

    // The defect itself: no part sits a part-width away from every other part of its type.
    for (i, page) in result.pages.iter().enumerate() {
        for part_index in 0..3 {
            let gap = loneliest(page, part_index);
            assert!(
                gap < 0.5,
                "page {i}: a type-{part_index} part is {gap:.2} part widths from its nearest neighbour"
            );
        }
    }
    for (i, page) in result.pages.iter().enumerate() {
        let mut counts = [0usize; 3];
        for p in &page.placements {
            counts[p.part_index] += 1;
        }
        println!("page {i}: {counts:?} util {:.3}", page.utilisation);
    }

    // Deterministic.
    assert_eq!(nest(&qa_request()).combined_svg, result.combined_svg);
}
