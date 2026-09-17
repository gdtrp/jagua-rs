//! CUTL-198: the row/column fill (`HORIZONTAL` / `VERTICAL`) from each start corner.
//!
//! For a rectangle set, an L-shape set and a 3-type mixed set, every direction × corner must give
//! a layout with no overlapping bounding-box cells, every part inside the sheet, only cardinal
//! rotations, every requested part placed, and the corner layouts related to `TOP_LEFT` by the
//! reflection of cells (parts keep their own orientation — never a mirror). Output is
//! deterministic (a second run is byte-identical).

use jagua_utils::{
    AdaptiveNestingStrategy, FillDirection, NestingResult, PackingMode, PageResult, PartInput,
    PlacedPartInfo, StartCorner, nest_auto,
};
use std::path::PathBuf;

const RECT_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/rect_100x150.svg");
const MIXED_A: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_a.svg");
const MIXED_B: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_b.svg");
const MIXED_C: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_c.svg");
/// An L-shape whose centroid sits well off its bbox centre, so a plain mirror of positions would
/// be caught. Bbox 100 × 120, min corner at the origin.
const L_SVG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 120">
  <path d="M 0,0 L 100,0 L 100,40 L 40,40 L 40,120 L 0,120 Z" fill="black"/>
</svg>"#;
/// A part that fits the test sheet in no orientation.
const OVERSIZE_SVG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2000 100">
  <path d="M 0,0 L 2000,0 L 2000,100 L 0,100 Z" fill="black"/>
</svg>"#;

const BIN_W: f32 = 1000.0;
const BIN_H: f32 = 500.0;
const SPACING: f32 = 2.0;
const EPS: f32 = 1e-3;

const DIRECTIONS: [FillDirection; 2] = [FillDirection::Horizontal, FillDirection::Vertical];
const CORNERS: [StartCorner; 4] = [
    StartCorner::TopLeft,
    StartCorner::TopRight,
    StartCorner::BottomLeft,
    StartCorner::BottomRight,
];

/// A part type with its bbox (all fixtures have their bbox min corner at the input origin, so a
/// placement's `centroid_x/y` is the centroid offset inside the bbox).
#[derive(Clone)]
struct Fixture {
    svg: &'static [u8],
    w: f32,
    h: f32,
}

const RECT: Fixture = Fixture {
    svg: RECT_SVG,
    w: 100.0,
    h: 150.0,
};
const L_SHAPE: Fixture = Fixture {
    svg: L_SVG.as_bytes(),
    w: 100.0,
    h: 120.0,
};
const MIX: [Fixture; 3] = [
    Fixture {
        svg: MIXED_A,
        w: 100.0,
        h: 100.0,
    },
    Fixture {
        svg: MIXED_B,
        w: 120.0,
        h: 80.0,
    },
    Fixture {
        svg: MIXED_C,
        w: 200.0,
        h: 50.0,
    },
];

fn part(f: &Fixture, count: usize, id: &str) -> PartInput {
    PartInput {
        svg_bytes: f.svg.to_vec(),
        count,
        item_id: Some(id.to_string()),
        allowed_rotations: None,
    }
}

fn nest(
    parts: &[PartInput],
    direction: FillDirection,
    corner: StartCorner,
    rotations: usize,
) -> NestingResult {
    let strategy = AdaptiveNestingStrategy::new().with_layout(direction, corner);
    nest_auto(
        &strategy,
        BIN_W,
        BIN_H,
        SPACING,
        parts,
        rotations,
        PackingMode::Auto,
        None,
    )
    .expect("nest")
}

fn write_output(case: &str, result: &NestingResult) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("cutl198")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (i, svg) in result.page_svgs.iter().enumerate() {
        std::fs::write(dir.join(format!("page-{i}.svg")), svg).unwrap();
    }
    std::fs::write(dir.join("combined.svg"), &result.combined_svg).unwrap();
    std::fs::write(
        dir.join("pages.json"),
        serde_json::to_string_pretty(&result.pages).unwrap(),
    )
    .unwrap();
}

/// Axis-aligned cell `(x0, y0, x1, y1)` of a placed part's rotated bounding box, in bin coords.
fn cell(p: &PlacedPartInfo, fixtures: &[Fixture]) -> (f32, f32, f32, f32) {
    let f = &fixtures[p.part_index];
    let (s, c) = p.rotation.to_radians().sin_cos();
    let corners = [
        (-p.centroid_x, -p.centroid_y),
        (f.w - p.centroid_x, -p.centroid_y),
        (f.w - p.centroid_x, f.h - p.centroid_y),
        (-p.centroid_x, f.h - p.centroid_y),
    ];
    let mut r = (
        f32::INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NEG_INFINITY,
    );
    for (u, v) in corners {
        let x = p.x + u * c - v * s;
        let y = p.y + u * s + v * c;
        r.0 = r.0.min(x);
        r.1 = r.1.min(y);
        r.2 = r.2.max(x);
        r.3 = r.3.max(y);
    }
    r
}

fn cells(page: &PageResult, fixtures: &[Fixture]) -> Vec<(f32, f32, f32, f32)> {
    page.placements.iter().map(|p| cell(p, fixtures)).collect()
}

fn check_page(page: &PageResult, fixtures: &[Fixture], label: &str) {
    let cs = cells(page, fixtures);
    for (p, c) in page.placements.iter().zip(&cs) {
        assert!(
            c.0 >= -EPS && c.1 >= -EPS && c.2 <= BIN_W + EPS && c.3 <= BIN_H + EPS,
            "{label}: {p:?} cell {c:?} outside the sheet"
        );
        let r = p.rotation.rem_euclid(360.0);
        assert!(
            r.abs() < 1e-3 || (r - 90.0).abs() < 1e-3,
            "{label}: rotation {} is not cardinal",
            p.rotation
        );
    }
    for i in 0..cs.len() {
        for j in (i + 1)..cs.len() {
            let (a, b) = (cs[i], cs[j]);
            let gap_x = (b.0 - a.2).max(a.0 - b.2);
            let gap_y = (b.1 - a.3).max(a.1 - b.3);
            assert!(
                gap_x >= SPACING - EPS || gap_y >= SPACING - EPS,
                "{label}: cells {a:?} and {b:?} overlap or sit closer than spacing"
            );
        }
    }
}

fn check_result(result: &NestingResult, fixtures: &[Fixture], total: usize, label: &str) {
    assert_eq!(result.parts_placed, total, "{label}: not every part placed");
    assert_eq!(result.total_parts_requested, total);
    assert!(
        result.sheets_total_estimate.is_some(),
        "{label}: must take the deterministic fill path"
    );
    assert_eq!(result.pages.len(), result.page_svgs.len());
    for (i, page) in result.pages.iter().enumerate() {
        check_page(page, fixtures, &format!("{label} page {i}"));
    }
}

/// Cells rounded and sorted, so two layouts can be compared as sets per page.
fn cell_set(page: &PageResult, fixtures: &[Fixture]) -> Vec<(i64, i64, i64, i64, usize)> {
    let q = |v: f32| (v * 100.0).round() as i64;
    let mut v: Vec<_> = page
        .placements
        .iter()
        .map(|p| {
            let c = cell(p, fixtures);
            (q(c.0), q(c.1), q(c.2), q(c.3), p.part_index)
        })
        .collect();
    v.sort();
    v
}

fn reflect_x(c: (i64, i64, i64, i64, usize)) -> (i64, i64, i64, i64, usize) {
    let w = (BIN_W * 100.0).round() as i64;
    (w - c.2, c.1, w - c.0, c.3, c.4)
}

fn reflect_y(c: (i64, i64, i64, i64, usize)) -> (i64, i64, i64, i64, usize) {
    let h = (BIN_H * 100.0).round() as i64;
    (c.0, h - c.3, c.2, h - c.1, c.4)
}

/// Every direction × corner for one part set: validity, corner relation, determinism.
fn run_set(name: &str, fixtures: &[Fixture], parts: &[PartInput]) {
    let total: usize = parts.iter().map(|p| p.count).sum();
    for direction in DIRECTIONS {
        let top_left = nest(parts, direction, StartCorner::TopLeft, 4);
        let label = format!("{name} {direction:?} TopLeft");
        check_result(&top_left, fixtures, total, &label);
        write_output(&format!("{name}_{direction:?}_TopLeft"), &top_left);

        for corner in CORNERS.iter().skip(1) {
            let r = nest(parts, direction, *corner, 4);
            let label = format!("{name} {direction:?} {corner:?}");
            check_result(&r, fixtures, total, &label);
            write_output(&format!("{name}_{direction:?}_{corner:?}"), &r);
            assert_eq!(r.pages.len(), top_left.pages.len(), "{label}: page count");
            for (i, (page, tl)) in r.pages.iter().zip(&top_left.pages).enumerate() {
                // Rotations are the part's own: the same multiset as at TopLeft.
                let mut rots: Vec<i64> = page
                    .placements
                    .iter()
                    .map(|p| p.rotation.round() as i64)
                    .collect();
                let mut tl_rots: Vec<i64> = tl
                    .placements
                    .iter()
                    .map(|p| p.rotation.round() as i64)
                    .collect();
                rots.sort();
                tl_rots.sort();
                assert_eq!(rots, tl_rots, "{label} page {i}: rotations changed");
                let mut expected: Vec<_> = cell_set(tl, fixtures)
                    .into_iter()
                    .map(|c| {
                        let c = if corner.is_right() { reflect_x(c) } else { c };
                        if corner.is_bottom() { reflect_y(c) } else { c }
                    })
                    .collect();
                expected.sort();
                assert_eq!(
                    cell_set(page, fixtures),
                    expected,
                    "{label} page {i}: cells are not the reflection of TopLeft's"
                );
            }
        }

        // Deterministic: a second run is byte-identical.
        let again = nest(parts, direction, StartCorner::TopLeft, 4);
        assert_eq!(
            again.combined_svg, top_left.combined_svg,
            "{label}: not deterministic"
        );
    }
}

#[test]
fn rectangles_every_direction_and_corner() {
    // 0°: 9 × 3 = 27 per sheet ⇒ 2 full sheets + 6 leftovers.
    run_set("rect", &[RECT], &[part(&RECT, 60, "rect")]);
}

#[test]
fn l_shapes_every_direction_and_corner() {
    // Irregular outline (would be the lattice under STAIRCASE): 0° ⇒ 9 × 4 = 36 ⇒ 1 full + 4.
    run_set("lshape", &[L_SHAPE], &[part(&L_SHAPE, 40, "L")]);
}

#[test]
fn mixed_three_types_every_direction_and_corner() {
    let parts = vec![
        part(&MIX[0], 30, "a"),
        part(&MIX[1], 25, "b"),
        part(&MIX[2], 40, "c"),
    ];
    run_set("mixed", &MIX, &parts);
}

#[test]
fn full_sheets_of_a_type_are_identical_and_the_block_is_compact() {
    let r = nest(
        &[part(&RECT, 60, "rect")],
        FillDirection::Horizontal,
        StartCorner::TopLeft,
        4,
    );
    assert_eq!(r.pages.len(), 3);
    assert_eq!(
        r.page_svgs[0], r.page_svgs[1],
        "full sheets must be byte-identical"
    );
    assert_eq!(r.pages[0].parts_placed, 27);
    assert_eq!(r.pages[2].parts_placed, 6);
    // The leftover page grows along x: 3 per strip ⇒ 2 strips, x ∈ [0, 202], full height used.
    let cs = cells(&r.pages[2], &[RECT]);
    let max_x = cs.iter().map(|c| c.2).fold(0.0, f32::max);
    assert!((max_x - 202.0).abs() < EPS, "block reaches x = {max_x}");

    let v = nest(
        &[part(&RECT, 60, "rect")],
        FillDirection::Vertical,
        StartCorner::TopLeft,
        4,
    );
    let cs = cells(&v.pages[2], &[RECT]);
    let max_y = cs.iter().map(|c| c.3).fold(0.0, f32::max);
    // 9 per strip along x ⇒ 6 leftovers fit in one strip of height 150.
    assert!((max_y - 150.0).abs() < EPS, "block reaches y = {max_y}");
}

#[test]
fn grain_locked_parts_keep_their_orientation() {
    for (allowed, expected) in [(vec![0.0f32], 0.0f32), (vec![90.0], 90.0)] {
        let mut p = part(&RECT, 30, "rect");
        p.allowed_rotations = Some(allowed.clone());
        // amount_of_rotations = 0 must not override an explicit per-part list.
        let r = nest(&[p], FillDirection::Horizontal, StartCorner::BottomRight, 0);
        check_result(&r, &[RECT], 30, &format!("locked {allowed:?}"));
        for pl in r.pages.iter().flat_map(|pg| &pg.placements) {
            assert!(
                (pl.rotation.rem_euclid(360.0) - expected).abs() < 1e-3,
                "locked to {allowed:?} but placed at {}",
                pl.rotation
            );
        }
    }
}

#[test]
fn an_oversize_type_is_left_unplaced_without_failing_the_request() {
    let oversize = PartInput {
        svg_bytes: OVERSIZE_SVG.as_bytes().to_vec(),
        count: 3,
        item_id: Some("big".into()),
        allowed_rotations: None,
    };
    let r = nest(
        &[part(&RECT, 10, "rect"), oversize],
        FillDirection::Horizontal,
        StartCorner::TopLeft,
        4,
    );
    assert_eq!(r.parts_placed, 10);
    assert_eq!(r.total_parts_requested, 13);
    assert!(r.sheets_total_estimate.is_some());
    assert!(
        r.pages
            .iter()
            .flat_map(|p| &p.placements)
            .all(|p| p.part_index == 0)
    );
}

#[test]
fn more_than_four_types_still_take_the_fill_path() {
    let fixtures = [
        MIX[0].clone(),
        MIX[1].clone(),
        MIX[2].clone(),
        RECT,
        L_SHAPE,
    ];
    let parts: Vec<PartInput> = fixtures
        .iter()
        .enumerate()
        .map(|(i, f)| part(f, 7, &format!("t{i}")))
        .collect();
    let r = nest(&parts, FillDirection::Vertical, StartCorner::TopRight, 4);
    check_result(&r, &fixtures, 35, "five types");
}
