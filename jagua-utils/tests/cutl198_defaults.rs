//! CUTL-198: the defaults reproduce today's output byte for byte.
//!
//! `with_layout(STAIRCASE, TOP_LEFT)` — and STAIRCASE with any other corner, since the corner is
//! ignored under STAIRCASE — must give exactly the result of a strategy that never heard of the
//! layout fields, for every deterministic class (grid, pairing, lattice, mixed) and for max-fit.

use jagua_utils::{
    AdaptiveNestingStrategy, FillDirection, NestingResult, PackingMode, PartInput, StartCorner,
    nest_auto, nest_max_fit_auto,
};

const RECT_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/rect_100x150.svg");
const TRIANGLE_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/triangle_300x100.svg");
const CIRCLE_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/circle_d120.svg");
const MIXED_A: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_a.svg");
const MIXED_B: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_b.svg");

fn part(svg: &[u8], count: usize, id: &str) -> PartInput {
    PartInput {
        svg_bytes: svg.to_vec(),
        count,
        item_id: Some(id.to_string()),
        allowed_rotations: None,
    }
}

fn strategies() -> Vec<(&'static str, AdaptiveNestingStrategy)> {
    vec![
        (
            "explicit defaults",
            AdaptiveNestingStrategy::new()
                .with_layout(FillDirection::Staircase, StartCorner::TopLeft),
        ),
        (
            "staircase + bottom-right (corner ignored)",
            AdaptiveNestingStrategy::new()
                .with_layout(FillDirection::Staircase, StartCorner::BottomRight),
        ),
    ]
}

fn assert_same(label: &str, baseline: &NestingResult, other: &NestingResult) {
    assert_eq!(
        other.combined_svg, baseline.combined_svg,
        "{label}: combined_svg"
    );
    assert_eq!(other.page_svgs, baseline.page_svgs, "{label}: page_svgs");
    assert_eq!(
        serde_json::to_string(&other.pages).unwrap(),
        serde_json::to_string(&baseline.pages).unwrap(),
        "{label}: pages"
    );
    assert_eq!(other.parts_placed, baseline.parts_placed);
    assert_eq!(other.sheets_total_estimate, baseline.sheets_total_estimate);
}

fn check_nest(case: &str, parts: &[PartInput], bin_w: f32, bin_h: f32) {
    let baseline = nest_auto(
        &AdaptiveNestingStrategy::new(),
        bin_w,
        bin_h,
        2.0,
        parts,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("baseline");
    assert!(
        baseline.sheets_total_estimate.is_some(),
        "{case}: the baseline must be a deterministic fast path for a meaningful byte-compare"
    );
    for (name, strategy) in strategies() {
        let r = nest_auto(
            &strategy,
            bin_w,
            bin_h,
            2.0,
            parts,
            4,
            PackingMode::Auto,
            None,
        )
        .expect("nest");
        assert_same(&format!("{case} / {name}"), &baseline, &r);
    }
}

#[test]
fn grid_class_is_byte_identical() {
    check_nest("grid", &[part(RECT_SVG, 50, "rect")], 1000.0, 500.0);
}

#[test]
fn pairing_class_is_byte_identical() {
    check_nest("pairing", &[part(TRIANGLE_SVG, 30, "tri")], 1000.0, 500.0);
}

#[test]
fn lattice_class_is_byte_identical() {
    check_nest("lattice", &[part(CIRCLE_SVG, 40, "circle")], 1000.0, 500.0);
}

#[test]
fn mixed_class_is_byte_identical() {
    check_nest(
        "mixed",
        &[part(MIXED_A, 60, "a"), part(MIXED_B, 70, "b")],
        1000.0,
        500.0,
    );
}

#[test]
fn max_fit_is_byte_identical() {
    for (case, svg) in [("rect", RECT_SVG), ("triangle", TRIANGLE_SVG)] {
        let p = part(svg, 1, case);
        let baseline = nest_max_fit_auto(
            &AdaptiveNestingStrategy::new(),
            1000.0,
            500.0,
            2.0,
            &p,
            4,
            PackingMode::Auto,
            None,
        )
        .expect("baseline");
        for (name, strategy) in strategies() {
            let r = nest_max_fit_auto(
                &strategy,
                1000.0,
                500.0,
                2.0,
                &p,
                4,
                PackingMode::Auto,
                None,
            )
            .expect("max fit");
            assert_same(&format!("max-fit {case} / {name}"), &baseline, &r);
        }
    }
}
