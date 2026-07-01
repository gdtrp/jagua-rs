//! CUTL-160 WS-7: "max parts per sheet" must equal the periodic full-sheet count.
//!
//! The QA bug (`docs/bugfix/image (10)`): the UI showed "max per sheet: 44" yet a sheet held 45,
//! because max_fit and the real nest were different code paths. With both rectangular paths going
//! through the same deterministic grid stencil, the max can never be exceeded. Renders to
//! `test_output/cutl160/maxfit_consistency/`.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto, nest_max_fit_auto};
use std::path::PathBuf;

const RECT_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/rect_100x150.svg");

fn write_output(case: &str, result: &jagua_utils::NestingResult) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("cutl160")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create output dir");
    for (i, page) in result.page_svgs.iter().enumerate() {
        std::fs::write(dir.join(format!("page-{i}.svg")), page).expect("write page");
    }
    std::fs::write(dir.join("combined.svg"), &result.combined_svg).expect("write combined");
}

#[test]
fn max_fit_cap_equals_periodic_full_sheet_and_is_never_exceeded() {
    let strategy = AdaptiveNestingStrategy::new();
    let (bin_w, bin_h, spacing) = (980.0f32, 2000.0f32, 2.0f32);

    // max_fit: deterministic single-sheet maximum for this rectangle.
    let mf_part = PartInput {
        svg_bytes: RECT_SVG.to_vec(),
        count: 1, // count is ignored by max_fit (it saturates)
        item_id: Some("rect".into()),
        allowed_rotations: None,
    };
    let max_fit = nest_max_fit_auto(
        &strategy,
        bin_w,
        bin_h,
        spacing,
        &mf_part,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("max_fit");
    write_output("maxfit_consistency", &max_fit);

    assert_eq!(max_fit.pages.len(), 1, "max_fit returns a single sheet");
    let cap = max_fit.pages[0].parts_placed;
    assert!(cap > 0);

    // Periodic nest of many copies: every full sheet must hold exactly `cap` — never more.
    let qty = cap * 3 + 7;
    let part = PartInput {
        svg_bytes: RECT_SVG.to_vec(),
        count: qty,
        item_id: Some("rect".into()),
        allowed_rotations: None,
    };
    let periodic = nest_auto(
        &strategy,
        bin_w,
        bin_h,
        spacing,
        std::slice::from_ref(&part),
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest");

    for pg in &periodic.pages {
        assert!(
            pg.parts_placed <= cap,
            "page {} has {} parts > max_fit cap {cap}",
            pg.page_index,
            pg.parts_placed
        );
    }
    assert_eq!(
        periodic.pages[0].parts_placed, cap,
        "full sheet == max_fit cap"
    );
    assert_eq!(periodic.parts_placed, qty, "all parts placed");
}
