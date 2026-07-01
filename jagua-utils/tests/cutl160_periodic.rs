//! CUTL-160 WS-3: periodic packing (identical sheets + remainder).
//!
//! The QA "N different sheets" case (`docs/bugfix/image (7)`, 29 different layouts; `image (10)`
//! max-per-sheet bug): bulk identical parts must produce `K` byte-identical full sheets plus one
//! remainder sheet, not `K` different layouts. Renders to `test_output/cutl160/periodic_rect/`.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};
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
    std::fs::write(
        dir.join("pages.json"),
        serde_json::to_string_pretty(&result.pages).expect("serialize pages"),
    )
    .expect("write pages.json");
}

#[test]
fn bulk_rectangles_repeat_one_stencil_plus_remainder() {
    let strategy = AdaptiveNestingStrategy::new();
    let qty = 401usize;
    let part = PartInput {
        svg_bytes: RECT_SVG.to_vec(),
        count: qty,
        item_id: Some("rect".into()),
        allowed_rotations: None,
    };
    let result = nest_auto(
        &strategy,
        980.0,
        2000.0,
        2.0,
        std::slice::from_ref(&part),
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest");

    write_output("periodic_rect", &result);

    // Total placed equals the requested quantity (all parts accounted for).
    assert_eq!(result.parts_placed, qty, "all parts placed");

    let cap = result.pages[0].parts_placed;
    assert!(cap > 0, "stencil must place at least one part");
    let expected_full = qty / cap;
    let expected_rem = qty % cap;
    let expected_pages = expected_full + usize::from(expected_rem > 0);
    assert_eq!(
        result.pages.len(),
        expected_pages,
        "K full + remainder pages"
    );

    // Every full sheet is byte-identical to the stencil (the core fix).
    let stencil_svg = &result.page_svgs[0];
    for i in 0..expected_full {
        assert_eq!(
            &result.page_svgs[i], stencil_svg,
            "full sheet {i} must be byte-identical to the stencil"
        );
        assert_eq!(
            result.pages[i].parts_placed, cap,
            "full sheet {i} holds cap parts"
        );
    }
    if expected_rem > 0 {
        let last = result.pages.last().unwrap();
        assert_eq!(last.parts_placed, expected_rem, "remainder count");
        assert!(
            last.parts_placed <= cap,
            "remainder never exceeds the per-sheet max (cap={cap})"
        );
    }

    // WS-7: no sheet ever exceeds the stencil cap (the 44-vs-45 bug is impossible here).
    for pg in &result.pages {
        assert!(pg.parts_placed <= cap, "page {} exceeds cap", pg.page_index);
    }
}
