//! CUTL-160 WS-4: pairing fast path for half-bbox parts (right triangles).
//!
//! The QA «косынка» case (`docs/bugfix/image (8)/(11)`): a right triangle (area = ½ bbox) packed at
//! ~66% with a misaligned last row. Pairing two triangles into their bbox rectangle and grid-packing
//! that lifts density well above the LBF baseline and makes every full sheet identical. Renders to
//! `test_output/cutl160/pairing_triangle/`.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};
use std::path::PathBuf;

const TRIANGLE_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/triangle_300x100.svg");

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
fn right_triangles_pair_into_a_dense_regular_layout() {
    let strategy = AdaptiveNestingStrategy::new();
    let qty = 258usize;
    let part = PartInput {
        svg_bytes: TRIANGLE_SVG.to_vec(),
        count: qty,
        item_id: Some("kosynka".into()),
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

    write_output("pairing_triangle", &result);

    assert_eq!(result.parts_placed, qty, "all triangles placed");

    // Full sheets are identical (periodic) and far denser than the ~66% LBF baseline.
    let cap = result.pages[0].parts_placed;
    let expected_full = qty / cap;
    let stencil = &result.page_svgs[0];
    for i in 0..expected_full {
        assert_eq!(&result.page_svgs[i], stencil, "full sheet {i} identical");
    }
    assert!(
        result.pages[0].utilisation > 0.80,
        "paired triangles should exceed 80% density (LBF was ~66%), got {:.3}",
        result.pages[0].utilisation
    );

    // Only the two paired orientations 0° / 180° appear.
    for p in result.pages.iter().flat_map(|pg| &pg.placements) {
        let r = p.rotation.rem_euclid(360.0);
        let ok = (r - 0.0).abs() < 1.0 || (r - 180.0).abs() < 1.0 || (r - 360.0).abs() < 1.0;
        assert!(ok, "unexpected rotation {r} in pairing output");
    }
}
