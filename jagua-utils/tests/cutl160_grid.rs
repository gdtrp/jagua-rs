//! CUTL-160 WS-2: rectangular grid fast path.
//!
//! The QA "frames" case (`docs/bugfix/image (5)/(6)`): rectangular parts were scattered at random
//! angles (~22/sheet, 66% density). The grid packer must place them in a dense cardinal-rotation
//! grid instead. Renders to `test_output/cutl160/grid_frames/` for visual validation.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};
use std::path::PathBuf;

const FRAME_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/frame_200x295.svg");

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
fn frames_pack_in_a_dense_cardinal_grid() {
    let strategy = AdaptiveNestingStrategy::new();
    // 200x295 frame on a 980x2000 sheet, spacing 2mm. A naive single-orientation grid fits 24;
    // the two-orientation packer rotates to fit 27.
    let part = PartInput {
        svg_bytes: FRAME_SVG.to_vec(),
        count: 27,
        item_id: Some("frame".into()),
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

    write_output("grid_frames", &result);

    assert_eq!(result.pages.len(), 1, "single sheet expected");
    assert_eq!(
        result.parts_placed, 27,
        "grid must fit 27 (beats LBF's ~22)"
    );
    assert!(
        result.utilisation > 0.8,
        "dense grid should exceed 80% utilisation, got {:.3}",
        result.utilisation
    );
    // Cardinal rotations only (WS-8): every placement is at 0° or 90°.
    for p in result.pages.iter().flat_map(|pg| &pg.placements) {
        let r = p.rotation.rem_euclid(360.0);
        let cardinal = (r - 0.0).abs() < 1.0
            || (r - 90.0).abs() < 1.0
            || (r - 180.0).abs() < 1.0
            || (r - 270.0).abs() < 1.0;
        assert!(cardinal, "off-axis rotation {r} in grid output");
    }
}
