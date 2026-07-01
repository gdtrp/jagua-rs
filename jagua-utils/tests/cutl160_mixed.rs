//! CUTL-160 WS-5: mixed-parts grouping.
//!
//! 2–4 rectangular part types: each type produces its own run of identical full sheets and the
//! leftovers of all types are co-packed onto shared remainder sheets (deterministic shelf packer).
//! Renders to `test_output/cutl160/mixed/`.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};
use std::path::PathBuf;

const A: &[u8] = include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_a.svg");
const B: &[u8] = include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_b.svg");
const C: &[u8] = include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/mixed_c.svg");

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

fn parts() -> Vec<PartInput> {
    vec![
        PartInput {
            svg_bytes: A.to_vec(),
            count: 400,
            item_id: Some("a".into()),
            allowed_rotations: None,
        },
        PartInput {
            svg_bytes: B.to_vec(),
            count: 300,
            item_id: Some("b".into()),
            allowed_rotations: None,
        },
        PartInput {
            svg_bytes: C.to_vec(),
            count: 250,
            item_id: Some("c".into()),
            allowed_rotations: None,
        },
    ]
}

fn run() -> jagua_utils::NestingResult {
    let strategy = AdaptiveNestingStrategy::new();
    nest_auto(
        &strategy,
        980.0,
        2000.0,
        2.0,
        &parts(),
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest")
}

#[test]
fn mixed_types_place_all_parts_deterministically() {
    let result = run();
    write_output("mixed", &result);

    let total: usize = parts().iter().map(|p| p.count).sum();
    assert_eq!(result.parts_placed, total, "all mixed parts placed");

    // Multiple full sheets (per type) plus remainder pages.
    assert!(
        result.pages.len() >= 5,
        "expected several sheets, got {}",
        result.pages.len()
    );

    // Every placement maps to one of the three types.
    for p in result.pages.iter().flat_map(|pg| &pg.placements) {
        assert!(p.part_index < 3, "part_index {} out of range", p.part_index);
    }

    // Deterministic: a second run is byte-identical.
    let again = run();
    assert_eq!(
        result.combined_svg, again.combined_svg,
        "mixed packing must be deterministic"
    );
}
