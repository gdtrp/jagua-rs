//! CUTL-160 WS-6: determinate progress enabler.
//!
//! The deterministic fast paths know the total sheet count up front, so `sheets_total_estimate` is
//! populated — the frontend can render a determinate progress bar (percent ≈ sheets / sheetsTotal)
//! instead of the indeterminate spinner from `docs/bugfix/image (3)`.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};

const RECT_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/rect_100x150.svg");

#[test]
fn periodic_reports_total_sheets_up_front() {
    let strategy = AdaptiveNestingStrategy::new();
    let part = PartInput {
        svg_bytes: RECT_SVG.to_vec(),
        count: 401,
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

    assert_eq!(
        result.sheets_total_estimate,
        Some(result.pages.len()),
        "fast paths must report the total sheet count up front"
    );
}
