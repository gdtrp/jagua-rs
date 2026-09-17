//! CUTL-198: max-fit under a row/column fill reports the fill packer's own full sheet, so it is
//! ≥ every actual sheet of the periodic run with the same direction (the WS-7 / G4 invariant).

use jagua_utils::{
    AdaptiveNestingStrategy, FillDirection, PackingMode, PartInput, StartCorner, nest_auto,
    nest_max_fit_auto,
};

const RECT_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl160/rect_100x150.svg");
const CIRCLE_SVG: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/circle_d120.svg");
const L_SVG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 120">
  <path d="M 0,0 L 100,0 L 100,40 L 40,40 L 40,120 L 0,120 Z" fill="black"/>
</svg>"#;

fn part(svg: &[u8], count: usize) -> PartInput {
    PartInput {
        svg_bytes: svg.to_vec(),
        count,
        item_id: Some("p".into()),
        allowed_rotations: None,
    }
}

#[test]
fn max_fit_matches_the_periodic_full_sheet() {
    for (name, svg) in [
        ("rect", RECT_SVG),
        ("circle", CIRCLE_SVG),
        ("lshape", L_SVG.as_bytes()),
    ] {
        for direction in [FillDirection::Horizontal, FillDirection::Vertical] {
            for corner in [StartCorner::TopLeft, StartCorner::BottomRight] {
                let strategy = AdaptiveNestingStrategy::new().with_layout(direction, corner);
                let max_fit = nest_max_fit_auto(
                    &strategy,
                    1000.0,
                    500.0,
                    2.0,
                    &part(svg, 1),
                    4,
                    PackingMode::Auto,
                    None,
                )
                .expect("max fit");
                let label = format!("{name} {direction:?} {corner:?}");
                assert_eq!(max_fit.pages.len(), 1, "{label}: one page");
                let cap = max_fit.pages[0].parts_placed;
                assert!(cap > 0, "{label}");
                assert_eq!(max_fit.parts_placed, cap);
                assert_eq!(max_fit.total_parts_requested, cap);

                let periodic = nest_auto(
                    &strategy,
                    1000.0,
                    500.0,
                    2.0,
                    &[part(svg, 3 * cap + 5)],
                    4,
                    PackingMode::Auto,
                    None,
                )
                .expect("nest");
                assert_eq!(periodic.parts_placed, 3 * cap + 5, "{label}");
                assert_eq!(
                    periodic.pages.len(),
                    4,
                    "{label}: 3 full sheets + 1 remainder"
                );
                for (i, page) in periodic.pages.iter().enumerate() {
                    assert!(
                        page.parts_placed <= cap,
                        "{label} page {i}: {} > max-fit {cap}",
                        page.parts_placed
                    );
                }
                for i in 0..3 {
                    assert_eq!(
                        periodic.pages[i].parts_placed, cap,
                        "{label}: full sheet {i}"
                    );
                    assert_eq!(
                        periodic.page_svgs[i], max_fit.page_svgs[0],
                        "{label}: a full sheet must be the max-fit stencil, byte for byte"
                    );
                }
                assert_eq!(periodic.pages[3].parts_placed, 5, "{label}");
            }
        }
    }
}
