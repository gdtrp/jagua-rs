//! End-to-end tests for offcut (free-space) detection (JG-OFF-1).
//!
//! These run the real nesting strategies on SVG inputs and assert that, when an
//! [`OffcutPolicy`] is set, the final result carries per-page offcuts — and that, when no
//! policy is set, offcuts stay empty (backwards-compatible behaviour).

#[cfg(test)]
mod tests {
    use jagua_utils::svg_nesting::{
        AdaptiveNestingStrategy, NestingResult, NestingStrategy, Offcut, OffcutPolicy, OffcutShape,
        PartInput, SimpleNestingStrategy,
    };
    use std::sync::Mutex;

    /// A small square SVG that packs easily, leaving plenty of free space in a large bin.
    const SQUARE_SVG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M 0,0 L 100,0 L 100,100 L 0,100 Z" fill="black"/>
</svg>"#;

    /// An irregular real part used for the polygon path.
    const FORK_SVG: &[u8] = include_bytes!("../../jagua-sqs-processor/tests/testdata/fork.svg");

    /// A real production part (~220×357mm, irregular) used for a realistic multi-part render.
    const PROD1_SVG: &[u8] = include_bytes!("../../jagua-sqs-processor/tests/testdata/prod-1.svg");

    /// The three real parts from the user's offcut report (1500×3000 sheet, spacing 10,
    /// qty 5 each). Part A ≈400×165 rounded, B ≈67×67 square, C ≈240×141.
    const OFFCUT_PART_A: &[u8] =
        include_bytes!("../../jagua-sqs-processor/tests/testdata/offcut_part_a.svg");
    const OFFCUT_PART_B: &[u8] =
        include_bytes!("../../jagua-sqs-processor/tests/testdata/offcut_part_b.svg");
    const OFFCUT_PART_C: &[u8] =
        include_bytes!("../../jagua-sqs-processor/tests/testdata/offcut_part_c.svg");

    /// A large square that, packed bottom-left with no spacing, hugs the corner so the free
    /// space (`bin − hull`) is a simply-connected L-shaped polygon rather than a
    /// bin-with-hole.
    const BIG_SQUARE_SVG: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 800">
    <path d="M 0,0 L 800,0 L 800,800 L 0,800 Z" fill="black"/>
</svg>"#;

    const BIN_W: f32 = 2000.0;
    const BIN_H: f32 = 1000.0;

    fn square_parts(count: usize) -> Vec<PartInput> {
        vec![PartInput {
            svg_bytes: SQUARE_SVG.as_bytes().to_vec(),
            count,
            item_id: None,
            allowed_rotations: None,
        }]
    }

    fn rect_policy() -> OffcutPolicy {
        OffcutPolicy {
            min_offcut_width_mm: 200.0,
            min_offcut_height_mm: 200.0,
            shape: OffcutShape::Rectangle,
            kerf_mm: 0.0,
        }
    }

    fn within_bin(x: f32, y: f32) -> bool {
        (-1e-2..=BIN_W + 1e-2).contains(&x) && (-1e-2..=BIN_H + 1e-2).contains(&y)
    }

    #[test]
    fn final_result_has_offcuts_when_policy_set() {
        let strategy = AdaptiveNestingStrategy::new().with_offcut_policy(rect_policy());
        let result = strategy
            .nest(BIN_W, BIN_H, 5.0, &square_parts(2), 4, None)
            .expect("nesting should succeed");

        assert!(!result.pages.is_empty(), "expected at least one page");
        let total: usize = result.pages.iter().map(|p| p.offcuts.len()).sum();
        assert!(total > 0, "expected offcuts on the final layout");

        for page in &result.pages {
            for o in &page.offcuts {
                match o {
                    Offcut::Rect { width, height, .. } => {
                        assert!(
                            *width >= 200.0 && *height >= 200.0,
                            "offcut below threshold: {o:?}"
                        );
                    }
                    other => panic!("rectangle policy must yield RECT, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn no_policy_means_no_offcuts() {
        let strategy = AdaptiveNestingStrategy::new();
        let result = strategy
            .nest(BIN_W, BIN_H, 5.0, &square_parts(2), 4, None)
            .expect("nesting should succeed");

        assert!(
            result.pages.iter().all(|p| p.offcuts.is_empty()),
            "no policy must leave offcuts empty"
        );
    }

    #[test]
    fn intermediate_callbacks_have_empty_offcuts() {
        use std::sync::Arc;
        // Capture every streamed improvement; each must have empty offcuts (offcuts are
        // computed only on the final result), while the returned final must have them.
        let captured: Arc<Mutex<Vec<NestingResult>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&captured);
        let cb = move |r: NestingResult| {
            sink.lock().unwrap().push(r);
            Ok(())
        };

        let strategy = AdaptiveNestingStrategy::new().with_offcut_policy(rect_policy());
        // Many small parts in a large bin tends to stream several improving layouts.
        let result = strategy
            .nest(BIN_W, BIN_H, 5.0, &square_parts(40), 4, Some(Box::new(cb)))
            .expect("nesting should succeed");

        for (i, intermediate) in captured.lock().unwrap().iter().enumerate() {
            assert!(
                intermediate.pages.iter().all(|p| p.offcuts.is_empty()),
                "streamed intermediate #{i} must have empty offcuts"
            );
        }

        let final_total: usize = result.pages.iter().map(|p| p.offcuts.len()).sum();
        assert!(final_total > 0, "final result must carry offcuts");
    }

    #[test]
    fn quadrilateral_policy_yields_polygon_offcuts() {
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::Quadrilateral,
            kerf_mm: 0.0,
        };
        let parts = vec![PartInput {
            svg_bytes: BIG_SQUARE_SVG.as_bytes().to_vec(),
            count: 1,
            item_id: None,
            allowed_rotations: None,
        }];
        let strategy = AdaptiveNestingStrategy::new().with_offcut_policy(policy);
        // No spacing so the 1000x1000 part hugs the bottom-left corner and three edges.
        let result = strategy
            .nest(BIN_W, BIN_H, 0.0, &parts, 4, None)
            .expect("nesting should succeed");

        let polys: Vec<&Offcut> = result.pages.iter().flat_map(|p| p.offcuts.iter()).collect();
        assert!(
            !polys.is_empty(),
            "expected polygon offcuts beside the part"
        );
        for o in polys {
            match o {
                Offcut::Poly { vertices, .. } => {
                    assert!(vertices.len() >= 3, "degenerate polygon: {o:?}");
                    for v in vertices {
                        assert!(within_bin(v.x, v.y), "vertex outside bin: {v:?}");
                    }
                }
                other => panic!("quadrilateral policy must yield POLY, got {other:?}"),
            }
        }
    }

    #[test]
    fn quadrilateral_floating_part_consumes_free_space() {
        // A single part floating in the interior: the polygon offcut must capture ALL the
        // free space (sheet minus the part), whether geo expresses the part as an interior
        // ring (hole) or a pinched exterior. We verify by area: total free area is strictly
        // less than the whole sheet (the part is excluded) but still most of it.
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::Quadrilateral,
            kerf_mm: 0.0,
        };
        let parts = vec![PartInput {
            svg_bytes: FORK_SVG.to_vec(),
            count: 1,
            item_id: None,
            allowed_rotations: None,
        }];
        let strategy = AdaptiveNestingStrategy::new().with_offcut_policy(policy);
        let result = strategy
            .nest(BIN_W, BIN_H, 50.0, &parts, 4, None)
            .expect("nesting should succeed");

        let offcuts: Vec<&Offcut> = result.pages.iter().flat_map(|p| p.offcuts.iter()).collect();
        assert!(!offcuts.is_empty(), "expected a free-space polygon");

        let mut free_area = 0.0f64;
        for o in offcuts {
            match o {
                Offcut::Poly { vertices, holes } => {
                    for v in vertices.iter().chain(holes.iter().flatten()) {
                        assert!(within_bin(v.x, v.y), "vertex outside bin: {v:?}");
                    }
                    free_area += ring_area(vertices);
                    for h in holes {
                        free_area -= ring_area(h);
                    }
                }
                other => panic!("quadrilateral policy must yield POLY, got {other:?}"),
            }
        }

        let bin_area = (BIN_W * BIN_H) as f64;
        assert!(
            free_area < bin_area - 1000.0,
            "free area {free_area} should exclude the part (bin {bin_area})"
        );
        assert!(
            free_area > 0.9 * bin_area,
            "free area {free_area} should be most of the sheet"
        );
    }

    /// Absolute shoelace area of an open polygon ring.
    fn ring_area(ring: &[jagua_utils::svg_nesting::OffcutVertex]) -> f64 {
        let n = ring.len();
        if n < 3 {
            return 0.0;
        }
        let mut acc = 0.0f64;
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            acc += a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64;
        }
        (acc / 2.0).abs()
    }

    #[test]
    fn simple_strategy_offcuts() {
        // With a policy: offcuts present.
        let with = SimpleNestingStrategy::new().with_offcut_policy(rect_policy());
        let result = with
            .nest(BIN_W, BIN_H, 5.0, &square_parts(1), 4, None)
            .expect("nesting should succeed");
        let total: usize = result.pages.iter().map(|p| p.offcuts.len()).sum();
        assert!(
            total > 0,
            "simple strategy should produce offcuts with a policy"
        );

        // Without a policy: none.
        let without = SimpleNestingStrategy::new();
        let result = without
            .nest(BIN_W, BIN_H, 5.0, &square_parts(1), 4, None)
            .expect("nesting should succeed");
        assert!(
            result.pages.iter().all(|p| p.offcuts.is_empty()),
            "simple strategy must leave offcuts empty without a policy"
        );
    }

    #[test]
    fn prod1_30_parts_rectangle_offcuts() {
        // Realistic render: 30 copies of a real production part on a 2000×1000 sheet, with a
        // RECTANGLE offcut policy. Writes the result SVGs (with the offcut overlay drawn on)
        // to `test_output/offcuts/` for manual inspection.
        let parts = vec![PartInput {
            svg_bytes: PROD1_SVG.to_vec(),
            count: 30,
            item_id: Some("prod-1".into()),
            allowed_rotations: None,
        }];
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::Rectangle,
            kerf_mm: 5.0,
        };
        // SimpleNestingStrategy is deterministic and bounded — good for a reproducible render.
        let result = SimpleNestingStrategy::new()
            .with_offcut_policy(policy)
            .nest(BIN_W, BIN_H, 5.0, &parts, 4, None)
            .expect("nesting should succeed");

        assert!(result.parts_placed > 0, "expected parts to be placed");

        let out_dir = std::path::Path::new("test_output/offcuts");
        std::fs::create_dir_all(out_dir).unwrap();
        std::fs::write(
            out_dir.join("prod1_rect_combined.svg"),
            &result.combined_svg,
        )
        .unwrap();
        for (i, svg) in result.page_svgs.iter().enumerate() {
            std::fs::write(out_dir.join(format!("prod1_rect_page{i}.svg")), svg).unwrap();
        }

        // Every reported offcut must be a RECT meeting the threshold and lying within the bin.
        let total: usize = result.pages.iter().map(|p| p.offcuts.len()).sum();
        assert!(total > 0, "expected at least one offcut across the pages");
        for o in result.pages.iter().flat_map(|p| p.offcuts.iter()) {
            match o {
                Offcut::Rect {
                    x,
                    y,
                    width,
                    height,
                } => {
                    assert!(
                        *width >= 100.0 && *height >= 100.0,
                        "offcut below threshold: {o:?}"
                    );
                    assert!(within_bin(*x, *y) && within_bin(x + width, y + height));
                }
                other => panic!("rectangle policy must yield RECT, got {other:?}"),
            }
        }
    }

    #[test]
    fn prod1_30_parts_quadrilateral_offcuts() {
        // Same realistic render as the rectangle test, but with a QUADRILATERAL policy: the
        // offcut hugs the convex hull of the packed parts instead of their bounding box.
        // Writes its own SVGs to `test_output/offcuts/` for side-by-side comparison.
        let parts = vec![PartInput {
            svg_bytes: PROD1_SVG.to_vec(),
            count: 30,
            item_id: Some("prod-1".into()),
            allowed_rotations: None,
        }];
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::Quadrilateral,
            kerf_mm: 5.0,
        };
        let result = SimpleNestingStrategy::new()
            .with_offcut_policy(policy)
            .nest(BIN_W, BIN_H, 5.0, &parts, 4, None)
            .expect("nesting should succeed");

        assert!(result.parts_placed > 0, "expected parts to be placed");

        let out_dir = std::path::Path::new("test_output/offcuts");
        std::fs::create_dir_all(out_dir).unwrap();
        std::fs::write(
            out_dir.join("prod1_quad_combined.svg"),
            &result.combined_svg,
        )
        .unwrap();
        for (i, svg) in result.page_svgs.iter().enumerate() {
            std::fs::write(out_dir.join(format!("prod1_quad_page{i}.svg")), svg).unwrap();
        }

        // Any reported offcut must be a polygon whose vertices lie within the bin.
        for o in result.pages.iter().flat_map(|p| p.offcuts.iter()) {
            match o {
                Offcut::Poly { vertices, holes } => {
                    assert!(vertices.len() >= 3, "degenerate polygon: {o:?}");
                    for v in vertices.iter().chain(holes.iter().flatten()) {
                        assert!(within_bin(v.x, v.y), "vertex outside bin: {v:?}");
                    }
                }
                other => panic!("quadrilateral policy must yield POLY, got {other:?}"),
            }
        }
    }

    #[test]
    fn default_serialization_omits_offcuts() {
        // A page with empty offcuts must serialize without an `offcuts` key, keeping the
        // wire response byte-identical to the pre-feature build.
        use jagua_utils::svg_nesting::PageResult;
        let page = PageResult {
            page_index: 0,
            utilisation: 0.42,
            svg_url: None,
            parts_placed: 0,
            placements: Vec::new(),
            offcuts: Vec::new(),
        };
        let json = serde_json::to_string(&page).unwrap();
        assert!(
            !json.contains("offcuts"),
            "empty offcuts must be omitted from JSON: {json}"
        );
    }

    #[test]
    fn report_repro_three_parts_1500x3000_rectangle() {
        // Reproduces the user's offcut report: parts A/B/C ×5 on a 1500×3000 sheet, spacing
        // 10, RECTANGLE policy. Writes the result SVG (with offcut + kerf-band overlay) to
        // `test_output/offcuts/` for visual validation, then asserts the two reported fixes:
        //   #3 offcuts reach the TRUE sheet edges (no spacing gap), and
        //   #2 the free space is captured as MULTIPLE rectangles (not just the one big strip).
        const W: f32 = 1500.0;
        const H: f32 = 3000.0;
        const SPACING: f32 = 10.0;

        let parts = vec![
            PartInput {
                svg_bytes: OFFCUT_PART_A.to_vec(),
                count: 5,
                item_id: Some("A".into()),
                allowed_rotations: None,
            },
            PartInput {
                svg_bytes: OFFCUT_PART_B.to_vec(),
                count: 5,
                item_id: Some("B".into()),
                allowed_rotations: None,
            },
            PartInput {
                svg_bytes: OFFCUT_PART_C.to_vec(),
                count: 5,
                item_id: Some("C".into()),
                allowed_rotations: None,
            },
        ];
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::Rectangle,
            // Non-zero kerf so the shaded kerf band renders for visual inspection.
            kerf_mm: 10.0,
        };

        let result = AdaptiveNestingStrategy::new()
            .with_offcut_policy(policy)
            .nest(W, H, SPACING, &parts, 4, None)
            .expect("nesting should succeed");

        assert_eq!(result.parts_placed, 15, "all 15 parts should fit");

        let out_dir = std::path::Path::new("test_output/offcuts");
        std::fs::create_dir_all(out_dir).unwrap();
        std::fs::write(
            out_dir.join("report_repro_combined.svg"),
            &result.combined_svg,
        )
        .unwrap();
        for (i, svg) in result.page_svgs.iter().enumerate() {
            std::fs::write(out_dir.join(format!("report_repro_page{i}.svg")), svg).unwrap();
        }

        let offcuts: Vec<&Offcut> = result.pages.iter().flat_map(|p| p.offcuts.iter()).collect();
        assert!(!offcuts.is_empty(), "expected offcuts");

        // All offcuts are rectangles inside the TRUE sheet (0..W, 0..H) — note the bounds are
        // the real sheet, not the spacing-inset collision box.
        let mut max_right = 0.0f32;
        let mut max_top = 0.0f32;
        let mut min_bottom = H;
        for o in &offcuts {
            match o {
                Offcut::Rect {
                    x,
                    y,
                    width,
                    height,
                } => {
                    assert!(
                        *width >= 100.0 && *height >= 100.0,
                        "below threshold: {o:?}"
                    );
                    assert!(*x >= -1e-2 && *y >= -1e-2, "offcut outside sheet: {o:?}");
                    assert!(
                        x + width <= W + 1e-2 && y + height <= H + 1e-2,
                        "offcut outside sheet: {o:?}"
                    );
                    max_right = max_right.max(x + width);
                    max_top = max_top.max(y + height);
                    min_bottom = min_bottom.min(*y);
                }
                other => panic!("rectangle policy must yield RECT, got {other:?}"),
            }
        }

        // #3: an offcut reaches the true right edge and the true top/bottom — no spacing gap.
        assert!(
            max_right >= W - 1.0,
            "no offcut reaches the right sheet edge (max_right={max_right}, W={W})"
        );
        assert!(
            max_top >= H - 1.0 && min_bottom <= 1.0,
            "offcuts don't reach the true top/bottom (top={max_top}, bottom={min_bottom})"
        );

        // #2: the free space is captured as more than one rectangle (e.g. the big right
        // column PLUS the block below the packed left column).
        assert!(
            offcuts.len() >= 2,
            "expected multiple offcut rectangles, got {}: {offcuts:?}",
            offcuts.len()
        );

        // The kerf band is drawn onto the page SVG.
        let page0 = String::from_utf8_lossy(&result.page_svgs[0]);
        assert!(
            page0.contains("id=\"offcut_kerf\""),
            "kerf band group missing from page SVG"
        );
    }

    #[test]
    fn report_repro_three_parts_rectangle_merged() {
        // Same scenario as the rectangle repro, but RECTANGLE_MERGED: the rectangle
        // decomposition is unioned into a single connected outline so the kerf band wraps only
        // the true material perimeter (no kerf on the internal rectangle-split lines). Writes
        // its SVG to `test_output/offcuts/` for side-by-side visual comparison.
        const W: f32 = 1500.0;
        const H: f32 = 3000.0;
        const SPACING: f32 = 10.0;

        let parts = vec![
            PartInput {
                svg_bytes: OFFCUT_PART_A.to_vec(),
                count: 5,
                item_id: Some("A".into()),
                allowed_rotations: None,
            },
            PartInput {
                svg_bytes: OFFCUT_PART_B.to_vec(),
                count: 5,
                item_id: Some("B".into()),
                allowed_rotations: None,
            },
            PartInput {
                svg_bytes: OFFCUT_PART_C.to_vec(),
                count: 5,
                item_id: Some("C".into()),
                allowed_rotations: None,
            },
        ];
        let policy = OffcutPolicy {
            min_offcut_width_mm: 100.0,
            min_offcut_height_mm: 100.0,
            shape: OffcutShape::RectangleMerged,
            kerf_mm: 10.0,
        };

        let result = AdaptiveNestingStrategy::new()
            .with_offcut_policy(policy)
            .nest(W, H, SPACING, &parts, 4, None)
            .expect("nesting should succeed");

        assert_eq!(result.parts_placed, 15, "all 15 parts should fit");

        let out_dir = std::path::Path::new("test_output/offcuts");
        std::fs::create_dir_all(out_dir).unwrap();
        std::fs::write(
            out_dir.join("report_repro_merged_combined.svg"),
            &result.combined_svg,
        )
        .unwrap();
        for (i, svg) in result.page_svgs.iter().enumerate() {
            std::fs::write(
                out_dir.join(format!("report_repro_merged_page{i}.svg")),
                svg,
            )
            .unwrap();
        }

        let offcuts: Vec<&Offcut> = result.pages.iter().flat_map(|p| p.offcuts.iter()).collect();
        assert!(!offcuts.is_empty(), "expected offcuts");

        // The merged result is a single connected polygon (the free space is one region).
        assert_eq!(
            offcuts.len(),
            1,
            "RECTANGLE_MERGED should collapse to one polygon, got {}: {offcuts:?}",
            offcuts.len()
        );

        // It is a POLY that reaches the true sheet edges.
        let (mut max_right, mut max_top, mut min_bottom) = (0.0f32, 0.0f32, H);
        match offcuts[0] {
            Offcut::Poly { vertices, .. } => {
                assert!(vertices.len() >= 4, "degenerate polygon: {:?}", offcuts[0]);
                for v in vertices {
                    assert!(
                        within_repro_sheet(v.x, v.y, W, H),
                        "vertex outside sheet: {v:?}"
                    );
                    max_right = max_right.max(v.x);
                    max_top = max_top.max(v.y);
                    min_bottom = min_bottom.min(v.y);
                }
            }
            other => panic!("RECTANGLE_MERGED must yield POLY, got {other:?}"),
        }
        assert!(
            max_right >= W - 1.0,
            "offcut doesn't reach right edge ({max_right})"
        );
        assert!(
            max_top >= H - 1.0 && min_bottom <= 1.0,
            "offcut doesn't reach true top/bottom (top={max_top}, bottom={min_bottom})"
        );

        let page0 = String::from_utf8_lossy(&result.page_svgs[0]);
        assert!(page0.contains("id=\"offcut_kerf\""), "kerf band missing");
    }

    fn within_repro_sheet(x: f32, y: f32, w: f32, h: f32) -> bool {
        (-1e-2..=w + 1e-2).contains(&x) && (-1e-2..=h + 1e-2).contains(&y)
    }
}
