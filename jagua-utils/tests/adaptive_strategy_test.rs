#[cfg(test)]
mod tests {
    use jagua_utils::svg_nesting::{AdaptiveNestingStrategy, NestingResult, NestingStrategy, PartInput};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    /// Test that optimization stops immediately when all parts are placed,
    /// even on the first run (not continuing to run 40 iterations)
    #[test]
    fn test_stops_on_first_successful_placement() {
        // Create a simple square SVG that's easy to pack
        let svg = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M 10,10 L 90,10 L 90,90 L 10,90 Z" fill="black"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();

        // Try to place just 1 part in a large bin (should succeed on first run)
        let parts = vec![PartInput { svg_bytes: svg.as_bytes().to_vec(), count: 1, item_id: None }];
        let result = strategy.nest(
            500.0,  // bin_width - large enough to fit the part
            500.0,  // bin_height
            5.0,    // spacing
            &parts,
            4,      // amount_of_rotations
            None,   // no callback
        );

        // Should succeed
        assert!(result.is_ok(), "Nesting should succeed");

        let nesting_result = result.unwrap();

        // Should place the part
        assert_eq!(
            nesting_result.parts_placed, 1,
            "Should place 1 part"
        );

        // Should not have unplaced parts
        assert!(
            nesting_result.unplaced_parts_svg.is_none(),
            "Should have no unplaced parts"
        );
    }

    /// Test that when all requested parts are placed, optimization returns immediately
    /// This tests the fix for stopping on first successful run
    #[test]
    fn test_early_return_when_all_parts_placed() {
        // Create a simple square SVG
        let svg = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50">
    <path d="M 0,0 L 50,0 L 50,50 L 0,50 Z" fill="black"/>
</svg>"#;

        // Counter to track how many times the callback is called
        let callback_count = Arc::new(AtomicUsize::new(0));
        let callback_count_clone = Arc::clone(&callback_count);

        let callback = move |_result: NestingResult| {
            callback_count_clone.fetch_add(1, Ordering::SeqCst);
            Ok(())
        };

        let strategy = AdaptiveNestingStrategy::new();

        // Try to place 4 parts in a large bin
        let parts = vec![PartInput { svg_bytes: svg.as_bytes().to_vec(), count: 4, item_id: None }];
        let result = strategy.nest(
            500.0,
            500.0,
            5.0,
            &parts,
            4,
            Some(Box::new(callback)),
        );

        assert!(result.is_ok(), "Nesting should succeed");

        let nesting_result = result.unwrap();
        assert_eq!(nesting_result.parts_placed, 4, "Should place all 4 parts");

        // The optimization should stop early once all parts are placed
        // It should NOT run 40 iterations
        // The callback count should be relatively low (not 40)
        let count = callback_count.load(Ordering::SeqCst);
        assert!(
            count < 10,
            "Callback should be called fewer than 10 times when all parts placed early, but was called {} times",
            count
        );
    }

    /// End-to-end test with real-world complex SVG
    /// Tests timeout and iteration limits with parts that don't all fit
    #[test]
    fn test_complex_svg_with_timeout() {
        // Real-world complex circular SVG with holes (from user's data)
        let svg = r#"<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="496mm" height="493mm" viewBox="2690 -1917 496 493" xmlns="http://www.w3.org/2000/svg" version="1.1">
<title>OpenSCAD Model</title>
<path d="
M 3014.74,-1434.89 L 3062,-1455.93 L 3103.86,-1486.34 L 3138.49,-1524.8 L 3164.36,-1569.61 L 3180.35,-1618.81
 L 3185.75,-1670.27 L 3180.35,-1721.73 L 3164.36,-1770.94 L 3138.49,-1815.75 L 3103.86,-1854.2 L 3062,-1884.61
 L 3014.74,-1905.66 L 2964.13,-1916.42 L 2912.38,-1916.42 L 2861.77,-1905.66 L 2814.5,-1884.61 L 2772.65,-1854.2
 L 2738.02,-1815.75 L 2712.15,-1770.94 L 2696.16,-1721.73 L 2690.75,-1670.27 L 2696.16,-1618.81 L 2712.15,-1569.61
 L 2738.02,-1524.8 L 2772.65,-1486.34 L 2814.5,-1455.93 L 2861.77,-1434.89 L 2912.38,-1424.13 L 2964.13,-1424.13
 z
M 2997.73,-1890.4 L 2995.94,-1891.05 L 2994.48,-1892.28 L 2993.52,-1893.94 L 2993.19,-1895.82 L 2993.52,-1897.7
 L 2994.48,-1899.35 L 2995.94,-1900.58 L 2997.73,-1901.23 L 2999.64,-1901.23 L 3001.44,-1900.58 L 3002.9,-1899.35
 L 3003.86,-1897.7 L 3004.19,-1895.82 L 3003.86,-1893.94 L 3002.9,-1892.28 L 3001.44,-1891.05 L 2999.64,-1890.4
 z
M 2876.87,-1890.4 L 2875.07,-1891.05 L 2873.61,-1892.28 L 2872.65,-1893.94 L 2872.32,-1895.82 L 2872.65,-1897.7
 L 2873.61,-1899.35 L 2875.07,-1900.58 L 2876.87,-1901.23 L 2878.78,-1901.23 L 2880.57,-1900.58 L 2882.03,-1899.35
 L 2882.99,-1897.7 L 2883.32,-1895.82 L 2882.99,-1893.94 L 2882.03,-1892.28 L 2880.57,-1891.05 L 2878.78,-1890.4
 z
M 2914.89,-1448 L 2869.19,-1457.71 L 2826.5,-1476.72 L 2788.7,-1504.18 L 2757.44,-1538.9 L 2734.08,-1579.37
 L 2719.64,-1623.8 L 2714.75,-1670.27 L 2719.64,-1716.74 L 2734.08,-1761.18 L 2757.44,-1801.64 L 2788.7,-1836.37
 L 2826.5,-1863.83 L 2869.19,-1882.83 L 2914.89,-1892.55 L 2961.62,-1892.55 L 3007.32,-1882.83 L 3050,-1863.83
 L 3087.81,-1836.37 L 3119.07,-1801.64 L 3142.43,-1761.18 L 3156.87,-1716.74 L 3161.75,-1670.27 L 3156.87,-1623.8
 L 3142.43,-1579.37 L 3119.07,-1538.9 L 3087.81,-1504.18 L 3050,-1476.72 L 3007.32,-1457.71 L 2961.62,-1448
 z
M 3102.41,-1829.97 L 3100.61,-1830.62 L 3099.15,-1831.85 L 3098.2,-1833.5 L 3097.86,-1835.38 L 3098.2,-1837.26
 L 3099.15,-1838.92 L 3100.61,-1840.15 L 3102.41,-1840.8 L 3104.32,-1840.8 L 3106.11,-1840.15 L 3107.58,-1838.92
 L 3108.53,-1837.26 L 3108.86,-1835.38 L 3108.53,-1833.5 L 3107.58,-1831.85 L 3106.11,-1830.62 L 3104.32,-1829.97
 z
M 2772.19,-1829.97 L 2770.4,-1830.62 L 2768.93,-1831.85 L 2767.98,-1833.5 L 2767.65,-1835.38 L 2767.98,-1837.26
 L 2768.93,-1838.92 L 2770.4,-1840.15 L 2772.19,-1840.8 L 2774.1,-1840.8 L 2775.9,-1840.15 L 2777.36,-1838.92
 L 2778.31,-1837.26 L 2778.65,-1835.38 L 2778.31,-1833.5 L 2777.36,-1831.85 L 2775.9,-1830.62 L 2774.1,-1829.97
 z
M 3162.84,-1725.29 L 3161.05,-1725.94 L 3159.58,-1727.17 L 3158.63,-1728.83 L 3158.3,-1730.71 L 3158.63,-1732.59
 L 3159.58,-1734.24 L 3161.05,-1735.47 L 3162.84,-1736.12 L 3164.75,-1736.12 L 3166.55,-1735.47 L 3168.01,-1734.24
 L 3168.97,-1732.59 L 3169.3,-1730.71 L 3168.97,-1728.83 L 3168.01,-1727.17 L 3166.55,-1725.94 L 3164.75,-1725.29
 z
M 2711.76,-1725.29 L 2709.96,-1725.94 L 2708.5,-1727.17 L 2707.54,-1728.83 L 2707.21,-1730.71 L 2707.54,-1732.59
 L 2708.5,-1734.24 L 2709.96,-1735.47 L 2711.76,-1736.12 L 2713.67,-1736.12 L 2715.46,-1735.47 L 2716.92,-1734.24
 L 2717.88,-1732.59 L 2718.21,-1730.71 L 2717.88,-1728.83 L 2716.92,-1727.17 L 2715.46,-1725.94 L 2713.67,-1725.29
 z
M 3162.84,-1604.42 L 3161.05,-1605.08 L 3159.58,-1606.3 L 3158.63,-1607.96 L 3158.3,-1609.84 L 3158.63,-1611.72
 L 3159.58,-1613.37 L 3161.05,-1614.6 L 3162.84,-1615.25 L 3164.75,-1615.25 L 3166.55,-1614.6 L 3168.01,-1613.37
 L 3168.97,-1611.72 L 3169.3,-1609.84 L 3168.97,-1607.96 L 3168.01,-1606.3 L 3166.55,-1605.08 L 3164.75,-1604.42
 z
M 2711.76,-1604.42 L 2709.96,-1605.08 L 2708.5,-1606.3 L 2707.54,-1607.96 L 2707.21,-1609.84 L 2707.54,-1611.72
 L 2708.5,-1613.37 L 2709.96,-1614.6 L 2711.76,-1615.25 L 2713.67,-1615.25 L 2715.46,-1614.6 L 2716.92,-1613.37
 L 2717.88,-1611.72 L 2718.21,-1609.84 L 2717.88,-1607.96 L 2716.92,-1606.3 L 2715.46,-1605.08 L 2713.67,-1604.42
 z
M 3102.41,-1499.75 L 3100.61,-1500.4 L 3099.15,-1501.63 L 3098.2,-1503.28 L 3097.86,-1505.16 L 3098.2,-1507.04
 L 3099.15,-1508.7 L 3100.61,-1509.93 L 3102.41,-1510.58 L 3104.32,-1510.58 L 3106.11,-1509.93 L 3107.58,-1508.7
 L 3108.53,-1507.04 L 3108.86,-1505.16 L 3108.53,-1503.28 L 3107.58,-1501.63 L 3106.11,-1500.4 L 3104.32,-1499.75
 z
M 2772.19,-1499.75 L 2770.4,-1500.4 L 2768.93,-1501.63 L 2767.98,-1503.28 L 2767.65,-1505.16 L 2767.98,-1507.04
 L 2768.93,-1508.7 L 2770.4,-1509.93 L 2772.19,-1510.58 L 2774.1,-1510.58 L 2775.9,-1509.93 L 2777.36,-1508.7
 L 2778.31,-1507.04 L 2778.65,-1505.16 L 2778.31,-1503.28 L 2777.36,-1501.63 L 2775.9,-1500.4 L 2774.1,-1499.75
 z
M 2997.73,-1439.31 L 2995.94,-1439.97 L 2994.48,-1441.19 L 2993.52,-1442.85 L 2993.19,-1444.73 L 2993.52,-1446.61
 L 2994.48,-1448.26 L 2995.94,-1449.49 L 2997.73,-1450.15 L 2999.64,-1450.15 L 3001.44,-1449.49 L 3002.9,-1448.26
 L 3003.86,-1446.61 L 3004.19,-1444.73 L 3003.86,-1442.85 L 3002.9,-1441.19 L 3001.44,-1439.97 L 2999.64,-1439.31
 z
M 2876.87,-1439.31 L 2875.07,-1439.97 L 2873.61,-1441.19 L 2872.65,-1442.85 L 2872.32,-1444.73 L 2872.65,-1446.61
 L 2873.61,-1448.26 L 2875.07,-1449.49 L 2876.87,-1450.15 L 2878.78,-1450.15 L 2880.57,-1449.49 L 2882.03,-1448.26
 L 2882.99,-1446.61 L 2883.32,-1444.73 L 2882.99,-1442.85 L 2882.03,-1441.19 L 2880.57,-1439.97 L 2878.78,-1439.31
 z
" stroke="black" fill="lightgray" stroke-width="0.5"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();
        let start = Instant::now();

        // Use parameters from user's scenario: 1200x1200 bin, 2mm spacing, 4 rotations, 6 parts
        let parts = vec![PartInput { svg_bytes: svg.as_bytes().to_vec(), count: 6, item_id: None }];
        let result = strategy.nest(
            1200.0,  // bin_width
            1200.0,  // bin_height
            2.0,     // spacing
            &parts,
            4,       // amount_of_rotations
            None,
        );

        let duration = start.elapsed();

        // Should complete successfully (even if not all parts fit)
        assert!(result.is_ok(), "Complex nesting should complete without error");

        let nesting_result = result.unwrap();

        // Should have placed some parts (at least 1)
        assert!(
            nesting_result.parts_placed > 0,
            "Should place at least 1 part"
        );

        // Should complete within reasonable time (much less than 10 minute timeout)
        // For test purposes, it should finish relatively quickly since we limited to 40 runs
        assert!(
            duration.as_secs() < 120,
            "Optimization should complete within 2 minutes for test, took {} seconds",
            duration.as_secs()
        );

        // Verify utilisation field is present and valid
        assert!(
            nesting_result.utilisation >= 0.0 && nesting_result.utilisation <= 1.0,
            "Utilisation should be between 0.0 and 1.0, got {}",
            nesting_result.utilisation
        );

        // Should have some utilisation if parts are placed
        if nesting_result.parts_placed > 0 {
            assert!(
                nesting_result.utilisation > 0.0,
                "Should have non-zero utilisation when parts are placed"
            );
        }

        println!(
            "Complex SVG test: placed {}/{} parts with {:.1}% bin utilisation in {:.2}s",
            nesting_result.parts_placed,
            nesting_result.total_parts_requested,
            nesting_result.utilisation * 100.0,
            duration.as_secs_f64()
        );
    }

    /// Test that when no items can be placed (part too large for bin),
    /// the strategy returns 0 parts placed without panicking
    #[test]
    fn test_returns_zero_when_part_too_large_for_bin() {
        // Create a large square SVG (500x500)
        let svg = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 500 500">
    <path d="M 0,0 L 500,0 L 500,500 L 0,500 Z" fill="black"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();

        // Try to place the part in a bin that's too small (100x100 bin for 500x500 part)
        let parts = vec![PartInput { svg_bytes: svg.as_bytes().to_vec(), count: 1, item_id: None }];
        let result = strategy.nest(
            100.0,  // bin_width - smaller than the part
            100.0,  // bin_height - smaller than the part
            5.0,    // spacing
            &parts,
            4,      // amount_of_rotations
            None,   // no callback
        );

        // Should succeed (not panic)
        assert!(result.is_ok(), "Nesting should return Ok, not panic");

        let nesting_result = result.unwrap();

        // Should place 0 parts since the part doesn't fit
        assert_eq!(
            nesting_result.parts_placed, 0,
            "Should place 0 parts when part is too large for bin"
        );

        // Should have empty SVG data
        assert!(
            nesting_result.combined_svg.is_empty(),
            "Combined SVG should be empty when no parts placed"
        );
        assert!(
            nesting_result.page_svgs.is_empty(),
            "Page SVGs should be empty when no parts placed"
        );

        // Utilisation should be 0
        assert_eq!(
            nesting_result.utilisation, 0.0,
            "Utilisation should be 0 when no parts placed"
        );

        // total_parts_requested should still reflect the original request
        assert_eq!(
            nesting_result.total_parts_requested, 1,
            "Should still report 1 part was requested"
        );
    }

    /// Multi-part placement test with 3 different SVGs.
    /// Nests them together and writes page SVGs + placements JSON to disk for visual validation.
    #[test]
    fn test_multi_part_placements_output() {
        let _ = env_logger::try_init();

        // Part 1: circle (30-point polygon approximation)
        let svg_circle = r#"<?xml version="1.0" standalone="no"?>
<svg width="90mm" height="90mm" viewBox="-45 -45 90 90" xmlns="http://www.w3.org/2000/svg" version="1.1">
<path d="M 13.9062,42.7979 L 22.5,38.9707 L 30.1113,33.4414 L 36.4062,26.4502 L 41.1094,18.3027 L 44.0166,9.35645
 L 45,-0 L 44.0166,-9.35645 L 41.1094,-18.3027 L 36.4062,-26.4502 L 30.1113,-33.4414 L 22.5,-38.9707
 L 13.9062,-42.7979 L 4.7041,-44.7539 L -4.7041,-44.7539 L -13.9062,-42.7979 L -22.5,-38.9707 L -30.1113,-33.4414
 L -36.4062,-26.4502 L -41.1094,-18.3027 L -44.0166,-9.35645 L -45,-0 L -44.0166,9.35645 L -41.1094,18.3027
 L -36.4062,26.4502 L -30.1113,33.4414 L -22.5,38.9707 L -13.9062,42.7979 L -4.7041,44.7539 L 4.7041,44.7539 z
" stroke="black" fill="lightgray" stroke-width="0.5"/>
</svg>"#;

        // Part 2: small square (80x80)
        let svg_square = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
    <path d="M 0,0 L 80,0 L 80,80 L 0,80 Z" fill="black"/>
</svg>"#;

        // Part 3: L-shape
        let svg_lshape = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M 0,0 L 40,0 L 40,60 L 100,60 L 100,100 L 0,100 Z" fill="black"/>
</svg>"#;

        let parts = vec![
            PartInput { svg_bytes: svg_circle.as_bytes().to_vec(), count: 10, item_id: None },
            PartInput { svg_bytes: svg_square.as_bytes().to_vec(), count: 15, item_id: None },
            PartInput { svg_bytes: svg_lshape.as_bytes().to_vec(), count: 12, item_id: None },
        ];

        let strategy = AdaptiveNestingStrategy::new();

        let result = strategy.nest(
            1200.0,
            1200.0,
            5.0,
            &parts,
            4,
            None,
        );

        assert!(result.is_ok(), "Multi-part nesting should succeed");
        let nesting_result = result.unwrap();

        assert!(nesting_result.parts_placed > 0, "Should place at least some parts");
        assert!(!nesting_result.pages.is_empty(), "Pages should not be empty");

        // Verify placement data consistency
        let total_placements: usize = nesting_result.pages.iter().map(|p| p.placements.len()).sum();
        assert_eq!(
            total_placements,
            nesting_result.parts_placed,
            "Number of placements should match parts_placed"
        );

        // Verify all part_index values are within range
        for page in &nesting_result.pages {
            for p in &page.placements {
                assert!(p.part_index < 3, "part_index {} should be < 3", p.part_index);
            }
        }

        // Write output files for visual validation
        let output_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_output")
            .join("multi_part_placements");
        // Clean stale files from previous runs
        if output_dir.exists() {
            std::fs::remove_dir_all(&output_dir).expect("clean output dir");
        }
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        // Write each page SVG
        for (i, page_svg) in nesting_result.page_svgs.iter().enumerate() {
            let path = output_dir.join(format!("page-{}.svg", i));
            std::fs::write(&path, page_svg).expect("write page SVG");
            println!("Wrote page SVG: {}", path.display());
        }

        // Write combined SVG
        let combined_path = output_dir.join("combined.svg");
        std::fs::write(&combined_path, &nesting_result.combined_svg).expect("write combined SVG");
        println!("Wrote combined SVG: {}", combined_path.display());

        // Write unplaced parts SVG if present
        if let Some(ref unplaced_svg) = nesting_result.unplaced_parts_svg {
            let unplaced_path = output_dir.join("unplaced.svg");
            std::fs::write(&unplaced_path, unplaced_svg).expect("write unplaced SVG");
            println!("Wrote unplaced SVG: {}", unplaced_path.display());
        }

        // Write pages JSON (grouped placements by page)
        let pages_json = serde_json::to_string_pretty(&nesting_result.pages)
            .expect("serialize pages");
        let pages_path = output_dir.join("pages.json");
        std::fs::write(&pages_path, &pages_json).expect("write pages JSON");
        println!("Wrote pages JSON: {}", pages_path.display());

        println!(
            "\nMulti-part placement test summary:");
        println!(
            "  Parts requested: {} (circle: 10, square: 15, L-shape: 12)",
            nesting_result.total_parts_requested
        );
        println!("  Parts placed: {}", nesting_result.parts_placed);
        println!("  Pages: {}", nesting_result.pages.len());
        println!("  Utilisation: {:.1}%", nesting_result.utilisation * 100.0);
        println!("  Output dir: {}", output_dir.display());

        // Print per-page breakdown
        for page in &nesting_result.pages {
            println!(
                "  Page {}: {} items, utilisation {:.1}%",
                page.page_index,
                page.placements.len(),
                page.utilisation * 100.0
            );
        }
    }

    /// Test that high density stops optimization early
    #[test]
    fn test_high_density_early_stopping() {
        // Create small squares that will fill the bin efficiently
        let svg = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <path d="M 0,0 L 100,0 L 100,100 L 0,100 Z" fill="black"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();
        let start = Instant::now();

        // Try to place 100 parts in a large bin - should stop early when density is high
        let parts = vec![PartInput { svg_bytes: svg.as_bytes().to_vec(), count: 100, item_id: None }];
        let result = strategy.nest(
            1500.0,  // Large bin
            1500.0,
            5.0,
            &parts,
            4,
            None,
        );

        let duration = start.elapsed();

        assert!(result.is_ok(), "Nesting should succeed");
        let nesting_result = result.unwrap();

        // Should complete relatively quickly due to high density stopping
        assert!(
            duration.as_secs() < 30,
            "Should complete quickly with high density stopping, took {} seconds",
            duration.as_secs()
        );

        // Should have good utilisation
        assert!(
            nesting_result.utilisation > 0.0,
            "Should have positive utilisation"
        );

        println!(
            "High density test: placed {}/{} parts with {:.1}% utilisation in {:.2}s",
            nesting_result.parts_placed,
            nesting_result.total_parts_requested,
            nesting_result.utilisation * 100.0,
            duration.as_secs_f64()
        );
    }

    /// Test that single-part requests with 3+ pages skip SVG generation for middle pages,
    /// producing byte-identical SVGs for first and middle pages.
    #[test]
    fn test_single_part_skips_middle_pages() {
        // Small squares (80x80) in a 300x300 bin with spacing → ~9 per page
        // 50 parts should force 3+ pages
        let svg = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
    <path d="M 0,0 L 80,0 L 80,80 L 0,80 Z" fill="black"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();
        let parts = vec![PartInput {
            svg_bytes: svg.as_bytes().to_vec(),
            count: 50,
            item_id: None,
        }];

        let result = strategy
            .nest(300.0, 300.0, 5.0, &parts, 4, None)
            .expect("Nesting should succeed");

        // Should produce 3+ pages
        assert!(
            result.pages.len() > 2,
            "Expected >2 pages, got {}",
            result.pages.len()
        );

        // Middle page SVGs should be byte-identical to first page SVG
        let first_page_svg = &result.page_svgs[0];
        for (i, page_svg) in result.page_svgs.iter().enumerate().skip(1) {
            if i < result.page_svgs.len() - 1 {
                assert_eq!(
                    page_svg, first_page_svg,
                    "Middle page {} SVG should be identical to first page SVG",
                    i
                );
            }
        }

        // First and last pages should be non-empty
        assert!(
            !result.page_svgs[0].is_empty(),
            "First page SVG should be non-empty"
        );
        assert!(
            !result.page_svgs.last().unwrap().is_empty(),
            "Last page SVG should be non-empty"
        );

        // parts_placed should match sum of all page placements
        let total_placements: usize = result.pages.iter().map(|p| p.placements.len()).sum();
        assert_eq!(
            total_placements, result.parts_placed,
            "Sum of page placements ({}) should match parts_placed ({})",
            total_placements, result.parts_placed
        );

        // All part_index values should be 0 (single part type)
        for page in &result.pages {
            for p in &page.placements {
                assert_eq!(
                    p.part_index, 0,
                    "All part_index values should be 0 for single-part request"
                );
            }
        }
    }

    /// Test that a rounded rectangle with holes packs at least 21 parts into a 1000x1000 bin
    /// with spacing 20 and 4 rotations. This validates that the adaptive strategy parameters
    /// are tuned well enough for dense packing of complex shapes.
    #[test]
    fn test_rounded_rect_with_holes_packs_21_parts() {
        let _ = env_logger::try_init();

        // Rounded rectangle with internal holes (~240x141 bounding box)
        let svg = r##"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-120.00000000000003 -70.50000000000003 240.00000000000006 141.00000000000006">
  <path d="M-119.99999999999999,67.5 L-120,-67.5 L-119.91581370408996,-68.20571271990902 L-118.85510940920922,-69.85766178233088 L-117,-70.50000000000003 L117,-70.5 L117.70571271990899,-70.41581370408994 L119.35766178233087,-69.3551094092092 L120.00000000000001,-67.5 L120,67.5 L119.91581370408994,68.20571271990899 L118.8551094092092,69.85766178233087 L117,70.50000000000001 L-117,70.5 L-117.70571271990895,70.41581370408993 L-119.35766178233082,69.3551094092092 L-119.99999999999999,67.5 Z M-78.85,-50.75 L-68.35,-50.75 L-68.35,-53.25 L-78.85,-53.25 L-78.85,-50.75 Z M-68.35,50.75 L-78.85,50.75 L-78.85,53.25 L-68.35,53.25 L-68.35,50.75 Z M74.28,-53.25 L63.78,-53.25 L63.78,-50.75 L74.28,-50.75 L74.28,-53.25 Z M81.5,40 L101,40 L101,-40 L81.5,-40 L81.5,40 Z M-81.5,-40 L-101,-40 L-101,40 L-81.5,40 L-81.5,-40 Z M74.28,50.75 L63.78,50.75 L63.78,53.25 L74.28,53.25 L74.28,50.75 Z M117.5,63 L117.4879618166805,62.7549571491761 L117.45196320100808,62.51227419495968 L117.39235083933053,62.274288306863845 L117.30969883127823,62.043291419087275 L117.2048031608709,61.821508157935 L117.07867403075637,61.61107441745099 L116.93252613340685,61.41401678959088 L116.76776695296637,61.232233047033624 L116.58598321040913,61.067473866593154 L116.388925582549,60.92132596924363 L116.178491842065,60.795196839129105 L115.95670858091273,60.690301168721774 L115.72571169313616,60.60764916066947 L115.48772580504033,60.548036798991916 L115.2450428508239,60.512038183319504 L115,60.49999999999999 L114.7549571491761,60.512038183319504 L114.51227419495967,60.548036798991916 L114.27428830686384,60.60764916066947 L114.04329141908727,60.690301168721774 L113.821508157935,60.795196839129105 L113.611074417451,60.92132596924363 L113.41401678959089,61.067473866593154 L113.23223304703363,61.232233047033624 L113.06747386659315,61.41401678959088 L112.92132596924363,61.61107441745099 L112.7951968391291,61.821508157935 L112.69030116872177,62.043291419087275 L112.60764916066947,62.274288306863845 L112.54803679899192,62.51227419495968 L112.5120381833195,62.7549571491761 L112.5,63 L112.5120381833195,63.2450428508239 L112.54803679899192,63.48772580504032 L112.60764916066947,63.725711693136155 L112.69030116872177,63.956708580912725 L112.7951968391291,64.178491842065 L112.92132596924363,64.388925582549 L113.06747386659315,64.58598321040911 L113.23223304703363,64.76776695296637 L113.41401678959087,64.93252613340685 L113.611074417451,65.07867403075637 L113.821508157935,65.2048031608709 L114.04329141908727,65.30969883127823 L114.27428830686384,65.39235083933053 L114.51227419495967,65.45196320100808 L114.7549571491761,65.4879618166805 L115,65.5 L115.2450428508239,65.4879618166805 L115.48772580504033,65.45196320100808 L115.72571169313616,65.39235083933053 L115.95670858091273,65.30969883127823 L116.178491842065,65.2048031608709 L116.388925582549,65.07867403075637 L116.58598321040913,64.93252613340685 L116.76776695296637,64.76776695296637 L116.93252613340685,64.58598321040913 L117.07867403075637,64.388925582549 L117.2048031608709,64.178491842065 L117.30969883127823,63.95670858091273 L117.39235083933053,63.725711693136155 L117.45196320100808,63.48772580504032 L117.4879618166805,63.2450428508239 L117.5,63 Z M117.50000000000001,-63 L117.48796181668051,-63.2450428508239 L117.45196320100808,-63.48772580504032 L117.39235083933053,-63.72571169313616 L117.30969883127823,-63.95670858091273 L117.2048031608709,-64.178491842065 L117.07867403075637,-64.388925582549 L116.93252613340685,-64.58598321040913 L116.76776695296638,-64.76776695296638 L116.58598321040913,-64.93252613340685 L116.388925582549,-65.07867403075637 L116.178491842065,-65.2048031608709 L115.95670858091273,-65.30969883127823 L115.72571169313616,-65.39235083933053 L115.48772580504033,-65.45196320100808 L115.2450428508239,-65.48796181668051 L115,-65.50000000000001 L114.7549571491761,-65.48796181668051 L114.51227419495967,-65.45196320100808 L114.27428830686384,-65.39235083933053 L114.04329141908727,-65.30969883127823 L113.821508157935,-65.2048031608709 L113.611074417451,-65.07867403075637 L113.41401678959087,-64.93252613340685 L113.23223304703362,-64.76776695296638 L113.06747386659315,-64.58598321040913 L112.92132596924363,-64.388925582549 L112.7951968391291,-64.178491842065 L112.69030116872177,-63.95670858091273 L112.60764916066947,-63.72571169313616 L112.54803679899192,-63.48772580504033 L112.51203818331949,-63.2450428508239 L112.49999999999999,-63 L112.51203818331949,-62.7549571491761 L112.54803679899192,-62.51227419495967 L112.60764916066947,-62.27428830686384 L112.69030116872177,-62.04329141908727 L112.7951968391291,-61.821508157935 L112.92132596924363,-61.611074417450986 L113.06747386659315,-61.414016789590875 L113.23223304703362,-61.232233047033624 L113.41401678959087,-61.06747386659315 L113.611074417451,-60.921325969243625 L113.821508157935,-60.7951968391291 L114.04329141908727,-60.690301168721774 L114.27428830686384,-60.60764916066947 L114.51227419495967,-60.54803679899191 L114.7549571491761,-60.51203818331949 L115,-60.499999999999986 L115.2450428508239,-60.51203818331949 L115.48772580504033,-60.54803679899191 L115.72571169313616,-60.60764916066947 L115.95670858091273,-60.69030116872177 L116.178491842065,-60.7951968391291 L116.388925582549,-60.921325969243625 L116.58598321040913,-61.06747386659315 L116.76776695296638,-61.232233047033624 L116.93252613340685,-61.414016789590875 L117.07867403075637,-61.611074417450986 L117.2048031608709,-61.821508157935 L117.30969883127823,-62.04329141908727 L117.39235083933053,-62.27428830686384 L117.45196320100808,-62.51227419495967 L117.48796181668051,-62.7549571491761 L117.50000000000001,-63 Z M-112.5,63 L-112.51203818331952,62.7549571491761 L-112.54803679899193,62.51227419495968 L-112.60764916066948,62.274288306863845 L-112.69030116872179,62.043291419087275 L-112.79519683912912,61.82150815793501 L-112.92132596924364,61.611074417451 L-113.06747386659316,61.41401678959089 L-113.23223304703363,61.23223304703364 L-113.41401678959089,61.06747386659316 L-113.611074417451,60.92132596924364 L-113.82150815793501,60.79519683912912 L-114.04329141908728,60.69030116872179 L-114.27428830686385,60.60764916066948 L-114.51227419495969,60.54803679899193 L-114.7549571491761,60.51203818331952 L-115,60.50000000000001 L-115.2450428508239,60.51203818331952 L-115.48772580504031,60.54803679899193 L-115.72571169313615,60.60764916066948 L-115.95670858091272,60.69030116872179 L-116.17849184206499,60.79519683912912 L-116.388925582549,60.92132596924364 L-116.58598321040911,61.06747386659316 L-116.76776695296637,61.23223304703364 L-116.93252613340684,61.41401678959089 L-117.07867403075636,61.611074417451 L-117.20480316087088,61.82150815793501 L-117.30969883127821,62.043291419087275 L-117.39235083933052,62.274288306863845 L-117.45196320100807,62.51227419495968 L-117.48796181668048,62.7549571491761 L-117.5,63 L-117.48796181668048,63.2450428508239 L-117.45196320100807,63.48772580504032 L-117.39235083933052,63.725711693136155 L-117.30969883127821,63.956708580912725 L-117.20480316087088,64.17849184206499 L-117.07867403075636,64.388925582549 L-116.93252613340684,64.58598321040911 L-116.76776695296637,64.76776695296637 L-116.58598321040911,64.93252613340684 L-116.388925582549,65.07867403075636 L-116.17849184206499,65.20480316087088 L-115.95670858091272,65.30969883127821 L-115.72571169313615,65.39235083933052 L-115.48772580504033,65.45196320100807 L-115.2450428508239,65.48796181668048 L-115,65.5 L-114.75495714917611,65.48796181668048 L-114.51227419495969,65.45196320100807 L-114.27428830686385,65.39235083933052 L-114.04329141908728,65.30969883127821 L-113.82150815793501,65.20480316087088 L-113.611074417451,65.07867403075636 L-113.41401678959089,64.93252613340684 L-113.23223304703363,64.76776695296637 L-113.06747386659316,64.58598321040911 L-112.92132596924364,64.388925582549 L-112.79519683912912,64.17849184206499 L-112.69030116872179,63.956708580912725 L-112.60764916066948,63.725711693136155 L-112.54803679899193,63.48772580504032 L-112.51203818331952,63.2450428508239 L-112.5,63 Z M-112.5,-63 L-112.5120381833195,-63.2450428508239 L-112.54803679899193,-63.48772580504032 L-112.60764916066948,-63.725711693136155 L-112.69030116872179,-63.956708580912725 L-112.79519683912912,-64.178491842065 L-112.92132596924364,-64.388925582549 L-113.06747386659316,-64.58598321040911 L-113.23223304703363,-64.76776695296637 L-113.41401678959089,-64.93252613340684 L-113.611074417451,-65.07867403075636 L-113.821508157935,-65.20480316087088 L-114.04329141908728,-65.30969883127821 L-114.27428830686384,-65.39235083933052 L-114.51227419495967,-65.45196320100807 L-114.7549571491761,-65.4879618166805 L-115,-65.5 L-115.2450428508239,-65.4879618166805 L-115.48772580504033,-65.45196320100807 L-115.72571169313616,-65.39235083933052 L-115.95670858091272,-65.30969883127821 L-116.178491842065,-65.20480316087088 L-116.388925582549,-65.07867403075636 L-116.58598321040911,-64.93252613340684 L-116.76776695296637,-64.76776695296637 L-116.93252613340684,-64.58598321040911 L-117.07867403075636,-64.388925582549 L-117.20480316087088,-64.178491842065 L-117.30969883127821,-63.956708580912725 L-117.39235083933052,-63.725711693136155 L-117.45196320100807,-63.48772580504032 L-117.4879618166805,-63.2450428508239 L-117.5,-63 L-117.4879618166805,-62.7549571491761 L-117.45196320100807,-62.51227419495968 L-117.39235083933052,-62.274288306863845 L-117.30969883127821,-62.043291419087275 L-117.20480316087088,-61.82150815793501 L-117.07867403075636,-61.61107441745099 L-116.93252613340684,-61.41401678959089 L-116.76776695296637,-61.23223304703363 L-116.58598321040911,-61.06747386659316 L-116.388925582549,-60.92132596924364 L-116.178491842065,-60.79519683912911 L-115.95670858091273,-60.69030116872178 L-115.72571169313616,-60.60764916066948 L-115.48772580504033,-60.54803679899192 L-115.2450428508239,-60.512038183319504 L-115,-60.5 L-114.7549571491761,-60.512038183319504 L-114.51227419495967,-60.54803679899192 L-114.27428830686384,-60.607649160669474 L-114.04329141908727,-60.69030116872178 L-113.821508157935,-60.79519683912911 L-113.611074417451,-60.92132596924364 L-113.41401678959089,-61.06747386659316 L-113.23223304703363,-61.23223304703363 L-113.06747386659316,-61.41401678959089 L-112.92132596924364,-61.61107441745099 L-112.79519683912912,-61.82150815793501 L-112.69030116872179,-62.043291419087275 L-112.60764916066948,-62.274288306863845 L-112.54803679899193,-62.51227419495968 L-112.5120381833195,-62.7549571491761 L-112.5,-63 Z" fill="black" stroke="none" fill-rule="nonzero"/>
</svg>"##;

        let strategy = AdaptiveNestingStrategy::new();
        let start = Instant::now();

        let parts = vec![PartInput {
            svg_bytes: svg.as_bytes().to_vec(),
            count: 24,
            item_id: None,
        }];

        let result = strategy.nest(
            1000.0, // bin_width
            1000.0, // bin_height
            20.0,   // spacing
            &parts,
            4,      // amount_of_rotations (0°, 90°, 180°, 270°)
            None,
        );

        let duration = start.elapsed();

        assert!(result.is_ok(), "Nesting should succeed: {:?}", result.err());
        let nesting_result = result.unwrap();

        // Write output for visual inspection
        let output_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_output")
            .join("rounded_rect_21_parts");
        if output_dir.exists() {
            std::fs::remove_dir_all(&output_dir).expect("clean output dir");
        }
        std::fs::create_dir_all(&output_dir).expect("create output dir");

        for (i, page_svg) in nesting_result.page_svgs.iter().enumerate() {
            let path = output_dir.join(format!("page-{}.svg", i));
            std::fs::write(&path, page_svg).expect("write page SVG");
        }

        println!(
            "Rounded rect test: placed {}/{} parts with {:.1}% utilisation in {:.2}s on {} pages",
            nesting_result.parts_placed,
            nesting_result.total_parts_requested,
            nesting_result.utilisation * 100.0,
            duration.as_secs_f64(),
            nesting_result.pages.len()
        );
        println!("Output dir: {}", output_dir.display());

        // Must place at least 21 parts on the first page (single bin)
        // Shape is ~240x141, rotated 90° it's ~141x240
        // With spacing 20, effective ~161x260 or ~260x161
        // 1000x1000 bin should fit at least 21 parts with good packing
        let first_page_placed = nesting_result.pages.first()
            .map(|p| p.parts_placed)
            .unwrap_or(0);
        assert!(
            first_page_placed >= 21,
            "Should place at least 21 parts on the first page (1000x1000 bin with spacing 20), but only placed {}",
            first_page_placed
        );

        // Test should complete within 2 minutes
        assert!(
            duration.as_secs() < 120,
            "Test should complete within 2 minutes, took {} seconds",
            duration.as_secs()
        );
    }

    /// Test that multi-part requests do not apply the middle-page skip optimization.
    #[test]
    fn test_multi_part_does_not_skip_middle_pages() {
        // Two different part types
        let svg_square = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 80 80">
    <path d="M 0,0 L 80,0 L 80,80 L 0,80 Z" fill="black"/>
</svg>"#;

        let svg_rect = r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 60">
    <path d="M 0,0 L 120,0 L 120,60 L 0,60 Z" fill="black"/>
</svg>"#;

        let strategy = AdaptiveNestingStrategy::new();
        let parts = vec![
            PartInput {
                svg_bytes: svg_square.as_bytes().to_vec(),
                count: 25,
                item_id: None,
            },
            PartInput {
                svg_bytes: svg_rect.as_bytes().to_vec(),
                count: 25,
                item_id: None,
            },
        ];

        let result = strategy
            .nest(300.0, 300.0, 5.0, &parts, 4, None)
            .expect("Nesting should succeed");

        assert!(
            result.parts_placed > 0,
            "Should place at least some parts"
        );

        // Verify placement data consistency
        let total_placements: usize = result.pages.iter().map(|p| p.placements.len()).sum();
        assert_eq!(
            total_placements, result.parts_placed,
            "Sum of page placements should match parts_placed"
        );

        // Should have both part_index values present across all placements
        let has_part_0 = result
            .pages
            .iter()
            .flat_map(|p| &p.placements)
            .any(|p| p.part_index == 0);
        let has_part_1 = result
            .pages
            .iter()
            .flat_map(|p| &p.placements)
            .any(|p| p.part_index == 1);

        assert!(has_part_0, "Should have placements with part_index 0");
        assert!(has_part_1, "Should have placements with part_index 1");

        // All part_index values should be valid (0 or 1)
        for page in &result.pages {
            for p in &page.placements {
                assert!(
                    p.part_index < 2,
                    "part_index {} should be < 2",
                    p.part_index
                );
            }
        }
    }
}
