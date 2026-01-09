#[cfg(test)]
mod tests {
    use jagua_utils::svg_nesting::{AdaptiveNestingStrategy, NestingResult, NestingStrategy};
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
        let result = strategy.nest(
            500.0,  // bin_width - large enough to fit the part
            500.0,  // bin_height
            5.0,    // spacing
            svg.as_bytes(),
            1,      // amount_of_parts - just 1 part
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
        let result = strategy.nest(
            500.0,
            500.0,
            5.0,
            svg.as_bytes(),
            4,  // Request 4 parts
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
        let result = strategy.nest(
            1200.0,  // bin_width
            1200.0,  // bin_height
            2.0,     // spacing
            svg.as_bytes(),
            6,       // amount_of_parts
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
        let result = strategy.nest(
            100.0,  // bin_width - smaller than the part
            100.0,  // bin_height - smaller than the part
            5.0,    // spacing
            svg.as_bytes(),
            1,      // amount_of_parts
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
        let result = strategy.nest(
            1500.0,  // Large bin
            1500.0,
            5.0,
            svg.as_bytes(),
            100,  // Request many parts
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
}
