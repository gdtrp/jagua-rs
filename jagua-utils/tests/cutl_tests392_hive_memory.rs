//! cutl-tests#392 / #396: the request that OOM-killed jagua on staging (run 4a930f88, 2026-10-25).
//!
//! Two 450 × 1800 plates from `batch/hive.dxf`, 150 pieces each, on 1250 × 2500, spacing 2,
//! 4 rotations. One plate carries 2,215 Ø15 holes (~142k vertices, a 2.5 MB SVG); the other is a
//! plain rectangle. jagua peaked at 1204 MiB on it and was OOM-killed in its 1 GiB pod.
//!
//! The fixtures are cutl-backend's own conversions of the two shapes. This binary installs dhat's
//! counting allocator, so it measures the peak heap of the nesting call deterministically. dhat
//! allows one profiler per process, so this file holds exactly one test.

use jagua_utils::{AdaptiveNestingStrategy, PackingMode, PartInput, nest_auto};

#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

const WITH_HOLES: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl-tests-392/hive_with_holes.svg");
const PLAIN: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl-tests-392/hive_plain.svg");

const MIB: usize = 1024 * 1024;

fn part(svg: &[u8], count: usize, id: &str) -> PartInput {
    PartInput {
        svg_bytes: svg.to_vec(),
        count,
        item_id: Some(id.to_string()),
        allowed_rotations: None,
    }
}

#[test]
fn staging_request_4a930f88_places_every_part_within_bounded_memory() {
    let _profiler = dhat::Profiler::builder().testing().build();
    let parts = vec![
        part(WITH_HOLES, 150, "hive-with-holes"),
        part(PLAIN, 150, "hive-plain"),
    ];

    let result = nest_auto(
        &AdaptiveNestingStrategy::new(),
        1250.0,
        2500.0,
        2.0,
        &parts,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest");

    let stats = dhat::HeapStats::get();
    let page_bytes: usize = result.page_svgs.iter().map(|p| p.len()).sum();
    eprintln!(
        "hive 2 × 150: {} pages, {} / {} placed, page SVG bytes {} MiB, peak heap {} MiB, live {} MiB",
        result.pages.len(),
        result.parts_placed,
        result.total_parts_requested,
        page_bytes / MIB,
        stats.max_bytes / MIB,
        stats.curr_bytes / MIB,
    );
    assert_eq!(result.total_parts_requested, 300);
    assert_eq!(
        result.parts_placed, 300,
        "every plate fits a 1250 × 2500 sheet"
    );

    // Each type's full sheets are copies of one stencil, so they must share one buffer. Before the
    // fix every one of the 150 pages was its own 2.5 MB string, three times over.
    let mut distinct: Vec<*const u8> = result.page_svgs.iter().map(|p| p.as_ptr()).collect();
    distinct.sort();
    distinct.dedup();
    assert!(
        distinct.len() <= 3,
        "{} distinct page buffers for {} pages: identical sheets must share one",
        distinct.len(),
        result.page_svgs.len(),
    );

    // Measured 24 MiB after the fix, 1404 MiB before (staging: 1204 MiB RSS, OOM at 1 GiB).
    assert!(
        stats.max_bytes < 64 * MIB,
        "peak heap {} MiB: the 1 GiB worker pod must survive this request with room to spare",
        stats.max_bytes / MIB,
    );
}
