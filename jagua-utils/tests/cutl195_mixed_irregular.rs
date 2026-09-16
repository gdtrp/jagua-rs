//! CUTL-195: group mode must not re-nest every sheet from scratch.
//!
//! Replays the staging request behind the ticket (correlation `e33add54…`, 2026-09-14): a 100×60
//! rounded rectangle and a Ø120 circle, 120 of each on a 500×600 sheet, spacing 2, 4 rotations.
//! Nested separately they gave 2 + 7 identical sheets plus a remainder each; nested together the
//! request fell to LBF (the circle is not rectangular), which produced 10 *different* sheets in
//! ~107 s. Now each type keeps its own identical full sheets and only the leftovers share sheets.
//! Renders to `test_output/cutl195/`.

use jagua_utils::{AdaptiveNestingStrategy, NestingResult, PackingMode, PartInput, nest_auto};
use std::path::PathBuf;

const RRECT: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/rrect_100x60.svg");
const CIRCLE: &[u8] =
    include_bytes!("../../jagua-sqs-processor/tests/testdata/cutl195/circle_d120.svg");

fn part(svg: &[u8], count: usize, id: &str) -> PartInput {
    PartInput {
        svg_bytes: svg.to_vec(),
        count,
        item_id: Some(id.into()),
        allowed_rotations: None,
    }
}

fn nest(parts: &[PartInput], bin_w: f32, bin_h: f32) -> NestingResult {
    nest_auto(
        &AdaptiveNestingStrategy::new(),
        bin_w,
        bin_h,
        2.0,
        parts,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest")
}

fn write_output(case: &str, result: &NestingResult) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("cutl195")
        .join(case);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create output dir");
    for (i, page) in result.page_svgs.iter().enumerate() {
        std::fs::write(dir.join(format!("page-{i}.svg")), page).expect("write page");
    }
    std::fs::write(dir.join("combined.svg"), &result.combined_svg).expect("write combined");
}

/// `(x, y, rotation, part_index)` of a page, sorted — the geometric content of a sheet.
fn layout_of(page: &jagua_utils::PageResult) -> Vec<(i64, i64, i64, usize)> {
    let mut v: Vec<_> = page
        .placements
        .iter()
        .map(|p| {
            (
                (p.x * 1000.0).round() as i64,
                (p.y * 1000.0).round() as i64,
                (p.rotation * 1000.0).round() as i64,
                p.part_index,
            )
        })
        .collect();
    v.sort();
    v
}

/// Count how many leading pages of `pages` are byte-identical to the first one.
fn identical_run(pages: &[Vec<u8>]) -> usize {
    pages.iter().take_while(|p| **p == pages[0]).count()
}

#[test]
fn group_mode_keeps_each_types_identical_full_sheets() {
    let (bin_w, bin_h) = (500.0, 600.0);
    let a = part(RRECT, 120, "rrect");
    let b = part(CIRCLE, 120, "circle");

    // Reference: each type on its own (the "по отдельности всё красиво" runs).
    let alone_a = nest(std::slice::from_ref(&a), bin_w, bin_h);
    let alone_b = nest(std::slice::from_ref(&b), bin_w, bin_h);
    assert_eq!(alone_a.parts_placed, 120);
    assert_eq!(alone_b.parts_placed, 120);
    let full_a = identical_run(&alone_a.page_svgs);
    let full_b = identical_run(&alone_b.page_svgs);
    assert!(
        full_a >= 2,
        "rrect alone: expected ≥2 identical full sheets, got {full_a}"
    );
    assert!(
        full_b >= 2,
        "circle alone: expected ≥2 identical full sheets, got {full_b}"
    );
    assert_eq!(
        alone_a.pages.len(),
        full_a + 1,
        "rrect alone: full sheets + 1 remainder"
    );
    assert_eq!(
        alone_b.pages.len(),
        full_b + 1,
        "circle alone: full sheets + 1 remainder"
    );

    // Group mode.
    let mixed = nest(&[a.clone(), b.clone()], bin_w, bin_h);
    write_output("rrect_circle_500x600", &mixed);
    assert_eq!(mixed.parts_placed, 240, "all parts placed");
    assert_eq!(
        mixed.sheets_total_estimate,
        Some(mixed.pages.len()),
        "deterministic path reports its sheet count"
    );

    // The dominant sheets are the single-type stencils, repeated: first `full_a` pages are the
    // rrect sheet, the next `full_b` the circle sheet — same placements as when nested alone.
    let pages = &mixed.pages;
    assert!(
        pages.len() > full_a + full_b,
        "got only {} sheets",
        pages.len()
    );
    let strip_idx = |v: Vec<(i64, i64, i64, usize)>| -> Vec<(i64, i64, i64)> {
        v.into_iter().map(|(x, y, r, _)| (x, y, r)).collect()
    };
    for (i, page) in pages.iter().enumerate().take(full_a) {
        assert_eq!(
            layout_of(page),
            layout_of(&alone_a.pages[0]),
            "sheet {i} must be the rrect stencil"
        );
        assert_eq!(
            mixed.page_svgs[i], mixed.page_svgs[0],
            "rrect sheets byte-identical"
        );
    }
    for (i, page) in pages.iter().enumerate().skip(full_a).take(full_b) {
        // part_index differs between the single-type (0) and mixed (1) runs; compare geometry.
        assert_eq!(
            strip_idx(layout_of(page)),
            strip_idx(layout_of(&alone_b.pages[0])),
            "sheet {i} must be the circle stencil"
        );
        assert_eq!(
            mixed.page_svgs[i], mixed.page_svgs[full_a],
            "circle sheets byte-identical"
        );
    }

    // Remainder: at most as many sheets as the two single-type remainders, never more sheets in
    // total than cutting the types separately.
    let remainder_sheets = pages.len() - full_a - full_b;
    assert!(
        (1..=2).contains(&remainder_sheets),
        "expected 1–2 remainder sheets, got {remainder_sheets}"
    );
    assert!(pages.len() <= alone_a.pages.len() + alone_b.pages.len());
    let leftover_parts: usize = pages[full_a + full_b..]
        .iter()
        .map(|p| p.parts_placed)
        .sum();
    assert_eq!(
        leftover_parts,
        240 - full_a * pages[0].parts_placed - full_b * pages[full_a].parts_placed
    );

    // Every placement stays inside the sheet (band packing shifts stencil strips upward).
    for p in pages.iter().flat_map(|pg| &pg.placements) {
        assert!(
            p.x >= 0.0 && p.x <= bin_w && p.y >= 0.0 && p.y <= bin_h,
            "{p:?} outside sheet"
        );
    }

    // Deterministic: a second run is byte-identical.
    let again = nest(&[a, b], bin_w, bin_h);
    assert_eq!(
        mixed.combined_svg, again.combined_svg,
        "group mode must be deterministic"
    );
}

/// The "сменное задание" (shift task) case from the same ticket: 500 + 300 of the same two shapes on
/// a 1200×800 sheet took LBF 6.5 minutes and produced 9 unrelated sheets.
#[test]
fn shift_task_case_is_deterministic_and_places_everything() {
    let parts = [part(RRECT, 500, "rrect"), part(CIRCLE, 300, "circle")];
    let result = nest(&parts, 1200.0, 800.0);
    write_output("rrect_circle_1200x800", &result);
    assert_eq!(result.parts_placed, 800);
    assert!(
        result.sheets_total_estimate.is_some(),
        "must take the deterministic path"
    );
    // Both types have several identical full sheets.
    let run_a = identical_run(&result.page_svgs);
    assert!(run_a >= 2, "expected identical rrect sheets, got {run_a}");
    let run_b = identical_run(&result.page_svgs[run_a..]);
    assert!(run_b >= 2, "expected identical circle sheets, got {run_b}");
}

/// A small irregular mix where no type fills a sheet keeps today's LBF co-packing (the band
/// packer would only stack two short strips).
#[test]
fn small_irregular_mix_still_uses_the_general_optimiser() {
    let parts = [part(RRECT, 5, "rrect"), part(CIRCLE, 3, "circle")];
    let result = nest_auto(
        &AdaptiveNestingStrategy::new().with_time_budget(std::time::Duration::from_secs(20)),
        500.0,
        600.0,
        2.0,
        &parts,
        4,
        PackingMode::Auto,
        None,
    )
    .expect("nest");
    assert_eq!(result.parts_placed, 8);
    assert!(
        result.sheets_total_estimate.is_none(),
        "expected the general LBF path for a sub-sheet mix"
    );
}
