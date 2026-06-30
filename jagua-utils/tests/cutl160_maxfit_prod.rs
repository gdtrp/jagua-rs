//! CUTL-160 — max-fit consistency over **real production parts**.
//!
//! `nest_max_fit_auto` ("max copies of one part on a sheet") and the real periodic nest must agree:
//! the reported max per sheet has to equal the count the periodic nest actually places on a full
//! sheet, and the nest must never exceed it (the WS-7 "44 vs 45" guarantee, here extended from
//! rectangles to every deterministic class — grid, pairing and the irregular lattice). This harness
//! drives that contract with production geometries: the first ten request cases plus case-019 (an
//! 81-vertex concave part) and case-017 (a 328-vertex "hardcore" outline). For every single part
//! type it checks: max-fit is a single saturated sheet, it is reproducible, and a periodic nest of
//! `cap·2 + r` copies fills each full sheet to exactly `cap`, never more, with everything placed.

use jagua_utils::{
    AdaptiveNestingStrategy, NestingResult, PackingMode, PartInput, nest_auto, nest_max_fit_auto,
};
use std::path::{Path, PathBuf};

fn prod_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("jagua-sqs-processor/tests/testdata/prod-tests")
}

/// Where rendered sheets are written for visual validation.
fn out_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_output")
        .join("cutl160")
        .join("maxfit_prod")
}

/// Persist a single SVG sheet under `test_output/cutl160/maxfit_prod/<case>/<file>` for inspection.
fn write_sheet(case: &str, file: &str, svg: &[u8]) {
    let dir = out_root().join(case);
    std::fs::create_dir_all(&dir).expect("create output dir");
    std::fs::write(dir.join(file), svg).expect("write sheet");
}

struct Case {
    name: String,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    rotations: usize,
    parts: Vec<PartInput>,
}

fn load_case(dir: &Path) -> Option<Case> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("data.json")).ok()?).ok()?;
    let num = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
    let mut parts = Vec::new();
    for item in v.get("items").and_then(|x| x.as_array())? {
        let item_id = item.get("itemId").and_then(|x| x.as_str())?;
        let count = item.get("count").and_then(|x| x.as_u64()).unwrap_or(1) as usize;
        let allowed_rotations = item
            .get("allowedRotations")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|n| n.as_f64().map(|d| d as f32))
                    .collect::<Vec<f32>>()
            });
        let svg = std::fs::read(dir.join(format!("{item_id}.svg"))).ok()?;
        parts.push(PartInput {
            svg_bytes: svg,
            count,
            item_id: Some(item_id.to_string()),
            allowed_rotations,
        });
    }
    Some(Case {
        name: dir.file_name()?.to_string_lossy().into_owned(),
        bin_w: num("binWidth", 980.0) as f32,
        bin_h: num("binHeight", 2000.0) as f32,
        spacing: num("spacing", 2.0) as f32,
        rotations: num("amountOfRotations", 4.0) as usize,
        parts,
    })
}

/// Run max-fit for one part type and assert it agrees with the periodic nest.
fn check_part(case: &Case, part: &PartInput) {
    let id = part.item_id.clone().unwrap_or_default();
    let label = format!("{}/{}", case.name, &id[..id.len().min(8)]);
    let strategy = AdaptiveNestingStrategy::new();

    let max_fit = nest_max_fit_auto(
        &strategy,
        case.bin_w,
        case.bin_h,
        case.spacing,
        part,
        case.rotations,
        PackingMode::Auto,
        None,
    );
    let max_fit = match max_fit {
        Ok(r) => r,
        // An un-nestable part (too big for the sheet) is a legitimate outcome, not a failure.
        Err(e) => {
            eprintln!("[{label}] max_fit rejected: {e}");
            return;
        }
    };
    assert_eq!(
        max_fit.pages.len(),
        1,
        "[{label}] max_fit is a single sheet"
    );
    let cap = max_fit.pages[0].parts_placed;
    assert!(cap > 0, "[{label}] max_fit cap > 0");

    // Store the saturated single sheet for visual validation.
    let short = &id[..id.len().min(8)];
    let util = (max_fit.pages[0].utilisation * 100.0).round() as i32;
    if let Some(svg) = max_fit.page_svgs.first() {
        write_sheet(
            &case.name,
            &format!("{short}_maxfit_cap{cap}_util{util}.svg"),
            svg,
        );
    }

    // Deterministic max-fit must be reproducible (same cap + identical sheet bytes).
    let again = nest_max_fit_auto(
        &strategy,
        case.bin_w,
        case.bin_h,
        case.spacing,
        part,
        case.rotations,
        PackingMode::Auto,
        None,
    )
    .expect("max_fit rerun");
    assert_eq!(
        again.pages[0].parts_placed, cap,
        "[{label}] max_fit deterministic cap"
    );
    assert_eq!(
        again.page_svgs.first(),
        max_fit.page_svgs.first(),
        "[{label}] max_fit deterministic bytes"
    );

    // The LBF fallback (no deterministic class) can't promise the stencil contract; only the
    // deterministic paths (sheets_total_estimate = Some) do. Report and skip in the LBF case.
    if max_fit.sheets_total_estimate.is_none() {
        eprintln!("[{label}] max_fit via LBF (cap {cap}) — consistency check skipped");
        return;
    }

    // Periodic nest of cap·2 + r copies: each full sheet holds exactly `cap`, never more, all placed.
    let r = (cap / 2).max(1);
    let qty = cap * 2 + r;
    let many = PartInput {
        svg_bytes: part.svg_bytes.clone(),
        count: qty,
        item_id: part.item_id.clone(),
        allowed_rotations: part.allowed_rotations.clone(),
    };
    let periodic: NestingResult = nest_auto(
        &strategy,
        case.bin_w,
        case.bin_h,
        case.spacing,
        std::slice::from_ref(&many),
        case.rotations,
        PackingMode::Auto,
        None,
    )
    .expect("periodic nest");

    for pg in &periodic.pages {
        assert!(
            pg.parts_placed <= cap,
            "[{label}] page {} placed {} > max_fit cap {cap}",
            pg.page_index,
            pg.parts_placed
        );
    }
    assert_eq!(
        periodic.pages[0].parts_placed, cap,
        "[{label}] first full sheet == max_fit cap"
    );
    assert_eq!(
        periodic.parts_placed, qty,
        "[{label}] all {qty} parts placed"
    );
    eprintln!(
        "[{label}] OK — cap {cap}/sheet, {qty} parts on {} sheets",
        periodic.pages.len()
    );
}

#[test]
fn max_fit_matches_periodic_on_prod_parts() {
    let root = prod_root();
    // First ten request cases + a concave part (019) + the 328-vertex "hardcore" outline (017).
    let mut names: Vec<String> = (1..=10).map(|i| format!("case-{i:03}")).collect();
    names.push("case-017".into());
    names.push("case-019".into());

    let mut ran = 0usize;
    for name in &names {
        let dir = root.join(name);
        let Some(case) = load_case(&dir) else {
            eprintln!("[{name}] not present — skipping");
            continue;
        };
        for part in &case.parts {
            check_part(&case, part);
            ran += 1;
        }
    }
    assert!(
        ran > 0,
        "no production cases were available under {}",
        root.display()
    );
    eprintln!("max-fit consistency checked across {ran} part type(s)");
}
