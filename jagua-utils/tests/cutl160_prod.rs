//! CUTL-160 — batch test over **real production requests** (generated from a CSV export).
//!
//! `scripts/gen_prod_cases.py` turns each CSV row into
//! `jagua-sqs-processor/tests/testdata/prod-tests/case-NNN/` (`data.json` +
//! `<itemId>.svg`, fetched from the request's private S3 url). This harness discovers every case at
//! runtime, runs `nest_auto`, writes the rendered sheets to `<case>/out/`, and emits a per-case
//! report (`prod-tests/REPORT.md` + `REPORT.csv`). It never aborts on a single case — LBF/rejected
//! outcomes are recorded, not failed; only a genuine fast-path bug (placed < requested, or
//! non-deterministic output) fails the test.

use jagua_utils::{AdaptiveNestingStrategy, NestingResult, PackingMode, PartInput, nest_auto};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn prod_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("jagua-sqs-processor/tests/testdata/prod-tests")
}

struct Case {
    name: String,
    csv_row: u64,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    rotations: usize,
    parts: Vec<PartInput>,
    total: usize,
}

fn load_case(dir: &Path) -> Option<Case> {
    let data_path = dir.join("data.json");
    if !data_path.exists() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&data_path).ok()?).ok()?;
    let name = dir.file_name()?.to_string_lossy().into_owned();
    let num = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
    let mut parts = Vec::new();
    let mut total = 0;
    for item in v.get("items").and_then(|x| x.as_array())? {
        let item_id = item.get("itemId").and_then(|x| x.as_str())?;
        let count = item.get("count").and_then(|x| x.as_u64())? as usize;
        // allowedRotations: present ⇒ Some(degrees); empty ⇒ any rotation (contract); absent ⇒ None.
        let allowed_rotations = item
            .get("allowedRotations")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|n| n.as_f64().map(|d| d as f32))
                    .collect::<Vec<f32>>()
            });
        let svg = std::fs::read(dir.join(format!("{item_id}.svg"))).ok()?;
        total += count;
        parts.push(PartInput {
            svg_bytes: svg,
            count,
            item_id: Some(item_id.to_string()),
            allowed_rotations,
        });
    }
    Some(Case {
        name,
        csv_row: v.get("csvRow").and_then(|x| x.as_u64()).unwrap_or(0),
        bin_w: num("binWidth", 980.0) as f32,
        bin_h: num("binHeight", 2000.0) as f32,
        spacing: num("spacing", 2.0) as f32,
        rotations: num("amountOfRotations", 4.0) as usize,
        parts,
        total,
    })
}

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Ok,       // deterministic fast path, all parts placed
    General,  // routed to LBF (informational)
    Rejected, // impossible request (e.g. part larger than bin)
    Bug,      // fast path but placed < requested or non-deterministic — a real defect
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Ok => "OK",
            Status::General => "LBF",
            Status::Rejected => "REJECTED",
            Status::Bug => "BUG",
        }
    }
}

struct Report {
    name: String,
    csv_row: u64,
    types: usize,
    total: usize,
    placed: usize,
    sheets: usize,
    density: f32,
    ms: u128,
    status: Status,
    note: String,
}

fn write_case_artifacts(case: &Case, result: &NestingResult) {
    let dir = prod_root().join(&case.name).join("out");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("out dir");
    let sheets_dir = dir.join("sheets");
    std::fs::create_dir_all(&sheets_dir).expect("sheets dir");

    let dominant = |p: &jagua_utils::PageResult| -> String {
        let mut c: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for pl in &p.placements {
            *c.entry(pl.item_id.as_str()).or_default() += 1;
        }
        c.into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(id, _)| id.chars().take(8).collect())
            .unwrap_or_else(|| "empty".into())
    };

    let mut index = format!(
        "# {} — {} sheets, {}/{} parts, {:.1}% util\n\n| run | sheets | each | parts/sheet | dominant | file |\n|---|---|---|---|---|---|\n",
        case.name,
        result.pages.len(),
        result.parts_placed,
        case.total,
        result.utilisation * 100.0,
    );
    let mut i = 0;
    let mut run_no = 0;
    while i < result.page_svgs.len() {
        let svg = &result.page_svgs[i];
        let mut j = i + 1;
        while j < result.page_svgs.len() && result.page_svgs[j] == *svg {
            j += 1;
        }
        let page = &result.pages[i];
        let fname = format!(
            "run{run_no:02}_sheets{i:04}-{:04}_x{}_{}parts_{}.svg",
            j - 1,
            j - i,
            page.parts_placed,
            dominant(page)
        );
        std::fs::write(sheets_dir.join(&fname), svg).expect("sheet");
        index.push_str(&format!(
            "| {run_no} | {i}–{} | ×{} | {} | {} | sheets/{fname} |\n",
            j - 1,
            j - i,
            page.parts_placed,
            dominant(page)
        ));
        run_no += 1;
        i = j;
    }
    std::fs::write(dir.join("index.md"), index).expect("index");
    // Full combined view only for small cases (avoid huge files for hundred-sheet runs).
    if result.pages.len() <= 6 {
        std::fs::write(dir.join("combined.svg"), &result.combined_svg).expect("combined");
    }
}

fn run_case(case: &Case) -> Report {
    let types = case.parts.len();
    let mut rep = Report {
        name: case.name.clone(),
        csv_row: case.csv_row,
        types,
        total: case.total,
        placed: 0,
        sheets: 0,
        density: 0.0,
        ms: 0,
        status: Status::Ok,
        note: String::new(),
    };
    // The LBF path only checks its time budget *between* waves, so a single huge wave (e.g. a tiny
    // irregular part ×thousands) can run unbounded. A deadline cancellation checker stops it
    // mid-solve; `with_time_budget` still gates the between-wave guard. Deterministic packers ignore
    // both (they finish in well under a second).
    let cancel_deadline = Instant::now() + Duration::from_secs(8);
    let strategy = AdaptiveNestingStrategy::with_cancellation_checker(Box::new(move || {
        Instant::now() >= cancel_deadline
    }))
    .with_time_budget(Duration::from_secs(5));
    let t0 = Instant::now();
    let outcome = nest_auto(
        &strategy,
        case.bin_w,
        case.bin_h,
        case.spacing,
        &case.parts,
        case.rotations,
        PackingMode::Auto,
        None,
    );
    rep.ms = t0.elapsed().as_millis();

    let result = match outcome {
        Ok(r) => r,
        Err(e) => {
            rep.status = Status::Rejected;
            rep.note = e.to_string();
            return rep;
        }
    };
    rep.placed = result.parts_placed;
    rep.sheets = result.pages.len();
    rep.density = result.utilisation;
    write_case_artifacts(case, &result);

    if result.sheets_total_estimate.is_none() {
        // General LBF path: informational only (may not place all within the budget).
        rep.status = Status::General;
        return rep;
    }
    // Deterministic fast path: must place everything and be reproducible.
    if result.parts_placed != case.total {
        rep.status = Status::Bug;
        rep.note = format!("fast path placed {}/{}", result.parts_placed, case.total);
        return rep;
    }
    let again = nest_auto(
        &AdaptiveNestingStrategy::new(),
        case.bin_w,
        case.bin_h,
        case.spacing,
        &case.parts,
        case.rotations,
        PackingMode::Auto,
        None,
    );
    let deterministic = matches!(&again, Ok(r2)
        if r2.parts_placed == result.parts_placed
            && r2.pages.len() == result.pages.len()
            && r2.page_svgs.first() == result.page_svgs.first());
    if !deterministic {
        rep.status = Status::Bug;
        rep.note = "non-deterministic across runs".into();
    }
    rep
}

/// Batch report over all generated production cases. Runs 138 real requests; LBF-routed cases are
/// bounded by a deadline so the run can't hang.
#[test]
fn production_cases_report() {
    let root = prod_root();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|_| panic!("missing {}", root.display()))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir() && p.join("data.json").exists())
        .collect();
    dirs.sort();
    // Optional single-case filter for debugging: CUTL160_CASE=case-104.
    let only = std::env::var("CUTL160_CASE").ok();
    let cases: Vec<Case> = dirs
        .iter()
        .filter_map(|d| load_case(d))
        .filter(|c| only.as_ref().is_none_or(|o| &c.name == o))
        .collect();
    assert!(!cases.is_empty(), "no cases under {}", root.display());

    let mut reports: Vec<Report> = Vec::new();
    for (k, case) in cases.iter().enumerate() {
        let r = run_case(case);
        eprintln!(
            "[{:>3}/{}] {} row{} {} types — {}/{} on {} sheets, {:.0}% ({}) {}ms{}",
            k + 1,
            cases.len(),
            r.name,
            r.csv_row,
            r.types,
            r.placed,
            r.total,
            r.sheets,
            r.density * 100.0,
            r.status.label(),
            r.ms,
            if r.note.is_empty() {
                String::new()
            } else {
                format!(" — {}", r.note)
            },
        );
        reports.push(r);
    }

    // --- write the report (markdown + csv) ---
    let mut md = String::from(
        "# CUTL-160 production batch report\n\n| case | csv row | types | parts | placed | packer | sheets | density | ms | status | note |\n|---|---|---|---|---|---|---|---|---|---|---|\n",
    );
    let mut csv =
        String::from("case,csv_row,types,parts,placed,packer,sheets,density_pct,ms,status,note\n");
    let (mut n_ok, mut n_lbf, mut n_rej, mut n_bug) = (0, 0, 0, 0);
    for r in &reports {
        let packer = match r.status {
            Status::General => "LBF",
            Status::Rejected => "-",
            _ => "deterministic",
        };
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {:.0}% | {} | {} | {} |\n",
            r.name,
            r.csv_row,
            r.types,
            r.total,
            r.placed,
            packer,
            r.sheets,
            r.density * 100.0,
            r.ms,
            r.status.label(),
            r.note,
        ));
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{:.1},{},{},{}\n",
            r.name,
            r.csv_row,
            r.types,
            r.total,
            r.placed,
            packer,
            r.sheets,
            r.density * 100.0,
            r.ms,
            r.status.label(),
            r.note.replace(',', ";"),
        ));
        match r.status {
            Status::Ok => n_ok += 1,
            Status::General => n_lbf += 1,
            Status::Rejected => n_rej += 1,
            Status::Bug => n_bug += 1,
        }
    }
    let summary = format!(
        "\n**{} cases**: {} deterministic OK, {} LBF, {} rejected, {} BUG.\n",
        reports.len(),
        n_ok,
        n_lbf,
        n_rej,
        n_bug,
    );
    md.push_str(&summary);
    std::fs::write(root.join("REPORT.md"), &md).expect("REPORT.md");
    std::fs::write(root.join("REPORT.csv"), &csv).expect("REPORT.csv");
    eprintln!("{summary}\nreport -> {}", root.join("REPORT.md").display());

    let bugs: Vec<&Report> = reports.iter().filter(|r| r.status == Status::Bug).collect();
    assert!(
        bugs.is_empty(),
        "{} fast-path case(s) have defects: {}",
        bugs.len(),
        bugs.iter()
            .map(|r| format!("{} ({})", r.name, r.note))
            .collect::<Vec<_>>()
            .join(", "),
    );
}
