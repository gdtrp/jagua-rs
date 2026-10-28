//! RFC-030 N0 (dxf-processor): the oracle over the production corpus.
//!
//! Every case of `$ORACLE_CORPUS/case-NNN/` (`data.json` + `<itemId>.svg`) runs through the
//! production router (`nest_auto`, or `nest_max_fit_auto` when the request asked for max-fit) and
//! its answer is written to `$ORACLE_OUT/case-NNN/`. `$ORACLE_MODE` picks the run:
//! - `oracle` (default): oracle mode with a trace: wave of 1, no wall-clock check, the route,
//!   every LBF run and every sample decision recorded (`oracle.json`, `trace.jsonl`);
//! - `compare`: oracle mode without the trace (the library solver), checked against the traced
//!   run's `oracle.json`: the traced copy of LBF must be a copy, not a fork;
//! - `production`: the production strategy as a 1-CPU pod runs it (`NEST_RUN_PARALLELISM=1` set by
//!   the caller, the 600 s budget, wall-clock checks live) → `production.json`, the quality
//!   reference.
//! `$ORACLE_CASE` restricts the run to one case. `$ORACLE_WINDOW=a..b` writes sample events a..b in
//! full. Cases run one at a time: the warning counter is a process-wide logger.
//!
//! Gated behind the `oracle` feature: it needs the corpus, which customer drawings keep outside
//! every repository.
#![cfg(feature = "oracle")]

use jagua_utils::svg_nesting::oracle::{Oracle, OracleConfig, OracleReport};
use jagua_utils::{
    AdaptiveNestingStrategy, FillDirection, NestingResult, OffcutPolicy, OffcutShape, PackingMode,
    PartInput, StartCorner, nest_auto, nest_max_fit_auto,
};
use serde_json::{Value, json};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

struct Case {
    name: String,
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    rotations: usize,
    max_fit: bool,
    fill: Option<(FillDirection, StartCorner)>,
    parts: Vec<PartInput>,
    total: usize,
}

fn env_path(k: &str) -> PathBuf {
    PathBuf::from(std::env::var(k).unwrap_or_else(|_| panic!("{k} is not set")))
}

fn load_case(dir: &Path) -> Option<Case> {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("data.json")).ok()?).ok()?;
    let num = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
    let mut parts = Vec::new();
    let mut total = 0;
    for item in v.get("items").and_then(Value::as_array)? {
        let item_id = item.get("itemId").and_then(Value::as_str)?;
        let count = item.get("count").and_then(Value::as_u64)? as usize;
        // allowedRotations: present ⇒ Some(degrees); empty ⇒ any rotation; absent ⇒ None.
        let allowed_rotations = item.get("allowedRotations").and_then(Value::as_array).map(|a| {
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
    let direction = match v.get("fillDirection").and_then(Value::as_str) {
        Some("HORIZONTAL") => Some(FillDirection::Horizontal),
        Some("VERTICAL") => Some(FillDirection::Vertical),
        _ => None,
    };
    let corner = match v.get("startCorner").and_then(Value::as_str) {
        Some("TOP_RIGHT") => StartCorner::TopRight,
        Some("BOTTOM_LEFT") => StartCorner::BottomLeft,
        Some("BOTTOM_RIGHT") => StartCorner::BottomRight,
        _ => StartCorner::TopLeft,
    };
    Some(Case {
        name: dir.file_name()?.to_string_lossy().into_owned(),
        bin_w: num("binWidth", 980.0) as f32,
        bin_h: num("binHeight", 2000.0) as f32,
        spacing: num("spacing", 2.0) as f32,
        rotations: num("amountOfRotations", 4.0) as usize,
        max_fit: v.get("maxFit").and_then(Value::as_bool).unwrap_or(false),
        fill: direction.map(|d| (d, corner)),
        parts,
        total,
    })
}

/// The offcut policy the backend sends with every request (`NestingRequestFactory`): it shapes
/// the final answer only, never a decision.
fn backend_offcut_policy() -> OffcutPolicy {
    OffcutPolicy {
        min_offcut_width_mm: 100.0,
        min_offcut_height_mm: 100.0,
        shape: OffcutShape::RectangleMerged,
        kerf_mm: 0.0,
    }
}

fn strategy_for(case: &Case, base: AdaptiveNestingStrategy) -> AdaptiveNestingStrategy {
    let s = base.with_offcut_policy(backend_offcut_policy());
    match case.fill {
        Some((d, c)) => s.with_layout(d, c),
        None => s,
    }
}

fn run(case: &Case, strategy: &AdaptiveNestingStrategy) -> anyhow::Result<NestingResult> {
    if case.max_fit {
        nest_max_fit_auto(
            strategy,
            case.bin_w,
            case.bin_h,
            case.spacing,
            &case.parts[0],
            case.rotations,
            PackingMode::Auto,
            None,
        )
    } else {
        nest_auto(
            strategy,
            case.bin_w,
            case.bin_h,
            case.spacing,
            &case.parts,
            case.rotations,
            PackingMode::Auto,
            None,
        )
    }
}

// ----- warnings: repairs and inflation back-offs, deduplicated per case -----

#[derive(Default)]
struct Warnings {
    repair: BTreeSet<String>,
    backoff: BTreeSet<String>,
    other: BTreeSet<String>,
}

fn warnings() -> &'static Mutex<Warnings> {
    static W: OnceLock<Mutex<Warnings>> = OnceLock::new();
    W.get_or_init(|| Mutex::new(Warnings::default()))
}

struct Counter;

impl log::Log for Counter {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Warn
    }
    fn log(&self, r: &log::Record<'_>) {
        if r.level() > log::Level::Warn {
            return;
        }
        let msg = r.args().to_string();
        let mut w = warnings().lock().unwrap();
        // parsing.rs:1045-1087 (sanitize_polygon) and :1140-1156 (build_inflatable_shape).
        if msg.contains("Self-intersecting outer boundary")
            || msg.contains("repair collapsed")
            || msg.contains("Could not fully repair")
        {
            w.repair.insert(msg);
        } else if msg.contains("part inflation") {
            w.backoff.insert(msg);
        } else {
            w.other.insert(msg);
        }
    }
    fn flush(&self) {}
}

fn install_counter() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        log::set_boxed_logger(Box::new(Counter)).expect("logger");
        log::set_max_level(log::LevelFilter::Warn);
    });
    *warnings().lock().unwrap() = Warnings::default();
}

// ----- output -----

fn bits(f: f32) -> String {
    format!("{:08x}", f.to_bits())
}

fn result_json(r: &NestingResult) -> Value {
    let pages: Vec<Value> = r
        .pages
        .iter()
        .map(|p| {
            json!({
                "index": p.page_index,
                "utilisation": bits(p.utilisation),
                "partsPlaced": p.parts_placed,
                "placements": p.placements.iter().map(|q| json!({
                    "itemId": q.item_id, "partIndex": q.part_index,
                    "x": bits(q.x), "y": bits(q.y), "rotation": bits(q.rotation),
                    "centroidX": bits(q.centroid_x), "centroidY": bits(q.centroid_y),
                })).collect::<Vec<_>>(),
                "offcuts": serde_json::to_value(&p.offcuts).unwrap_or(Value::Null),
            })
        })
        .collect();
    json!({
        "partsPlaced": r.parts_placed,
        "partsRequested": r.total_parts_requested,
        "sheets": r.pages.len(),
        "utilisation": bits(r.utilisation),
        "materialUtilisation": bits(r.material_utilisation),
        "sheetsTotalEstimate": r.sheets_total_estimate,
        "unplacedSvg": r.unplaced_parts_svg.is_some(),
        "pages": pages,
    })
}

fn report_json(rep: &OracleReport) -> Value {
    json!({
        "route": rep.route.as_ref().map(|r| json!({
            "class": r.class, "packer": r.packer, "fellBack": r.fell_back,
        })),
        "guardsFired": rep.guards_fired,
        "sampleEvents": rep.sample_events,
        "traceHash": format!("{:016x}", rep.trace_hash),
        "runs": rep.runs.iter().map(|r| json!({
            "run": r.run_no, "loop": r.loop_idx, "seed": r.seed, "lsFrac": r.ls_frac,
            "placements": r.placements, "samples": r.samples, "placed": r.placed,
        })).collect::<Vec<_>>(),
    })
}

fn warnings_json() -> Value {
    let w = warnings().lock().unwrap();
    json!({
        "repair": w.repair.iter().collect::<Vec<_>>(),
        "backoff": w.backoff.iter().collect::<Vec<_>>(),
        "other": w.other.iter().collect::<Vec<_>>(),
    })
}

fn versions() -> Value {
    json!({
        "rustc": std::env::var("ORACLE_RUSTC").unwrap_or_default(),
        "libc": std::env::var("ORACLE_LIBC").unwrap_or_default(),
        "image": std::env::var("ORACLE_IMAGE").unwrap_or_default(),
        "jaguaCommit": std::env::var("ORACLE_COMMIT").unwrap_or_default(),
    })
}

fn window() -> std::ops::Range<u64> {
    std::env::var("ORACLE_WINDOW")
        .ok()
        .and_then(|w| {
            let (a, b) = w.split_once("..")?;
            Some(a.parse().ok()?..b.parse().ok()?)
        })
        .unwrap_or(0..0)
}

fn cases() -> Vec<PathBuf> {
    let root = env_path("ORACLE_CORPUS");
    let only = std::env::var("ORACLE_CASE").ok();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root)
        .expect("corpus")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("data.json").exists())
        .filter(|p| match &only {
            Some(c) => p.file_name().is_some_and(|n| n.to_string_lossy() == *c),
            None => true,
        })
        .collect();
    dirs.sort();
    dirs
}

#[test]
fn oracle_corpus() {
    install_counter();
    let mode = std::env::var("ORACLE_MODE").unwrap_or_else(|_| "oracle".into());
    let out_root = env_path("ORACLE_OUT");
    let mut failures = Vec::new();
    let dirs = cases();
    assert!(!dirs.is_empty(), "no case under $ORACLE_CORPUS");

    for dir in dirs {
        let Some(case) = load_case(&dir) else {
            failures.push(format!("{}: unreadable case", dir.display()));
            continue;
        };
        let out = out_root.join(&case.name);
        std::fs::create_dir_all(&out).unwrap();
        *warnings().lock().unwrap() = Warnings::default();
        let started = Instant::now();

        match mode.as_str() {
            "oracle" => {
                let trace = std::fs::File::create(out.join("trace.jsonl")).unwrap();
                let oracle = Oracle::with_trace(
                    OracleConfig {
                        sample_window: window(),
                        ..OracleConfig::default()
                    },
                    Box::new(std::io::BufWriter::new(trace)),
                );
                let strategy =
                    strategy_for(&case, AdaptiveNestingStrategy::new().with_oracle(oracle.clone()));
                let result = run(&case, &strategy);
                let rep = oracle.finish().unwrap();
                if rep.guards_fired > 0 {
                    failures.push(format!("{}: {} time guards fired", case.name, rep.guards_fired));
                }
                let doc = json!({
                    "case": case.name, "mode": mode, "partsTotal": case.total,
                    "ms": started.elapsed().as_millis() as u64,
                    "oracle": report_json(&rep),
                    "result": result.as_ref().map(result_json).unwrap_or(Value::Null),
                    "error": result.as_ref().err().map(|e| format!("{e:#}")),
                    "warnings": warnings_json(),
                    "versions": versions(),
                });
                std::fs::write(out.join("oracle.json"), serde_json::to_vec_pretty(&doc).unwrap())
                    .unwrap();
                eprintln!(
                    "{}: {:?} placed {:?} samples {} hash {:016x}",
                    case.name,
                    rep.route.map(|r| r.packer),
                    result.as_ref().map(|r| r.parts_placed).ok(),
                    rep.sample_events,
                    rep.trace_hash
                );
            }
            "compare" => {
                let traced: Value = match std::fs::read(out.join("oracle.json")) {
                    Ok(b) => serde_json::from_slice(&b).unwrap(),
                    Err(_) => {
                        failures.push(format!("{}: no traced oracle.json to compare", case.name));
                        continue;
                    }
                };
                let oracle = Oracle::new(OracleConfig::default());
                let strategy =
                    strategy_for(&case, AdaptiveNestingStrategy::new().with_oracle(oracle.clone()));
                let result = run(&case, &strategy);
                let rep = oracle.finish().unwrap();
                let got_result = result.as_ref().map(result_json).unwrap_or(Value::Null);
                let got_runs = report_json(&rep)["runs"].clone();
                if got_result != traced["result"] {
                    failures.push(format!("{}: untraced layout differs from the traced", case.name));
                }
                if got_runs != traced["oracle"]["runs"] {
                    failures.push(format!("{}: untraced runs differ from the traced", case.name));
                }
                eprintln!("{}: compared", case.name);
            }
            "production" => {
                let strategy = strategy_for(&case, AdaptiveNestingStrategy::new());
                let result = run(&case, &strategy);
                let doc = json!({
                    "case": case.name, "mode": mode, "partsTotal": case.total,
                    "ms": started.elapsed().as_millis() as u64,
                    "parallelism": std::env::var("NEST_RUN_PARALLELISM").unwrap_or_default(),
                    "result": result.as_ref().map(result_json).unwrap_or(Value::Null),
                    "error": result.as_ref().err().map(|e| format!("{e:#}")),
                    "warnings": warnings_json(),
                    "versions": versions(),
                });
                std::fs::write(
                    out.join("production.json"),
                    serde_json::to_vec_pretty(&doc).unwrap(),
                )
                .unwrap();
                eprintln!("{}: production {} ms", case.name, started.elapsed().as_millis());
            }
            other => panic!("unknown ORACLE_MODE {other}"),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
