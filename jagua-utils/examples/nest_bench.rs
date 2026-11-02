//! RFC-030 N11: one timed run of jagua-rs on one corpus case, as one JSON line on stdout. Built with
//! `cargo build --release --example nest_bench` (the workspace's release profile: opt-level 3, fat
//! LTO; an example so it may use the dev-dependencies).
//!
//! nest_bench <case dir> <oracle|production> [first_run]
//!
//! - `oracle`: counted samples, no clock (the N4/N5 parity mode): the samples are reported, so wall
//!   time per sample compares the engines at equal work.
//! - `production`: as jagua-sqs-processor runs a request without maxSeconds (the 600 s execution
//!   timeout as a cancellation checker): the time to the first improvement and to the final.
//!
//! Offcut detection is off (ORACLE_NO_OFFCUTS's reason: it changes no layout and costs hours on the
//! large cases, JK-6; N6a measured it apart). One run per process, so the peak RSS (VmHWM) and the
//! CPU time (utime + stime) read from /proc/self are this run's.
use jagua_utils::svg_nesting::oracle::{Oracle, OracleConfig};
use jagua_utils::{
    AdaptiveNestingStrategy, FillDirection, NestingResult, PackingMode, PartInput, StartCorner, nest_auto,
    nest_max_fit_auto,
};
use serde_json::{Value, json};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Case {
    bin_w: f32,
    bin_h: f32,
    spacing: f32,
    rotations: usize,
    max_fit: bool,
    fill: Option<(FillDirection, StartCorner)>,
    parts: Vec<PartInput>,
}

/// The case as tests/oracle_prod.rs load_case reads it.
fn load_case(dir: &Path) -> Case {
    let v: Value = serde_json::from_slice(&std::fs::read(dir.join("data.json")).expect("data.json")).expect("json");
    let num = |k: &str, d: f64| v.get(k).and_then(Value::as_f64).unwrap_or(d);
    let mut parts = Vec::new();
    for item in v["items"].as_array().expect("items") {
        let item_id = item["itemId"].as_str().expect("itemId");
        let allowed_rotations = item
            .get("allowedRotations")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|n| n.as_f64().map(|d| d as f32)).collect::<Vec<f32>>());
        parts.push(PartInput {
            svg_bytes: std::fs::read(dir.join(format!("{item_id}.svg"))).expect("part svg"),
            count: item["count"].as_u64().expect("count") as usize,
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
    Case {
        bin_w: num("binWidth", 980.0) as f32,
        bin_h: num("binHeight", 2000.0) as f32,
        spacing: num("spacing", 2.0) as f32,
        rotations: num("amountOfRotations", 4.0) as usize,
        max_fit: v.get("maxFit").and_then(Value::as_bool).unwrap_or(false),
        fill: direction.map(|d| (d, corner)),
        parts,
    }
}

fn run(case: &Case, s: AdaptiveNestingStrategy, cb: jagua_utils::svg_nesting::ImprovementCallback) -> anyhow::Result<NestingResult> {
    let s = match case.fill {
        Some((d, c)) => s.with_layout(d, c),
        None => s,
    };
    if case.max_fit {
        nest_max_fit_auto(&s, case.bin_w, case.bin_h, case.spacing, &case.parts[0], case.rotations, PackingMode::Auto, Some(cb))
    } else {
        nest_auto(&s, case.bin_w, case.bin_h, case.spacing, &case.parts, case.rotations, PackingMode::Auto, Some(cb))
    }
}

/// utime + stime of this process in ms, and VmHWM in KiB (Linux /proc).
fn proc_usage() -> (u64, u64) {
    let stat = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let after = stat.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<&str> = after.split_whitespace().collect();
    let ticks = |i: usize| f.get(i).and_then(|s| s.parse::<u64>().ok()).unwrap_or(0);
    let cpu_ms = (ticks(11) + ticks(12)) * 10; // utime, stime in USER_HZ = 100
    let hwm = std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1).and_then(|n| n.parse().ok()))
        .unwrap_or(0);
    (cpu_ms, hwm)
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 3, "usage: nest_bench <case dir> <oracle|production> [first_run]");
    let dir = Path::new(&a[1]);
    let mode = a[2].as_str();
    let first_run: usize = a.get(3).map_or(Ok(1), |s| s.parse())?;
    let case = load_case(dir);
    let start = Instant::now();
    let first: Arc<Mutex<Option<u64>>> = Arc::new(Mutex::new(None));
    let f2 = first.clone();
    let cb: jagua_utils::svg_nesting::ImprovementCallback = Box::new(move |_r| {
        let mut f = f2.lock().unwrap();
        if f.is_none() {
            *f = Some(start.elapsed().as_millis() as u64);
        }
        Ok(())
    });
    let (result, samples) = match mode {
        "oracle" => {
            let oracle = Oracle::new(OracleConfig { first_run, ..OracleConfig::default() });
            let s = AdaptiveNestingStrategy::new().with_oracle(oracle.clone());
            let r = run(&case, s, cb);
            let rep = oracle.finish()?;
            (r, Some(rep.runs.iter().map(|x| x.samples as u64).sum::<u64>()))
        }
        "production" => {
            let timeout = Duration::from_secs(600);
            let s = AdaptiveNestingStrategy::with_cancellation_checker(Box::new(move || start.elapsed() > timeout));
            (run(&case, s, cb), None)
        }
        m => anyhow::bail!("unknown mode {m}"),
    };
    let ms = start.elapsed().as_millis() as u64;
    let (cpu_ms, rss_kib) = proc_usage();
    let first_ms = *first.lock().unwrap();
    println!(
        "{}",
        json!({
            "engine": "jagua", "case": dir.file_name().map(|n| n.to_string_lossy().into_owned()),
            "mode": mode, "firstRun": first_run, "ms": ms, "firstImprovementMs": first_ms,
            "cpuMs": cpu_ms, "peakRssKiB": rss_kib, "samples": samples,
            "sheets": result.as_ref().map(|r| r.pages.len()).ok(),
            "partsPlaced": result.as_ref().map(|r| r.parts_placed).ok(),
            "utilisation": result.as_ref().map(|r| r.utilisation as f64).ok(),
            "error": result.as_ref().err().map(|e| format!("{e:#}")),
        })
    );
    Ok(())
}
