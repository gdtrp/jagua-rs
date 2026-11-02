//! The Sparrow oracle (RFC-030 N10): one run on a strip instance (or a warm start: instance plus
//! solution), stopped by evaluation counts, with the decision trace of `sparrow::oracle`.
//!
//! oracle <input.json> <trace.jsonl> <final.json> <seed> <exploration evals> <compression evals> [workers]
//!
//! Configuration: Sparrow's defaults, except that simplification is off (the inputs are inflated
//! polygons, as the Go engine passes them), compression decays by failures (the clock-based decay
//! is the other nondeterminism), and each phase stops after its evaluation budget.
use jagua_rs::io::import::Importer;
use rand::rngs::Xoshiro256PlusPlus;
use rand::SeedableRng;
use sparrow::config::{ShrinkDecayStrategy, DEFAULT_SPARROW_CONFIG};
use sparrow::consts::DEFAULT_FAIL_DECAY_RATIO_CMPR;
use sparrow::optimizer::optimize;
use sparrow::oracle;
use sparrow::util::io;
use sparrow::util::listener::DummySolListener;
use std::path::Path;
use std::time::Duration;

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    anyhow::ensure!(a.len() >= 7, "usage: oracle <input> <trace> <final> <seed> <expl evals> <cmpr evals> [workers]");
    let (seed, expl, cmpr): (u64, u64, u64) = (a[4].parse()?, a[5].parse()?, a[6].parse()?);
    let workers: usize = a.get(7).map_or(Ok(3), |w| w.parse())?;

    let mut config = DEFAULT_SPARROW_CONFIG;
    config.poly_simpl_tolerance = None;
    config.narrow_concavity_cutoff_ratio = None;
    config.expl_cfg.time_limit = Duration::from_secs(expl);
    config.cmpr_cfg.time_limit = Duration::from_secs(cmpr);
    config.cmpr_cfg.shrink_decay = ShrinkDecayStrategy::FailureBased(DEFAULT_FAIL_DECAY_RATIO_CMPR);
    config.expl_cfg.separator_config.n_workers = workers;
    config.cmpr_cfg.separator_config.n_workers = workers;

    let (ext_instance, ext_solution) = io::read_spp_input(Path::new(&a[1]))?;
    let importer = Importer::new(config.cde_config, config.poly_simpl_tolerance, config.narrow_concavity_cutoff_ratio);
    let instance = jagua_rs::probs::spp::io::import_instance(&importer, &ext_instance)?;
    let initial = ext_solution.map(|e| jagua_rs::probs::spp::io::import_solution(&instance, &e)).transpose()?;

    oracle::start(&a[2])?;
    oracle::line(&format!(
        "{{\"k\":\"cfg\",\"seed\":{seed},\"expl\":{expl},\"cmpr\":{cmpr},\"workers\":{workers},\"warm\":{}}}",
        initial.is_some()
    ));
    let mut term = oracle::CountTerminator::default();
    let sol = optimize(
        instance,
        Xoshiro256PlusPlus::seed_from_u64(seed),
        &mut DummySolListener,
        &mut term,
        &config.expl_cfg,
        &config.cmpr_cfg,
        initial.as_ref(),
    )?;
    for pi in sol.layout_snapshot().placed_items().values() {
        oracle::line(&format!("{{\"k\":\"place\",\"i\":{},\"dt\":\"{}\"}}", pi.item().idx(), oracle::dt(pi.d_transf())));
    }
    oracle::line(&format!(
        "{{\"k\":\"result\",\"w\":\"{}\",\"evals\":{}}}",
        oracle::hx(sol.strip_width()),
        oracle::evals()
    ));
    oracle::finish();
    let ext = jagua_rs::probs::spp::io::export(&sol, *sparrow::EPOCH);
    io::write_json(&ext, Path::new(&a[3]), log::Level::Info)?;
    Ok(())
}
