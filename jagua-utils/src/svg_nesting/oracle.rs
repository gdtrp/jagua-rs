//! Oracle mode (RFC-030 N0): a reproducible, count-based run of the production nesting with a
//! decision trace, so a port can be compared against this engine decision for decision.
//!
//! Nothing here runs unless a strategy is built with [`AdaptiveNestingStrategy::with_oracle`]:
//! the production path is unchanged. In oracle mode
//! - every wave runs `wave_size` seeds (production's pods have one CPU, so production waves are
//!   already one run wide) and the process-wide routine reservation is bypassed, so a concurrent
//!   request can no longer change the wave size and with it every later seed;
//! - the wall-clock checks (deadline, predictive guard, 120 s per run) are skipped and counted in
//!   [`OracleReport::guards_fired`], which must stay 0 for a run to count as an oracle;
//! - the router's choice and each LBF run (seed, `ls_frac`, samples, parts placed) are recorded;
//! - with a trace sink, LBF runs through a traced copy of `lbf`'s solver ([`traced_solve`]) that
//!   draws the random stream in the same order and hashes every sample decision.
//!
//! [`AdaptiveNestingStrategy::with_oracle`]: crate::AdaptiveNestingStrategy::with_oracle

use std::cmp::Ordering;
use std::io::Write;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use jagua_rs::collision_detection::CDEngine;
use jagua_rs::collision_detection::hazards::filter::{HazKeyFilter, HazardFilter, NoFilter};
use jagua_rs::entities::{Instance, Item};
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::probs::bpp::entities::{BPLayoutType, BPPlacement, BPProblem, BPSolution};
use lbf::config::LBFConfig;
use lbf::opt::lbf_bpp::LBFOptimizerBP;
use lbf::opt::loss::LBFLoss;
use lbf::opt::search::item_placement_order;
use lbf::samplers::ls_sampler::LSSampler;
use lbf::samplers::uniform_rect_sampler::UniformRectSampler;
use rand::Rng;

/// How an oracle run is set up.
#[derive(Clone, Debug)]
pub struct OracleConfig {
    /// Seeds per wave. 1 reproduces production (one CPU per pod).
    pub wave_size: usize,
    /// Sample events (0-based, counted over the whole request) written out in full, for
    /// inspecting a window located by the checkpoints. Empty: checkpoints and events only.
    pub sample_window: Range<u64>,
    /// Sample events per checkpoint line.
    pub checkpoint_every: u64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            wave_size: 1,
            sample_window: 0..0,
            checkpoint_every: 1000,
        }
    }
}

/// The packer that answered a request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RouteInfo {
    /// The classifier's bucket (`SingleHighFill`, …, `General`), or the forced mode / fill path.
    pub class: String,
    /// `grid`, `pairing`, `mixed`, `lattice`, `fill`, `lbf` (and the `max_fit_*` variants).
    pub packer: String,
    /// The class's own packer failed and the request fell back to LBF.
    pub fell_back: bool,
}

/// One LBF run (one seed of one wave).
#[derive(Clone, Debug, PartialEq)]
pub struct RunRecord {
    /// The wave's run number (`seed_offset`).
    pub run_no: usize,
    pub loop_idx: usize,
    pub seed: u64,
    pub ls_frac: f32,
    /// `n_samples` per item search.
    pub placements: usize,
    /// The optimizer's own `sample_counter` after the run.
    pub samples: usize,
    pub placed: usize,
}

/// What an oracle run observed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OracleReport {
    pub route: Option<RouteInfo>,
    pub runs: Vec<RunRecord>,
    /// Wall-clock checks that would have fired in production. Must be 0.
    pub guards_fired: usize,
    /// Sample events hashed into the trace.
    pub sample_events: u64,
    /// FNV-1a 64 over every sample event, in order.
    pub trace_hash: u64,
}

/// Shared oracle state: the test keeps one clone, the strategy the other.
#[derive(Clone)]
pub struct Oracle {
    pub(crate) config: OracleConfig,
    state: Arc<Mutex<OracleState>>,
}

struct OracleState {
    report: OracleReport,
    sink: Option<Box<dyn Write + Send>>,
    hash: Fnv64,
}

impl Oracle {
    /// An oracle without a trace: route, runs and guards only; LBF runs the library solver.
    pub fn new(config: OracleConfig) -> Self {
        Self::build(config, None)
    }

    /// An oracle that traces every LBF sample into `sink` (JSON lines).
    pub fn with_trace(config: OracleConfig, sink: Box<dyn Write + Send>) -> Self {
        Self::build(config, Some(sink))
    }

    fn build(config: OracleConfig, sink: Option<Box<dyn Write + Send>>) -> Self {
        assert!(config.wave_size > 0, "an oracle wave runs at least one seed");
        assert!(config.checkpoint_every > 0, "checkpoints need a period");
        Self {
            config,
            state: Arc::new(Mutex::new(OracleState {
                report: OracleReport::default(),
                sink,
                hash: Fnv64::new(),
            })),
        }
    }

    /// What was observed so far; writes the closing line of the trace once.
    pub fn finish(&self) -> std::io::Result<OracleReport> {
        let mut st = self.lock();
        st.report.trace_hash = st.hash.0;
        let line = format!(
            "{{\"k\":\"end\",\"n\":{},\"h\":\"{:016x}\"}}",
            st.report.sample_events, st.hash.0
        );
        if let Some(sink) = st.sink.as_mut() {
            writeln!(sink, "{line}")?;
            sink.flush()?;
        }
        Ok(st.report.clone())
    }

    pub(crate) fn traced(&self) -> bool {
        self.lock().sink.is_some()
    }

    /// The router's bucket for this request (the classifier's class, the forced mode or the fill
    /// path). Noted once, before any packer runs.
    pub(crate) fn note_class(&self, class: &str) {
        let mut st = self.lock();
        let route = st.report.route.get_or_insert_with(RouteInfo::default);
        route.class = class.to_string();
    }

    /// The packer about to answer. A second, different packer means the first one failed and the
    /// request fell back (in practice: to LBF).
    pub(crate) fn note_packer(&self, packer: &str) {
        let mut st = self.lock();
        let route = st.report.route.get_or_insert_with(RouteInfo::default);
        if !route.packer.is_empty() && route.packer != packer {
            route.fell_back = true;
        }
        route.packer = packer.to_string();
    }

    pub(crate) fn note_guard(&self) {
        self.lock().report.guards_fired += 1;
    }

    pub(crate) fn note_run(&self, run: RunRecord) {
        let mut st = self.lock();
        let line = format!(
            "{{\"k\":\"run\",\"run\":{},\"loop\":{},\"seed\":{},\"ls_frac\":{},\"placements\":{},\"samples\":{},\"placed\":{}}}",
            run.run_no,
            run.loop_idx,
            run.seed,
            run.ls_frac,
            run.placements,
            run.samples,
            run.placed
        );
        st.write_line(&line);
        st.report.runs.push(run);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, OracleState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Hash one sample decision; write it in full inside the window; checkpoint on the period.
    fn sample(&self, ev: &SampleEvent) {
        let mut st = self.lock();
        let n = st.report.sample_events;
        ev.hash_into(&mut st.hash);
        if self.config.sample_window.contains(&n) {
            let line = ev.json(n);
            st.write_line(&line);
        }
        st.report.sample_events = n + 1;
        if (n + 1) % self.config.checkpoint_every == 0 {
            let line = format!("{{\"k\":\"chk\",\"n\":{},\"h\":\"{:016x}\"}}", n + 1, st.hash.0);
            st.write_line(&line);
        }
    }

    fn event(&self, line: String) {
        self.lock().write_line(&line);
    }
}

impl OracleState {
    fn write_line(&mut self, line: &str) {
        if let Some(sink) = self.sink.as_mut()
            && let Err(e) = writeln!(sink, "{line}")
        {
            // The trace is evidence: a lost line must fail the comparison, not pass silently.
            panic!("oracle trace write failed: {e}");
        }
    }
}

/// FNV-1a, 64 bit. Chosen because a port reproduces it in a few lines with no library.
#[derive(Clone, Copy, Debug)]
pub struct Fnv64(pub u64);

impl Fnv64 {
    pub const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    pub const PRIME: u64 = 0x0000_0100_0000_01b3;

    pub fn new() -> Self {
        Self(Self::OFFSET)
    }

    pub fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= u64::from(*b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }
}

impl Default for Fnv64 {
    fn default() -> Self {
        Self::new()
    }
}

/// Which phase of `lbf::opt::search::search` drew the sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Uniform,
    Local,
}

/// One sample decision. Encoded little-endian for the hash, field by field, in this order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleEvent {
    pub run_no: u32,
    pub loop_idx: u32,
    pub item: u32,
    /// Position of the layout in `search_layouts`' scan order (open layouts, then bins with stock).
    pub layout: u32,
    pub phase: Phase,
    /// The sample's index within its phase.
    pub i: u32,
    pub rotation: f32,
    pub tx: f32,
    pub ty: f32,
    /// The surrogate collided (the sample was rejected before the polygon test).
    pub surrogate_hit: bool,
    /// The sample was cheap enough to test against the polygon.
    pub worth: bool,
    /// The polygon collided.
    pub poly_hit: bool,
    /// The sample became the new best.
    pub accepted: bool,
}

impl SampleEvent {
    fn hash_into(&self, h: &mut Fnv64) {
        h.write(&self.run_no.to_le_bytes());
        h.write(&self.loop_idx.to_le_bytes());
        h.write(&self.item.to_le_bytes());
        h.write(&self.layout.to_le_bytes());
        h.write(&[match self.phase {
            Phase::Uniform => 0,
            Phase::Local => 1,
        }]);
        h.write(&self.i.to_le_bytes());
        h.write(&self.rotation.to_bits().to_le_bytes());
        h.write(&self.tx.to_bits().to_le_bytes());
        h.write(&self.ty.to_bits().to_le_bytes());
        h.write(&[
            u8::from(self.surrogate_hit),
            u8::from(self.worth),
            u8::from(self.poly_hit),
            u8::from(self.accepted),
        ]);
    }

    fn json(&self, n: u64) -> String {
        format!(
            "{{\"k\":\"s\",\"n\":{},\"run\":{},\"loop\":{},\"item\":{},\"layout\":{},\"ph\":\"{}\",\"i\":{},\"r\":\"{:08x}\",\"x\":\"{:08x}\",\"y\":\"{:08x}\",\"sur\":{},\"worth\":{},\"poly\":{},\"acc\":{}}}",
            n,
            self.run_no,
            self.loop_idx,
            self.item,
            self.layout,
            match self.phase {
                Phase::Uniform => "U",
                Phase::Local => "L",
            },
            self.i,
            self.rotation.to_bits(),
            self.tx.to_bits(),
            self.ty.to_bits(),
            u8::from(self.surrogate_hit),
            u8::from(self.worth),
            u8::from(self.poly_hit),
            u8::from(self.accepted)
        )
    }
}

/// Identity of one LBF run inside a request.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RunCtx {
    pub run_no: usize,
    pub loop_idx: usize,
}

/// `LBFOptimizerBP::solve_with_cancellation`, copied from `lbf` at the production commit with
/// every sample decision traced. Same item order, same layout scan, same random draws in the same
/// order: the solution equals the library's (a test in `tests/oracle_prod.rs` proves it on the
/// corpus). Only the logging is left out.
pub(crate) fn traced_solve<C: Fn() -> bool>(
    opt: &mut LBFOptimizerBP,
    cancellation_checker: Option<&C>,
    oracle: &Oracle,
    ctx: RunCtx,
) -> BPSolution {
    'outer: for item_id in item_placement_order(&opt.instance) {
        if let Some(checker) = cancellation_checker
            && checker()
        {
            break 'outer;
        }
        let item = opt.instance.item(item_id);
        'inner: while opt.problem.item_demand_qtys[item_id] > 0 {
            if opt.problem.item_placed_qtys().sum::<usize>() % 100 == 0
                && let Some(checker) = cancellation_checker
                && checker()
            {
                break 'outer;
            }
            let placement = traced_search_layouts(
                &opt.problem,
                item,
                &opt.config,
                &mut opt.rng,
                &mut opt.sample_counter,
                oracle,
                ctx,
            );
            match placement {
                Some((i_opt, layout_pos)) => {
                    let (tx, ty) = i_opt.d_transf.translation();
                    oracle.event(format!(
                        "{{\"k\":\"place\",\"run\":{},\"loop\":{},\"item\":{},\"layout\":{},\"r\":\"{:08x}\",\"x\":\"{:08x}\",\"y\":\"{:08x}\"}}",
                        ctx.run_no,
                        ctx.loop_idx,
                        i_opt.item_id,
                        layout_pos,
                        i_opt.d_transf.rotation().to_bits(),
                        tx.to_bits(),
                        ty.to_bits()
                    ));
                    opt.problem.place_item(i_opt);
                }
                None => break 'inner,
            }
        }
    }
    opt.problem.save()
}

fn traced_search_layouts(
    problem: &BPProblem,
    item: &Item,
    config: &LBFConfig,
    rng: &mut impl Rng,
    sample_counter: &mut usize,
    oracle: &Oracle,
    ctx: RunCtx,
) -> Option<(BPPlacement, u32)> {
    let open_layouts = problem.layouts.keys().map(BPLayoutType::Open);
    let bins_with_stock = problem
        .bin_stock_qtys
        .iter()
        .enumerate()
        .filter_map(|(bin_id, qty)| match *qty > 0 {
            true => Some(BPLayoutType::Closed { bin_id }),
            false => None,
        });

    for (pos, layout_id) in open_layouts.chain(bins_with_stock).enumerate() {
        let cde = match layout_id {
            BPLayoutType::Open(lkey) => problem.layouts[lkey].cde(),
            BPLayoutType::Closed { bin_id } => problem.instance.container(bin_id).base_cde.as_ref(),
        };
        let at = Where {
            ctx,
            item: item.id as u32,
            layout: pos as u32,
        };
        let placement = match &item.min_quality {
            None => traced_search(cde, item, config, rng, sample_counter, &NoFilter, oracle, at),
            Some(min_quality) => {
                let filter = HazKeyFilter::from_irrelevant_qzones(*min_quality, &cde.hazards_map);
                traced_search(cde, item, config, rng, sample_counter, &filter, oracle, at)
            }
        };
        if let Some((d_transf, _)) = placement {
            return Some((
                BPPlacement {
                    layout_id,
                    item_id: item.id,
                    d_transf,
                },
                pos as u32,
            ));
        }
    }
    None
}

#[derive(Clone, Copy)]
struct Where {
    ctx: RunCtx,
    item: u32,
    layout: u32,
}

/// `lbf::opt::search::search` with every sample traced.
fn traced_search(
    cde: &CDEngine,
    item: &Item,
    config: &LBFConfig,
    rng: &mut impl Rng,
    sample_counter: &mut usize,
    filter: &impl HazardFilter,
    oracle: &Oracle,
    at: Where,
) -> Option<(DTransformation, LBFLoss)> {
    let surrogate = item.shape_cd.surrogate();
    let mut buffer = {
        let mut buffer = (*item.shape_cd).clone();
        buffer.surrogate = None;
        buffer
    };
    let mut best: Option<(DTransformation, LBFLoss)> = None;

    let ls_sample_budget = (config.n_samples as f32 * config.ls_frac) as usize;
    let uni_sample_budget = config.n_samples - ls_sample_budget;

    let mut bin_sampler = UniformRectSampler::new(cde.bbox(), item);
    for i in 0..uni_sample_budget {
        let d_transf = bin_sampler.sample(rng);
        let transf = d_transf.compose();
        let mut ev = event(at, Phase::Uniform, i, d_transf);
        if cde.detect_surrogate_collision(surrogate, &transf, filter) {
            ev.surrogate_hit = true;
        } else {
            buffer.transform_from(&item.shape_cd, &transf);
            let cost = LBFLoss::from_shape(&buffer);
            let worth_testing = match (best.as_ref(), &cost) {
                (Some((_, best_cost)), cost) => {
                    cost.partial_cmp(best_cost).unwrap() == Ordering::Less
                }
                (None, _) => true,
            };
            ev.worth = worth_testing;
            if worth_testing {
                ev.poly_hit = cde.detect_poly_collision(&buffer, filter);
                if !ev.poly_hit {
                    ev.accepted = true;
                    best = Some((d_transf, cost));
                    let tightened = cost.tighten_sample_bbox(bin_sampler.bbox);
                    bin_sampler = UniformRectSampler::new(tightened, item);
                }
            }
        }
        oracle.sample(&ev);
    }
    *sample_counter += uni_sample_budget;

    let (best_sample, best_cost) = best.as_mut()?;
    let mut ls_sampler = LSSampler::from_defaults(item, *best_sample, cde.bbox());
    for i in 0..ls_sample_budget {
        let d_transf = ls_sampler.sample(rng);
        let transf = d_transf.compose();
        let mut ev = event(at, Phase::Local, i, d_transf);
        if cde.detect_surrogate_collision(surrogate, &transf, filter) {
            ev.surrogate_hit = true;
        } else {
            buffer.transform_from(&item.shape_cd, &transf);
            let cost = LBFLoss::from_shape(&buffer);
            let worth_testing = cost < *best_cost;
            ev.worth = worth_testing;
            if worth_testing {
                ev.poly_hit = cde.detect_poly_collision(&buffer, filter);
                if !ev.poly_hit {
                    ev.accepted = true;
                    ls_sampler.shift_mean(d_transf);
                    (*best_sample, *best_cost) = (d_transf, cost);
                }
            }
        }
        oracle.sample(&ev);
        let progress_pct = i as f32 / ls_sample_budget as f32;
        ls_sampler.decay_stddev(progress_pct);
    }
    // `LSSampler::n_samples` is crate-private in `lbf`; it counts exactly one per `sample` call.
    *sample_counter += ls_sample_budget;

    best
}

fn event(at: Where, phase: Phase, i: usize, d: DTransformation) -> SampleEvent {
    let (tx, ty) = d.translation();
    SampleEvent {
        run_no: at.ctx.run_no as u32,
        loop_idx: at.ctx.loop_idx as u32,
        item: at.item,
        layout: at.layout,
        phase,
        i: i as u32,
        rotation: d.rotation(),
        tx,
        ty,
        surrogate_hit: false,
        worth: false,
        poly_hit: false,
        accepted: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Default)]
    struct Buf(Arc<Mutex<Vec<u8>>>);

    impl Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl Buf {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn ev(i: u32) -> SampleEvent {
        SampleEvent {
            run_no: 1,
            loop_idx: 0,
            item: 2,
            layout: 0,
            phase: Phase::Uniform,
            i,
            rotation: 1.5,
            tx: 10.25,
            ty: -3.0,
            surrogate_hit: false,
            worth: true,
            poly_hit: false,
            accepted: i == 0,
        }
    }

    #[test]
    fn fnv64_matches_the_reference_vectors() {
        // Published FNV-1a 64 test vectors (Fowler/Noll/Vo).
        let mut h = Fnv64::new();
        assert_eq!(h.0, 0xcbf29ce484222325);
        h.write(b"a");
        assert_eq!(h.0, 0xaf63dc4c8601ec8c);
        let mut h = Fnv64::new();
        h.write(b"foobar");
        assert_eq!(h.0, 0x85944171f73967e8);
        assert_eq!(Fnv64::default().0, Fnv64::OFFSET);
    }

    #[test]
    fn sample_events_hash_field_by_field_and_checkpoint_on_the_period() {
        let buf = Buf::default();
        let o = Oracle::with_trace(
            OracleConfig {
                wave_size: 1,
                sample_window: 1..2,
                checkpoint_every: 2,
            },
            Box::new(buf.clone()),
        );
        assert!(o.traced());
        for i in 0..3 {
            o.sample(&ev(i));
        }
        let report = o.finish().unwrap();
        assert_eq!(report.sample_events, 3);

        let mut want = Fnv64::new();
        for i in 0..3 {
            ev(i).hash_into(&mut want);
        }
        assert_eq!(report.trace_hash, want.0);

        let text = buf.text();
        let lines: Vec<&str> = text.lines().collect();
        // event 1 in full (the window), a checkpoint after 2 events, the end line.
        assert_eq!(lines.len(), 3, "{text}");
        assert!(lines[0].starts_with("{\"k\":\"s\",\"n\":1,"), "{}", lines[0]);
        assert!(lines[0].contains("\"r\":\"3fc00000\""), "{}", lines[0]);
        assert!(lines[0].contains("\"y\":\"c0400000\""), "{}", lines[0]);
        assert!(lines[1].starts_with("{\"k\":\"chk\",\"n\":2,"), "{}", lines[1]);
        assert_eq!(lines[2], format!("{{\"k\":\"end\",\"n\":3,\"h\":\"{:016x}\"}}", want.0));
    }

    #[test]
    fn a_changed_decision_changes_the_hash() {
        let mut a = Fnv64::new();
        ev(0).hash_into(&mut a);
        let mut b = Fnv64::new();
        let mut e = ev(0);
        e.accepted = false;
        e.hash_into(&mut b);
        assert_ne!(a.0, b.0);
        let mut c = Fnv64::new();
        let mut e = ev(0);
        e.phase = Phase::Local;
        e.hash_into(&mut c);
        assert_ne!(a.0, c.0);
        assert!(ev(0).json(0).contains("\"ph\":\"U\""));
        let mut l = ev(0);
        l.phase = Phase::Local;
        assert!(l.json(0).contains("\"ph\":\"L\""));
    }

    #[test]
    fn routes_record_the_fallback() {
        let o = Oracle::new(OracleConfig::default());
        assert!(!o.traced());
        o.note_class("SingleIrregular");
        o.note_packer("lattice");
        assert_eq!(
            o.finish().unwrap().route,
            Some(RouteInfo {
                class: "SingleIrregular".into(),
                packer: "lattice".into(),
                fell_back: false
            })
        );
        o.note_packer("lbf");
        o.note_packer("lbf");
        let r = o.finish().unwrap().route.unwrap();
        assert_eq!(
            (r.class.as_str(), r.packer.as_str(), r.fell_back),
            ("SingleIrregular", "lbf", true)
        );

        // Straight to LBF: no fall-back.
        let o = Oracle::new(OracleConfig::default());
        o.note_class("General");
        o.note_packer("lbf");
        assert!(!o.finish().unwrap().route.unwrap().fell_back);
    }

    #[test]
    fn runs_and_guards_are_counted() {
        let buf = Buf::default();
        let o = Oracle::with_trace(OracleConfig::default(), Box::new(buf.clone()));
        o.note_guard();
        o.note_run(RunRecord {
            run_no: 3,
            loop_idx: 1,
            seed: 3001,
            ls_frac: 0.5,
            placements: 50_000,
            samples: 123,
            placed: 7,
        });
        let report = o.finish().unwrap();
        assert_eq!(report.guards_fired, 1);
        assert_eq!(report.runs.len(), 1);
        assert!(buf.text().starts_with("{\"k\":\"run\",\"run\":3,\"loop\":1,\"seed\":3001,"));
    }

    #[test]
    #[should_panic(expected = "at least one seed")]
    fn a_zero_wave_is_refused() {
        Oracle::new(OracleConfig {
            wave_size: 0,
            ..OracleConfig::default()
        });
    }

    #[test]
    #[should_panic(expected = "oracle trace write failed")]
    fn a_lost_trace_line_fails_loudly() {
        struct Broken;
        impl Write for Broken {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::other("disk full"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let o = Oracle::with_trace(OracleConfig::default(), Box::new(Broken));
        o.event("{}".into());
    }
}
