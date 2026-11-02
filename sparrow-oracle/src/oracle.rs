//! The decision trace of the Sparrow oracle (RFC-030 N10). Instrumentation only: nothing here
//! changes a decision. The Go port (dxfcalc/internal/nestengine/sparrow) writes the same lines.
//!
//! - Every sample evaluation folds into its worker's FNV-1a 64 hash (`Traced`): the rotation,
//!   translation, the verdict's kind and loss, as 17 little-endian bytes.
//! - After each separator iteration, the iteration folds into the global hash: each worker's hash
//!   and weighted loss in worker order, the winning worker, the total loss (`iteration`). A "chk"
//!   line carries the global hash every `CHECKPOINT_EVERY` iterations; with ORACLE_FULL=1 every
//!   iteration is also written in full.
//! - The phase decisions are written as lines as they happen (`line`): a separation's start and
//!   end, a strip-width change, exploration's feasible and infeasible widths, a disruption's swap,
//!   a compression attempt, an LBF construction placement.
//! - The evaluation counter (`add_evals`, `evals`) drives `CountTerminator`: the run stops after a
//!   number of evaluations, never on the clock.

use crate::eval::sample_eval::{SampleEval, SampleEvaluator};
use crate::util::terminator::Terminator;
use jagua_rs::geometry::DTransformation;
use jagua_rs::Instant;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Duration;

pub const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const CHECKPOINT_EVERY: u64 = 100;

/// Folds bytes into an FNV-1a 64 hash.
pub fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(FNV_PRIME);
    }
}

/// The 17 bytes of one evaluation: rotation, x, y, the verdict's kind (0 clear, 1 collision,
/// 2 invalid) and its loss (0 when invalid).
pub fn eval_bytes(dt: DTransformation, e: SampleEval) -> [u8; 17] {
    let (kind, loss) = match e {
        SampleEval::Clear { loss } => (0u8, loss),
        SampleEval::Collision { loss } => (1u8, loss),
        SampleEval::Invalid => (2u8, 0.0),
    };
    let (x, y) = dt.translation();
    let mut b = [0u8; 17];
    b[0..4].copy_from_slice(&dt.rotation().to_bits().to_le_bytes());
    b[4..8].copy_from_slice(&x.to_bits().to_le_bytes());
    b[8..12].copy_from_slice(&y.to_bits().to_le_bytes());
    b[12] = kind;
    b[13..17].copy_from_slice(&loss.to_bits().to_le_bytes());
    b
}

/// A sample evaluator whose every evaluation folds into a hash.
pub struct Traced<'a, E: SampleEvaluator> {
    inner: E,
    h: &'a mut u64,
}

impl<'a, E: SampleEvaluator> Traced<'a, E> {
    pub fn new(inner: E, h: &'a mut u64) -> Self {
        Self { inner, h }
    }
}

impl<E: SampleEvaluator> SampleEvaluator for Traced<'_, E> {
    fn evaluate_sample(&mut self, dt: DTransformation, upper_bound: Option<SampleEval>) -> SampleEval {
        let e = self.inner.evaluate_sample(dt, upper_bound);
        fnv(self.h, &eval_bytes(dt, e));
        e
    }

    fn n_evals(&self) -> usize {
        self.inner.n_evals()
    }
}

struct Writer {
    out: BufWriter<File>,
    h: u64,
    n: u64,
    full: bool,
}

static TRACE: Mutex<Option<Writer>> = Mutex::new(None);
static EVALS: AtomicUsize = AtomicUsize::new(0);

/// Starts the trace at path.
pub fn start(path: &str) -> std::io::Result<()> {
    let out = BufWriter::new(File::create(path)?);
    let full = std::env::var("ORACLE_FULL").is_ok_and(|v| v == "1");
    *TRACE.lock().unwrap() = Some(Writer { out, h: FNV_OFFSET, n: 0, full });
    Ok(())
}

/// Writes one line (a JSON object without the newline), when a trace is open.
pub fn line(s: &str) {
    if let Some(w) = TRACE.lock().unwrap().as_mut() {
        writeln!(w.out, "{s}").expect("trace write");
    }
}

/// Folds one separator iteration into the global hash; `full` is its line under ORACLE_FULL.
pub fn iteration(bytes: &[u8], full: impl FnOnce(u64) -> String) {
    if let Some(w) = TRACE.lock().unwrap().as_mut() {
        fnv(&mut w.h, bytes);
        w.n += 1;
        if w.full {
            writeln!(w.out, "{}", full(w.n)).expect("trace write");
        }
        if w.n % CHECKPOINT_EVERY == 0 {
            writeln!(w.out, "{{\"k\":\"chk\",\"n\":{},\"h\":\"{:016x}\"}}", w.n, w.h).expect("trace write");
        }
    }
}

/// Ends the trace: the final "end" line with the iteration count and the hash.
pub fn finish() {
    if let Some(mut w) = TRACE.lock().unwrap().take() {
        writeln!(w.out, "{{\"k\":\"end\",\"n\":{},\"h\":\"{:016x}\"}}", w.n, w.h).expect("trace write");
        w.out.flush().expect("trace flush");
    }
}

/// Adds a separator iteration's evaluations to the counter.
pub fn add_evals(n: usize) {
    EVALS.fetch_add(n, Ordering::SeqCst);
}

/// The evaluations counted so far.
pub fn evals() -> usize {
    EVALS.load(Ordering::SeqCst)
}

/// f32 bits as the trace writes them.
pub fn hx(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

/// A transformation as the trace writes it: "r x y" in bits.
pub fn dt(d: DTransformation) -> String {
    let (x, y) = d.translation();
    format!("{} {} {}", hx(d.rotation()), hx(x), hx(y))
}

/// A terminator that stops a phase after a number of evaluations. `new_timeout(d)` reads
/// `d.as_secs()` as that number: the oracle sets each phase's "time limit" to its evaluation
/// budget, so Sparrow's phases stay unchanged.
#[derive(Debug, Clone, Default)]
pub struct CountTerminator {
    start: usize,
    budget: Option<usize>,
}

impl Terminator for CountTerminator {
    fn kill(&self) -> bool {
        self.budget.is_some_and(|b| evals() - self.start >= b)
    }

    fn new_timeout(&mut self, timeout: Duration) {
        self.start = evals();
        self.budget = Some(timeout.as_secs() as usize);
    }

    fn timeout_at(&self) -> Option<Instant> {
        None
    }
}
