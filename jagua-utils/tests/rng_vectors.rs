//! RFC-030 N3 (dxf-processor): writes the random streams the LBF search draws, from the real rand
//! 0.10 / rand_distr 0.6 / lbf code, so the Go port (dxfcalc/internal/nestengine/lbf/rng and
//! lbf/sample) is checked bit for bit. Each stream records its first values verbatim and an FNV-1a
//! 64 hash over all of them (little-endian bytes per value). jagua-rs itself is not touched.
//!
//!   cargo test -p jagua-utils --features oracle --test rng_vectors -- --nocapture
//!   (writes $ORACLE_OUT/rng_vectors.json)
#![cfg(feature = "oracle")]

use jagua_rs::entities::Item;
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::geometry::shape_modification::{ShapeModifyConfig, ShapeModifyMode};
use jagua_rs::geometry::{DTransformation, OriginalShape};
use lbf::samplers::ls_sampler::LSSampler;
use lbf::samplers::rotation_distr::UniformRotDistr;
use lbf::samplers::uniform_rect_sampler::UniformRectSampler;
use rand::prelude::SmallRng;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Open01, StandardNormal, StandardUniform, Uniform};
use serde_json::{Value, json};

const HEAD: usize = 64;

struct Stream {
    head: Vec<String>,
    hash: u64,
    count: usize,
}

impl Stream {
    fn new() -> Self {
        Stream { head: vec![], hash: 0xcbf29ce484222325, count: 0 }
    }
    fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.hash ^= x as u64;
            self.hash = self.hash.wrapping_mul(0x100000001b3);
        }
    }
    fn push(&mut self, text: String, b: &[u8]) {
        if self.head.len() < HEAD {
            self.head.push(text);
        }
        self.bytes(b);
        self.count += 1;
    }
    fn u64(&mut self, v: u64) {
        self.push(format!("{v:016x}"), &v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.push(format!("{v:08x}"), &v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }
    fn dt(&mut self, d: DTransformation) {
        let (r, (x, y)) = (d.rotation(), d.translation());
        let mut b = vec![];
        for v in [r, x, y] {
            b.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        self.push(format!("{:08x} {:08x} {:08x}", r.to_bits(), x.to_bits(), y.to_bits()), &b);
    }
    fn json(self, mut meta: Value) -> Value {
        meta["count"] = json!(self.count);
        meta["head"] = json!(self.head);
        meta["fnv"] = json!(format!("{:016x}", self.hash));
        meta
    }
}

fn bits(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

const SEEDS: [u64; 5] = [0, 1000, 1001, 1 << 63, u64::MAX];

fn raw_streams(out: &mut Vec<Value>) {
    for seed in SEEDS {
        let mut rng = SmallRng::seed_from_u64(seed);
        let mut s = Stream::new();
        for _ in 0..1_000_000 {
            s.u64(rng.next_u64());
        }
        out.push(s.json(json!({"kind": "u64", "seed": seed})));
    }
    let mut rng = SmallRng::seed_from_u64(7);
    let mut s = Stream::new();
    for _ in 0..100_000 {
        s.u32(rng.next_u32());
    }
    out.push(s.json(json!({"kind": "u32", "seed": 7u64})));
    let mut rng = SmallRng::seed_from_u64(11);
    let (mut a, mut b) = (Stream::new(), Stream::new());
    for _ in 0..100_000 {
        let v: f64 = rng.sample(StandardUniform);
        a.f64(v);
        let w: f64 = rng.sample(Open01);
        b.f64(w);
    }
    out.push(a.json(json!({"kind": "f64_standard_interleaved_open01", "seed": 11u64, "which": "standard"})));
    out.push(b.json(json!({"kind": "f64_standard_interleaved_open01", "seed": 11u64, "which": "open01"})));
}

fn uniform_streams(out: &mut Vec<Value>) {
    let ranges: [(f32, f32); 6] = [
        (0.0, 1200.0),
        (-3.5, 2500.25),
        (0.0, 2.0 * std::f32::consts::PI),
        (10.0, 10.000002),
        (-1e-3, 1e-3),
        (612.4999, 612.5001),
    ];
    for (i, (lo, hi)) in ranges.into_iter().enumerate() {
        let u = Uniform::new(lo, hi).unwrap();
        let mut rng = SmallRng::seed_from_u64(100 + i as u64);
        let mut s = Stream::new();
        for _ in 0..100_000 {
            s.f32(u.sample(&mut rng));
        }
        out.push(s.json(json!({"kind": "uniform_f32", "seed": 100 + i as u64, "low": bits(lo), "high": bits(hi)})));
    }
}

fn choose_streams(out: &mut Vec<Value>) {
    let lens: [usize; 11] = [1, 2, 3, 4, 5, 6, 7, 8, 1000, 3_000_000_000, 4_294_967_295];
    for (i, n) in lens.into_iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(200 + i as u64);
        let mut s = Stream::new();
        for _ in 0..100_000 {
            // IndexedRandom::choose draws exactly random_range(..len).
            let k: usize = rng.random_range(..n);
            s.u32(k as u32);
        }
        out.push(s.json(json!({"kind": "choose", "seed": 200 + i as u64, "len": n})));
    }
}

fn normal_streams(out: &mut Vec<Value>) {
    let mut rng = SmallRng::seed_from_u64(300);
    let mut s = Stream::new();
    for _ in 0..200_000 {
        let z: f64 = rng.sample(StandardNormal);
        s.f64(z);
    }
    out.push(s.json(json!({"kind": "standard_normal_f64", "seed": 300u64})));
    let params: [(f32, f32); 4] = [(0.0, 1.0), (5.0, 12.0), (0.0, 2.0 * std::f32::consts::PI / 180.0), (612.5, 0.3)];
    for (i, (m, sd)) in params.into_iter().enumerate() {
        let n = Normal::new(m, sd).unwrap();
        let mut rng = SmallRng::seed_from_u64(310 + i as u64);
        let mut s = Stream::new();
        for _ in 0..100_000 {
            s.f32(n.sample(&mut rng));
        }
        out.push(s.json(json!({"kind": "normal_f32", "seed": 310 + i as u64, "mean": bits(m), "sd": bits(sd)})));
    }
}

fn item(rot: RotationRange) -> Item {
    // The rotation mode is all the samplers read from the item; the shape is the 1200 × 800 sheet.
    let shape = SPolygon::from(Rect::try_new(0.0, 0.0, 1200.0, 800.0).unwrap());
    let os = OriginalShape {
        shape,
        pre_transform: DTransformation::empty(),
        modify_mode: ShapeModifyMode::Inflate,
        modify_config: ShapeModifyConfig { simplify_tolerance: None, offset: None, narrow_concavity_cutoff: None },
    };
    let cfg = SPSurrogateConfig { n_pole_limits: [(100, 0.0), (20, 0.75), (10, 0.90)], n_ff_poles: 2, n_ff_piers: 0 };
    Item::new(0, os, rot, None, cfg).unwrap()
}

fn rotations() -> Vec<(&'static str, RotationRange)> {
    use std::f32::consts::PI;
    vec![
        ("none", RotationRange::None),
        ("continuous", RotationRange::Continuous),
        ("discrete4", RotationRange::Discrete(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0])),
        ("discrete2", RotationRange::Discrete(vec![0.0, PI])),
    ]
}

fn sampler_streams(out: &mut Vec<Value>) {
    let bbox = Rect::try_new(3.25, -1.5, 1196.75, 798.5).unwrap();
    for (j, (name, rot)) in rotations().into_iter().enumerate() {
        let it = item(rot);
        // UniformRectSampler, rebuilt on a tightened box half way, as search() does.
        let mut rng = SmallRng::seed_from_u64(400 + j as u64);
        let mut s = Stream::new();
        let mut us = UniformRectSampler::new(bbox, &it);
        for i in 0..20_000 {
            if i == 10_000 {
                us = UniformRectSampler::new(Rect::try_new(3.25, -1.5, 611.0625, 798.5).unwrap(), &it);
            }
            s.dt(us.sample(&mut rng));
        }
        out.push(s.json(json!({"kind": "uniform_rect", "seed": 400 + j as u64, "rot": name})));
        // LSSampler as search() drives it: from_defaults, a mean shift every 97 samples, the
        // decay after every sample with pct = i / ls.
        let ls = 15_000usize;
        let start = DTransformation::new(if name == "continuous" { 0.7 } else { 0.0 }, (612.5, 400.25));
        let mut ls_s = LSSampler::from_defaults(&it, start, bbox);
        let mut rng = SmallRng::seed_from_u64(500 + j as u64);
        let mut s = Stream::new();
        for i in 0..ls {
            let d = ls_s.sample(&mut rng);
            s.dt(d);
            if i % 97 == 0 {
                ls_s.shift_mean(d);
            }
            ls_s.decay_stddev(i as f32 / ls as f32);
        }
        out.push(s.json(json!({"kind": "ls", "seed": 500 + j as u64, "rot": name, "ls": ls})));
    }
    let unif = UniformRotDistr::Range(Uniform::new(0.0, 2.0 * std::f32::consts::PI).unwrap());
    let mut rng = SmallRng::seed_from_u64(600);
    let mut s = Stream::new();
    for _ in 0..10_000 {
        s.f32(unif.sample(&mut rng));
    }
    out.push(s.json(json!({"kind": "uniform_rot", "seed": 600u64})));
}

#[test]
fn rng_vectors() {
    let mut out = vec![];
    raw_streams(&mut out);
    uniform_streams(&mut out);
    choose_streams(&mut out);
    normal_streams(&mut out);
    sampler_streams(&mut out);
    let consts = json!({
        "sd_rot": [bits(lbf::samplers::ls_sampler::SD_ROT.0), bits(lbf::samplers::ls_sampler::SD_ROT.1)],
        "sd_transl": [bits(lbf::samplers::ls_sampler::SD_TRANSL.0), bits(lbf::samplers::ls_sampler::SD_TRANSL.1)],
        "two_pi": bits(2.0 * std::f32::consts::PI),
        "ls_budget": [(50000f32 * 0.3f32) as usize, (50000f32 * 0.5f32) as usize, (50000f32 * 0.7f32) as usize],
    });
    let doc = json!({"generator": "jagua-utils/tests/rng_vectors.rs", "rand": "0.10.0", "rand_distr": "0.6.0", "consts": consts, "streams": out});
    let dir = std::env::var("ORACLE_OUT").expect("ORACLE_OUT");
    std::fs::write(format!("{dir}/rng_vectors.json"), serde_json::to_string_pretty(&doc).unwrap()).unwrap();
}
