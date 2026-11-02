//! Vectors of the rand 0.10 calls Sparrow makes that jagua's LBF does not (RFC-030 N10), from the
//! rand that Sparrow links: shuffle, IteratorRandom::choose on a filter, random_range on f32, and
//! random::<u64>. Each stream ends with the next u64, so the state after the call is compared too.
//!
//! rng_vectors > rng_sparrow_vectors.json
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::rngs::Xoshiro256PlusPlus;
use rand::{Rng, RngExt, SeedableRng};
use serde_json::{json, Value};

fn main() {
    let mut out: Vec<Value> = vec![];
    for (i, n) in [0usize, 1, 2, 3, 5, 8, 13, 20, 33, 64, 100, 257].into_iter().enumerate() {
        let seed = 700 + i as u64;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut v: Vec<usize> = (0..n).collect();
        v.shuffle(&mut rng);
        out.push(json!({"kind": "shuffle", "seed": seed, "n": n, "out": v, "next": rng.next_u64().to_string()}));
    }
    for (i, n) in [0usize, 1, 2, 3, 4, 7, 16, 31, 50, 129].into_iter().enumerate() {
        for keep in [1usize, 2, 3] {
            let seed = 800 + 10 * i as u64 + keep as u64;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            // a filter keeping every keep-th of n*keep elements: n candidates, size hint (0, n*keep)
            let chosen = (0..n * keep).filter(|x| x % keep == 0).choose(&mut rng).map(|x| x / keep);
            out.push(json!({"kind": "choose_filtered", "seed": seed, "n": n, "keep": keep,
                "out": chosen.map_or(-1i64, |c| c as i64), "next": rng.next_u64().to_string()}));
        }
    }
    for (i, (lo, hi)) in [(0.0f32, 1.0f32), (0.0, 555.18146), (-3.5, 2.25), (10.0, 10.0), (0.0, 1e-30), (-1e6, 1e6)].into_iter().enumerate() {
        let seed = 900 + i as u64;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let v: Vec<String> = (0..16).map(|_| format!("{:08x}", rng.random_range(lo..=hi).to_bits())).collect();
        out.push(json!({"kind": "sample_single_f32", "seed": seed, "low": format!("{:08x}", lo.to_bits()),
            "high": format!("{:08x}", hi.to_bits()), "out": v, "next": rng.next_u64().to_string()}));
    }
    for (i, (lo, hi)) in [(0.0f32, 1.0f32), (0.0, 734.0), (5.0, 900.0)].into_iter().enumerate() {
        let seed = 950 + i as u64;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let v: Vec<String> = (0..16).map(|_| format!("{:08x}", rng.random_range(lo..hi).to_bits())).collect();
        out.push(json!({"kind": "sample_single_f32_open", "seed": seed, "low": format!("{:08x}", lo.to_bits()),
            "high": format!("{:08x}", hi.to_bits()), "out": v, "next": rng.next_u64().to_string()}));
    }
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(990);
    let v: Vec<String> = (0..8).map(|_| rng.random::<u64>().to_string()).collect();
    out.push(json!({"kind": "random_u64", "seed": 990, "out": v, "next": rng.next_u64().to_string()}));
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}
