//! RFC-030 N3 (dxf-processor): records jagua-rs's collision verdicts on real parts, so the Go port
//! (dxfcalc/internal/nestengine/cde) is checked decision for decision. For each part SVG under
//! $KERNEL_SVG_DIR with at most 2,000 outline vertices, the item is built as the adaptive strategy
//! builds it (spacing 2, four discrete rotations); a 3000 × 2000 sheet deflated by 1 is the
//! container. Parts are placed on a grid where the polygon test allows, as a search would accept
//! them; then every part is queried at 400 transforms (inside, across the edges, outside, on the
//! placed parts) for the surrogate and the polygon verdicts. jagua-rs itself is not touched.
//!
//!   KERNEL_SVG_DIR=/svg cargo test -p jagua-utils --features oracle --test cde_vectors -- --nocapture
//!   (writes $ORACLE_OUT/cde_vectors.json)
#![cfg(feature = "oracle")]

use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::collision_detection::hazards::filter::NoFilter;
use jagua_rs::entities::{Container, Item, Layout};
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::geo_traits::TransformableFrom;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::geometry::shape_modification::{ShapeModifyConfig, ShapeModifyMode};
use jagua_rs::geometry::{DTransformation, OriginalShape};
use jagua_utils::svg_nesting::{
    build_inflatable_shape, calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path, reverse_winding,
    sanitize_polygon,
};
use serde_json::json;
use std::f32::consts::PI;

fn b(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}

const CFG: CDEConfig = CDEConfig {
    quadtree_depth: 5,
    cd_threshold: 16,
    item_surrogate_config: SPSurrogateConfig { n_pole_limits: [(100, 0.0), (20, 0.75), (10, 0.90)], n_ff_poles: 2, n_ff_piers: 0 },
};

fn item(id: usize, svg: &[u8]) -> Option<(Item, Vec<String>)> {
    let path = extract_path_from_svg_bytes(svg).unwrap();
    let (outer, _) = parse_svg_path(&path).unwrap();
    let wound = if calculate_signed_area(&outer) < 0.0 { reverse_winding(&outer) } else { outer };
    let sanitized = sanitize_polygon(wound);
    if sanitized.len() > 2000 {
        return None;
    }
    let pts = sanitized.iter().map(|p| format!("{} {}", b(p.0), b(p.1))).collect();
    let poly = SPolygon::new(sanitized).unwrap();
    let c = poly.centroid();
    let pre = DTransformation::new(0.0, (-c.0, -c.1));
    let cfg = ShapeModifyConfig { offset: Some(1.0), simplify_tolerance: None, narrow_concavity_cutoff: None };
    let os = build_inflatable_shape(poly, pre, ShapeModifyMode::Inflate, cfg).unwrap();
    let rot = RotationRange::Discrete(vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
    Some((Item::new(id, os, rot, None, CFG.item_surrogate_config).unwrap(), pts))
}

#[test]
fn cde_vectors() {
    let dir = std::env::var("KERNEL_SVG_DIR").expect("KERNEL_SVG_DIR");
    let mut paths: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "svg")).collect();
    paths.sort();
    let mut items = vec![];
    let mut parts = vec![];
    for p in &paths {
        if let Some((it, pts)) = item(items.len(), &std::fs::read(p).unwrap()) {
            parts.push(json!({"name": p.file_stem().unwrap().to_string_lossy(), "sanitized": pts}));
            items.push(it);
        }
    }
    let sheet = SPolygon::from(Rect::try_new(0.0, 0.0, 3000.0, 2000.0).unwrap());
    let cos = OriginalShape {
        shape: sheet,
        pre_transform: DTransformation::empty(),
        modify_mode: ShapeModifyMode::Deflate,
        modify_config: ShapeModifyConfig { offset: Some(1.0), simplify_tolerance: None, narrow_concavity_cutoff: None },
    };
    let container = Container::new(0, cos, vec![], CFG).unwrap();
    let mut layout = Layout::new(container);
    let rots = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0];

    // Place on a 7 × 5 grid, cycling parts and rotations, where the polygon test allows.
    let mut placed = vec![];
    let mut k = 0usize;
    for gy in 0..5 {
        for gx in 0..7 {
            let it = &items[k % items.len()];
            let dt = DTransformation::new(rots[k % 4], (200.0 + 420.0 * gx as f32, 200.0 + 390.0 * gy as f32));
            k += 1;
            let mut buf = (*it.shape_cd).clone();
            buf.surrogate = None;
            buf.transform_from(&it.shape_cd, &dt.compose());
            let sur = layout.cde().detect_surrogate_collision(it.shape_cd.surrogate(), &dt.compose(), &NoFilter);
            let poly = layout.cde().detect_poly_collision(&buf, &NoFilter);
            placed.push(json!({"item": it.id, "dt": [b(dt.rotation()), b(dt.translation().0), b(dt.translation().1)], "sur": sur, "poly": poly, "placed": !poly}));
            if !poly {
                layout.place_item(it, dt);
            }
        }
    }

    // Query every part at 400 transforms: a 20 × 20 grid over the engine's box grown by 10 %.
    let bb = layout.cde().bbox();
    let mut queries = vec![];
    for it in &items {
        let mut buf = (*it.shape_cd).clone();
        buf.surrogate = None;
        let mut out = vec![];
        for i in 0..20 {
            for j in 0..20 {
                let x = bb.x_min - 0.1 * bb.width() + 1.2 * bb.width() * (i as f32 / 19.0);
                let y = bb.y_min - 0.1 * bb.height() + 1.2 * bb.height() * (j as f32 / 19.0);
                let r = if (i + j) % 5 == 4 { 0.7 } else { rots[(i + j) % 4] };
                let dt = DTransformation::new(r, (x, y));
                let t = dt.compose();
                let sur = layout.cde().detect_surrogate_collision(it.shape_cd.surrogate(), &t, &NoFilter);
                buf.transform_from(&it.shape_cd, &t);
                let poly = layout.cde().detect_poly_collision(&buf, &NoFilter);
                out.push(format!("{} {} {} {} {}", b(r), b(x), b(y), u8::from(sur), u8::from(poly)));
            }
        }
        queries.push(json!({"item": it.id, "q": out}));
    }
    let doc = json!({"parts": parts, "placements": placed, "queries": queries, "n_hazards": layout.cde().hazards().count()});
    let out = std::env::var("ORACLE_OUT").expect("ORACLE_OUT");
    std::fs::write(format!("{out}/cde_vectors.json"), serde_json::to_string(&doc).unwrap()).unwrap();
}
