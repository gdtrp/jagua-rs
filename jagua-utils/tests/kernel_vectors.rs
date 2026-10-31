//! RFC-030 N3 (dxf-processor): records jagua-rs's kernel results on real part SVGs, so the Go port
//! (dxfcalc/internal/nestengine/geom2, buffer, jaguain) is checked value for value. For each SVG in
//! $KERNEL_SVG_DIR it runs jagua-utils' item build (extract, parse, wind, sanitize, SPolygon::new,
//! build_inflatable_shape) and records every intermediate as f32 bit patterns, plus transforms,
//! point-in-polygon, separations, the pole, the surrogate, the offsets and the simplification.
//! jagua-rs itself is not touched.
//!
//!   KERNEL_SVG_DIR=/svg cargo test -p jagua-utils --features oracle --test kernel_vectors -- --nocapture
//!   (writes $ORACLE_OUT/kernel_vectors/<name>.json)
#![cfg(feature = "oracle")]

use jagua_rs::geometry::fail_fast::{SPSurrogateConfig, compute_pole};
use jagua_rs::geometry::geo_traits::{CollidesWith, SeparationDistance, Transformable};
use jagua_rs::geometry::primitives::{Circle, Point, Rect, SPolygon};
use jagua_rs::geometry::shape_modification::{ShapeModifyConfig, ShapeModifyMode, offset_shape, simplify_shape};
use jagua_rs::geometry::{DTransformation, Transformation};
use jagua_utils::svg_nesting::{
    build_inflatable_shape, calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path, reverse_winding,
    sanitize_polygon,
};
use serde_json::{Value, json};

fn b(v: f32) -> String {
    format!("{:08x}", v.to_bits())
}
fn pts(p: &[Point]) -> Value {
    json!(p.iter().map(|q| format!("{} {}", b(q.0), b(q.1))).collect::<Vec<_>>())
}
fn rect(r: &Rect) -> Value {
    json!([b(r.x_min), b(r.y_min), b(r.x_max), b(r.y_max)])
}
fn circle(c: &Circle) -> Value {
    json!([b(c.center.0), b(c.center.1), b(c.radius)])
}
fn matrix(t: &Transformation) -> Value {
    json!(t.matrix().iter().flatten().map(|v| b(v.into_inner())).collect::<Vec<_>>())
}
fn err_kind(e: &anyhow::Error) -> String {
    let s = e.to_string();
    for k in ["at least 3 points", "duplicate points", "intersecting edges", "no area", "non-consecutive duplicate", "empty polygon", "no pole"] {
        if s.contains(k) {
            return k.to_string();
        }
    }
    s
}

fn spoly(p: &SPolygon) -> Value {
    let mut v = json!({
        "vertices": pts(&p.vertices), "bbox": rect(&p.bbox), "area": b(p.area), "diameter": b(p.diameter),
        "poi": circle(&p.poi), "centroid": [b(p.centroid().0), b(p.centroid().1)],
    });
    if let Some(s) = &p.surrogate {
        v["poles"] = json!(s.poles.iter().map(circle).collect::<Vec<_>>());
        v["hull"] = json!(s.convex_hull_indices);
        v["hull_area"] = json!(b(s.convex_hull_area));
    }
    v
}

const CFG: SPSurrogateConfig = SPSurrogateConfig { n_pole_limits: [(100, 0.0), (20, 0.75), (10, 0.90)], n_ff_poles: 2, n_ff_piers: 0 };

fn transforms() -> Vec<DTransformation> {
    use std::f32::consts::PI;
    vec![
        DTransformation::new(0.0, (0.0, 0.0)),
        DTransformation::new(PI / 2.0, (100.5, -3.25)),
        DTransformation::new(PI, (612.5, 400.25)),
        DTransformation::new(3.0 * PI / 2.0, (-17.125, 1234.0)),
        DTransformation::new(0.7, (1.0, 2.0)),
        DTransformation::new(5.1, (2500.0, 1250.0)),
    ]
}

fn probes(p: &SPolygon) -> Vec<Point> {
    // A 23 × 17 grid over the bounding box grown by 5 %, plus every vertex and every edge midpoint.
    let r = p.bbox;
    let (dx, dy) = (r.width() * 0.05, r.height() * 0.05);
    let mut out = vec![];
    for i in 0..23 {
        for j in 0..17 {
            let x = r.x_min - dx + (r.width() + 2.0 * dx) * (i as f32 / 22.0);
            let y = r.y_min - dy + (r.height() + 2.0 * dy) * (j as f32 / 16.0);
            out.push(Point(x, y));
        }
    }
    out.extend(p.vertices.iter().copied());
    out.extend(p.edge_iter().map(|e| e.centroid()));
    out
}

fn record(name: &str, svg: &[u8]) -> Value {
    let path = extract_path_from_svg_bytes(svg).unwrap();
    let (outer, holes) = parse_svg_path(&path).unwrap();
    let wound = if calculate_signed_area(&outer) < 0.0 { reverse_winding(&outer) } else { outer.clone() };
    let sanitized = sanitize_polygon(wound.clone());
    let mut doc = json!({"name": name, "parsed": pts(&outer), "holes": holes.len(), "wound": pts(&wound), "sanitized": pts(&sanitized)});
    let poly = match SPolygon::new(sanitized) {
        Ok(p) => p,
        Err(e) => {
            doc["spolygon_error"] = json!(err_kind(&e));
            return doc;
        }
    };
    doc["spolygon"] = spoly(&poly);
    let mut with_sur = poly.clone();
    with_sur.generate_surrogate(CFG).unwrap();
    doc["surrogate"] = spoly(&with_sur);
    // A second pole avoiding the first, as generate_surrogate_poles' first step.
    doc["pole2"] = circle(&compute_pole(&poly, &[poly.poi]).unwrap());

    let probes = probes(&poly);
    doc["probes"] = pts(&probes);
    doc["inside"] = json!(probes.iter().map(|q| u8::from(poly.collides_with(q))).collect::<Vec<_>>());
    doc["separation"] = json!(probes.iter().map(|q| {
        let (pos, d) = poly.sq_separation_distance(q);
        format!("{} {}", u8::from(pos == jagua_rs::geometry::geo_enums::GeoPosition::Interior), b(d))
    }).collect::<Vec<_>>());

    let mut tr = vec![];
    for dt in transforms() {
        let t = dt.compose();
        let moved = with_sur.transform_clone(&t);
        let inv = t.clone().inverse();
        let dec = t.decompose();
        tr.push(json!({"dt": [b(dt.rotation()), b(dt.translation().0), b(dt.translation().1)],
            "matrix": matrix(&t), "inverse": matrix(&inv),
            "decompose": [b(dec.rotation()), b(dec.translation().0), b(dec.translation().1)],
            "moved": spoly(&moved)}));
    }
    doc["transforms"] = json!(tr);

    let c = poly.centroid();
    let pre = DTransformation::new(0.0, (-c.0, -c.1));
    let centred = poly.transform_clone(&pre.compose());
    let mut offsets = vec![];
    for spacing in [0.2f32, 2.0, 6.0, 20.0] {
        let d = spacing / 2.0;
        let raw = match offset_shape(&centred, ShapeModifyMode::Inflate, d) {
            Ok(p) => spoly(&p),
            Err(e) => json!({"error": err_kind(&e)}),
        };
        let deflated = match offset_shape(&centred, ShapeModifyMode::Deflate, d) {
            Ok(p) => spoly(&p),
            Err(e) => json!({"error": err_kind(&e)}),
        };
        let cfg = ShapeModifyConfig { offset: Some(d), simplify_tolerance: None, narrow_concavity_cutoff: None };
        let built = match build_inflatable_shape(poly.clone(), pre, ShapeModifyMode::Inflate, cfg) {
            Ok(os) => {
                let mut internal = os.convert_to_internal().unwrap();
                internal.generate_surrogate(CFG).unwrap();
                json!({"offset": b(os.modify_config.offset.unwrap_or(0.0)), "internal": spoly(&internal)})
            }
            Err(e) => json!({"error": err_kind(&e)}),
        };
        offsets.push(json!({"spacing": b(spacing), "inflate": raw, "deflate": deflated, "item": built}));
    }
    doc["offsets"] = json!(offsets);
    doc["simplify_inflate"] = spoly(&simplify_shape(&poly, ShapeModifyMode::Inflate, 0.001));
    doc["simplify_deflate"] = spoly(&simplify_shape(&poly, ShapeModifyMode::Deflate, 0.001));
    doc["quadrants"] = json!(poly.bbox.inflate_to_square().quadrants().iter().map(rect).collect::<Vec<_>>());
    doc
}

#[test]
fn kernel_vectors() {
    let dir = std::env::var("KERNEL_SVG_DIR").expect("KERNEL_SVG_DIR");
    let out = format!("{}/kernel_vectors", std::env::var("ORACLE_OUT").expect("ORACLE_OUT"));
    std::fs::create_dir_all(&out).unwrap();
    let mut names: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "svg")).collect();
    names.sort();
    for p in names {
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        let doc = record(&name, &std::fs::read(&p).unwrap());
        std::fs::write(format!("{out}/{name}.json"), serde_json::to_string(&doc).unwrap()).unwrap();
        eprintln!("{name}: done");
    }
}
