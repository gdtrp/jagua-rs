//! RFC-030 N7: the depth-3 hole-nesting fixture's part, made with jagua-rs (owner, 2026-11-02: no
//! corpus request nests 3 deep). case-068's deepest real chain is 16dbc0b1 → c0b56c3b → 175724b7;
//! 175724b7 has one hole (a 7 mm circle as a 32-gon). The new part is that hole deflated by
//! jagua-rs's offset_shape by the request's spacing (2 mm) plus 0.5 mm, so, inflated by spacing/2,
//! it fits the hole shrunk by spacing/2 with 0.5 mm to spare. It is written as a DXF (one closed
//! LWPOLYLINE, mm) for the Go engine's reader.
//!
//! DEPTH3_CORPUS=~/repos-backup/nesting-corpus cargo test -p jagua-utils --test depth3_part -- --nocapture
//! writes $DEPTH3_CORPUS/derived/depth3-175724b7-hole.dxf (outside the repo: it derives from a
//! customer part).
use jagua_rs::geometry::primitives::{Point, SPolygon};
use jagua_rs::geometry::shape_modification::{ShapeModifyMode, offset_shape};
use jagua_utils::svg_nesting::{extract_path_from_svg_bytes, parse_svg_path};
use std::fmt::Write as _;
use std::path::Path;

#[test]
fn depth3_part() {
    let Ok(root) = std::env::var("DEPTH3_CORPUS") else {
        eprintln!("set DEPTH3_CORPUS");
        return;
    };
    let dir = Path::new(&root).join("cases").join("case-068");
    let svg = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.file_name().unwrap().to_string_lossy().starts_with("175724b7"))
        .expect("175724b7's SVG in case-068");
    let path = extract_path_from_svg_bytes(&std::fs::read(&svg).unwrap()).unwrap();
    let (_outer, holes) = parse_svg_path(&path).unwrap();
    assert_eq!(holes.len(), 1, "175724b7 has one hole");
    let mut hole: Vec<Point> = holes[0].clone();
    hole.dedup();
    if hole.first() == hole.last() {
        hole.pop();
    }
    let sp = SPolygon::new(hole).unwrap();
    let part = offset_shape(&sp, ShapeModifyMode::Deflate, 2.5).unwrap();
    let mut dxf = String::from("0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLWPOLYLINE\n8\n0\n");
    write!(dxf, "90\n{}\n70\n1\n", part.vertices.len()).unwrap();
    for p in &part.vertices {
        write!(dxf, "10\n{}\n20\n{}\n", p.0, p.1).unwrap();
    }
    dxf.push_str("0\nENDSEC\n0\nEOF\n");
    let out = Path::new(&root).join("derived");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join("depth3-175724b7-hole.dxf"), dxf).unwrap();
    eprintln!("hole area {:.3}, part area {:.3}, {} vertices, bbox {:?}", sp.area, part.area, part.vertices.len(), part.bbox);
}
