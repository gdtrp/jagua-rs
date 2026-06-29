//! Generates the SQS wire types from the vendored AsyncAPI spec (`asyncapi/jagua-rs.yaml`).
//!
//! The spec's `components.schemas` are the single source of truth for the request/response
//! contract. At build time we lift them into a draft-07 JSON Schema document, rewrite the
//! AsyncAPI `$ref`s, and hand them to typify, which emits idiomatic serde structs into
//! `$OUT_DIR/generated.rs` (included by `src/generated.rs`).

use std::{env, fs, path::PathBuf};

use serde_json::{json, Map, Value};

const SPEC_PATH: &str = "asyncapi/jagua-rs.yaml";

/// Wire schemas reused from jagua-utils instead of being generated, paired index-wise with
/// [`REPLACEMENT_PATHS`].
const REPLACED: &[&str] = &[
    "OffcutPolicy",
    "Offcut",
    "OffcutVertex",
    "NestingResponsePage",
    "NestingPlacement",
];
/// Rust paths typify substitutes for each [`REPLACED`] schema (same order).
const REPLACEMENT_PATHS: &[&str] = &[
    "jagua_utils::OffcutPolicy",
    "jagua_utils::Offcut",
    "jagua_utils::OffcutVertex",
    "jagua_utils::PageResult",
    "jagua_utils::PlacedPartInfo",
];

/// Rewrite every `$ref: '#/components/schemas/X'` to `#/definitions/X` (the location typify /
/// draft-07 expects), recursively.
fn rewrite_refs(v: &mut Value) {
    match v {
        Value::Object(map) => {
            if let Some(Value::String(s)) = map.get_mut("$ref") {
                if let Some(name) = s.strip_prefix("#/components/schemas/") {
                    *s = format!("#/definitions/{name}");
                }
            }
            for val in map.values_mut() {
                rewrite_refs(val);
            }
        }
        Value::Array(arr) => arr.iter_mut().for_each(rewrite_refs),
        _ => {}
    }
}

fn main() {
    println!("cargo:rerun-if-changed={SPEC_PATH}");

    let yaml = fs::read_to_string(SPEC_PATH).unwrap_or_else(|_| {
        panic!("AsyncAPI spec not found at {SPEC_PATH} — it is sourced from cutl-schemas, not vendored. \
                Run ./scripts/sync-schema.sh (needs gh auth) before building.")
    });
    let doc: Value = serde_yaml::from_str(&yaml).expect("parse AsyncAPI YAML");

    let schemas = doc
        .pointer("/components/schemas")
        .and_then(Value::as_object)
        .cloned()
        .expect("components.schemas missing from spec");

    let mut definitions = Map::new();
    for (name, mut schema) in schemas {
        rewrite_refs(&mut schema);
        definitions.insert(name, schema);
    }

    // These wire types already exist as canonical, serde-tested domain types in jagua-utils;
    // typify reuses them via with_replacement (below) so the wire stays byte-identical and we
    // don't carry duplicate generated copies. Stub their schema bodies to a bare object so
    // typify never deep-parses them — the replacement overrides the type entirely.
    for name in REPLACED {
        definitions.insert((*name).to_string(), json!({ "type": "object" }));
    }
    // Helper schemas that only support the replaced types (Offcut's enums / former oneOf
    // variants). Once their parents are replaced they're unreferenced, so drop them to keep
    // typify from emitting dead enum/struct code.
    for name in ["OffcutKind", "OffcutShape", "OffcutRect", "OffcutPoly"] {
        definitions.remove(name);
    }

    let root = json!({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "definitions": definitions,
    });

    let root_schema: schemars::schema::RootSchema =
        serde_json::from_value(root).expect("build root JSON Schema");

    let mut settings = typify::TypeSpaceSettings::default();
    for (name, path) in REPLACED.iter().zip(REPLACEMENT_PATHS) {
        settings.with_replacement(name, path, std::iter::empty());
    }
    let mut type_space = typify::TypeSpace::new(&settings);
    type_space
        .add_root_schema(root_schema)
        .expect("typify: add_root_schema");

    let tokens = type_space.to_stream();
    let text =
        prettyplease::unparse(&syn::parse2::<syn::File>(tokens).expect("parse generated tokens"));

    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("generated.rs");
    fs::write(&out, text).expect("write generated.rs");
}
