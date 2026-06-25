//! Wire types generated at build time from the AsyncAPI spec (`asyncapi/jagua-rs.yaml`) by
//! `build.rs` (typify). Do not edit — re-run `make sync-spec` + rebuild after the contract
//! changes. Nested domain types (`OffcutPolicy`, `Offcut`, `OffcutVertex`, `PageResult`,
//! `PlacedPartInfo`) are reused from `jagua_utils` via typify replacement, so the wire stays
//! byte-identical to those types' own serde.
#![allow(clippy::all)]
#![allow(dead_code)]
#![allow(missing_docs)]

include!(concat!(env!("OUT_DIR"), "/generated.rs"));
