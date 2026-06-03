//! Utility functions for jagua-rs
//!
//! This crate provides utility functions for working with jagua-rs,
//! including SVG nesting utilities.

pub mod svg_nesting;

pub use svg_nesting::{
    AdaptiveNestingStrategy, NestingResult, NestingStrategy, Offcut, OffcutPolicy, OffcutShape,
    OffcutVertex, PageResult, PartInput, PlacedPartInfo, SimpleNestingStrategy, nest_svg_parts,
};
