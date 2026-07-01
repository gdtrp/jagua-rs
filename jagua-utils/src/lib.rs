//! Utility functions for jagua-rs
//!
//! This crate provides utility functions for working with jagua-rs,
//! including SVG nesting utilities.

pub mod svg_nesting;

pub use svg_nesting::{
    AdaptiveNestingStrategy, NestingResult, NestingStrategy, Offcut, OffcutPolicy, OffcutShape,
    OffcutVertex, PackingMode, PageResult, PartInput, PlacedPartInfo, SimpleNestingStrategy,
    nest_auto, nest_max_fit_auto, nest_svg_parts,
};
