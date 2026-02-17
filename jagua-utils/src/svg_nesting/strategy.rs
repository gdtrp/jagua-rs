//! Nesting strategies for different optimization approaches

mod simple;
mod adaptive;

pub use simple::SimpleNestingStrategy;
pub use adaptive::AdaptiveNestingStrategy;

use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;

/// A single part type with its SVG bytes and count
#[derive(Debug, Clone)]
pub struct PartInput {
    /// SVG content as bytes
    pub svg_bytes: Vec<u8>,
    /// Number of copies of this part to nest
    pub count: usize,
    /// Optional user-provided correlation ID for this part type
    pub item_id: Option<String>,
}

/// Callback function type for sending intermediate improvements
/// Called when a better result is found during optimization
pub type ImprovementCallback = Box<dyn Fn(NestingResult) -> Result<()> + Send + Sync>;

/// Returns true when all items map to the same part index (single part type request).
/// Used to skip redundant SVG generation for middle pages that are visually identical.
pub(crate) fn is_single_part_type(item_id_to_part_idx: &[usize]) -> bool {
    match item_id_to_part_idx.first() {
        Some(&first) => item_id_to_part_idx.iter().all(|&idx| idx == first),
        None => true,
    }
}

/// Trait for nesting strategies that can be plugged into the nesting system
pub trait NestingStrategy: Send + Sync {
    /// Execute the nesting strategy
    ///
    /// # Arguments
    /// * `bin_width` - Width of the bin
    /// * `bin_height` - Height of the bin
    /// * `spacing` - Minimum spacing between parts
    /// * `parts` - Slice of part inputs, each with SVG bytes and count
    /// * `amount_of_rotations` - Number of discrete rotations to allow
    /// * `improvement_callback` - Optional callback to send intermediate improvements (called when better results are found)
    ///
    /// # Returns
    /// * [`NestingResult`] containing SVG bytes and placement metadata
    fn nest(
        &self,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
        parts: &[PartInput],
        amount_of_rotations: usize,
        improvement_callback: Option<ImprovementCallback>,
    ) -> Result<NestingResult>;
}
