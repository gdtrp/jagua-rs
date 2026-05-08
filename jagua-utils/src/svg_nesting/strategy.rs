//! Nesting strategies for different optimization approaches

mod adaptive;
mod simple;

pub use adaptive::AdaptiveNestingStrategy;
pub use simple::SimpleNestingStrategy;

use crate::svg_nesting::parsing::{
    calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path,
};
use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;

/// Sanity ceiling on the upper-bound count when computing max_fit.
const MAX_FIT_HARD_CAP: usize = 10_000;

/// Slack multiplier applied to the area-based upper-bound count for max_fit.
const MAX_FIT_AREA_SLACK: f32 = 1.5;

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

/// Compute the maximum number of copies of a single part that fit on one sheet.
///
/// The strategy is invoked with a generous upper-bound count (derived from
/// `bin_area / part_area * MAX_FIT_AREA_SLACK`, capped at `MAX_FIT_HARD_CAP`),
/// then the resulting `NestingResult` is truncated to its first page.
pub fn nest_max_fit_single_sheet(
    strategy: &dyn NestingStrategy,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    amount_of_rotations: usize,
    improvement_callback: Option<ImprovementCallback>,
) -> Result<NestingResult> {
    let path_data = extract_path_from_svg_bytes(&part.svg_bytes)?;
    let (polygon_points, _holes) = parse_svg_path(&path_data)?;
    let part_area = calculate_signed_area(&polygon_points).abs();
    if part_area <= 0.0 {
        anyhow::bail!("max_fit: part has non-positive area, cannot compute upper bound");
    }
    let bin_area = bin_width * bin_height;
    if bin_area <= 0.0 {
        anyhow::bail!(
            "max_fit: bin area is non-positive ({} x {})",
            bin_width,
            bin_height
        );
    }
    let raw = (bin_area / part_area * MAX_FIT_AREA_SLACK).ceil();
    let upper_bound = (raw as usize).clamp(1, MAX_FIT_HARD_CAP);

    let saturated_part = PartInput {
        svg_bytes: part.svg_bytes.clone(),
        count: upper_bound,
        item_id: part.item_id.clone(),
    };
    let parts = std::slice::from_ref(&saturated_part);

    let mut result = strategy.nest(
        bin_width,
        bin_height,
        spacing,
        parts,
        amount_of_rotations,
        improvement_callback,
    )?;

    truncate_to_first_page(&mut result);
    Ok(result)
}

/// Truncate a `NestingResult` to keep only its first page.
///
/// Used by [`nest_max_fit_single_sheet`] so callers see exactly one page even
/// when the upper-bound count overflows onto additional sheets.
fn truncate_to_first_page(result: &mut NestingResult) {
    if result.pages.is_empty() {
        result.parts_placed = 0;
        result.total_parts_requested = 0;
        result.page_svgs.clear();
        result.combined_svg.clear();
        result.unplaced_parts_svg = None;
        result.utilisation = 0.0;
        return;
    }
    result.pages.truncate(1);
    let first = &result.pages[0];
    result.parts_placed = first.parts_placed;
    result.total_parts_requested = first.parts_placed;
    result.utilisation = first.utilisation;
    if !result.page_svgs.is_empty() {
        result.page_svgs.truncate(1);
        result.combined_svg = result.page_svgs[0].clone();
    }
    result.unplaced_parts_svg = None;
}
