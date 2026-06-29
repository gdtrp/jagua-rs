//! Nesting strategies for different optimization approaches

mod adaptive;
mod simple;

pub use adaptive::AdaptiveNestingStrategy;
pub use simple::SimpleNestingStrategy;

use crate::svg_nesting::svg_generation::NestingResult;
use anyhow::Result;
use jagua_rs::geometry::geo_enums::RotationRange;

/// A single part type with its SVG bytes and count
#[derive(Debug, Clone)]
pub struct PartInput {
    /// SVG content as bytes
    pub svg_bytes: Vec<u8>,
    /// Number of copies of this part to nest
    pub count: usize,
    /// Optional user-provided correlation ID for this part type
    pub item_id: Option<String>,
    /// Optional grain-direction constraint: the exact set of rotations (in **degrees**)
    /// this part may be placed at. When `Some`, only these orientations are allowed,
    /// overriding the global `amount_of_rotations`. When `None`, the part follows the
    /// global rotation setting (today's behaviour). An empty list or a single `0` means
    /// "0° only" (no rotation), matching the core `ExtItem::allowed_orientations` semantics.
    pub allowed_rotations: Option<Vec<f32>>,
}

/// Resolve the rotation constraint for a single part.
///
/// When `allowed_rotations` is set (grain-direction control), only those angles are
/// permitted: the values are interpreted as **degrees** and converted into a
/// [`RotationRange::Discrete`]. An empty list, or a single `0`, means "0° only",
/// matching the core `ExtItem` import semantics. When `allowed_rotations` is `None`,
/// the supplied `fallback` (derived from the global `amount_of_rotations`) is used,
/// preserving today's behaviour.
pub(crate) fn resolve_rotation_range(
    allowed_rotations: &Option<Vec<f32>>,
    fallback: &RotationRange,
) -> RotationRange {
    match allowed_rotations {
        Some(angles) => {
            if angles.is_empty() || (angles.len() == 1 && angles[0] == 0.0) {
                RotationRange::None
            } else {
                RotationRange::Discrete(angles.iter().map(|deg| deg.to_radians()).collect())
            }
        }
        None => fallback.clone(),
    }
}

/// For the bounding-box fit pre-check: decide which of the two axis-aligned
/// orientations a part may use — original (`w×h`) and/or 90°-swapped (`h×w`) —
/// given its optional grain constraint. Returns `(allow_original, allow_swapped)`.
///
/// Unconstrained parts (`None`) may use both. A grain-locked part may only use an
/// orientation its allowed-angle set actually reaches. Non-cardinal angles (e.g. 45°)
/// can't be judged by an axis-aligned bbox, so we stay permissive there and let the
/// optimiser decide rather than wrongly rejecting the part up front.
pub(crate) fn fit_orientations(allowed_rotations: &Option<Vec<f32>>) -> (bool, bool) {
    let angles = match allowed_rotations {
        None => return (true, true),
        Some(a) if a.is_empty() => return (true, false), // 0° only
        Some(a) => a,
    };
    let (mut allow_original, mut allow_swapped) = (false, false);
    for &deg in angles {
        let norm = deg.rem_euclid(180.0);
        if !(1.0..=179.0).contains(&norm) {
            allow_original = true; // ~0° / ~180°
        } else if (norm - 90.0).abs() < 1.0 {
            allow_swapped = true; // ~90° / ~270°
        } else {
            // Non-cardinal grain angle: not judgeable via axis-aligned bbox.
            allow_original = true;
            allow_swapped = true;
        }
    }
    (allow_original, allow_swapped)
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

/// Pack as many copies of a single part as possible onto one sheet.
///
/// Thin wrapper around [`AdaptiveNestingStrategy::nest_max_fit`] kept for
/// backwards compatibility with existing callers. Forces single-bin packing
/// inside the strategy so the optimiser concentrates on bin-0 density rather
/// than spilling overflow onto subsequent bins.
pub fn nest_max_fit_single_sheet(
    strategy: &AdaptiveNestingStrategy,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    part: &PartInput,
    amount_of_rotations: usize,
    improvement_callback: Option<ImprovementCallback>,
) -> Result<NestingResult> {
    strategy.nest_max_fit(
        bin_width,
        bin_height,
        spacing,
        part,
        amount_of_rotations,
        improvement_callback,
    )
}
