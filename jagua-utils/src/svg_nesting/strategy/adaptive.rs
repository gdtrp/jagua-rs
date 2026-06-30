//! Adaptive nesting strategy that starts with lower parameters and adaptively increases them

use crate::svg_nesting::{
    offcut::{OffcutPolicy, apply_offcuts},
    parsing::{
        build_inflatable_shape, calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path,
        reverse_winding, sanitize_polygon,
    },
    strategy::{NestingStrategy, PartInput, is_single_part_type},
    svg_generation::{
        NestingResult, PageResult, PlacedPartInfo, combine_svg_documents, post_process_svg_multi,
    },
};
use anyhow::Result;
use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::entities::{Container, Instance, Item, Layout};
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::OriginalShape;
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::geometry::shape_modification::ShapeModifyMode;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::{SvgDrawOptions, s_layout_to_svg};
use jagua_rs::probs::bpp::entities::{BPInstance, BPSolution, Bin};
use lbf::config::LBFConfig;
use lbf::opt::lbf_bpp::LBFOptimizerBP;
use rand::SeedableRng;
use rand::prelude::SmallRng;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Process-wide ceiling on the number of optimization routines (parallel seed
/// runs) executing concurrently across ALL in-flight nesting executions.
///
/// Each execution fans a wave of independent runs out across CPU cores. Without
/// a global cap, a burst of simultaneous requests (e.g. many `max_fit` SQS
/// messages) would each try to fan out, flooding the shared rayon pool with far
/// more queued work than there are cores and slowing every request down. This
/// bounds the total in-flight fan-out so the machine stays responsive under
/// load.
const MAX_GLOBAL_ROUTINES: usize = 100;

/// Per-execution reserved routine counts, keyed by a unique execution id. The
/// sum of the values is held at or below [`MAX_GLOBAL_ROUTINES`]. An execution
/// reserves before each wave and releases (via [`RoutineReservation`]'s `Drop`)
/// once the wave completes, so the budget is shared fairly over time.
static GLOBAL_ROUTINES: OnceLock<Mutex<HashMap<u64, usize>>> = OnceLock::new();

/// Accessor for the lazily-initialised global routine map.
fn global_routines() -> &'static Mutex<HashMap<u64, usize>> {
    GLOBAL_ROUTINES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Monotonic source of unique execution ids (one per `nest_inner` call).
static EXECUTION_ID_SEQ: AtomicU64 = AtomicU64::new(0);

/// RAII reservation of a slice of the global routine budget. Holds the granted
/// count for the lifetime of one wave; releases it on drop so a panic or early
/// return can't leak the reservation and starve later executions.
struct RoutineReservation {
    execution_id: u64,
    /// Number of routines actually granted (>= 1, may be < requested when the
    /// global budget is contended).
    granted: usize,
}

impl Drop for RoutineReservation {
    fn drop(&mut self) {
        if let Ok(mut map) = global_routines().lock() {
            map.remove(&self.execution_id);
        }
    }
}

/// Reserve up to `desired` routines for `execution_id` from the shared global
/// budget. Grants `min(desired, remaining)` but always at least 1, so an
/// execution can never be starved into making zero progress even when the
/// budget is fully subscribed.
fn reserve_routines(execution_id: u64, desired: usize) -> RoutineReservation {
    let mut map = global_routines().lock().unwrap_or_else(|e| e.into_inner());
    // This execution's previous wave (if any) is released before the next wave
    // reserves, but guard against a stale entry just in case.
    map.remove(&execution_id);
    let in_use: usize = map.values().copied().sum();
    let available = MAX_GLOBAL_ROUTINES.saturating_sub(in_use);
    let granted = desired.min(available).max(1);
    map.insert(execution_id, granted);
    RoutineReservation {
        execution_id,
        granted,
    }
}

/// Adaptive nesting strategy that starts with lower parameters and adaptively increases them
/// based on results. Sends intermediate improvements via callback.
pub struct AdaptiveNestingStrategy {
    /// Optional function to check if optimization should be cancelled
    cancellation_checker: Option<Box<dyn Fn() -> bool + Send + Sync>>,
    /// When set, the final layout is scanned for reusable offcuts.
    offcut_policy: Option<OffcutPolicy>,
    /// Optional per-request wall-clock budget. When set, overrides the default time budget
    /// (42s for max_fit, 600s for normal nesting).
    time_budget: Option<Duration>,
}

impl AdaptiveNestingStrategy {
    /// Create a new adaptive nesting strategy
    pub fn new() -> Self {
        Self {
            cancellation_checker: None,
            offcut_policy: None,
            time_budget: None,
        }
    }

    /// Create a new adaptive nesting strategy with cancellation checking
    pub fn with_cancellation_checker(
        cancellation_checker: Box<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self {
            cancellation_checker: Some(cancellation_checker),
            offcut_policy: None,
            time_budget: None,
        }
    }

    /// Enable offcut detection on the final layout using `policy`.
    pub fn with_offcut_policy(mut self, policy: OffcutPolicy) -> Self {
        self.offcut_policy = Some(policy);
        self
    }

    /// Override the optimization wall-clock budget (caps both the normal and max_fit paths).
    pub fn with_time_budget(mut self, budget: Duration) -> Self {
        self.time_budget = Some(budget);
        self
    }

    /// Populate per-page offcuts on a final result from the solution that produced it.
    /// No-op when no offcut policy is set. Called only at the final return sites so that
    /// streamed intermediate results keep empty offcuts.
    fn finalize_offcuts(
        &self,
        result: &mut NestingResult,
        solution: &BPSolution,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
    ) {
        if let Some(policy) = self.offcut_policy {
            apply_offcuts(result, solution, &policy, bin_width, bin_height, spacing);
        }
    }

    /// Check if optimization should be cancelled
    fn is_cancelled(&self) -> bool {
        self.cancellation_checker
            .as_ref()
            .map(|checker| checker())
            .unwrap_or(false)
    }

    /// Calculate average bin utilisation (density) from a solution
    /// Returns a ratio between 0.0 and 1.0 representing how much of the total bin area is occupied
    #[allow(dead_code)]
    fn calculate_bin_density(&self, solution: &BPSolution, bin_width: f32, bin_height: f32) -> f32 {
        let single_bin_area = bin_width * bin_height;
        let num_bins = solution.layout_snapshots.len();

        if single_bin_area <= 0.0 || num_bins == 0 {
            return 0.0;
        }

        let total_bin_area = single_bin_area * num_bins as f32;

        // Sum up the bounding box areas of all placed items across all bins
        let used_area: f32 = solution
            .layout_snapshots
            .values()
            .flat_map(|ls| &ls.placed_items)
            .map(|(_key, item)| {
                let bbox = &item.shape.bbox;
                bbox.width() * bbox.height()
            })
            .sum();

        // Return utilisation ratio clamped to [0.0, 1.0]
        (used_area / total_bin_area).clamp(0.0, 1.0)
    }

    /// Run a single optimization run with given parameters
    fn run_single_optimization(
        &self,
        instance: &BPInstance,
        cde_config: &CDEConfig,
        spacing: f32,
        loops: usize,
        placements: usize,
        seed_offset: usize,
        ls_frac: f32,
    ) -> Result<(usize, BPSolution)> {
        let mut best_solution = None;
        let mut best_placed = 0;

        for loop_idx in 0..loops {
            // Check cancellation at the start of each loop iteration
            if self.is_cancelled() {
                log::debug!("Cancellation detected in optimization loop, breaking");
                break;
            }

            let seed = (seed_offset * 1000 + loop_idx) as u64;
            let lbf_config = LBFConfig {
                cde_config: *cde_config,
                poly_simpl_tolerance: Some(0.001),
                min_item_separation: Some(spacing),
                prng_seed: Some(seed),
                n_samples: placements,
                ls_frac,
                narrow_concavity_cutoff: None,
                svg_draw_options: Default::default(),
            };

            let mut optimizer =
                LBFOptimizerBP::new(instance.clone(), lbf_config, SmallRng::seed_from_u64(seed));

            // Let each run complete fully (cutting LBF mid-solve yields a sparse
            // partial layout). The wall-clock cap is honoured *between* waves by
            // the predictive guard in nest_inner, which refuses to start a wave
            // that the running average says won't finish before the deadline.
            let cancellation_fn = || self.is_cancelled();
            let solution = optimizer.solve_with_cancellation(Some(&cancellation_fn));

            let placed: usize = solution
                .layout_snapshots
                .values()
                .map(|ls| ls.placed_items.len())
                .sum();

            if placed > best_placed {
                best_placed = placed;
                best_solution = Some(solution);
            }

            // If we've placed all items, no need to continue
            if placed >= instance.total_item_qty() {
                break;
            }
        }

        match best_solution {
            Some(solution) => Ok((best_placed, solution)),
            None => anyhow::bail!("No items could be placed in the bin"),
        }
    }

    /// Run `wave_size` independent optimizations concurrently and return the
    /// best `(placed, solution)` among them.
    ///
    /// Each run is fully independent (its own cloned instance + RNG seed), so
    /// they parallelise cleanly across CPU cores via rayon's global pool. This
    /// lets the adaptive search explore many seeds within a tight wall-clock
    /// budget instead of one-at-a-time (~20s each for dense max_fit inputs).
    /// `ls_frac` is varied per run within the wave, same {0.3, 0.5, 0.7} cycle
    /// as the sequential path.
    fn run_parallel_wave(
        &self,
        instance: &BPInstance,
        cde_config: &CDEConfig,
        spacing: f32,
        loops: usize,
        placements: usize,
        base_seed: usize,
        wave_size: usize,
    ) -> Result<(usize, BPSolution)> {
        let results: Vec<(usize, BPSolution)> = (0..wave_size)
            .into_par_iter()
            .filter_map(|i| {
                let run_no = base_seed + i;
                let ls_frac = match run_no % 3 {
                    1 => 0.3,
                    2 => 0.5,
                    _ => 0.7,
                };
                self.run_single_optimization(
                    instance, cde_config, spacing, loops, placements, run_no, ls_frac,
                )
                .ok()
            })
            .collect();

        // Best = most parts placed, ties broken by fewer pages.
        let mut best: Option<(usize, BPSolution)> = None;
        for (placed, solution) in results {
            let pages = solution.layout_snapshots.len();
            let better = match &best {
                None => true,
                Some((bp, bsol)) => {
                    placed > *bp || (placed == *bp && pages < bsol.layout_snapshots.len())
                }
            };
            if better {
                best = Some((placed, solution));
            }
        }
        best.ok_or_else(|| anyhow::anyhow!("No items could be placed in the bin"))
    }

    /// Generate SVG from solution
    /// `item_id_to_holes` maps each item ID to its holes slice
    /// `item_id_to_part_idx` maps each item ID to its part index (index into the parts array)
    /// `item_id_to_part_id` maps each item ID to its user-provided part ID (itemId from request)
    fn generate_svg_from_solution(
        &self,
        solution: &BPSolution,
        instance: &BPInstance,
        item_id_to_holes: &[&[Vec<jagua_rs::geometry::primitives::Point>]],
        item_id_to_part_idx: &[usize],
        item_id_to_part_id: &[Option<String>],
        bin_width: f32,
        bin_height: f32,
        total_parts_requested: usize,
    ) -> Result<NestingResult> {
        // Count items directly from the solution
        let total_items_placed: usize = solution
            .layout_snapshots
            .values()
            .map(|ls| ls.placed_items.len())
            .sum();

        log::debug!("Optimization complete: {} parts placed", total_items_placed);

        // Generate SVG output and extract placement data
        let svg_options = SvgDrawOptions::default();
        let mut page_svg_strings: Vec<String> = Vec::new();
        let mut page_svgs: Vec<Vec<u8>> = Vec::new();
        let mut pages: Vec<PageResult> = Vec::new();

        let mut layout_entries: Vec<_> = solution.layout_snapshots.iter().collect();
        layout_entries.sort_by_key(|(_, layout_snapshot)| layout_snapshot.container.id);

        let num_pages = layout_entries.len();
        let skip_middle = is_single_part_type(item_id_to_part_idx) && num_pages >= 3;

        for (page_index, (layout_key, layout_snapshot)) in layout_entries.iter().enumerate() {
            let is_first = page_index == 0;
            let is_last = page_index == num_pages - 1;

            if skip_middle && !is_first && !is_last {
                // Clone first page's SVG for middle pages (visually identical for single-part)
                let first_svg_str = page_svg_strings[0].clone();
                page_svg_strings.push(first_svg_str.clone());
                page_svgs.push(first_svg_str.into_bytes());
            } else {
                let svg_doc = s_layout_to_svg(
                    layout_snapshot,
                    instance,
                    svg_options,
                    &format!("Layout {:?} - {} items", layout_key, total_items_placed),
                );
                let svg_str = svg_doc.to_string();
                let processed_svg = post_process_svg_multi(&svg_str, item_id_to_holes);
                page_svg_strings.push(processed_svg.clone());
                page_svgs.push(processed_svg.into_bytes());
            }

            // Compute per-page utilisation using actual polygon area (matches SVG density)
            let page_util = layout_snapshot.density(instance);

            // Always extract real placement data (cheap, and item_id values differ per page)
            let mut page_placements = Vec::new();
            for (_key, placed_item) in layout_snapshot.placed_items.iter() {
                let (x, y) = placed_item.d_transf.translation();
                let rotation = placed_item.d_transf.rotation().to_degrees();
                let internal_id = placed_item.item_id;
                let part_index = item_id_to_part_idx.get(internal_id).copied().unwrap_or(0);
                let item_id = item_id_to_part_id
                    .get(internal_id)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| internal_id.to_string());
                let centroid = instance.item(internal_id).shape_orig.centroid();
                page_placements.push(PlacedPartInfo {
                    item_id,
                    part_index,
                    x,
                    y,
                    rotation,
                    centroid_x: centroid.x(),
                    centroid_y: centroid.y(),
                });
            }

            pages.push(PageResult {
                page_index,
                utilisation: page_util,
                svg_url: None,
                parts_placed: page_placements.len(),
                placements: page_placements,
                // Offcuts stay empty here: this runs for every candidate (incl. streamed
                // intermediates). They are populated only on the final result via the
                // finalize pass in `nest_inner`.
                offcuts: Vec::new(),
            });
        }

        // Combine all page SVGs into a single valid SVG document
        let combined_svg = combine_svg_documents(&page_svg_strings, bin_width, bin_height);

        // Verify the count matches what's actually in the SVG
        use regex::Regex;
        let re_item_use = Regex::new(r##"<use[^>]*href=["']#item_\d+["']"##).unwrap();
        let items_in_svg = re_item_use.find_iter(&combined_svg).count();

        // Use the actual count from SVG as the source of truth
        let corrected_count = items_in_svg;

        if corrected_count != total_items_placed {
            log::warn!(
                "Count mismatch detected: SVG contains {} item <use> tags, but optimizer reports {}",
                corrected_count,
                total_items_placed
            );
        }

        // Calculate average bin utilisation
        let utilisation = if pages.is_empty() {
            0.0
        } else {
            pages.iter().map(|p| p.utilisation).sum::<f32>() / pages.len() as f32
        };

        log::debug!("Average bin utilisation: {:.1}%", utilisation * 100.0);

        Ok(NestingResult {
            combined_svg: combined_svg.into_bytes(),
            page_svgs,
            parts_placed: corrected_count,
            total_parts_requested,
            unplaced_parts_svg: None,
            utilisation,
            pages,
            sheets_total_estimate: None,
        })
    }
}

impl Default for AdaptiveNestingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveNestingStrategy {
    /// Pack as many copies of a single part as possible onto one sheet.
    ///
    /// Forces `bin_stock = 1` so the optimizer has nowhere to spill — every
    /// adaptive iteration is dedicated to maximising bin-0 density. The result
    /// is then truncated to its first page (extras are reported as unplaced
    /// internally and discarded).
    pub fn nest_max_fit(
        &self,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
        part: &PartInput,
        amount_of_rotations: usize,
        improvement_callback: Option<crate::svg_nesting::strategy::ImprovementCallback>,
    ) -> Result<NestingResult> {
        // Saturate the part count: the optimizer will pack as many as fit on
        // bin 0 and report the rest as unplaced. Hard cap to avoid pathological
        // memory usage on tiny parts in huge bins.
        const MAX_FIT_PART_COUNT: usize = 10_000;
        let saturated = PartInput {
            svg_bytes: part.svg_bytes.clone(),
            count: MAX_FIT_PART_COUNT,
            item_id: part.item_id.clone(),
            allowed_rotations: part.allowed_rotations.clone(),
        };
        let parts = std::slice::from_ref(&saturated);

        // Wall-clock cap comes solely from the per-request time budget (e.g. the SQS
        // `maxSeconds` field). When unset, nest_inner applies its default (600s) — there is
        // no separate max_fit-specific cap. The cap is enforced *between* waves: nest_inner's
        // predictive guard refuses to start a wave the running per-wave average says won't
        // finish before the deadline (cutting LBF mid-solve yields a sparse layout).
        let deadline = self.time_budget.map(|b| Instant::now() + b);

        // Single fixed orientation for now; the parallel adaptive search inside
        // nest_inner explores many seeds within the deadline and keeps the best.
        let rotations = if amount_of_rotations == 0 { 0 } else { 1 };

        // Wrap the user callback so streamed intermediate improvements report
        // truncated single-page numbers.
        let wrapped_cb: Option<crate::svg_nesting::strategy::ImprovementCallback> =
            improvement_callback.map(|cb| {
                Box::new(move |mut intermediate: NestingResult| -> Result<()> {
                    truncate_to_first_page(&mut intermediate);
                    cb(intermediate)
                }) as crate::svg_nesting::strategy::ImprovementCallback
            });

        let mut result = self.nest_inner(
            bin_width,
            bin_height,
            spacing,
            parts,
            rotations,
            wrapped_cb,
            Some(1), // bin_stock = 1: forces single-bin packing
            deadline,
        )?;
        truncate_to_first_page(&mut result);
        Ok(result)
    }
}

impl NestingStrategy for AdaptiveNestingStrategy {
    fn nest(
        &self,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
        parts: &[PartInput],
        amount_of_rotations: usize,
        improvement_callback: Option<crate::svg_nesting::strategy::ImprovementCallback>,
    ) -> Result<NestingResult> {
        // A per-request time budget overrides the default 600s internal deadline.
        let deadline = self.time_budget.map(|b| Instant::now() + b);
        self.nest_inner(
            bin_width,
            bin_height,
            spacing,
            parts,
            amount_of_rotations,
            improvement_callback,
            None,
            deadline,
        )
    }
}

/// Truncate a `NestingResult` to keep only its first page. Used by `nest_max_fit`.
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

impl AdaptiveNestingStrategy {
    /// Internal nest entry point. `bin_stock_override = Some(n)` caps available
    /// bins to `n` (used by max_fit to force single-bin packing).
    fn nest_inner(
        &self,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
        parts: &[PartInput],
        amount_of_rotations: usize,
        improvement_callback: Option<crate::svg_nesting::strategy::ImprovementCallback>,
        bin_stock_override: Option<usize>,
        deadline: Option<Instant>,
    ) -> Result<NestingResult> {
        // Parse all part SVGs and build per-part data
        struct ParsedPart {
            item_shape: OriginalShape,
            processed_holes: Vec<Vec<jagua_rs::geometry::primitives::Point>>,
            count: usize,
            item_id: Option<String>,
            allowed_rotations: Option<Vec<f32>>,
        }

        let cde_config = CDEConfig {
            quadtree_depth: 5,
            cd_threshold: 16,
            item_surrogate_config: SPSurrogateConfig {
                n_pole_limits: [(100, 0.0), (20, 0.75), (10, 0.90)],
                n_ff_poles: 2,
                n_ff_piers: 0,
            },
        };

        let importer = Importer::new(cde_config, None, Some(spacing), None);

        let mut parsed_parts = Vec::with_capacity(parts.len());
        for (part_idx, part) in parts.iter().enumerate() {
            let path_data = extract_path_from_svg_bytes(&part.svg_bytes)?;
            let (polygon_points, holes) = parse_svg_path(&path_data)?;

            log::debug!(
                "Parsed SVG path for part {} (adaptive): {} outer boundary points, {} holes",
                part_idx,
                polygon_points.len(),
                holes.len()
            );

            let outer_area = calculate_signed_area(&polygon_points);
            let polygon_points = if outer_area < 0.0 {
                reverse_winding(&polygon_points)
            } else {
                polygon_points
            };

            let mut processed_holes = Vec::new();
            for hole in holes.iter() {
                let hole_area = calculate_signed_area(hole);
                let processed_hole = if hole_area > 0.0 {
                    reverse_winding(hole)
                } else {
                    hole.clone()
                };
                processed_holes.push(processed_hole);
            }

            let polygon_points = sanitize_polygon(polygon_points);
            let polygon = SPolygon::new(polygon_points)?;
            let centroid = polygon.centroid();
            let pre_transform = DTransformation::new(0.0, (-centroid.x(), -centroid.y()));

            let item_shape = build_inflatable_shape(
                polygon,
                pre_transform,
                ShapeModifyMode::Inflate,
                importer.shape_modify_config,
            )?;

            parsed_parts.push(ParsedPart {
                item_shape,
                processed_holes,
                count: part.count,
                item_id: part.item_id.clone(),
                allowed_rotations: part.allowed_rotations.clone(),
            });
        }

        // For max_fit the caller saturates the count (e.g. 10_000 copies) and
        // lets the optimizer place as many as fit. But LbfOptimizer::solve() does
        // work proportional to the *requested* quantity: every copy that can't
        // fit still burns a full sampling pass (and is rendered into the unplaced
        // grid). With thousands of impossible copies, each run takes minutes and
        // the optimization appears to hang.
        //
        // The number of parts that can physically fit is bounded by
        // bin_area / part_area (a strict upper bound: real packing with spacing
        // and geometric waste always places fewer). Clamp each part's count to
        // that bound plus a small margin so the optimizer still has more copies
        // than it can ever place, without churning through impossible extras.
        if bin_stock_override.is_some() {
            let bin_area = bin_width * bin_height;
            for parsed in parsed_parts.iter_mut() {
                let part_area = parsed.item_shape.shape.area.max(f32::MIN_POSITIVE);
                let upper_bound = ((bin_area / part_area).ceil() as usize).saturating_add(2);
                parsed.count = parsed.count.min(upper_bound);
            }
        }

        let total_parts_requested: usize = parsed_parts.iter().map(|p| p.count).sum();

        // Validate that each part can physically fit inside the bin (accounting for spacing)
        let effective_bin_w = bin_width - spacing;
        let effective_bin_h = bin_height - spacing;
        if effective_bin_w <= 0.0 || effective_bin_h <= 0.0 {
            anyhow::bail!(
                "Bin ({:.2} x {:.2}) is too small for the given spacing ({:.2}). \
                 Effective bin dimensions would be {:.2} x {:.2}. \
                 Please increase the bin dimensions or reduce the spacing.",
                bin_width,
                bin_height,
                spacing,
                effective_bin_w,
                effective_bin_h,
            );
        }
        for (part_idx, parsed) in parsed_parts.iter().enumerate() {
            let bbox = &parsed.item_shape.shape.bbox;
            let item_w = bbox.width();
            let item_h = bbox.height();
            let part_label = parsed
                .item_id
                .as_deref()
                .map(|id| format!("'{}'", id))
                .unwrap_or_else(|| format!("#{}", part_idx));

            // Check if item fits in at least one orientation its grain constraint
            // permits (original or rotated 90°).
            let (allow_original, allow_swapped) =
                crate::svg_nesting::strategy::fit_orientations(&parsed.allowed_rotations);
            let fits_original =
                allow_original && item_w <= effective_bin_w && item_h <= effective_bin_h;
            let fits_rotated =
                allow_swapped && item_h <= effective_bin_w && item_w <= effective_bin_h;
            if !fits_original && !fits_rotated {
                anyhow::bail!(
                    "Part {} (size {:.2} x {:.2}) is too large to fit in the bin ({:.2} x {:.2}) \
                     with spacing {:.2} (effective bin area: {:.2} x {:.2}). \
                     Please increase the bin dimensions or reduce the part size/spacing.",
                    part_label,
                    item_w,
                    item_h,
                    bin_width,
                    bin_height,
                    spacing,
                    effective_bin_w,
                    effective_bin_h,
                );
            }
        }

        // Build container
        let bin_rect = Rect::try_new(0.0, 0.0, bin_width, bin_height)?;
        let bin_polygon = SPolygon::from(bin_rect);
        let container_shape = OriginalShape {
            shape: bin_polygon,
            pre_transform: DTransformation::empty(),
            modify_mode: ShapeModifyMode::Deflate,
            modify_config: importer.shape_modify_config,
        };

        let container_template = Container::new(0, container_shape, vec![], cde_config)?;

        // Build rotation range
        const MAX_ROTATIONS: usize = 4;
        let rotation_count = if amount_of_rotations == 0 {
            0
        } else {
            amount_of_rotations.clamp(1, MAX_ROTATIONS)
        };

        let rotation_range = if rotation_count == 0 {
            RotationRange::None
        } else if rotation_count == 1 {
            RotationRange::Discrete(vec![0.0])
        } else {
            let rotations: Vec<f32> = (0..rotation_count)
                .map(|i| (i as f32 * 2.0 * std::f32::consts::PI) / (rotation_count as f32))
                .collect();
            RotationRange::Discrete(rotations)
        };

        // Create one Item per part type with quantity = count.
        // This avoids expensive surrogate generation for each individual copy.
        let mut items = Vec::with_capacity(parsed_parts.len());
        let mut item_id_to_part_idx: Vec<usize> = Vec::with_capacity(parsed_parts.len());
        let mut item_id_to_part_id: Vec<Option<String>> = Vec::with_capacity(parsed_parts.len());
        for (part_idx, parsed) in parsed_parts.iter().enumerate() {
            // Per-part grain constraint overrides the global rotation range when set.
            let item_rotation_range = crate::svg_nesting::strategy::resolve_rotation_range(
                &parsed.allowed_rotations,
                &rotation_range,
            );
            let item = Item::new(
                part_idx,
                parsed.item_shape.clone(),
                item_rotation_range,
                None,
                cde_config.item_surrogate_config,
            )?;
            items.push((item, parsed.count));
            item_id_to_part_idx.push(part_idx);
            item_id_to_part_id.push(parsed.item_id.clone());
        }

        // Build per-item holes mapping (item_id → holes slice)
        let item_id_to_holes: Vec<&[Vec<jagua_rs::geometry::primitives::Point>]> =
            item_id_to_part_idx
                .iter()
                .map(|&part_idx| parsed_parts[part_idx].processed_holes.as_slice())
                .collect();

        // Stock = total_parts_requested by default (worst case: each item needs
        // its own bin). max_fit forces stock = 1 to concentrate optimisation
        // on a single sheet.
        let bin_stock = bin_stock_override.unwrap_or(total_parts_requested);
        let bin = Bin::new(container_template.clone(), bin_stock, 0);
        let instance = BPInstance::new(items, vec![bin]);

        // Adaptive optimization loop
        let optimization_start = Instant::now();
        // Unique id for this execution, used to track its share of the global
        // routine budget (see [`GLOBAL_ROUTINES`]).
        let execution_id = EXECUTION_ID_SEQ.fetch_add(1, Ordering::Relaxed);
        let is_max_fit = bin_stock_override.is_some();
        // For high item counts, scale down loops and samples to keep total work per run manageable.
        // max_fit gets a 2x budget since the user explicitly asked for the
        // densest possible single-sheet packing (quality over speed).
        const HIGH_COUNT_SAMPLE_BUDGET_DEFAULT: usize = 2_000_000;
        const HIGH_COUNT_SAMPLE_BUDGET_MAX_FIT: usize = 4_000_000;
        let high_count_sample_budget = if is_max_fit {
            HIGH_COUNT_SAMPLE_BUDGET_MAX_FIT
        } else {
            HIGH_COUNT_SAMPLE_BUDGET_DEFAULT
        };
        let base_placements: usize = 50000;
        // For budget scaling, use the bin stock cap when set (max_fit feeds a
        // saturated count that vastly overstates the items that will actually
        // be placed). Without this we'd under-budget per-item placement
        // attempts and produce sparse packings.
        let effective_count = match bin_stock_override {
            Some(_) => total_parts_requested.min(200),
            None => total_parts_requested,
        };
        let mut loops;
        let mut placements;
        if effective_count > 50 {
            loops = 1.max(50 / effective_count);
            placements = (high_count_sample_budget / (effective_count * loops).max(1))
                .max(1000)
                .min(base_placements);
        } else {
            loops = 3;
            placements = base_placements;
        }

        let mut best_result: Option<NestingResult> = None;
        // The solution that produced `best_result`, retained so offcuts can be computed on
        // the final layout only (see `finalize_offcuts`).
        let mut best_solution: Option<BPSolution> = None;
        let mut best_placed = 0;
        let mut best_pages = usize::MAX;
        let mut total_runs = 0;
        let mut batches_without_improvement = 0;
        // Slowest single run observed so far — feeds the predictive time guard
        // below, which uses the previous waves' durations to decide whether
        // there is room for another wave before the deadline is exhausted.
        let mut max_single_run_secs: u64 = 0;
        // Number of independent runs explored concurrently per wave. Defaults to
        // the machine's parallelism (override via NEST_RUN_PARALLELISM), clamped
        // so a single request can't spawn an unbounded fan-out.
        let run_parallelism: usize = std::env::var("NEST_RUN_PARALLELISM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
            })
            .clamp(1, 32);
        // Wall-clock safety ceiling for the whole optimization. The search
        // normally stops far earlier via convergence (no improvement across
        // several parallel waves); this cap only bounds pathological inputs and
        // is kept under the SQS visibility timeout so a job can never overrun
        // into redelivery (which spawned the duplicate concurrent runs we saw in
        // production). max_fit and regular nesting share the same ceiling — the
        // user wants the densest achievable packing, so we let it explore as long
        // as it keeps finding better results.
        let time_budget_secs: u64 = MAX_TOTAL_OPTIMIZATION_SECONDS;
        // Absolute wall-clock deadline. max_fit supplies a tight one (its 55s
        // frontend cap); otherwise we derive it from this call's start and the
        // 600s safety ceiling.
        let deadline =
            deadline.unwrap_or_else(|| optimization_start + Duration::from_secs(time_budget_secs));
        // Wider exploration budget for max_fit: extra adaptive iterations can
        // find seeds/orderings that interlock the last few parts.
        let max_total_runs: usize = if is_max_fit { 100 } else { 60 };
        const MAX_RUNS_WITHOUT_IMPROVEMENT: usize = 15;
        // Stop once several consecutive parallel waves fail to improve the best
        // result: each non-improving batch bumps the parameters and retries;
        // after this many in a row the search has converged. This guarantees
        // termination instead of grinding to the time cap. Since each wave now
        // explores `run_parallelism` seeds at once, a non-improving wave is
        // strong evidence of a plateau — but the user wants the densest packing,
        // so we keep exploring for a few waves before declaring convergence.
        const MAX_BATCHES_WITHOUT_IMPROVEMENT: usize = 5;
        // Cap how far the escalation grows `loops`. Without this, every quiet
        // batch keeps adding +2 loops, making each subsequent run progressively
        // slower; capping keeps the extra exploration batches affordable.
        const MAX_ESCALATED_LOOPS: usize = 7;
        const MAX_RUN_DURATION_SECONDS: u64 = 120;
        // Wall-clock safety ceiling, shared by max_fit and regular nesting. Kept
        // under the SQS visibility timeout so a job can't overrun into redelivery.
        const MAX_TOTAL_OPTIMIZATION_SECONDS: u64 = 600;
        const HIGH_DENSITY_THRESHOLD: f32 = 0.65;

        'outer: loop {
            let elapsed_total = optimization_start.elapsed().as_secs();
            if Instant::now() >= deadline {
                log::info!(
                    "Reached maximum optimization time ({} seconds), stopping with {} parts placed",
                    elapsed_total,
                    best_placed
                );
                break 'outer;
            }

            if self.is_cancelled() {
                log::info!("Cancellation detected, stopping adaptive optimization");
                break 'outer;
            }

            if total_runs >= max_total_runs {
                log::info!("Reached maximum total runs ({}), stopping", max_total_runs);
                break 'outer;
            }

            let mut improved_this_batch = false;
            let mut should_stop_due_to_timeout = false;
            let mut cancelled = false;
            for batch_run in 0..MAX_RUNS_WITHOUT_IMPROVEMENT {
                if self.is_cancelled() {
                    log::info!("Cancellation detected before starting run, stopping optimization");
                    cancelled = true;
                    break;
                }

                if total_runs >= max_total_runs {
                    break;
                }

                // Predictive time guard: if the slowest run so far would push the
                // total past the budget, return the best result we already have
                // instead of starting another long run. This keeps execution time
                // bounded well under the SQS visibility window so a slow request
                // can't overrun and trigger redelivery (which spawns duplicate
                // concurrent optimizations).
                let elapsed_now = optimization_start.elapsed().as_secs();
                if Instant::now() + Duration::from_secs(max_single_run_secs) >= deadline
                    && let Some(mut best) = best_result.take()
                {
                    log::info!(
                        "Projected next run (~{}s) would exceed the deadline at {}s elapsed; \
                         returning best result early ({} parts on {} pages)",
                        max_single_run_secs,
                        elapsed_now,
                        best_placed,
                        best_pages
                    );
                    if let Some(sol) = best_solution.as_ref() {
                        self.finalize_offcuts(&mut best, sol, bin_width, bin_height, spacing);
                    }
                    return Ok(best);
                }

                let run_start = Instant::now();
                total_runs += 1;

                // Vary ls_frac across iterations to explore both biased and
                // diversified local search regimes. Empirically the
                // {0.3, 0.5, 0.7} cycle finds tighter packings than a fixed
                // value because some shapes need more aggressive local search
                // to interlock cutouts/concavities.
                let ls_frac = match total_runs % 3 {
                    1 => 0.3,
                    2 => 0.5,
                    _ => 0.7,
                };

                log::info!(
                    "Run {}/{} (batch {}/{}): loops={}, placements={}, rotations={}, ls_frac={}",
                    total_runs,
                    max_total_runs,
                    batch_run + 1,
                    MAX_RUNS_WITHOUT_IMPROVEMENT,
                    loops,
                    placements,
                    amount_of_rotations,
                    ls_frac
                );

                // Explore several independent seeds concurrently and keep the
                // best. The wave is bounded two ways: locally so cumulative runs
                // never exceed `max_total_runs`, and globally by reserving a
                // slice of the shared routine budget so concurrent executions
                // can't collectively oversubscribe the machine under heavy load.
                let desired_wave = run_parallelism
                    .min(max_total_runs.saturating_sub(total_runs).saturating_add(1))
                    .max(1);
                let reservation = reserve_routines(execution_id, desired_wave);
                let wave_size = reservation.granted;
                if wave_size < desired_wave {
                    log::debug!(
                        "Global routine budget contended: wanted {} routines, granted {}",
                        desired_wave,
                        wave_size
                    );
                }
                let optimization_result = self.run_parallel_wave(
                    &instance,
                    &cde_config,
                    spacing,
                    loops,
                    placements,
                    total_runs,
                    wave_size,
                );
                drop(reservation);
                total_runs += wave_size.saturating_sub(1);
                let _ = ls_frac;

                let (_placed, solution) = match optimization_result {
                    Ok(r) => r,
                    Err(e) => {
                        // A wave only errors when every run in it placed nothing.
                        // If an earlier wave already produced a packing, never
                        // throw it away — return the stored best instead of an
                        // empty result. Only the very first wave (no best yet)
                        // returns the empty placeholder.
                        log::warn!("Wave placed no items: {}", e);
                        if let Some(mut best) = best_result.take() {
                            log::info!(
                                "Returning previous best result ({} parts on {} pages) after a barren wave",
                                best_placed,
                                best_pages
                            );
                            if let Some(sol) = best_solution.as_ref() {
                                self.finalize_offcuts(
                                    &mut best, sol, bin_width, bin_height, spacing,
                                );
                            }
                            return Ok(best);
                        }
                        return Ok(NestingResult {
                            combined_svg: Vec::new(),
                            page_svgs: Vec::new(),
                            parts_placed: 0,
                            total_parts_requested,
                            unplaced_parts_svg: None,
                            utilisation: 0.0,
                            pages: Vec::new(),
                            sheets_total_estimate: None,
                        });
                    }
                };

                let run_duration = run_start.elapsed();
                max_single_run_secs = max_single_run_secs.max(run_duration.as_secs());
                let run_exceeded_time_limit = run_duration.as_secs() > MAX_RUN_DURATION_SECONDS;
                if run_exceeded_time_limit {
                    log::warn!(
                        "Run took {} seconds, exceeding limit of {} seconds. Will stop after processing result.",
                        run_duration.as_secs(),
                        MAX_RUN_DURATION_SECONDS
                    );
                }

                let mut result = self.generate_svg_from_solution(
                    &solution,
                    &instance,
                    &item_id_to_holes,
                    &item_id_to_part_idx,
                    &item_id_to_part_id,
                    bin_width,
                    bin_height,
                    total_parts_requested,
                )?;

                // Handle unplaced parts SVG generation. Skip in max_fit mode
                // (bin_stock_override set) where "unplaced" is expected and
                // discarded by the caller — generating a giant grid SVG for
                // 9k+ unplaced parts is pure waste.
                if bin_stock_override.is_none() && result.parts_placed < total_parts_requested {
                    let mut unplaced_layout = Layout::new(container_template.clone());
                    let unplaced_count = total_parts_requested - result.parts_placed;

                    // Use first part's shape for grid sizing (approximate)
                    let first_shape = &parsed_parts[0].item_shape;
                    let part_bbox = &first_shape.shape.bbox;
                    let part_width = part_bbox.width();
                    let part_height = part_bbox.height();
                    let cols = ((bin_width - spacing) / (part_width + spacing))
                        .floor()
                        .max(1.0) as usize;

                    let parts_per_page = result.parts_placed.max(1);
                    let remainder = unplaced_count % parts_per_page;
                    let items_to_render = if remainder == 0 && unplaced_count > 0 {
                        parts_per_page
                    } else {
                        remainder
                    };

                    if unplaced_count > items_to_render {
                        log::info!(
                            "Last page shows {} of {} unplaced items ({} items/page)",
                            items_to_render,
                            unplaced_count,
                            parts_per_page
                        );
                    }

                    let total_grid_width =
                        (cols as f32 * part_width) + ((cols.saturating_sub(1)) as f32 * spacing);
                    let offset_x = (bin_width - total_grid_width) / 2.0;
                    let rows = ((items_to_render as f32 / cols as f32).ceil()) as usize;
                    let total_grid_height =
                        (rows as f32 * part_height) + ((rows.saturating_sub(1)) as f32 * spacing);
                    let offset_y = (bin_height - total_grid_height) / 2.0;

                    // Use item 0 as template for the grid (simplification for unplaced display)
                    let item_template = instance.item(0);

                    for i in 0..items_to_render {
                        let row = i / cols;
                        let col = i % cols;
                        let grid_x =
                            offset_x + (col as f32 * (part_width + spacing)) + part_width / 2.0;
                        let grid_y =
                            offset_y + (row as f32 * (part_height + spacing)) + part_height / 2.0;
                        let d_transf = DTransformation::new(0.0, (grid_x, grid_y));
                        unplaced_layout.place_item(item_template, d_transf);
                    }

                    let unplaced_snapshot = unplaced_layout.save();
                    let svg_options = SvgDrawOptions {
                        highlight_cd_shapes: false,
                        ..SvgDrawOptions::default()
                    };
                    let label = if unplaced_count > items_to_render {
                        format!(
                            "Unplaced parts: {} (showing {})",
                            unplaced_count, items_to_render
                        )
                    } else {
                        format!("Unplaced parts: {}", unplaced_count)
                    };
                    let unplaced_svg_doc =
                        s_layout_to_svg(&unplaced_snapshot, &instance, svg_options, &label);
                    let unplaced_svg_str = unplaced_svg_doc.to_string();
                    let processed_unplaced_svg =
                        post_process_svg_multi(&unplaced_svg_str, &item_id_to_holes);
                    result.unplaced_parts_svg = Some(processed_unplaced_svg.into_bytes());
                }

                let num_pages = result.page_svgs.len();

                log::info!(
                    "Run {} completed: {} parts placed on {} pages (utilisation {:.1}%), best so far: {} placed on {} pages, in {:.2}s",
                    total_runs,
                    result.parts_placed,
                    num_pages,
                    result.utilisation * 100.0,
                    best_placed,
                    best_pages,
                    run_duration.as_secs_f64()
                );

                // A result is better if it places more parts, or the same parts on fewer pages
                let is_improvement = result.parts_placed > best_placed
                    || (result.parts_placed == best_placed && num_pages < best_pages);

                if is_improvement {
                    improved_this_batch = true;
                    best_placed = result.parts_placed;
                    best_pages = num_pages;
                    best_result = Some(result.clone());
                    // Retain the producing solution so the final result (and only it) can
                    // be scanned for offcuts. The streamed `result.clone()` below keeps its
                    // empty offcuts.
                    best_solution = Some(solution.clone());

                    log::info!(
                        "New best result: {} parts placed on {} pages",
                        best_placed,
                        best_pages
                    );

                    if let Some(ref callback) = improvement_callback
                        && let Err(e) = callback(result.clone())
                    {
                        log::warn!("Failed to send improvement callback: {}", e);
                    }
                }

                // Stop early if all parts placed AND either:
                // - only 1 page used (can't reduce further), or
                // - bins are well-utilized (packing is already dense)
                if result.parts_placed >= total_parts_requested
                    && (num_pages <= 1 || result.utilisation >= HIGH_DENSITY_THRESHOLD)
                {
                    log::info!(
                        "All {} parts placed on {} pages (utilisation {:.1}%), stopping optimization",
                        result.parts_placed,
                        num_pages,
                        result.utilisation * 100.0
                    );
                    let mut final_result = best_result.take().unwrap_or(result);
                    let final_solution = best_solution.as_ref().unwrap_or(&solution);
                    self.finalize_offcuts(
                        &mut final_result,
                        final_solution,
                        bin_width,
                        bin_height,
                        spacing,
                    );
                    return Ok(final_result);
                }

                if run_exceeded_time_limit {
                    log::info!("Stopping adaptive optimization due to run timeout");
                    should_stop_due_to_timeout = true;
                    break;
                }

                if self.is_cancelled() {
                    log::info!("Cancellation detected after completing run, stopping optimization");
                    cancelled = true;
                    break;
                }
            }

            if cancelled || self.is_cancelled() {
                log::info!("Cancellation detected after batch completion, stopping optimization");
                break 'outer;
            }

            if should_stop_due_to_timeout {
                log::info!("Stopping adaptive optimization due to run timeout");
                break 'outer;
            }

            if !improved_this_batch {
                batches_without_improvement += 1;
                // Once the best solution stops improving across whole batches the
                // search has converged — keep escalating once, then stop. Without
                // this the loop grinds through every remaining run / the 600s cap
                // with ever-larger (slower) parameters even though it already
                // found its best answer, which is what made max_fit "run forever".
                if batches_without_improvement >= MAX_BATCHES_WITHOUT_IMPROVEMENT {
                    log::info!(
                        "No improvement for {} consecutive batches, optimization converged \
                         ({} parts placed on {} pages), stopping",
                        batches_without_improvement,
                        best_placed,
                        best_pages
                    );
                    break 'outer;
                }
                loops = (loops + 2).min(MAX_ESCALATED_LOOPS);
                // Cap placements so total work per run stays bounded
                let max_placements = if effective_count > 50 {
                    (high_count_sample_budget / (effective_count * loops).max(1))
                        .clamp(1000, 500000)
                } else {
                    500000
                };
                placements = (placements * 2).min(max_placements);
                log::info!(
                    "No improvement after {} runs, increasing parameters: loops={}, placements={}",
                    MAX_RUNS_WITHOUT_IMPROVEMENT,
                    loops,
                    placements
                );
            } else {
                batches_without_improvement = 0;
            }
        }

        match best_result.take() {
            Some(mut best) => {
                if let Some(sol) = best_solution.as_ref() {
                    self.finalize_offcuts(&mut best, sol, bin_width, bin_height, spacing);
                }
                Ok(best)
            }
            None => Ok(NestingResult {
                combined_svg: Vec::new(),
                page_svgs: Vec::new(),
                parts_placed: 0,
                total_parts_requested,
                unplaced_parts_svg: None,
                utilisation: 0.0,
                pages: Vec::new(),
                sheets_total_estimate: None,
            }),
        }
    }
}
