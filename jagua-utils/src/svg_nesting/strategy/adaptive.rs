//! Adaptive nesting strategy that starts with lower parameters and adaptively increases them

use crate::svg_nesting::{
    parsing::{
        calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path, reverse_winding,
    },
    strategy::{NestingStrategy, PartInput},
    svg_generation::{PageResult, PlacedPartInfo, combine_svg_documents, NestingResult, post_process_svg_multi},
};
use anyhow::Result;
use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::entities::{Container, Item, Layout};
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::OriginalShape;
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::primitives::{Rect, SPolygon};
use jagua_rs::geometry::shape_modification::ShapeModifyMode;
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::{SvgDrawOptions, s_layout_to_svg};
use jagua_rs::probs::bpp::entities::{BPInstance, Bin, BPSolution};
use lbf::config::LBFConfig;
use lbf::opt::lbf_bpp::LBFOptimizerBP;
use rand::SeedableRng;
use rand::prelude::SmallRng;
use std::time::Instant;

/// Adaptive nesting strategy that starts with lower parameters and adaptively increases them
/// based on results. Sends intermediate improvements via callback.
pub struct AdaptiveNestingStrategy {
    /// Optional function to check if optimization should be cancelled
    cancellation_checker: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}

impl AdaptiveNestingStrategy {
    /// Create a new adaptive nesting strategy
    pub fn new() -> Self {
        Self {
            cancellation_checker: None,
        }
    }

    /// Create a new adaptive nesting strategy with cancellation checking
    pub fn with_cancellation_checker(
        cancellation_checker: Box<dyn Fn() -> bool + Send + Sync>,
    ) -> Self {
        Self {
            cancellation_checker: Some(cancellation_checker),
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
    fn calculate_bin_density(
        &self,
        solution: &BPSolution,
        bin_width: f32,
        bin_height: f32,
    ) -> f32 {
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
        (used_area / total_bin_area).min(1.0).max(0.0)
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
                cde_config: cde_config.clone(),
                poly_simpl_tolerance: Some(0.001),
                min_item_separation: Some(spacing),
                prng_seed: Some(seed),
                n_samples: placements,
                ls_frac: 0.2,
                narrow_concavity_cutoff_ratio: None,
                svg_draw_options: Default::default(),
            };

            let mut optimizer =
                LBFOptimizerBP::new(instance.clone(), lbf_config, SmallRng::seed_from_u64(seed));

            // Use solve_with_cancellation to allow early termination within the LBF optimizer
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

    /// Generate SVG from solution
    /// `item_id_to_holes` maps each item ID to its holes slice
    /// `item_id_to_part_idx` maps each item ID to its part index (index into the parts array)
    fn generate_svg_from_solution(
        &self,
        solution: &BPSolution,
        instance: &BPInstance,
        item_id_to_holes: &[&[Vec<jagua_rs::geometry::primitives::Point>]],
        item_id_to_part_idx: &[usize],
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

        for (page_index, (layout_key, layout_snapshot)) in layout_entries.iter().enumerate() {
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

            // Compute per-page utilisation using actual polygon area (matches SVG density)
            let page_util = layout_snapshot.density(instance);

            // Extract placement data from this layout
            let mut page_placements = Vec::new();
            for (_key, placed_item) in layout_snapshot.placed_items.iter() {
                let (x, y) = placed_item.d_transf.translation();
                let rotation = placed_item.d_transf.rotation().to_degrees();
                let item_id = placed_item.item_id;
                let part_index = item_id_to_part_idx.get(item_id).copied().unwrap_or(0);
                page_placements.push(PlacedPartInfo {
                    item_id,
                    part_index,
                    x,
                    y,
                    rotation,
                });
            }

            pages.push(PageResult {
                page_index,
                utilisation: page_util,
                svg_url: None,
                placements: page_placements,
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
            total_parts_requested: total_parts_requested,
            unplaced_parts_svg: None,
            utilisation,
            pages,
        })
    }
}

impl Default for AdaptiveNestingStrategy {
    fn default() -> Self {
        Self::new()
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
        // Parse all part SVGs and build per-part data
        struct ParsedPart {
            item_shape: OriginalShape,
            processed_holes: Vec<Vec<jagua_rs::geometry::primitives::Point>>,
            count: usize,
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

        let importer = Importer::new(cde_config.clone(), None, Some(spacing), None);

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

            let polygon = SPolygon::new(polygon_points)?;
            let centroid = polygon.centroid();
            let pre_transform = DTransformation::new(0.0, (-centroid.x(), -centroid.y()));

            let item_shape = OriginalShape {
                shape: polygon,
                pre_transform,
                modify_mode: ShapeModifyMode::Inflate,
                modify_config: importer.shape_modify_config,
            };

            parsed_parts.push(ParsedPart {
                item_shape,
                processed_holes,
                count: part.count,
            });
        }

        let total_parts_requested: usize = parsed_parts.iter().map(|p| p.count).sum();

        // Build container
        let bin_rect = Rect::try_new(0.0, 0.0, bin_width, bin_height)?;
        let bin_polygon = SPolygon::from(bin_rect);
        let container_shape = OriginalShape {
            shape: bin_polygon,
            pre_transform: DTransformation::empty(),
            modify_mode: ShapeModifyMode::Deflate,
            modify_config: importer.shape_modify_config,
        };

        let container_template = Container::new(0, container_shape, vec![], cde_config.clone())?;

        // Build rotation range
        const MAX_ROTATIONS: usize = 4;
        let rotation_count = if amount_of_rotations == 0 {
            0
        } else {
            amount_of_rotations.max(1).min(MAX_ROTATIONS)
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

        // Create items with consecutive IDs across all parts, tracking item_id → part_index
        let mut items = Vec::with_capacity(total_parts_requested);
        let mut item_id_to_part_idx: Vec<usize> = Vec::with_capacity(total_parts_requested);
        let mut item_id = 0;
        for (part_idx, parsed) in parsed_parts.iter().enumerate() {
            for _ in 0..parsed.count {
                let item = Item::new(
                    item_id,
                    parsed.item_shape.clone(),
                    rotation_range.clone(),
                    None,
                    cde_config.item_surrogate_config,
                )?;
                items.push((item, 1));
                item_id_to_part_idx.push(part_idx);
                item_id += 1;
            }
        }

        // Build per-item holes mapping (item_id → holes slice)
        let item_id_to_holes: Vec<&[Vec<jagua_rs::geometry::primitives::Point>]> =
            item_id_to_part_idx
                .iter()
                .map(|&part_idx| parsed_parts[part_idx].processed_holes.as_slice())
                .collect();

        // Stock = total_parts_requested ensures enough bins are available
        // (worst case: each item needs its own bin)
        let bin = Bin::new(container_template.clone(), total_parts_requested, 0);
        let instance = BPInstance::new(items, vec![bin]);

        // Adaptive optimization loop
        let optimization_start = Instant::now();
        let mut loops = 1;
        let mut placements = 10000;

        let mut best_result: Option<NestingResult> = None;
        let mut best_placed = 0;
        let mut best_pages = usize::MAX;
        let mut total_runs = 0;
        const MAX_TOTAL_RUNS: usize = 40;
        const MAX_RUNS_WITHOUT_IMPROVEMENT: usize = 10;
        const MAX_RUN_DURATION_SECONDS: u64 = 60;
        const MAX_TOTAL_OPTIMIZATION_SECONDS: u64 = 600;
        const HIGH_DENSITY_THRESHOLD: f32 = 0.50;

        'outer: loop {
            let elapsed_total = optimization_start.elapsed().as_secs();
            if elapsed_total >= MAX_TOTAL_OPTIMIZATION_SECONDS {
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

            if total_runs >= MAX_TOTAL_RUNS {
                log::info!("Reached maximum total runs ({}), stopping", MAX_TOTAL_RUNS);
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

                if total_runs >= MAX_TOTAL_RUNS {
                    break;
                }

                let run_start = Instant::now();
                total_runs += 1;

                log::info!(
                    "Run {}/{} (batch {}/{}): loops={}, placements={}, rotations={}",
                    total_runs,
                    MAX_TOTAL_RUNS,
                    batch_run + 1,
                    MAX_RUNS_WITHOUT_IMPROVEMENT,
                    loops,
                    placements,
                    amount_of_rotations
                );

                let optimization_result = self.run_single_optimization(
                    &instance,
                    &cde_config,
                    spacing,
                    loops,
                    placements,
                    total_runs,
                );

                let (_placed, solution) = match optimization_result {
                    Ok(r) => r,
                    Err(e) => {
                        log::warn!("No items could be placed: {}", e);
                        return Ok(NestingResult {
                            combined_svg: Vec::new(),
                            page_svgs: Vec::new(),
                            parts_placed: 0,
                            total_parts_requested,
                            unplaced_parts_svg: None,
                            utilisation: 0.0,
                            pages: Vec::new(),
                        });
                    }
                };

                let run_duration = run_start.elapsed();
                if run_duration.as_secs() > MAX_RUN_DURATION_SECONDS {
                    log::warn!(
                        "Run took {} seconds, exceeding limit of {} seconds. Stopping.",
                        run_duration.as_secs(),
                        MAX_RUN_DURATION_SECONDS
                    );
                    should_stop_due_to_timeout = true;
                    break;
                }

                let mut result = self.generate_svg_from_solution(
                    &solution,
                    &instance,
                    &item_id_to_holes,
                    &item_id_to_part_idx,
                    bin_width,
                    bin_height,
                    total_parts_requested,
                )?;

                // Handle unplaced parts SVG generation
                if result.parts_placed < total_parts_requested {
                    use jagua_rs::entities::Instance;
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
                    let mut svg_options = SvgDrawOptions::default();
                    svg_options.highlight_cd_shapes = false;
                    let label = if unplaced_count > items_to_render {
                        format!("Unplaced parts: {} (showing {})", unplaced_count, items_to_render)
                    } else {
                        format!("Unplaced parts: {}", unplaced_count)
                    };
                    let unplaced_svg_doc = s_layout_to_svg(
                        &unplaced_snapshot,
                        &instance,
                        svg_options,
                        &label,
                    );
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

                    log::info!(
                        "New best result: {} parts placed on {} pages",
                        best_placed,
                        best_pages
                    );

                    if let Some(ref callback) = improvement_callback {
                        if let Err(e) = callback(result.clone()) {
                            log::warn!("Failed to send improvement callback: {}", e);
                        }
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
                    return Ok(best_result.unwrap_or(result));
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
                loops += 1;
                placements = (placements * 2).min(200000);
                log::info!(
                    "No improvement after {} runs, increasing parameters: loops={}, placements={}",
                    MAX_RUNS_WITHOUT_IMPROVEMENT,
                    loops,
                    placements
                );
            }
        }

        Ok(best_result.unwrap_or_else(|| {
            NestingResult {
                combined_svg: Vec::new(),
                page_svgs: Vec::new(),
                parts_placed: 0,
                total_parts_requested,
                unplaced_parts_svg: None,
                utilisation: 0.0,
                pages: Vec::new(),
            }
        }))
    }
}

