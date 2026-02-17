//! Simple single-run nesting strategy

use crate::svg_nesting::{
    parsing::{
        calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path, reverse_winding,
    },
    strategy::{NestingStrategy, PartInput, is_single_part_type},
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
use jagua_rs::probs::bpp::entities::{BPInstance, Bin};
use lbf::config::LBFConfig;
use lbf::opt::lbf_bpp::LBFOptimizerBP;
use rand::SeedableRng;
use rand::prelude::SmallRng;

/// Simple nesting strategy that runs the optimizer once with default parameters
pub struct SimpleNestingStrategy;

impl SimpleNestingStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SimpleNestingStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl NestingStrategy for SimpleNestingStrategy {
    fn nest(
        &self,
        bin_width: f32,
        bin_height: f32,
        spacing: f32,
        parts: &[PartInput],
        amount_of_rotations: usize,
        _improvement_callback: Option<crate::svg_nesting::strategy::ImprovementCallback>,
    ) -> Result<NestingResult> {
        // Parse all part SVGs
        struct ParsedPart {
            item_shape: OriginalShape,
            processed_holes: Vec<Vec<jagua_rs::geometry::primitives::Point>>,
            count: usize,
            item_id: Option<String>,
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

        let importer = Importer::new(cde_config.clone(), Some(0.001), Some(spacing), None);

        let mut parsed_parts = Vec::with_capacity(parts.len());
        for (part_idx, part) in parts.iter().enumerate() {
            let path_data = extract_path_from_svg_bytes(&part.svg_bytes)?;
            let (polygon_points, holes) = parse_svg_path(&path_data)?;

            log::debug!(
                "Parsed SVG path for part {} (simple): {} outer boundary points, {} holes",
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
            for (i, hole) in holes.iter().enumerate() {
                let hole_area = calculate_signed_area(hole);
                let processed_hole = if hole_area > 0.0 {
                    log::debug!(
                        "  Reversing hole {} (was counter-clockwise, area: {})",
                        i,
                        hole_area
                    );
                    reverse_winding(hole)
                } else {
                    log::debug!("  Hole {} is clockwise (area: {})", i, hole_area);
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
                item_id: part.item_id.clone(),
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

        // Create items with consecutive IDs across all parts
        let mut items = Vec::with_capacity(total_parts_requested);
        let mut item_id_to_part_idx: Vec<usize> = Vec::with_capacity(total_parts_requested);
        let mut item_id_to_part_id: Vec<Option<String>> = Vec::with_capacity(total_parts_requested);
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
                item_id_to_part_id.push(parsed.item_id.clone());
                item_id += 1;
            }
        }

        // Build per-item holes mapping
        let item_id_to_holes: Vec<&[Vec<jagua_rs::geometry::primitives::Point>]> =
            item_id_to_part_idx
                .iter()
                .map(|&part_idx| parsed_parts[part_idx].processed_holes.as_slice())
                .collect();

        // Stock = total_parts_requested ensures enough bins are available
        let bin = Bin::new(container_template.clone(), total_parts_requested, 0);
        let instance = BPInstance::new(items, vec![bin]);

        // Run optimizer with multiple seeds
        let mut best_solution = None;
        let mut best_placed = 0;

        for seed in 0..10 {
            let lbf_config = LBFConfig {
                cde_config: cde_config.clone(),
                poly_simpl_tolerance: Some(0.001),
                min_item_separation: Some(spacing),
                prng_seed: Some(seed),
                n_samples: 200000,
                ls_frac: 0.2,
                narrow_concavity_cutoff_ratio: None,
                svg_draw_options: Default::default(),
            };

            let mut optimizer =
                LBFOptimizerBP::new(instance.clone(), lbf_config, SmallRng::seed_from_u64(seed));
            let solution = optimizer.solve();

            let placed: usize = solution
                .layout_snapshots
                .values()
                .map(|ls| ls.placed_items.len())
                .sum();

            if placed > best_placed {
                best_placed = placed;
                best_solution = Some(solution);
            }

            if placed >= total_parts_requested {
                break;
            }
        }

        let solution = best_solution.expect("At least one optimization run should succeed");

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
        let skip_middle = is_single_part_type(&item_id_to_part_idx) && num_pages >= 3;

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
                    &instance,
                    svg_options,
                    &format!("Layout {:?} - {} items", layout_key, total_items_placed),
                );
                let svg_str = svg_doc.to_string();
                let processed_svg = post_process_svg_multi(&svg_str, &item_id_to_holes);
                page_svg_strings.push(processed_svg.clone());
                page_svgs.push(processed_svg.into_bytes());
            }

            // Compute per-page utilisation using actual polygon area (matches SVG density)
            let page_util = layout_snapshot.density(&instance);

            // Always extract real placement data (cheap, and item_id values differ per page)
            let mut page_placements = Vec::new();
            for (_key, placed_item) in layout_snapshot.placed_items.iter() {
                let (x, y) = placed_item.d_transf.translation();
                let rotation = placed_item.d_transf.rotation().to_degrees();
                let internal_id = placed_item.item_id;
                let part_index = item_id_to_part_idx.get(internal_id).copied().unwrap_or(0);
                let item_id = item_id_to_part_id.get(internal_id)
                    .cloned()
                    .flatten()
                    .unwrap_or_else(|| internal_id.to_string());
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
                parts_placed: page_placements.len(),
                placements: page_placements,
            });
        }

        let combined_svg = combine_svg_documents(&page_svg_strings, bin_width, bin_height);

        use regex::Regex;
        let re_item_use = Regex::new(r##"<use[^>]*href=["']#item_\d+["']"##).unwrap();
        let items_in_svg = re_item_use.find_iter(&combined_svg).count();
        let corrected_count = items_in_svg;

        if corrected_count != total_items_placed {
            log::warn!(
                "Count mismatch detected: SVG contains {} item <use> tags, but optimizer reports {}",
                corrected_count,
                total_items_placed
            );
        }

        // Generate unplaced parts SVG
        let unplaced_count = total_parts_requested.saturating_sub(corrected_count);
        let unplaced_parts_svg = if unplaced_count > 0 {
            use jagua_rs::entities::Instance;

            let mut unplaced_layout = Layout::new(container_template.clone());

            let first_shape = &parsed_parts[0].item_shape;
            let part_bbox = &first_shape.shape.bbox;
            let part_width = part_bbox.width();
            let part_height = part_bbox.height();

            let cols = ((bin_width - spacing) / (part_width + spacing)).floor().max(1.0) as usize;
            let rows = ((unplaced_count as f32 / cols as f32).ceil()) as usize;

            let total_grid_width = (cols as f32 * part_width) + ((cols.saturating_sub(1)) as f32 * spacing);
            let total_grid_height = (rows as f32 * part_height) + ((rows.saturating_sub(1)) as f32 * spacing);
            let offset_x = (bin_width - total_grid_width) / 2.0;
            let offset_y = (bin_height - total_grid_height) / 2.0;

            let item_template = instance.item(0);

            for i in 0..unplaced_count {
                let row = i / cols;
                let col = i % cols;
                let grid_x = offset_x + (col as f32 * (part_width + spacing)) + part_width / 2.0;
                let grid_y = offset_y + (row as f32 * (part_height + spacing)) + part_height / 2.0;
                let d_transf = DTransformation::new(0.0, (grid_x, grid_y));
                unplaced_layout.place_item(item_template, d_transf);
            }

            let unplaced_snapshot = unplaced_layout.save();
            let mut svg_options = SvgDrawOptions::default();
            svg_options.highlight_cd_shapes = false;
            let unplaced_svg_doc = s_layout_to_svg(
                &unplaced_snapshot,
                &instance,
                svg_options,
                &format!("Unplaced parts: {}", unplaced_count),
            );
            let unplaced_svg_str = unplaced_svg_doc.to_string();
            let processed_unplaced_svg = post_process_svg_multi(&unplaced_svg_str, &item_id_to_holes);
            Some(processed_unplaced_svg.into_bytes())
        } else {
            None
        };

        // Calculate average bin utilisation across all pages
        let utilisation = if pages.is_empty() {
            0.0
        } else {
            pages.iter().map(|p| p.utilisation).sum::<f32>() / pages.len() as f32
        };

        Ok(NestingResult {
            combined_svg: combined_svg.into_bytes(),
            page_svgs,
            parts_placed: corrected_count,
            total_parts_requested,
            unplaced_parts_svg,
            utilisation,
            pages,
        })
    }
}
