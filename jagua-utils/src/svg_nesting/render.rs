//! Shared infrastructure for the deterministic fast-path packers (grid / periodic / pairing).
//!
//! These packers compute placements with closed-form geometry instead of running the LBF
//! optimizer, then render them through the *same* jagua-rs path the optimizer output uses
//! (`Layout` → `save` → `s_layout_to_svg` → `post_process_svg_multi`), so the resulting
//! `NestingResult` / `PageResult` / SVG is shape-identical to the General path.
//!
//! The parse + container build here mirrors `strategy/adaptive.rs` (`nest_inner`); it is kept
//! separate rather than refactored out of `adaptive.rs` so the working General path is not
//! disturbed.

use crate::svg_nesting::parsing::{
    build_inflatable_shape, calculate_signed_area, extract_path_from_svg_bytes, parse_svg_path,
    reverse_winding, sanitize_polygon,
};
use crate::svg_nesting::strategy::PartInput;
use crate::svg_nesting::svg_generation::{
    NestingResult, PageResult, PlacedPartInfo, combine_svg_documents, post_process_svg_multi,
};
use anyhow::Result;
use jagua_rs::collision_detection::CDEConfig;
use jagua_rs::entities::{Container, Instance, Item, Layout};
use jagua_rs::geometry::fail_fast::SPSurrogateConfig;
use jagua_rs::geometry::geo_enums::RotationRange;
use jagua_rs::geometry::primitives::{Point, Rect, SPolygon};
use jagua_rs::geometry::shape_modification::ShapeModifyMode;
use jagua_rs::geometry::{DTransformation, OriginalShape};
use jagua_rs::io::import::Importer;
use jagua_rs::io::svg::{SvgDrawOptions, s_layout_to_svg};
use jagua_rs::probs::bpp::entities::{BPInstance, Bin};

/// The CDE configuration shared by every nesting path (identical to `adaptive.rs::nest_inner`).
pub(crate) fn default_cde_config() -> CDEConfig {
    CDEConfig {
        quadtree_depth: 5,
        cd_threshold: 16,
        item_surrogate_config: SPSurrogateConfig {
            n_pole_limits: [(100, 0.0), (20, 0.75), (10, 0.90)],
            n_ff_poles: 2,
            n_ff_piers: 0,
        },
    }
}

/// Cheap geometric measurements of a part, used only for classification (no `Item`/`SPolygon`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PartMetrics {
    pub bbox_w: f32,
    pub bbox_h: f32,
    pub area: f32,
    /// Number of distinct vertices on the outer ring after sanitisation (a right triangle → 3).
    pub n_vertices: usize,
    pub n_holes: usize,
}

impl PartMetrics {
    /// `area / (bbox_w * bbox_h)` — 1.0 for a perfect rectangle, ~0.5 for a right triangle.
    pub fn rectangularity(&self) -> f32 {
        let bbox_area = (self.bbox_w * self.bbox_h).max(f32::MIN_POSITIVE);
        (self.area / bbox_area).clamp(0.0, 1.0)
    }
}

/// Parse just enough of a part's SVG (outer ring + holes) to measure bbox, area and vertex count.
/// Cheap: no `SPolygon::new`, no surrogate/`Item` construction.
pub(crate) fn measure_part(part: &PartInput) -> Result<PartMetrics> {
    let path_data = extract_path_from_svg_bytes(&part.svg_bytes)?;
    let (polygon_points, holes) = parse_svg_path(&path_data)?;
    let points = sanitize_polygon(polygon_points);
    let bbox = SPolygon::generate_bounding_box(&points);
    Ok(PartMetrics {
        bbox_w: bbox.width(),
        bbox_h: bbox.height(),
        area: calculate_signed_area(&points).abs(),
        n_vertices: dedup_vertex_count(&points),
        n_holes: holes.len(),
    })
}

/// Count distinct vertices, treating a repeated closing vertex as one (SVG `Z` often duplicates
/// the start point). Used to spot near-triangular outlines.
fn dedup_vertex_count(points: &[Point]) -> usize {
    let n = points.len();
    if n >= 2 {
        let first = points[0];
        let last = points[n - 1];
        if (first.x() - last.x()).abs() < 1e-6 && (first.y() - last.y()).abs() < 1e-6 {
            return n - 1;
        }
    }
    n
}

/// Full parse of one part type (private): the inflated shape + holes the renderer needs.
struct ParsedPart {
    item_shape: OriginalShape,
    holes: Vec<Vec<Point>>,
    item_id: Option<String>,
    bbox_w: f32,
    bbox_h: f32,
    /// Centroid minus bbox-min (input units), i.e. where the centroid sits inside the bbox. The
    /// renderer places parts by centroid, so the pairing packer uses this to align a part's bbox to
    /// a target cell.
    cx_off: f32,
    cy_off: f32,
}

fn parse_part(part: &PartInput, importer: &Importer) -> Result<ParsedPart> {
    let path_data = extract_path_from_svg_bytes(&part.svg_bytes)?;
    let (polygon_points, holes) = parse_svg_path(&path_data)?;

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
    let bbox: Rect = SPolygon::generate_bounding_box(&polygon_points);
    let bbox_w = bbox.width();
    let bbox_h = bbox.height();

    let polygon = SPolygon::new(polygon_points)?;
    let centroid = polygon.centroid();
    let cx_off = centroid.x() - bbox.x_min;
    let cy_off = centroid.y() - bbox.y_min;
    let pre_transform = DTransformation::new(0.0, (-centroid.x(), -centroid.y()));
    let item_shape = build_inflatable_shape(
        polygon,
        pre_transform,
        ShapeModifyMode::Inflate,
        importer.shape_modify_config,
    )?;

    Ok(ParsedPart {
        item_shape,
        holes: processed_holes,
        item_id: part.item_id.clone(),
        bbox_w,
        bbox_h,
        cx_off,
        cy_off,
    })
}

/// A prepared part type: the geometry a fast packer needs to compute placements.
pub(crate) struct PreparedPart {
    /// Axis-aligned bounding-box width of the outer outline (input units).
    pub bbox_w: f32,
    /// Axis-aligned bounding-box height of the outer outline (input units).
    pub bbox_h: f32,
    /// Centroid X relative to bbox-min (the renderer places by centroid).
    pub cx_off: f32,
    /// Centroid Y relative to bbox-min.
    pub cy_off: f32,
}

/// A single placement computed by a fast packer: which part type, and its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Placement {
    /// Index into the prepared-parts slice.
    pub part_idx: usize,
    /// Rotation in radians applied about the (centred) shape origin.
    pub rotation: f32,
    /// Target translation of the shape centroid, in bin coordinates.
    pub x: f32,
    pub y: f32,
}

/// Everything the renderer needs to turn pages of [`Placement`]s into a `NestingResult`.
pub(crate) struct RenderContext {
    container: Container,
    instance: BPInstance,
    /// Per-part-index holes, for `post_process_svg_multi`.
    holes: Vec<Vec<Vec<Point>>>,
    /// Per-part-index user-provided ids (echoed into placements).
    item_ids: Vec<Option<String>>,
    bin_width: f32,
    bin_height: f32,
}

/// Parse all parts and build the container + `BPInstance` in one pass (single shared importer).
///
/// Returns the prepared parts (for the packer's placement math) and the [`RenderContext`] used to
/// render the computed placements. `rotation_ranges[i]` is part `i`'s rotation range (fast paths
/// pass cardinal ranges). `bin_stock` only needs to be ≥ the page count drawn.
pub(crate) fn prepare(
    parts: &[PartInput],
    rotation_ranges: &[RotationRange],
    bin_width: f32,
    bin_height: f32,
    bin_stock: usize,
) -> Result<(Vec<PreparedPart>, RenderContext)> {
    let cde_config = default_cde_config();
    // No min_item_separation: parts are not inflated and the container is not deflated; the fast
    // packers space placements by adding `spacing` to their pitch directly.
    let importer = Importer::new(cde_config, None, None, None);

    let parsed = parts
        .iter()
        .map(|p| parse_part(p, &importer))
        .collect::<Result<Vec<_>>>()?;

    let bin_rect = Rect::try_new(0.0, 0.0, bin_width, bin_height)?;
    let bin_polygon = SPolygon::from(bin_rect);
    let container_shape = OriginalShape {
        shape: bin_polygon,
        pre_transform: DTransformation::empty(),
        modify_mode: ShapeModifyMode::Deflate,
        modify_config: importer.shape_modify_config,
    };
    let container = Container::new(0, container_shape, vec![], cde_config)?;

    let mut items: Vec<(Item, usize)> = Vec::with_capacity(parsed.len());
    let mut holes: Vec<Vec<Vec<Point>>> = Vec::with_capacity(parsed.len());
    let mut item_ids: Vec<Option<String>> = Vec::with_capacity(parsed.len());
    let mut prepared: Vec<PreparedPart> = Vec::with_capacity(parsed.len());
    for (idx, p) in parsed.into_iter().enumerate() {
        let item = Item::new(
            idx,
            p.item_shape,
            rotation_ranges[idx].clone(),
            None,
            cde_config.item_surrogate_config,
        )?;
        items.push((item, 1));
        holes.push(p.holes);
        item_ids.push(p.item_id);
        prepared.push(PreparedPart {
            bbox_w: p.bbox_w,
            bbox_h: p.bbox_h,
            cx_off: p.cx_off,
            cy_off: p.cy_off,
        });
    }

    let bin = Bin::new(container.clone(), bin_stock.max(1), 0);
    let instance = BPInstance::new(items, vec![bin]);

    Ok((
        prepared,
        RenderContext {
            container,
            instance,
            holes,
            item_ids,
            bin_width,
            bin_height,
        },
    ))
}

/// Render one page worth of placements into a post-processed SVG string + its `PlacedPartInfo`s.
fn render_one_page(
    ctx: &RenderContext,
    placements: &[Placement],
    page_index: usize,
) -> (String, Vec<PlacedPartInfo>, f32) {
    let mut layout = Layout::new(ctx.container.clone());
    for pl in placements {
        let item = ctx.instance.item(pl.part_idx);
        let d_transf = DTransformation::new(pl.rotation, (pl.x, pl.y));
        layout.place_item(item, d_transf);
    }
    let snapshot = layout.save();

    let svg_doc = s_layout_to_svg(
        &snapshot,
        &ctx.instance,
        SvgDrawOptions::default(),
        &format!("Page {} - {} items", page_index, placements.len()),
    );
    let holes_refs: Vec<&[Vec<Point>]> = ctx.holes.iter().map(|h| h.as_slice()).collect();
    let processed = post_process_svg_multi(&svg_doc.to_string(), &holes_refs);

    let mut infos = Vec::with_capacity(placements.len());
    for pl in placements {
        let centroid = ctx.instance.item(pl.part_idx).shape_orig.centroid();
        let item_id = ctx.item_ids[pl.part_idx]
            .clone()
            .unwrap_or_else(|| pl.part_idx.to_string());
        infos.push(PlacedPartInfo {
            item_id,
            part_index: pl.part_idx,
            x: pl.x,
            y: pl.y,
            rotation: pl.rotation.to_degrees(),
            centroid_x: centroid.x(),
            centroid_y: centroid.y(),
        });
    }

    let utilisation = snapshot.density(&ctx.instance);
    (processed, infos, utilisation)
}

/// Assemble a `NestingResult` for a periodic packing: `full_sheets` byte-identical copies of the
/// `stencil` page plus an optional `remainder` page. The stencil is rendered **once** and its SVG
/// string + placements are cloned for every full sheet — this is what makes bulk runs fast and the
/// full sheets visually identical (the core fix for "K different sheets").
pub(crate) fn render_periodic(
    ctx: &RenderContext,
    stencil: &[Placement],
    full_sheets: usize,
    remainder: &[Placement],
    total_parts_requested: usize,
) -> NestingResult {
    let mut page_svg_strings: Vec<String> = Vec::new();
    let mut page_svgs: Vec<Vec<u8>> = Vec::new();
    let mut page_results: Vec<PageResult> = Vec::new();
    let mut total_placed = 0;

    if full_sheets > 0 && !stencil.is_empty() {
        let (svg, infos, util) = render_one_page(ctx, stencil, 0);
        for i in 0..full_sheets {
            page_svg_strings.push(svg.clone());
            page_svgs.push(svg.clone().into_bytes());
            total_placed += infos.len();
            page_results.push(PageResult {
                page_index: i,
                utilisation: util,
                svg_url: None,
                parts_placed: infos.len(),
                placements: infos.clone(),
                offcuts: Vec::new(),
            });
        }
    }

    if !remainder.is_empty() {
        let idx = page_results.len();
        let (svg, infos, util) = render_one_page(ctx, remainder, idx);
        total_placed += infos.len();
        page_svg_strings.push(svg.clone());
        page_svgs.push(svg.into_bytes());
        page_results.push(PageResult {
            page_index: idx,
            utilisation: util,
            svg_url: None,
            parts_placed: infos.len(),
            placements: infos,
            offcuts: Vec::new(),
        });
    }

    let combined_svg =
        combine_svg_documents(&page_svg_strings, ctx.bin_width, ctx.bin_height).into_bytes();
    let utilisation = if page_results.is_empty() {
        0.0
    } else {
        page_results.iter().map(|p| p.utilisation).sum::<f32>() / page_results.len() as f32
    };

    NestingResult {
        combined_svg,
        page_svgs,
        parts_placed: total_placed,
        total_parts_requested,
        unplaced_parts_svg: None,
        utilisation,
        sheets_total_estimate: Some(page_results.len()),
        pages: page_results,
    }
}

/// Render an explicit list of pages (each its own `Vec<Placement>`), used by the mixed-parts
/// packer. Consecutive byte-identical pages are rendered once and cloned, so repeated full sheets
/// stay cheap and visually identical.
pub(crate) fn render_page_list(
    ctx: &RenderContext,
    pages: &[Vec<Placement>],
    total_parts_requested: usize,
) -> NestingResult {
    let mut page_svg_strings: Vec<String> = Vec::with_capacity(pages.len());
    let mut page_svgs: Vec<Vec<u8>> = Vec::with_capacity(pages.len());
    let mut page_results: Vec<PageResult> = Vec::with_capacity(pages.len());
    let mut total_placed = 0;

    let mut cached: Option<(Vec<Placement>, String, Vec<PlacedPartInfo>, f32)> = None;
    for (idx, placements) in pages.iter().enumerate() {
        let (svg, infos, util) = match &cached {
            Some((prev, svg, infos, util)) if prev == placements => {
                (svg.clone(), infos.clone(), *util)
            }
            _ => {
                let rendered = render_one_page(ctx, placements, idx);
                cached = Some((
                    placements.clone(),
                    rendered.0.clone(),
                    rendered.1.clone(),
                    rendered.2,
                ));
                rendered
            }
        };
        total_placed += infos.len();
        page_svg_strings.push(svg.clone());
        page_svgs.push(svg.into_bytes());
        page_results.push(PageResult {
            page_index: idx,
            utilisation: util,
            svg_url: None,
            parts_placed: infos.len(),
            placements: infos,
            offcuts: Vec::new(),
        });
    }

    let combined_svg =
        combine_svg_documents(&page_svg_strings, ctx.bin_width, ctx.bin_height).into_bytes();
    let utilisation = if page_results.is_empty() {
        0.0
    } else {
        page_results.iter().map(|p| p.utilisation).sum::<f32>() / page_results.len() as f32
    };

    NestingResult {
        combined_svg,
        page_svgs,
        parts_placed: total_placed,
        total_parts_requested,
        unplaced_parts_svg: None,
        utilisation,
        sheets_total_estimate: Some(page_results.len()),
        pages: page_results,
    }
}
