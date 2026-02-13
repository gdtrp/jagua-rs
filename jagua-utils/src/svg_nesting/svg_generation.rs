//! SVG generation and post-processing utilities

use anyhow::Result;
use jagua_rs::geometry::primitives::{Point, SPolygon};
use jagua_rs::geometry::DTransformation;
use jagua_rs::geometry::geo_traits::Transformable;
use jagua_rs::geometry::OriginalShape;
use serde::{Deserialize, Serialize};

/// Per-item placement data describing where a part was placed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlacedPartInfo {
    /// Internal item ID assigned by the optimizer.
    pub item_id: usize,
    /// Which PartInput this came from (0-based index into the parts array).
    pub part_index: usize,
    /// Translation X in bin coordinates.
    pub x: f32,
    /// Translation Y in bin coordinates.
    pub y: f32,
    /// Rotation in degrees.
    pub rotation: f32,
}

/// Per-page result grouping utilisation, placements, and optional SVG URL for a single sheet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageResult {
    /// Page index (0-based).
    pub page_index: usize,
    /// Utilisation ratio (0.0 to 1.0) for this page.
    pub utilisation: f32,
    /// S3 URL to the SVG for this page (populated by the SQS processor).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub svg_url: Option<String>,
    /// Placements on this page.
    pub placements: Vec<PlacedPartInfo>,
}

/// Result data returned after nesting SVG parts.
#[derive(Clone, Debug)]
pub struct NestingResult {
    /// Combined SVG of all pages as bytes.
    pub combined_svg: Vec<u8>,
    /// Individual page SVGs (ordered by container id).
    pub page_svgs: Vec<Vec<u8>>,
    /// Number of parts placed.
    pub parts_placed: usize,
    /// Total number of parts requested (original amount_of_parts).
    pub total_parts_requested: usize,
    /// SVG for unplaced parts (if any), showing remaining parts in a grid layout.
    pub unplaced_parts_svg: Option<Vec<u8>>,
    /// Average bin utilisation ratio (0.0 to 1.0) across all pages.
    pub utilisation: f32,
    /// Per-page results: utilisation and placements grouped by page.
    pub pages: Vec<PageResult>,
}

/// Converts points to SVG path data
pub fn points_to_svg_path(points: &[Point]) -> String {
    if points.is_empty() {
        return String::new();
    }
    let mut path = format!("M {},{}", points[0].0, points[0].1);
    for point in points.iter().skip(1) {
        path.push_str(&format!(" L {},{}", point.0, point.1));
    }
    path.push_str(" z");
    path
}

/// Post-processes SVG to add holes to items and adjust colors
/// - Adds holes to each item's path (with opposite winding direction)
/// - Changes item fill color to white
/// - Changes stroke color to gray
/// - Makes holes transparent
/// Note: Holes should be in the original coordinate system (same as outer boundary in item definition)
pub fn post_process_svg(svg_str: &str, holes: &[Vec<Point>]) -> String {
    use regex::Regex;

    let mut result = svg_str.to_string();

    // Change item fill color to white and remove fill-opacity (make fully opaque)
    let re_fill = Regex::new(r##"fill="#FFC879""##).unwrap();
    result = re_fill
        .replace_all(&result, r##"fill="white""##)
        .to_string();

    // Remove fill-opacity from item paths (make fully opaque white)
    let re_fill_opacity = Regex::new(
        r##"(<g id="item_\d+">\s*<path[^>]*fill="white")[^>]*fill-opacity="[^"]*"([^>]*/>)"##,
    )
    .unwrap();
    result = re_fill_opacity
        .replace_all(&result, r##"${1}${2}"##)
        .to_string();

    // Change stroke color to gray for item paths
    // Match stroke="black" in item paths
    let re_stroke = Regex::new(r##"(<g id="item_\d+">\s*<path[^>]*stroke=")black(")"##).unwrap();
    result = re_stroke
        .replace_all(&result, r##"${1}gray${2}"##)
        .to_string();

    // Make container/bin transparent (remove fill or set to transparent)
    // Match: <g id="container_0"><path d="..." fill="#CC824A" ... />
    let re_container_fill =
        Regex::new(r##"(<g id="container_\d+">\s*<path[^>]*fill=")#CC824A(")"##).unwrap();
    result = re_container_fill
        .replace_all(&result, r##"${1}transparent${2}"##)
        .to_string();

    // If no holes, just return with color change
    if holes.is_empty() {
        return result;
    }

    // Note: The SVG generator applies the inverse of pre_transform to the original shape,
    // so the outer boundary in the item definition is in the original coordinate system.
    // Therefore, holes should also be in the original coordinate system (no transformation needed).

    // For each item definition, add holes to the path
    // Match: <g id="item_N"><path d="PATH_DATA" ... />
    let re_item = Regex::new(r##"(<g id="item_\d+">\s*<path d=")([^"]+)(" [^>]*/>)"##).unwrap();

    let mut matches_found = 0;
    result = re_item
        .replace_all(&result, |caps: &regex::Captures| -> String {
            matches_found += 1;
            let item_start = caps.get(1).unwrap().as_str();
            let outer_path = caps.get(2).unwrap().as_str();
            let item_end = caps.get(3).unwrap().as_str();

            // Build the combined path with outer boundary and holes
            let mut combined_path = outer_path.to_string();

            // Add holes with opposite winding direction (they'll be cut out)
            // Holes are in the original coordinate system, same as the outer boundary
            for (i, hole) in holes.iter().enumerate() {
                let hole_path = points_to_svg_path(hole);
                combined_path.push_str(&format!(" {}", hole_path));
                log::debug!("  Added hole {} to item path ({} points)", i, hole.len());
            }

            format!("{}{}{}", item_start, combined_path, item_end)
        })
        .to_string();

    log::debug!("Added holes to {} item definitions", matches_found);

    result
}

/// Generates an SVG showing unplaced parts arranged in a simple grid layout
///
/// # Arguments
/// * `unplaced_count` - Number of unplaced parts to show
/// * `item_shape_orig` - The OriginalShape of a single part (same as used for placed parts)
/// * `_item_shape_cd` - The collision detection shape (internal shape after modifications) - currently unused
/// * `bin_width` - Width of the bin
/// * `bin_height` - Height of the bin
/// * `spacing` - Spacing between parts
/// * `holes` - Holes to add to each part (for post-processing)
///
/// # Returns
/// SVG string with unplaced parts arranged in a grid
pub fn generate_unplaced_parts_svg(
    unplaced_count: usize,
    item_shape_orig: &OriginalShape,
    _item_shape_cd: &SPolygon,
    bin_width: f32,
    bin_height: f32,
    spacing: f32,
    holes: &[Vec<Point>],
) -> Result<String> {
    if unplaced_count == 0 {
        // Return empty bin SVG if no unplaced parts
        return Ok(format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {} {}">
  <g id="container_0">
    <path d="M 0,0 L {},0 L {},{} L 0,{} z" fill="transparent" stroke="gray" stroke-width="1"/>
  </g>
  <text x="{}" y="{}" font-size="{}" font-family="monospace">Unplaced parts: 0</text>
</svg>"#,
            bin_width,
            bin_height,
            bin_width,
            bin_width,
            bin_height,
            bin_height,
            bin_width * 0.02,
            bin_height * 0.05,
            bin_width * 0.02
        ));
    }

    // Calculate grid layout using the original shape bbox
    let part_bbox = &item_shape_orig.shape.bbox;
    let part_width = part_bbox.width();
    let part_height = part_bbox.height();
    
    // Calculate how many parts fit per row/column
    let cols = ((bin_width - spacing) / (part_width + spacing)).floor().max(1.0) as usize;
    let rows = ((unplaced_count as f32 / cols as f32).ceil()) as usize;
    
    // Adjust spacing to center the grid
    let total_grid_width = (cols as f32 * part_width) + ((cols.saturating_sub(1)) as f32 * spacing);
    let total_grid_height = (rows as f32 * part_height) + ((rows.saturating_sub(1)) as f32 * spacing);
    let offset_x = (bin_width - total_grid_width) / 2.0;
    let offset_y = (bin_height - total_grid_height) / 2.0;

    // Calculate stroke width similar to placed parts (matching s_layout_to_svg)
    let vbox_width = bin_width;
    let vbox_height = bin_height;
    let stroke_width = f32::min(vbox_width, vbox_height) * 0.001 * 2.2; // Match the multiplier used in s_layout_to_svg

    // Build SVG with item definitions matching the structure from s_layout_to_svg
    // Apply inverse pre_transform to center the shape at origin (like s_layout_to_svg does)
    // The item definition should be centered at origin, then we translate to grid positions
    let pre_transform = item_shape_orig.pre_transform.compose();
    let inverse_pre_transform = pre_transform.inverse();
    let centered_shape = item_shape_orig.shape.transform_clone(&inverse_pre_transform);
    let item_path_data = polygon_to_svg_path(&centered_shape);
    
    // Calculate viewBox similar to s_layout_to_svg (with 10% padding)
    let viewbox_padding = 0.1;
    let viewbox_x = -bin_width * viewbox_padding;
    let viewbox_y = -bin_height * viewbox_padding;
    let viewbox_width = bin_width * (1.0 + 2.0 * viewbox_padding);
    let viewbox_height = bin_height * (1.0 + 2.0 * viewbox_padding);
    
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str(&format!("<svg viewBox=\"{} {} {} {}\" xmlns=\"http://www.w3.org/2000/svg\">\n", viewbox_x, viewbox_y, viewbox_width, viewbox_height));
    
    // Container group (matching s_layout_to_svg structure)
    svg.push_str("  <g id=\"container_0\">\n");
    svg.push_str(&format!("    <path d=\"M 0,0 L {},0 L {},{} L 0,{} z\" fill=\"transparent\" stroke=\"black\" stroke-width=\"{}\"/>\n", bin_width, bin_width, bin_height, bin_height, stroke_width * 2.0));
    svg.push_str("  </g>\n");
    
    // Items group (matching s_layout_to_svg structure)
    svg.push_str("  <g id=\"items\">\n");
    svg.push_str("    <defs>\n");
    svg.push_str("      <g id=\"item_0\">\n");
    svg.push_str(&format!("        <path d=\"{}\" fill=\"#FFC879\" stroke-width=\"{}\" fill-rule=\"nonzero\" stroke=\"black\" fill-opacity=\"0.5\"/>\n", item_path_data, stroke_width));
    svg.push_str("      </g>\n");
    svg.push_str("    </defs>\n");

    // Add each unplaced part in grid layout using <use> elements (like placed parts)
    // The item definition is now centered at origin (0,0), so we just translate to grid positions
    for i in 0..unplaced_count {
        let row = i / cols;
        let col = i % cols;
        
        // Calculate grid position (center of grid cell) in bin coordinates
        let grid_x = offset_x + (col as f32 * (part_width + spacing)) + part_width / 2.0;
        let grid_y = offset_y + (row as f32 * (part_height + spacing)) + part_height / 2.0;
        
        // Since the item definition is centered at origin, we just translate to grid position
        svg.push_str(&format!("    <use href=\"#item_0\" transform=\"translate({} {})\"/>\n", grid_x, grid_y));
    }
    
    svg.push_str("  </g>\n");
    
    // Add empty groups to match structure
    svg.push_str("  <g id=\"quality_zones\"/>\n");
    svg.push_str("  <g id=\"optionals\">\n");
    svg.push_str("    <g id=\"highlight_cd_shapes\"/>\n");
    svg.push_str("    <g id=\"surrogates\">\n");
    svg.push_str("      <defs/>\n");
    svg.push_str("    </g>\n");
    svg.push_str("    <g id=\"collision_lines\"/>\n");
    svg.push_str("  </g>\n");
    
    // Add label text (matching s_layout_to_svg)
    svg.push_str(&format!("  <text font-family=\"monospace\" font-size=\"{}\" font-weight=\"500\" x=\"0\" y=\"-15\">Unplaced parts: {}</text>\n", bin_width * 0.025, unplaced_count));

    svg.push_str("</svg>");

    // Apply post-processing (add holes, adjust colors)
    let processed_svg = post_process_svg(&svg, holes);

    Ok(processed_svg)
}

/// Post-processes SVG with per-item holes based on item ID
/// Each item gets holes from the corresponding entry in `item_id_to_holes`.
/// Items whose ID is out of range get no holes.
pub fn post_process_svg_multi(svg_str: &str, item_id_to_holes: &[&[Vec<Point>]]) -> String {
    use regex::Regex;

    let mut result = svg_str.to_string();

    // Change item fill color to white and remove fill-opacity (make fully opaque)
    let re_fill = Regex::new(r##"fill="#FFC879""##).unwrap();
    result = re_fill
        .replace_all(&result, r##"fill="white""##)
        .to_string();

    // Remove fill-opacity from item paths (make fully opaque white)
    let re_fill_opacity = Regex::new(
        r##"(<g id="item_\d+">\s*<path[^>]*fill="white")[^>]*fill-opacity="[^"]*"([^>]*/>)"##,
    )
    .unwrap();
    result = re_fill_opacity
        .replace_all(&result, r##"${1}${2}"##)
        .to_string();

    // Change stroke color to gray for item paths
    let re_stroke = Regex::new(r##"(<g id="item_\d+">\s*<path[^>]*stroke=")black(")"##).unwrap();
    result = re_stroke
        .replace_all(&result, r##"${1}gray${2}"##)
        .to_string();

    // Make container/bin transparent
    let re_container_fill =
        Regex::new(r##"(<g id="container_\d+">\s*<path[^>]*fill=")#CC824A(")"##).unwrap();
    result = re_container_fill
        .replace_all(&result, r##"${1}transparent${2}"##)
        .to_string();

    // For each item definition, add the correct holes based on item ID
    let re_item =
        Regex::new(r##"(<g id="item_(\d+)">\s*<path d=")([^"]+)(" [^>]*/>)"##).unwrap();

    result = re_item
        .replace_all(&result, |caps: &regex::Captures| -> String {
            let item_start = caps.get(1).unwrap().as_str();
            let item_id: usize = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
            let outer_path = caps.get(3).unwrap().as_str();
            let item_end = caps.get(4).unwrap().as_str();

            // Look up holes for this item ID
            let holes = item_id_to_holes.get(item_id).copied().unwrap_or(&[]);

            if holes.is_empty() {
                return format!("{}{}{}", item_start, outer_path, item_end);
            }

            let mut combined_path = outer_path.to_string();
            for hole in holes {
                let hole_path = points_to_svg_path(hole);
                combined_path.push_str(&format!(" {}", hole_path));
            }

            format!("{}{}{}", item_start, combined_path, item_end)
        })
        .to_string();

    result
}

/// Extracts the inner content from an SVG document (everything inside <svg>...</svg>)
/// Removes XML declaration and root <svg> tags, returning just the content
fn extract_svg_inner_content(svg_str: &str) -> String {
    use regex::Regex;
    
    // Remove XML declaration if present
    let re_xml_decl = Regex::new(r##"<\?xml[^>]*\?>\s*"##).unwrap();
    let mut content = re_xml_decl.replace_all(svg_str, "").to_string();
    
    // Extract content between <svg> and </svg> tags
    // Match opening <svg> tag (with all attributes) and capture everything until closing </svg>
    let re_svg_content = Regex::new(r##"(?s)<svg[^>]*>(.*)</svg>"##).unwrap();
    if let Some(caps) = re_svg_content.captures(&content) {
        content = caps.get(1).unwrap().as_str().to_string();
    }
    
    content
}

/// Combines multiple SVG documents into a single valid SVG document.
/// Pages are stacked vertically with a gap between them.
pub fn combine_svg_documents(svg_documents: &[String], bin_width: f32, bin_height: f32) -> String {
    let num_pages = svg_documents.len();
    let gap = bin_height * 0.02; // small gap between pages
    let total_height = num_pages as f32 * bin_height + (num_pages.saturating_sub(1)) as f32 * gap;

    let mut combined = String::new();
    combined.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    combined.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\">\n",
        bin_width, total_height
    ));

    for (i, svg_doc) in svg_documents.iter().enumerate() {
        let y_offset = i as f32 * (bin_height + gap);
        let inner_content = extract_svg_inner_content(svg_doc);
        combined.push_str(&format!(
            "<g transform=\"translate(0,{})\">\n{}\n</g>\n",
            y_offset, inner_content
        ));
    }

    combined.push_str("</svg>");
    combined
}

/// Converts an SPolygon to SVG path data
fn polygon_to_svg_path(polygon: &SPolygon) -> String {
    let vertices = &polygon.vertices;
    if vertices.is_empty() {
        return String::new();
    }
    
    let mut path = format!("M {},{}", vertices[0].0, vertices[0].1);
    for vertex in vertices.iter().skip(1) {
        path.push_str(&format!(" L {},{}", vertex.0, vertex.1));
    }
    path.push_str(" z");
    path
}
