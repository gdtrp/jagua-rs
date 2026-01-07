//! SVG parsing utilities for extracting geometry from SVG files

use anyhow::Result;
use jagua_rs::geometry::primitives::Point;

/// Number of segments to use when linearizing curves
const CURVE_SEGMENTS: usize = 16;

/// Number of segments to use when approximating circles
const CIRCLE_SEGMENTS: usize = 32;

/// Helper function to parse a coordinate pair from SVG path tokens
/// Handles both "x,y" format and "x y" format
/// Returns (x, y, tokens_consumed) where tokens_consumed is the number of tokens used
pub fn parse_coordinate_pair(
    coord_str: &str,
    start_idx: usize,
    tokens: &[&str],
) -> Result<(f32, f32, usize)> {
    if let Some(comma_idx) = coord_str.find(',') {
        let x: f32 = coord_str[..comma_idx].parse()?;
        let y_str = &coord_str[comma_idx + 1..];
        let y: f32 = y_str.trim_end_matches(',').parse()?;
        Ok((x, y, 1))
    } else {
        let x: f32 = coord_str.trim_end_matches(',').parse()?;
        if start_idx + 1 < tokens.len() {
            let y: f32 = tokens[start_idx + 1].trim_end_matches(',').parse()?;
            Ok((x, y, 2))
        } else {
            anyhow::bail!("Incomplete coordinate pair at token {}", start_idx);
        }
    }
}

/// Parse a single coordinate value from a token
fn parse_single_coord(token: &str) -> Result<f32> {
    token
        .trim_end_matches(',')
        .parse::<f32>()
        .map_err(|e| anyhow::anyhow!("Failed to parse coordinate '{}': {}", token, e))
}

/// Parse arc parameters which might be comma or space separated
/// Returns (rx, ry, x_rotation, large_arc, sweep, x, y, tokens_consumed)
fn parse_arc_params(
    tokens: &[&str],
    start_idx: usize,
) -> Result<(f32, f32, f32, bool, bool, f32, f32, usize)> {
    let mut values: Vec<f32> = Vec::new();
    let mut idx = start_idx;

    // We need 7 values: rx, ry, x-rotation, large-arc, sweep, x, y
    while values.len() < 7 && idx < tokens.len() {
        let token = tokens[idx];

        // Split by comma and parse each part
        for part in token.split(',') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Ok(val) = part.parse::<f32>() {
                values.push(val);
                if values.len() >= 7 {
                    break;
                }
            } else {
                // Not a number, stop parsing
                break;
            }
        }
        idx += 1;
    }

    if values.len() < 7 {
        anyhow::bail!(
            "Arc command requires 7 parameters, got {} at token {}",
            values.len(),
            start_idx
        );
    }

    Ok((
        values[0],           // rx
        values[1],           // ry
        values[2],           // x-rotation
        values[3] != 0.0,    // large-arc flag
        values[4] != 0.0,    // sweep flag
        values[5],           // x
        values[6],           // y
        idx - start_idx,     // tokens consumed
    ))
}

/// Converts a circle to SVG path data (approximated as a polygon)
pub fn circle_to_path(cx: f32, cy: f32, r: f32) -> String {
    let mut path = String::new();
    for i in 0..=CIRCLE_SEGMENTS {
        let angle = 2.0 * std::f32::consts::PI * (i as f32) / (CIRCLE_SEGMENTS as f32);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if i == 0 {
            path.push_str(&format!("M {},{}", x, y));
        } else {
            path.push_str(&format!(" L {},{}", x, y));
        }
    }
    path.push_str(" z");
    path
}

/// Extracts circle attributes from a circle element
pub fn extract_circle_attributes(circle_str: &str) -> Option<(f32, f32, f32)> {
    // Helper to extract attribute value - handles multiple formats
    let extract_attr = |attr_name: &str, text: &str| -> Option<f32> {
        // Try with double quotes: attr="value"
        let pattern1 = format!("{}=\"", attr_name);
        if let Some(start) = text.find(&pattern1) {
            let start = start + pattern1.len();
            if let Some(end) = text[start..].find('"') {
                let val_str = text[start..start + end].trim();
                if let Ok(val) = val_str.parse::<f32>() {
                    return Some(val);
                }
            }
        }
        // Try with single quotes: attr='value'
        let pattern2 = format!("{}='", attr_name);
        if let Some(start) = text.find(&pattern2) {
            let start = start + pattern2.len();
            if let Some(end) = text[start..].find('\'') {
                let val_str = text[start..start + end].trim();
                if let Ok(val) = val_str.parse::<f32>() {
                    return Some(val);
                }
            }
        }
        // Try without quotes: attr=value (no space after =)
        let pattern3 = format!("{}=", attr_name);
        if let Some(start) = text.find(&pattern3) {
            let after_eq = start + pattern3.len();
            // Skip if next char is a quote (already handled above)
            if let Some(first_char) = text[after_eq..].chars().next() {
                if first_char != '"' && first_char != '\'' {
                    let remaining = &text[after_eq..];
                    if let Some(end) = remaining.find(|c: char| c.is_whitespace() || c == '/' || c == '>') {
                        let val_str = remaining[..end].trim();
                        if let Ok(val) = val_str.parse::<f32>() {
                            return Some(val);
                        }
                    }
                }
            }
        }
        None
    };

    let cx = extract_attr("cx", circle_str);
    let cy = extract_attr("cy", circle_str);
    let r = extract_attr("r", circle_str);

    if let (Some(cx_val), Some(cy_val), Some(r_val)) = (cx, cy, r) {
        Some((cx_val, cy_val, r_val))
    } else {
        None
    }
}

/// Extracts path data from SVG XML bytes
/// Supports both <path> elements and <circle> elements
pub fn extract_path_from_svg_bytes(svg_bytes: &[u8]) -> Result<String> {
    let svg_str = std::str::from_utf8(svg_bytes)?;

    // First, try to find path elements
    if let Some(path_start) = svg_str.find("<path") {
        if let Some(d_start) = svg_str[path_start..].find("d=\"") {
            let d_start = path_start + d_start + 3;
            if let Some(d_end) = svg_str[d_start..].find('"') {
                let path_data = &svg_str[d_start..d_start + d_end];
                return Ok(path_data.to_string());
            }
        }
        if let Some(d_start) = svg_str[path_start..].find("d='") {
            let d_start = path_start + d_start + 3;
            if let Some(d_end) = svg_str[d_start..].find('\'') {
                let path_data = &svg_str[d_start..d_start + d_end];
                return Ok(path_data.to_string());
            }
        }
    }

    // If no path found, try to extract circles and convert them to paths
    let mut circles = Vec::new();
    let mut search_start = 0;

    while let Some(circle_start) = svg_str[search_start..].find("<circle") {
        let absolute_start = search_start + circle_start;
        if let Some(circle_end) = svg_str[absolute_start..].find("/>") {
            let circle_str = &svg_str[absolute_start..absolute_start + circle_end + 2];
            if let Some((cx, cy, r)) = extract_circle_attributes(circle_str) {
                circles.push((cx, cy, r));
            }
            search_start = absolute_start + circle_end + 2;
        } else if let Some(circle_end) = svg_str[absolute_start..].find("</circle>") {
            let circle_str = &svg_str[absolute_start..absolute_start + circle_end];
            if let Some((cx, cy, r)) = extract_circle_attributes(circle_str) {
                circles.push((cx, cy, r));
            }
            search_start = absolute_start + circle_end + 9;
        } else {
            break;
        }
    }

    if !circles.is_empty() {
        // Sort circles by radius (largest first)
        circles.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Use the largest circle as the outer boundary
        let (cx, cy, r) = circles[0];
        let outer_path = circle_to_path(cx, cy, r);

        // Add smaller circles as holes if they are contained within the largest circle
        let mut combined_path = outer_path;
        for (hole_cx, hole_cy, hole_r) in circles.iter().skip(1) {
            let distance = ((cx - hole_cx).powi(2) + (cy - hole_cy).powi(2)).sqrt();
            if distance + hole_r <= r {
                let hole_path = circle_to_path(*hole_cx, *hole_cy, *hole_r);
                combined_path.push(' ');
                combined_path.push_str(&hole_path);
            }
        }

        return Ok(combined_path);
    }

    anyhow::bail!("Could not find path data or circles in SVG bytes");
}

/// Linearize a cubic Bezier curve into line segments
/// p0 = start point, p1 = control point 1, p2 = control point 2, p3 = end point
fn linearize_cubic_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    p3: (f32, f32),
    segments: usize,
) -> Vec<(f32, f32)> {
    let mut points = Vec::with_capacity(segments);
    for i in 1..=segments {
        let t = i as f32 / segments as f32;
        let t2 = t * t;
        let t3 = t2 * t;
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;

        let x = mt3 * p0.0 + 3.0 * mt2 * t * p1.0 + 3.0 * mt * t2 * p2.0 + t3 * p3.0;
        let y = mt3 * p0.1 + 3.0 * mt2 * t * p1.1 + 3.0 * mt * t2 * p2.1 + t3 * p3.1;
        points.push((x, y));
    }
    points
}

/// Linearize a quadratic Bezier curve into line segments
/// p0 = start point, p1 = control point, p2 = end point
fn linearize_quadratic_bezier(
    p0: (f32, f32),
    p1: (f32, f32),
    p2: (f32, f32),
    segments: usize,
) -> Vec<(f32, f32)> {
    let mut points = Vec::with_capacity(segments);
    for i in 1..=segments {
        let t = i as f32 / segments as f32;
        let mt = 1.0 - t;

        let x = mt * mt * p0.0 + 2.0 * mt * t * p1.0 + t * t * p2.0;
        let y = mt * mt * p0.1 + 2.0 * mt * t * p1.1 + t * t * p2.1;
        points.push((x, y));
    }
    points
}

/// Linearize an elliptical arc into line segments
/// Based on the SVG arc parameterization
fn linearize_arc(
    start: (f32, f32),
    rx: f32,
    ry: f32,
    x_axis_rotation: f32,
    large_arc_flag: bool,
    sweep_flag: bool,
    end: (f32, f32),
    segments: usize,
) -> Vec<(f32, f32)> {
    // Handle degenerate cases
    if rx.abs() < 1e-10 || ry.abs() < 1e-10 {
        return vec![end];
    }

    let rx = rx.abs();
    let ry = ry.abs();

    // Convert rotation to radians
    let phi = x_axis_rotation.to_radians();
    let cos_phi = phi.cos();
    let sin_phi = phi.sin();

    // Step 1: Compute (x1', y1') - transformed start point
    let dx = (start.0 - end.0) / 2.0;
    let dy = (start.1 - end.1) / 2.0;
    let x1_prime = cos_phi * dx + sin_phi * dy;
    let y1_prime = -sin_phi * dx + cos_phi * dy;

    // Step 2: Compute (cx', cy') - transformed center
    let mut rx = rx;
    let mut ry = ry;

    // Ensure radii are large enough
    let lambda = (x1_prime * x1_prime) / (rx * rx) + (y1_prime * y1_prime) / (ry * ry);
    if lambda > 1.0 {
        let sqrt_lambda = lambda.sqrt();
        rx *= sqrt_lambda;
        ry *= sqrt_lambda;
    }

    let sq = {
        let num = rx * rx * ry * ry - rx * rx * y1_prime * y1_prime - ry * ry * x1_prime * x1_prime;
        let den = rx * rx * y1_prime * y1_prime + ry * ry * x1_prime * x1_prime;
        if den.abs() < 1e-10 || num < 0.0 {
            0.0
        } else {
            (num / den).sqrt()
        }
    };

    let sign = if large_arc_flag == sweep_flag { -1.0 } else { 1.0 };
    let cx_prime = sign * sq * rx * y1_prime / ry;
    let cy_prime = sign * sq * -ry * x1_prime / rx;

    // Step 3: Compute (cx, cy) from (cx', cy')
    let cx = cos_phi * cx_prime - sin_phi * cy_prime + (start.0 + end.0) / 2.0;
    let cy = sin_phi * cx_prime + cos_phi * cy_prime + (start.1 + end.1) / 2.0;

    // Step 4: Compute theta1 and dtheta
    let ux = (x1_prime - cx_prime) / rx;
    let uy = (y1_prime - cy_prime) / ry;
    let vx = (-x1_prime - cx_prime) / rx;
    let vy = (-y1_prime - cy_prime) / ry;

    let n = (ux * ux + uy * uy).sqrt();
    let theta1 = if uy < 0.0 { -1.0 } else { 1.0 } * (ux / n).clamp(-1.0, 1.0).acos();

    let n = ((ux * ux + uy * uy) * (vx * vx + vy * vy)).sqrt();
    let p = ux * vx + uy * vy;
    let mut dtheta = if ux * vy - uy * vx < 0.0 { -1.0 } else { 1.0 } * (p / n).clamp(-1.0, 1.0).acos();

    if !sweep_flag && dtheta > 0.0 {
        dtheta -= 2.0 * std::f32::consts::PI;
    } else if sweep_flag && dtheta < 0.0 {
        dtheta += 2.0 * std::f32::consts::PI;
    }

    // Generate points along the arc
    let mut points = Vec::with_capacity(segments);
    for i in 1..=segments {
        let t = i as f32 / segments as f32;
        let theta = theta1 + t * dtheta;

        let x = cos_phi * rx * theta.cos() - sin_phi * ry * theta.sin() + cx;
        let y = sin_phi * rx * theta.cos() + cos_phi * ry * theta.sin() + cy;
        points.push((x, y));
    }

    points
}

/// Track the last command type for implicit coordinate handling
#[derive(Clone, Copy, PartialEq)]
enum LastCommand {
    None,
    MoveTo(bool),     // bool = is_relative
    LineTo(bool),
    HorizontalTo(bool),
    VerticalTo(bool),
    CurveTo(bool),
    SmoothCurveTo(bool),
    QuadTo(bool),
    SmoothQuadTo(bool),
    ArcTo(bool),
}

impl LastCommand {
    fn is_relative(&self) -> bool {
        match self {
            LastCommand::None => false,
            LastCommand::MoveTo(r) => *r,
            LastCommand::LineTo(r) => *r,
            LastCommand::HorizontalTo(r) => *r,
            LastCommand::VerticalTo(r) => *r,
            LastCommand::CurveTo(r) => *r,
            LastCommand::SmoothCurveTo(r) => *r,
            LastCommand::QuadTo(r) => *r,
            LastCommand::SmoothQuadTo(r) => *r,
            LastCommand::ArcTo(r) => *r,
        }
    }
}

/// Parses a single sub-path from SVG path tokens
/// Returns (points, next_index) where next_index is the index to continue parsing from
pub fn parse_subpath(tokens: &[&str], start_idx: usize) -> Result<(Vec<Point>, usize)> {
    let mut points = Vec::new();
    let mut current_x = 0.0f32;
    let mut current_y = 0.0f32;
    let mut start_x = 0.0f32;
    let mut start_y = 0.0f32;
    let mut last_command = LastCommand::None;
    let mut last_control_point: Option<(f32, f32)> = None;
    let mut i = start_idx;

    while i < tokens.len() {
        let token = tokens[i];

        match token {
            // MoveTo commands
            "M" | "m" => {
                let is_relative = token == "m";
                if i + 1 < tokens.len() {
                    let coord_str = tokens[i + 1];
                    let (x, y, consumed) = parse_coordinate_pair(coord_str, i + 1, tokens)?;
                    if is_relative {
                        current_x += x;
                        current_y += y;
                    } else {
                        current_x = x;
                        current_y = y;
                    }
                    start_x = current_x;
                    start_y = current_y;
                    points.push(Point(current_x, current_y));
                    // After M/m, implicit coords are treated as L/l
                    last_command = LastCommand::LineTo(is_relative);
                    last_control_point = None;
                    i += 1 + consumed;
                } else {
                    i += 1;
                }
            }

            // LineTo commands
            "L" | "l" => {
                let is_relative = token == "l";
                if i + 1 < tokens.len() {
                    let coord_str = tokens[i + 1];
                    let (x, y, consumed) = parse_coordinate_pair(coord_str, i + 1, tokens)?;
                    if is_relative {
                        current_x += x;
                        current_y += y;
                    } else {
                        current_x = x;
                        current_y = y;
                    }
                    points.push(Point(current_x, current_y));
                    last_command = LastCommand::LineTo(is_relative);
                    last_control_point = None;
                    i += 1 + consumed;
                } else {
                    i += 1;
                }
            }

            // Horizontal LineTo
            "H" | "h" => {
                let is_relative = token == "h";
                if i + 1 < tokens.len() {
                    let x = parse_single_coord(tokens[i + 1])?;
                    if is_relative {
                        current_x += x;
                    } else {
                        current_x = x;
                    }
                    points.push(Point(current_x, current_y));
                    last_command = LastCommand::HorizontalTo(is_relative);
                    last_control_point = None;
                    i += 2;
                } else {
                    i += 1;
                }
            }

            // Vertical LineTo
            "V" | "v" => {
                let is_relative = token == "v";
                if i + 1 < tokens.len() {
                    let y = parse_single_coord(tokens[i + 1])?;
                    if is_relative {
                        current_y += y;
                    } else {
                        current_y = y;
                    }
                    points.push(Point(current_x, current_y));
                    last_command = LastCommand::VerticalTo(is_relative);
                    last_control_point = None;
                    i += 2;
                } else {
                    i += 1;
                }
            }

            // Cubic Bezier CurveTo
            "C" | "c" => {
                let is_relative = token == "c";
                // C requires 3 coordinate pairs (6 values): x1,y1 x2,y2 x,y
                if i + 3 < tokens.len() {
                    let (x1, y1, c1) = parse_coordinate_pair(tokens[i + 1], i + 1, tokens)?;
                    let next_idx = i + 1 + c1;
                    let (x2, y2, c2) = parse_coordinate_pair(tokens[next_idx], next_idx, tokens)?;
                    let next_idx = next_idx + c2;
                    let (x, y, c3) = parse_coordinate_pair(tokens[next_idx], next_idx, tokens)?;

                    let (cp1, cp2, end) = if is_relative {
                        (
                            (current_x + x1, current_y + y1),
                            (current_x + x2, current_y + y2),
                            (current_x + x, current_y + y),
                        )
                    } else {
                        ((x1, y1), (x2, y2), (x, y))
                    };

                    let curve_points = linearize_cubic_bezier(
                        (current_x, current_y),
                        cp1,
                        cp2,
                        end,
                        CURVE_SEGMENTS,
                    );
                    for (px, py) in curve_points {
                        points.push(Point(px, py));
                    }
                    current_x = end.0;
                    current_y = end.1;
                    last_control_point = Some(cp2);
                    last_command = LastCommand::CurveTo(is_relative);
                    i = next_idx + c3;
                } else {
                    i += 1;
                }
            }

            // Smooth Cubic Bezier
            "S" | "s" => {
                let is_relative = token == "s";
                // S requires 2 coordinate pairs: x2,y2 x,y
                if i + 2 < tokens.len() {
                    let (x2, y2, c1) = parse_coordinate_pair(tokens[i + 1], i + 1, tokens)?;
                    let next_idx = i + 1 + c1;
                    let (x, y, c2) = parse_coordinate_pair(tokens[next_idx], next_idx, tokens)?;

                    // First control point is reflection of last control point
                    let cp1 = match last_control_point {
                        Some((lx, ly)) => (2.0 * current_x - lx, 2.0 * current_y - ly),
                        None => (current_x, current_y),
                    };

                    let (cp2, end) = if is_relative {
                        (
                            (current_x + x2, current_y + y2),
                            (current_x + x, current_y + y),
                        )
                    } else {
                        ((x2, y2), (x, y))
                    };

                    let curve_points = linearize_cubic_bezier(
                        (current_x, current_y),
                        cp1,
                        cp2,
                        end,
                        CURVE_SEGMENTS,
                    );
                    for (px, py) in curve_points {
                        points.push(Point(px, py));
                    }
                    current_x = end.0;
                    current_y = end.1;
                    last_control_point = Some(cp2);
                    last_command = LastCommand::SmoothCurveTo(is_relative);
                    i = next_idx + c2;
                } else {
                    i += 1;
                }
            }

            // Quadratic Bezier CurveTo
            "Q" | "q" => {
                let is_relative = token == "q";
                // Q requires 2 coordinate pairs: x1,y1 x,y
                if i + 2 < tokens.len() {
                    let (x1, y1, c1) = parse_coordinate_pair(tokens[i + 1], i + 1, tokens)?;
                    let next_idx = i + 1 + c1;
                    let (x, y, c2) = parse_coordinate_pair(tokens[next_idx], next_idx, tokens)?;

                    let (cp1, end) = if is_relative {
                        (
                            (current_x + x1, current_y + y1),
                            (current_x + x, current_y + y),
                        )
                    } else {
                        ((x1, y1), (x, y))
                    };

                    let curve_points = linearize_quadratic_bezier(
                        (current_x, current_y),
                        cp1,
                        end,
                        CURVE_SEGMENTS,
                    );
                    for (px, py) in curve_points {
                        points.push(Point(px, py));
                    }
                    current_x = end.0;
                    current_y = end.1;
                    last_control_point = Some(cp1);
                    last_command = LastCommand::QuadTo(is_relative);
                    i = next_idx + c2;
                } else {
                    i += 1;
                }
            }

            // Smooth Quadratic Bezier
            "T" | "t" => {
                let is_relative = token == "t";
                if i + 1 < tokens.len() {
                    let (x, y, consumed) = parse_coordinate_pair(tokens[i + 1], i + 1, tokens)?;

                    // Control point is reflection of last control point
                    let cp1 = match last_control_point {
                        Some((lx, ly)) => (2.0 * current_x - lx, 2.0 * current_y - ly),
                        None => (current_x, current_y),
                    };

                    let end = if is_relative {
                        (current_x + x, current_y + y)
                    } else {
                        (x, y)
                    };

                    let curve_points = linearize_quadratic_bezier(
                        (current_x, current_y),
                        cp1,
                        end,
                        CURVE_SEGMENTS,
                    );
                    for (px, py) in curve_points {
                        points.push(Point(px, py));
                    }
                    current_x = end.0;
                    current_y = end.1;
                    last_control_point = Some(cp1);
                    last_command = LastCommand::SmoothQuadTo(is_relative);
                    i += 1 + consumed;
                } else {
                    i += 1;
                }
            }

            // Elliptical Arc
            "A" | "a" => {
                let is_relative = token == "a";
                // A requires: rx ry x-axis-rotation large-arc-flag sweep-flag x y
                // Parameters may be comma or space separated
                if i + 1 < tokens.len() {
                    let (rx, ry, x_rotation, large_arc, sweep, x, y, consumed) =
                        parse_arc_params(tokens, i + 1)?;

                    let end = if is_relative {
                        (current_x + x, current_y + y)
                    } else {
                        (x, y)
                    };

                    let arc_points = linearize_arc(
                        (current_x, current_y),
                        rx,
                        ry,
                        x_rotation,
                        large_arc,
                        sweep,
                        end,
                        CURVE_SEGMENTS,
                    );
                    for (px, py) in arc_points {
                        points.push(Point(px, py));
                    }
                    current_x = end.0;
                    current_y = end.1;
                    last_control_point = None;
                    last_command = LastCommand::ArcTo(is_relative);
                    i += 1 + consumed;
                } else {
                    i += 1;
                }
            }

            // ClosePath
            "z" | "Z" => {
                if !points.is_empty() && points[0] != Point(start_x, start_y) {
                    points.push(Point(start_x, start_y));
                }
                i += 1;
                break;
            }

            // Handle implicit coordinates (continuation of previous command)
            _ => {
                // Try to parse as coordinate pair for implicit line continuation
                let parts: Vec<&str> = token.split(',').collect();
                let parsed = if parts.len() == 2 {
                    if let (Ok(x), Ok(y)) = (parts[0].parse::<f32>(), parts[1].parse::<f32>()) {
                        Some((x, y, 1))
                    } else {
                        None
                    }
                } else if let Ok(x) = token.trim_end_matches(',').parse::<f32>() {
                    if i + 1 < tokens.len() {
                        if let Ok(y) = tokens[i + 1].trim_end_matches(',').parse::<f32>() {
                            Some((x, y, 2))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some((x, y, consumed)) = parsed {
                    // Apply coordinates based on last command's relativity
                    let is_relative = last_command.is_relative();
                    if is_relative {
                        current_x += x;
                        current_y += y;
                    } else {
                        current_x = x;
                        current_y = y;
                    }

                    // If no command yet, treat as start of path
                    if last_command == LastCommand::None {
                        start_x = current_x;
                        start_y = current_y;
                        last_command = LastCommand::LineTo(false);
                    }

                    points.push(Point(current_x, current_y));
                    last_control_point = None;
                    i += consumed;
                } else {
                    // Unknown token, skip
                    i += 1;
                }
            }
        }
    }

    // Remove all duplicate points (not just consecutive ones)
    // This is necessary because SPolygon::new() requires all points to be unique
    let original_count = points.len();
    let first_point = points.first().cloned();
    let last_point = points.last().cloned();
    
    let mut cleaned_points = Vec::new();
    let mut seen_points = std::collections::HashSet::new();
    
    for point in points {
        // Use a tuple of integers to avoid floating-point equality issues in HashSet
        // We multiply by 1e6 and round to get sufficient precision
        let key = ((point.0 * 1e6) as i64, (point.1 * 1e6) as i64);
        if seen_points.insert(key) {
            cleaned_points.push(point);
        } else {
            log::trace!("Removing duplicate point: {:?}", point);
        }
    }

    log::trace!(
        "parse_subpath: original {} points (first: {:?}, last: {:?}), cleaned to {} points (removed {} duplicates)",
        original_count,
        first_point,
        last_point,
        cleaned_points.len(),
        original_count - cleaned_points.len()
    );

    // Remove closing point if it duplicates first point  
    // (this is redundant now but kept for extra safety)
    if cleaned_points.len() > 1 && cleaned_points[0] == cleaned_points[cleaned_points.len() - 1] {
        log::trace!("Removing duplicate closing point: {:?}", cleaned_points[cleaned_points.len() - 1]);
        cleaned_points.pop();
    }

    // Filter out very small edges
    // Note: We iterate without wrapping around to avoid creating duplicate first/last points
    const MIN_EDGE_LENGTH_SQ: f32 = 1e-10;
    let mut final_points = Vec::new();
    let cleaned_count = cleaned_points.len();
    
    for (idx, point) in cleaned_points.iter().enumerate() {
        // For the last point, check the edge back to the first point (closing edge)
        let next_idx = if idx == cleaned_points.len() - 1 {
            0
        } else {
            idx + 1
        };
        let next_point = &cleaned_points[next_idx];
        let dx = point.0 - next_point.0;
        let dy = point.1 - next_point.1;
        let edge_length_sq = dx * dx + dy * dy;
        
        // Keep the point if the edge from it is long enough
        if edge_length_sq > MIN_EDGE_LENGTH_SQ {
            final_points.push(*point);
        } else {
            // Skip this point because the edge is too small
            // But log it for debugging
            log::trace!(
                "Filtering out point ({}, {}) because edge to next point is too small (length_sq: {})",
                point.0, point.1, edge_length_sq
            );
        }
    }

    // Ensure minimum 3 points for a valid polygon
    // If fewer than 3 points, return empty (invalid polygon/line segment - skip it)
    if final_points.len() < 3 {
        log::debug!(
            "Sub-path has only {} points after filtering (minimum 3 required for polygon). \
             Original had {} points, cleaned to {}. Skipping this sub-path.",
            final_points.len(),
            original_count,
            cleaned_count
        );
        return Ok((Vec::new(), i));
    }

    // Final duplicate check (should not be needed but kept as safety)
    if final_points.len() > 1 && final_points[0] == final_points[final_points.len() - 1] {
        log::trace!("Removing duplicate closing point");
        final_points.pop();
    }

    Ok((final_points, i))
}

/// Tokenizes SVG path data by separating command letters from coordinates
/// Handles cases where commands are directly attached to coordinates (e.g., "M4282.687" -> "M" "4282.687")
fn tokenize_svg_path(path_data: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    
    for ch in path_data.chars() {
        match ch {
            // Command letters
            'M' | 'm' | 'L' | 'l' | 'H' | 'h' | 'V' | 'v' | 
            'C' | 'c' | 'S' | 's' | 'Q' | 'q' | 'T' | 't' | 
            'A' | 'a' | 'Z' | 'z' => {
                // Save current token if not empty
                if !current_token.is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
                // Add command as separate token
                tokens.push(ch.to_string());
            }
            // Whitespace
            ' ' | '\t' | '\n' | '\r' => {
                if !current_token.is_empty() {
                    tokens.push(current_token.trim().to_string());
                    current_token.clear();
                }
            }
            // Everything else (numbers, commas, dots, minus signs)
            _ => {
                current_token.push(ch);
            }
        }
    }
    
    // Don't forget the last token
    if !current_token.is_empty() {
        tokens.push(current_token.trim().to_string());
    }
    
    // Filter out empty tokens
    tokens.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Parses SVG path data and extracts polygon coordinates
/// The SVG path may contain multiple sub-paths (outer boundary and inner holes)
/// Returns (outer_boundary, holes) where outer_boundary is the sub-path with the largest absolute area
pub fn parse_svg_path(path_data: &str) -> Result<(Vec<Point>, Vec<Vec<Point>>)> {
    let token_strings = tokenize_svg_path(path_data);
    let tokens: Vec<&str> = token_strings.iter().map(|s| s.as_str()).collect();

    let mut all_subpaths = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        if tokens[i] == "M" || tokens[i] == "m" {
            let (points, next_idx) = parse_subpath(&tokens, i)?;
            if !points.is_empty() {
                all_subpaths.push(points);
            }
            i = next_idx;
        } else {
            i += 1;
        }
    }

    if all_subpaths.is_empty() {
        anyhow::bail!(
            "No valid polygons found in SVG path data. \
             All sub-paths were either empty or had fewer than 3 points \
             (lines and open paths cannot be used for nesting)."
        );
    }

    // Find the sub-path with the largest absolute area (outer boundary)
    let mut max_area = calculate_signed_area(&all_subpaths[0]).abs();
    let mut outer_boundary_idx = 0;
    for (idx, subpath) in all_subpaths.iter().enumerate().skip(1) {
        let area = calculate_signed_area(subpath).abs();
        if area > max_area {
            max_area = area;
            outer_boundary_idx = idx;
        }
    }

    let outer_boundary = all_subpaths.remove(outer_boundary_idx);
    let holes = all_subpaths;

    Ok((outer_boundary, holes))
}

/// Calculates the signed area of a polygon using the shoelace formula
/// Positive area = counter-clockwise, negative area = clockwise
pub fn calculate_signed_area(points: &[Point]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sigma: f32 = 0.0;
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        let (x_i, y_i) = points[i].into();
        let (x_j, y_j) = points[j].into();
        sigma += (y_i + y_j) * (x_i - x_j);
    }
    0.5 * sigma
}

/// Reverses the winding direction of a polygon
pub fn reverse_winding(points: &[Point]) -> Vec<Point> {
    points.iter().rev().cloned().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_path() {
        let path = "M 0,0 L 100,0 L 100,100 L 0,100 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert_eq!(outer.len(), 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_parse_relative_path() {
        let path = "m 0,0 l 100,0 l 0,100 l -100,0 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert_eq!(outer.len(), 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_parse_h_v_commands() {
        let path = "M 0,0 H 100 V 100 H 0 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert_eq!(outer.len(), 4);
        assert_eq!(outer[0], Point(0.0, 0.0));
        assert_eq!(outer[1], Point(100.0, 0.0));
        assert_eq!(outer[2], Point(100.0, 100.0));
        assert_eq!(outer[3], Point(0.0, 100.0));
    }

    #[test]
    fn test_parse_relative_h_v() {
        let path = "M 10,10 h 50 v 50 h -50 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert_eq!(outer.len(), 4);
        assert_eq!(outer[0], Point(10.0, 10.0));
        assert_eq!(outer[1], Point(60.0, 10.0));
        assert_eq!(outer[2], Point(60.0, 60.0));
        assert_eq!(outer[3], Point(10.0, 60.0));
    }

    #[test]
    fn test_parse_cubic_bezier() {
        let path = "M 0,0 C 50,0 100,50 100,100 L 0,100 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        // Should have start point + bezier segments + line point
        assert!(outer.len() >= 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_parse_quadratic_bezier() {
        let path = "M 0,0 Q 50,50 100,0 L 100,100 L 0,100 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert!(outer.len() >= 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_parse_arc() {
        // Arc command: A rx ry x-axis-rotation large-arc-flag sweep-flag x,y
        let path = "M 0,50 A 50 50 0 1 1 100,50 L 100,100 L 0,100 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert!(outer.len() >= 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_parse_smooth_curves() {
        // Smooth cubic bezier (S) and smooth quadratic bezier (T)
        let path = "M 0,0 C 10,10 20,10 30,0 S 50,0 60,10 L 60,50 L 0,50 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert!(outer.len() >= 4);
        assert!(holes.is_empty());

        let path2 = "M 0,0 Q 10,10 20,0 T 40,0 L 40,40 L 0,40 z";
        let (outer2, _) = parse_svg_path(path2).unwrap();
        assert!(outer2.len() >= 4);
    }

    #[test]
    fn test_mixed_commands() {
        // Complex path with multiple command types
        let path = "M 0,0 L 50,0 H 100 V 50 L 50,100 H 0 V 50 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert!(outer.len() >= 4);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_path_with_holes() {
        // Outer boundary with inner hole
        let path = "M 0,0 L 100,0 L 100,100 L 0,100 z M 25,25 L 75,25 L 75,75 L 25,75 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        assert_eq!(outer.len(), 4);
        assert_eq!(holes.len(), 1);
        assert_eq!(holes[0].len(), 4);
    }

    #[test]
    fn test_implicit_relative_coords() {
        // After 'l', implicit coords should also be relative
        let path = "M 0,0 l 10,0 10,0 10,0 z";
        let (outer, _) = parse_svg_path(path).unwrap();
        assert_eq!(outer[0], Point(0.0, 0.0));
        assert_eq!(outer[1], Point(10.0, 0.0));
        assert_eq!(outer[2], Point(20.0, 0.0));
        assert_eq!(outer[3], Point(30.0, 0.0));
    }

    #[test]
    fn test_circle_to_path() {
        let path = circle_to_path(50.0, 50.0, 25.0);
        assert!(path.starts_with("M "));
        assert!(path.ends_with(" z"));
        assert!(path.contains("L "));
    }

    #[test]
    fn test_extract_circle_attributes() {
        let circle1 = r#"<circle cx="50" cy="50" r="25"/>"#;
        let attrs = extract_circle_attributes(circle1).unwrap();
        assert_eq!(attrs, (50.0, 50.0, 25.0));

        let circle2 = r#"<circle cx='100' cy='100' r='50'/>"#;
        let attrs = extract_circle_attributes(circle2).unwrap();
        assert_eq!(attrs, (100.0, 100.0, 50.0));
    }

    #[test]
    fn test_calculate_signed_area() {
        // Counter-clockwise square -> positive area
        let ccw_square = vec![
            Point(0.0, 0.0),
            Point(100.0, 0.0),
            Point(100.0, 100.0),
            Point(0.0, 100.0),
        ];
        let area = calculate_signed_area(&ccw_square);
        assert!(area > 0.0);
        assert!((area - 10000.0).abs() < 0.001);

        // Clockwise square -> negative area
        let cw_square = vec![
            Point(0.0, 0.0),
            Point(0.0, 100.0),
            Point(100.0, 100.0),
            Point(100.0, 0.0),
        ];
        let area = calculate_signed_area(&cw_square);
        assert!(area < 0.0);
    }

    #[test]
    fn test_tokenize_svg_path_with_spaces() {
        // Test standard SVG path with spaces between commands and coordinates
        let path = "M 10,20 L 30,40 L 50,60 Z";
        let tokens = tokenize_svg_path(path);
        
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0], "M");
        assert_eq!(tokens[1], "10,20");
        assert_eq!(tokens[2], "L");
        assert_eq!(tokens[3], "30,40");
        assert_eq!(tokens[4], "L");
        assert_eq!(tokens[5], "50,60");
        assert_eq!(tokens[6], "Z");
    }

    #[test]
    fn test_tokenize_svg_path_without_spaces() {
        // Test compact SVG path without spaces (like fork.svg)
        let path = "M10,20L30,40L50,60Z";
        let tokens = tokenize_svg_path(path);
        
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0], "M");
        assert_eq!(tokens[1], "10,20");
        assert_eq!(tokens[2], "L");
        assert_eq!(tokens[3], "30,40");
        assert_eq!(tokens[4], "L");
        assert_eq!(tokens[5], "50,60");
        assert_eq!(tokens[6], "Z");
    }

    #[test]
    fn test_tokenize_svg_path_mixed_commands() {
        // Test path with various command types
        let path = "M100,200H150V250h-50v-50Z";
        let tokens = tokenize_svg_path(path);
        
        assert_eq!(tokens[0], "M");
        assert_eq!(tokens[1], "100,200");
        assert_eq!(tokens[2], "H");
        assert_eq!(tokens[3], "150");
        assert_eq!(tokens[4], "V");
        assert_eq!(tokens[5], "250");
        assert_eq!(tokens[6], "h");
        assert_eq!(tokens[7], "-50");
        assert_eq!(tokens[8], "v");
        assert_eq!(tokens[9], "-50");
        assert_eq!(tokens[10], "Z");
    }

    #[test]
    fn test_tokenize_svg_path_with_negative_numbers() {
        // Test path with negative coordinates (common in fork.svg)
        let path = "M4282.687,-295.234L4283.047,-213.278";
        let tokens = tokenize_svg_path(path);
        
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0], "M");
        assert_eq!(tokens[1], "4282.687,-295.234");
        assert_eq!(tokens[2], "L");
        assert_eq!(tokens[3], "4283.047,-213.278");
    }

    #[test]
    fn test_tokenize_svg_path_with_curves() {
        // Test path with curve commands
        let path = "M0,0C10,10 20,10 30,0S50,0 60,10";
        let tokens = tokenize_svg_path(path);
        
        assert!(tokens.contains(&"M".to_string()));
        assert!(tokens.contains(&"C".to_string()));
        assert!(tokens.contains(&"S".to_string()));
        // Verify commands are properly separated from coordinates
        assert_eq!(tokens[0], "M");
        assert_eq!(tokens[2], "C");
    }

    #[test]
    fn test_parse_path_with_duplicate_points() {
        // Test that duplicate points are removed
        // This path deliberately returns to an earlier point
        let path = "M 0,0 L 100,0 L 100,100 L 50,50 L 0,100 L 0,0 L 0,100 z";
        let (outer, _holes) = parse_svg_path(path).unwrap();
        
        // Check that no duplicate points exist
        use std::collections::HashSet;
        let unique_count = outer.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            unique_count,
            outer.len(),
            "All points should be unique after parsing"
        );
    }

    #[test]
    fn test_parse_path_consecutive_duplicates() {
        // Test that consecutive duplicate points are removed
        let path = "M 0,0 L 0,0 L 100,0 L 100,0 L 100,100 L 0,100 z";
        let (outer, _holes) = parse_svg_path(path).unwrap();
        
        // Should have 4 unique points (square)
        assert_eq!(outer.len(), 4);
    }

    #[test]
    fn test_parse_path_non_consecutive_duplicates() {
        // Test that non-consecutive duplicate points are removed
        let path = "M 0,0 L 50,0 L 100,0 L 100,50 L 100,100 L 50,100 L 0,100 L 0,50 L 50,0 z";
        // Point (50,0) appears at index 1 and index 8
        let (outer, _holes) = parse_svg_path(path).unwrap();
        
        // Check that the duplicate was removed
        use std::collections::HashSet;
        let unique_count = outer.iter().collect::<HashSet<_>>().len();
        assert_eq!(
            unique_count,
            outer.len(),
            "Non-consecutive duplicate at (50,0) should be removed"
        );
    }

    #[test]
    fn test_parse_compact_svg_notation() {
        // Test parsing SVG with compact notation (no spaces)
        let path = "M10,20L30,40L50,60L70,80Z";
        let result = parse_svg_path(path);
        
        assert!(result.is_ok(), "Should parse compact notation successfully");
        let (outer, _holes) = result.unwrap();
        assert_eq!(outer.len(), 4, "Should have 4 points");
        assert_eq!(outer[0], Point(10.0, 20.0));
        assert_eq!(outer[1], Point(30.0, 40.0));
        assert_eq!(outer[2], Point(50.0, 60.0));
        assert_eq!(outer[3], Point(70.0, 80.0));
    }

    #[test]
    fn test_parse_compact_svg_with_negative_coords() {
        // Test parsing compact SVG with negative coordinates
        let path = "M100,-200L150,-250L150,-300Z";
        let result = parse_svg_path(path);
        
        assert!(result.is_ok(), "Should parse compact notation with negative coords");
        let (outer, _holes) = result.unwrap();
        assert_eq!(outer.len(), 3);
        assert_eq!(outer[0], Point(100.0, -200.0));
        assert_eq!(outer[1], Point(150.0, -250.0));
        assert_eq!(outer[2], Point(150.0, -300.0));
    }

    #[test]
    fn test_parse_path_removes_tiny_edges() {
        // Test that very small edges are filtered out
        let path = "M 0,0 L 100,0 L 100.0000001,0.0000001 L 100,100 L 0,100 z";
        let (outer, _holes) = parse_svg_path(path).unwrap();
        
        // The point at (100.0000001, 0.0000001) should be filtered out
        // because the edge to/from it is too small
        assert!(outer.len() <= 5, "Tiny edges should be filtered out");
    }

    #[test]
    fn test_tokenize_empty_path() {
        let path = "";
        let tokens = tokenize_svg_path(path);
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenize_whitespace_only() {
        let path = "   \t\n  ";
        let tokens = tokenize_svg_path(path);
        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_parse_path_multiple_subpaths() {
        // Test parsing path with multiple M commands (multiple subpaths)
        let path = "M 0,0 L 100,0 L 100,100 L 0,100 z M 25,25 L 75,25 L 75,75 L 25,75 z";
        let (outer, holes) = parse_svg_path(path).unwrap();
        
        // Should identify one as outer boundary and one as hole
        assert_eq!(holes.len(), 1, "Should have 1 hole");
        assert!(outer.len() >= 3, "Outer boundary should have at least 3 points");
        assert!(holes[0].len() >= 3, "Hole should have at least 3 points");
    }

    #[test]
    fn test_parse_path_all_commands() {
        // Test path with all command types to ensure none are broken
        let path = "M 0,0 L 10,0 H 20 V 10 h 10 v 10 C 40,20 50,20 60,30 S 80,40 90,30 Q 100,20 110,30 T 130,30 A 10 10 0 0 1 150,30 Z";
        let result = parse_svg_path(path);
        
        assert!(result.is_ok(), "Should parse path with all command types: {:?}", result.err());
        let (outer, _holes) = result.unwrap();
        assert!(outer.len() >= 3, "Should have at least 3 points forming a polygon");
    }

    #[test]
    fn test_parse_fork_svg() {
        // Test the fork.svg file which has 4 sub-paths
        let svg_bytes = include_bytes!("../../../jagua-sqs-processor/tests/testdata/fork.svg");
        
        // Extract path data
        let path_data = extract_path_from_svg_bytes(svg_bytes).expect("Failed to extract path data");
        eprintln!("Path data length: {}", path_data.len());
        eprintln!("First 200 chars: {}", path_data.chars().take(200).collect::<String>());
        
        // Parse the path
        let result = parse_svg_path(&path_data);
        match &result {
            Ok((outer, holes)) => {
                eprintln!("Outer boundary points: {}", outer.len());
                eprintln!("First 3 points: {:?}", &outer[..3.min(outer.len())]);
                eprintln!("Last 3 points: {:?}", &outer[outer.len().saturating_sub(3)..]);
                
                // Check for duplicates
                use std::collections::HashSet;
                let unique_count = outer.iter().collect::<HashSet<_>>().len();
                if unique_count != outer.len() {
                    eprintln!("WARNING: {} duplicate points found!", outer.len() - unique_count);
                    // Find the duplicates
                    let mut seen = HashSet::new();
                    for (i, point) in outer.iter().enumerate() {
                        if !seen.insert(point) {
                            eprintln!("  Duplicate at index {}: {:?}", i, point);
                        }
                    }
                }
                
                eprintln!("Number of holes: {}", holes.len());
                for (i, hole) in holes.iter().enumerate() {
                    eprintln!("Hole {} points: {}", i, hole.len());
                }
                
                // Try to create SPolygon from the outer boundary to ensure it's valid
                use jagua_rs::geometry::primitives::SPolygon;
                match SPolygon::new(outer.clone()) {
                    Ok(poly) => {
                        eprintln!("Successfully created SPolygon");
                        eprintln!("  Area: {}", poly.area);
                        eprintln!("  Diameter: {}", poly.diameter);
                    }
                    Err(e) => {
                        eprintln!("Failed to create SPolygon: {}", e);
                        panic!("SPolygon creation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
        
        result.expect("Failed to parse fork.svg path");
    }
}
