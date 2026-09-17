//! Fill direction, start corner and the single rectangular remnant (CUTL-198).
//!
//! One nesting parameter, *direction*, with three values: `HORIZONTAL` and `VERTICAL` are
//! deterministic row/column fills that leave **one rectangular remnant** per page; `STAIRCASE`
//! is today's classify-and-route packing, untouched. The start corner says where the fill begins.
//!
//! # Coordinates
//!
//! The SVG output is y-down and unflipped (`layout_to_svg` emits the engine's raw numbers and
//! `combine_svg_documents` stacks pages downward), so engine `(0, 0)` renders **top-left**. The
//! corner enum is named as the viewer sees it: `TOP_LEFT` is `(0, 0)` — today's anchor —,
//! `TOP_RIGHT` is `(binWidth, 0)`, `BOTTOM_LEFT` is `(0, binHeight)`, `BOTTOM_RIGHT` is
//! `(binWidth, binHeight)`. Older notes that call high-y the "bottom strip" describe the engine's
//! own vocabulary, not the picture.

use serde::{Deserialize, Serialize};

/// How each sheet is filled (`NestingRequest.fillDirection`). Absent on the wire ⇒ [`Staircase`].
///
/// [`Staircase`]: FillDirection::Staircase
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FillDirection {
    /// The packed block grows along `binWidth` from the start corner's vertical edge: parts stack
    /// in vertical strips, strips advance along x. The remnant is one full-height strip on the far
    /// x side.
    Horizontal,
    /// The packed block grows along `binHeight` from the start corner's horizontal edge: parts
    /// advance in horizontal strips, strips stack along y. The remnant is one full-width strip on
    /// the far y side.
    Vertical,
    /// The packer's own densest fill — exactly today's classify-and-route behaviour, with today's
    /// offcut behaviour. The start corner is ignored (see [`StartCorner`]).
    #[default]
    Staircase,
}

/// The sheet corner the fill starts from (`NestingRequest.startCorner`), named as the SVG renders
/// it. Absent on the wire ⇒ [`TopLeft`], which reproduces today's output byte for byte.
///
/// Honoured only for [`FillDirection::Horizontal`] / [`FillDirection::Vertical`], where every part
/// owns a disjoint bounding-box cell that can be reflected while the part keeps its own
/// orientation. Under [`FillDirection::Staircase`] it is accepted and ignored: pairing places two
/// parts in one cell and the lattice / LBF packers interlock outlines, so reflecting cells there
/// would create overlaps, and mirroring outlines would produce a different part.
///
/// [`TopLeft`]: StartCorner::TopLeft
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StartCorner {
    /// Engine `(0, 0)` — today's anchor.
    #[default]
    TopLeft,
    /// Engine `(binWidth, 0)`.
    TopRight,
    /// Engine `(0, binHeight)`.
    BottomLeft,
    /// Engine `(binWidth, binHeight)`.
    BottomRight,
}

impl StartCorner {
    /// Whether the fill starts at the right-hand edge (x is reflected).
    pub fn is_right(self) -> bool {
        matches!(self, StartCorner::TopRight | StartCorner::BottomRight)
    }

    /// Whether the fill starts at the bottom edge (y is reflected).
    pub fn is_bottom(self) -> bool {
        matches!(self, StartCorner::BottomLeft | StartCorner::BottomRight)
    }
}

/// The request's layout choice: direction + start corner. `Default` is today's behaviour.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SheetFill {
    pub direction: FillDirection,
    pub corner: StartCorner,
}

impl SheetFill {
    pub fn new(direction: FillDirection, corner: StartCorner) -> Self {
        Self { direction, corner }
    }

    /// Today's packing: the classifier routes, the corner is ignored.
    pub fn is_staircase(&self) -> bool {
        self.direction == FillDirection::Staircase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_are_screaming_snake() {
        assert_eq!(
            serde_json::to_string(&FillDirection::Horizontal).unwrap(),
            "\"HORIZONTAL\""
        );
        assert_eq!(
            serde_json::to_string(&StartCorner::BottomRight).unwrap(),
            "\"BOTTOM_RIGHT\""
        );
        let d: FillDirection = serde_json::from_str("\"STAIRCASE\"").unwrap();
        assert_eq!(d, FillDirection::Staircase);
        let c: StartCorner = serde_json::from_str("\"TOP_LEFT\"").unwrap();
        assert_eq!(c, StartCorner::TopLeft);
    }

    #[test]
    fn defaults_are_todays_behaviour() {
        let fill = SheetFill::default();
        assert!(fill.is_staircase());
        assert_eq!(fill.corner, StartCorner::TopLeft);
        assert!(!fill.corner.is_right() && !fill.corner.is_bottom());
        assert!(StartCorner::BottomRight.is_right() && StartCorner::BottomRight.is_bottom());
    }
}
