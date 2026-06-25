//! Golden tests pinning the JSON wire to the AsyncAPI contract (`asyncapi/jagua-rs.yaml`).
//!
//! These exercise the generated wire types via the ergonomic DTOs' delegated (de)serialization:
//! the request null-tolerance (cancellation messages), the legacy/extension fields, and the exact
//! response shape (camelCase, the `improvement`/`final` literals, and the tagged offcut format).

use jagua_sqs_processor::{PageResult, PlacedPartInfo, SqsNestingRequest, SqsNestingResponse};
use jagua_utils::{Offcut, OffcutVertex};

/// Cancellation messages arrive with every nesting field set to explicit `null` (not omitted) and
/// `amountOfRotations: null`. The boundary must accept that and apply the rotation default of 8.
#[test]
fn request_cancellation_with_explicit_nulls() {
    let body = r#"{"correlationId":"c-1","binWidth":null,"binHeight":null,"spacing":null,
        "amountOfRotations":null,"cancelled":true,"parts":null}"#;
    let req: SqsNestingRequest = serde_json::from_str(body).unwrap();

    assert_eq!(req.correlation_id, "c-1");
    assert!(req.cancelled);
    assert_eq!(
        req.amount_of_rotations, 8,
        "null amountOfRotations ⇒ default 8"
    );
    assert!(req.bin_width.is_none());
    assert!(req.parts.is_none());
}

/// A normal multi-part request: omitted `cancelled` ⇒ false, the per-request `outputQueueUrl`
/// override is preserved, and part fields map across.
#[test]
fn request_multipart_with_output_queue_override() {
    let body = r#"{
        "correlationId":"c-2",
        "binWidth":1000,"binHeight":500,"spacing":5,
        "outputQueueUrl":"https://sqs/custom-queue",
        "parts":[{"svgUrl":"s3://b/p.svg","amountOfParts":3,"itemId":"part-A","allowedRotations":[0,180]}]
    }"#;
    let req: SqsNestingRequest = serde_json::from_str(body).unwrap();

    assert!(!req.cancelled, "absent cancelled ⇒ false");
    assert_eq!(
        req.output_queue_url.as_deref(),
        Some("https://sqs/custom-queue")
    );
    assert_eq!(req.bin_width, Some(1000.0));
    let parts = req.parts.expect("parts");
    assert_eq!(parts.len(), 1);
    assert_eq!(parts[0].item_id, "part-A");
    assert_eq!(parts[0].amount_of_parts, 3);
    assert_eq!(parts[0].allowed_rotations.as_deref(), Some(&[0, 180][..]));
}

/// The final response wire: camelCase keys, the literal `improvement`/`final` booleans, and the
/// tagged offcut shape (`{"kind":"RECT",…}`) inside per-page `offcuts`.
#[test]
fn final_response_wire_shape() {
    let response = SqsNestingResponse {
        correlation_id: "resp-1".to_string(),
        first_page_svg_url: Some("https://s3/first.svg".to_string()),
        last_page_svg_url: None,
        sheets: Some(1),
        page_svg_urls: Some(vec!["https://s3/p0.svg".to_string()]),
        pages: Some(vec![PageResult {
            page_index: 0,
            utilisation: 0.5,
            svg_url: Some("https://s3/p0.svg".to_string()),
            parts_placed: 7,
            placements: vec![PlacedPartInfo {
                item_id: "part-A".to_string(),
                part_index: 0,
                x: 1.0,
                y: 2.0,
                rotation: 90.0,
                centroid_x: 3.0,
                centroid_y: 4.0,
            }],
            offcuts: vec![Offcut::Rect {
                x: 500.0,
                y: 0.0,
                width: 500.0,
                height: 1000.0,
            }],
        }]),
        parts_placed: 7,
        utilisation: 0.5,
        is_improvement: false,
        is_final: true,
        timestamp: 1_700_000_000,
        error_message: None,
    };

    let json = serde_json::to_string(&response).unwrap();

    // Envelope: camelCase + the literal improvement/final keys.
    assert!(json.contains(r#""correlationId":"resp-1""#), "{json}");
    assert!(json.contains(r#""partsPlaced":7"#), "{json}");
    assert!(json.contains(r#""utilisation":0.5"#), "{json}");
    assert!(json.contains(r#""improvement":false"#), "{json}");
    assert!(json.contains(r#""final":true"#), "{json}");
    assert!(json.contains(r#""timestamp":1700000000"#), "{json}");
    // Per-page placement + tagged offcut.
    assert!(json.contains(r#""pageIndex":0"#), "{json}");
    assert!(json.contains(r#""centroidX":3.0"#), "{json}");
    assert!(
        json.contains(r#""kind":"RECT""#),
        "offcut must be tagged on kind: {json}"
    );
    assert!(json.contains(r#""width":500.0"#), "{json}");
}

/// An error response: the worker always emits correlationId/partsPlaced/utilisation/timestamp,
/// sets errorMessage, and round-trips back to the same struct.
#[test]
fn error_response_round_trips() {
    let response = SqsNestingResponse {
        correlation_id: "err-1".to_string(),
        first_page_svg_url: None,
        last_page_svg_url: None,
        sheets: None,
        page_svg_urls: None,
        pages: None,
        parts_placed: 0,
        utilisation: 0.0,
        is_improvement: false,
        is_final: true,
        timestamp: 42,
        error_message: Some("boom".to_string()),
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains(r#""errorMessage":"boom""#), "{json}");
    assert!(
        !json.contains("pages"),
        "absent pages must be omitted: {json}"
    );

    let back: SqsNestingResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back, response);
}

/// A POLY offcut keeps its `vertices`/`holes` wire shape end-to-end.
#[test]
fn poly_offcut_round_trips_through_response() {
    let response = SqsNestingResponse {
        correlation_id: "poly-1".to_string(),
        first_page_svg_url: None,
        last_page_svg_url: None,
        sheets: Some(1),
        page_svg_urls: None,
        pages: Some(vec![PageResult {
            page_index: 0,
            utilisation: 0.25,
            svg_url: None,
            parts_placed: 1,
            placements: vec![],
            offcuts: vec![Offcut::Poly {
                vertices: vec![
                    OffcutVertex { x: 0.0, y: 0.0 },
                    OffcutVertex { x: 100.0, y: 0.0 },
                    OffcutVertex { x: 50.0, y: 80.0 },
                ],
                holes: vec![],
            }],
        }]),
        parts_placed: 1,
        utilisation: 0.25,
        is_improvement: false,
        is_final: true,
        timestamp: 1,
        error_message: None,
    };

    let json = serde_json::to_string(&response).unwrap();
    assert!(json.contains(r#""kind":"POLY""#), "{json}");
    assert!(json.contains(r#""vertices":[{"x":0.0,"y":0.0}"#), "{json}");

    let back: SqsNestingResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back, response);
}
