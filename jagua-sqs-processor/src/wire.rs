//! Boundary between the generated AsyncAPI wire types ([`crate::generated`]) and the
//! processor's ergonomic DTOs ([`SqsNestingRequest`], [`SqsNestingResponse`], [`SvgPartSpec`]).
//!
//! The generated types are the single source of truth for the JSON wire (camelCase, spec-exact
//! numeric widths). The ergonomic DTOs keep the `f32`/`usize` shapes the engine wants and apply
//! the worker defaults. `Serialize`/`Deserialize` on the DTOs delegate here, so the wire is
//! governed entirely by the spec while `processor.rs` keeps using the friendly field types.

use serde::{Deserialize, Serialize};

use crate::generated;
use crate::processor::{SqsNestingRequest, SqsNestingResponse, SvgPartSpec};

/// Recursively drop object keys whose value is JSON `null`. In this contract `null` always means
/// "absent" (cancellation messages send `binWidth`/`spacing`/`parts`/… as `null`), and several
/// generated fields are non-`Option` (`parts: Vec`, `allowedRotations: Vec`) where an explicit
/// `null` would otherwise fail to deserialize. Stripping nulls makes absent and null equivalent.
fn strip_nulls(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|_, v| !v.is_null());
            for v in map.values_mut() {
                strip_nulls(v);
            }
        }
        serde_json::Value::Array(arr) => arr.iter_mut().for_each(strip_nulls),
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Request: generated -> ergonomic
// ---------------------------------------------------------------------------

impl From<generated::NestingRequestPart> for SvgPartSpec {
    fn from(p: generated::NestingRequestPart) -> Self {
        SvgPartSpec {
            item_id: p.item_id.unwrap_or_default(),
            svg_url: p.svg_url,
            amount_of_parts: p.amount_of_parts.max(0) as usize,
            // Empty (absent) ⇒ None ⇒ part follows the global rotation count.
            allowed_rotations: if p.allowed_rotations.is_empty() {
                None
            } else {
                Some(p.allowed_rotations)
            },
        }
    }
}

impl From<generated::NestingRequest> for SqsNestingRequest {
    fn from(g: generated::NestingRequest) -> Self {
        SqsNestingRequest {
            correlation_id: g.correlation_id.unwrap_or_default(),
            svg_base64: g.svg_base64,
            svg_url: g.svg_url,
            bin_width: g.bin_width.map(|v| v as f32),
            bin_height: g.bin_height.map(|v| v as f32),
            spacing: g.spacing.map(|v| v as f32),
            amount_of_parts: g.amount_of_parts.map(|v| v.max(0) as usize),
            parts: if g.parts.is_empty() {
                None
            } else {
                Some(g.parts.into_iter().map(SvgPartSpec::from).collect())
            },
            // Absent / null ⇒ worker default of 8.
            amount_of_rotations: g
                .amount_of_rotations
                .map(|v| v.max(0) as usize)
                .unwrap_or(8),
            output_queue_url: g.output_queue_url,
            // Absent ⇒ a normal (non-cancellation) nesting request.
            cancelled: g.cancelled.unwrap_or(false),
            max_fit: g.max_fit,
            bucket: g.bucket,
            s3_prefix: g.s3_prefix,
            offcut_policy: g.offcut_policy,
            max_seconds: g.max_seconds.map(|v| v.max(0) as u64),
        }
    }
}

// ---------------------------------------------------------------------------
// Request: ergonomic -> generated (serialize path; used by tests that emit requests)
// ---------------------------------------------------------------------------

impl From<&SvgPartSpec> for generated::NestingRequestPart {
    fn from(p: &SvgPartSpec) -> Self {
        generated::NestingRequestPart {
            item_id: Some(p.item_id.clone()),
            svg_url: p.svg_url.clone(),
            amount_of_parts: p.amount_of_parts as i32,
            allowed_rotations: p.allowed_rotations.clone().unwrap_or_default(),
        }
    }
}

impl From<&SqsNestingRequest> for generated::NestingRequest {
    fn from(r: &SqsNestingRequest) -> Self {
        generated::NestingRequest {
            correlation_id: Some(r.correlation_id.clone()),
            svg_base64: r.svg_base64.clone(),
            svg_url: r.svg_url.clone(),
            bin_width: r.bin_width.map(|v| v as f64),
            bin_height: r.bin_height.map(|v| v as f64),
            spacing: r.spacing.map(|v| v as f64),
            amount_of_parts: r.amount_of_parts.map(|v| v as i32),
            parts: r
                .parts
                .as_ref()
                .map(|ps| ps.iter().map(generated::NestingRequestPart::from).collect())
                .unwrap_or_default(),
            amount_of_rotations: Some(r.amount_of_rotations as i32),
            output_queue_url: r.output_queue_url.clone(),
            cancelled: Some(r.cancelled),
            max_fit: r.max_fit,
            bucket: r.bucket.clone(),
            s3_prefix: r.s3_prefix.clone(),
            offcut_policy: r.offcut_policy,
            max_seconds: r.max_seconds.map(|v| v as i32),
        }
    }
}

impl Serialize for SqsNestingRequest {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        generated::NestingRequest::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SqsNestingRequest {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = serde_json::Value::deserialize(deserializer)?;
        strip_nulls(&mut value);
        let generated: generated::NestingRequest =
            serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(SqsNestingRequest::from(generated))
    }
}

// ---------------------------------------------------------------------------
// Response: ergonomic <-> generated
// ---------------------------------------------------------------------------

impl From<&SqsNestingResponse> for generated::NestingResponse {
    fn from(r: &SqsNestingResponse) -> Self {
        generated::NestingResponse {
            correlation_id: r.correlation_id.clone(),
            first_page_svg_url: r.first_page_svg_url.clone(),
            last_page_svg_url: r.last_page_svg_url.clone(),
            sheets: r.sheets.map(|v| v as i32),
            sheets_total: r.sheets_total.map(|v| v as i32),
            page_svg_urls: r.page_svg_urls.clone().unwrap_or_default(),
            pages: r.pages.clone().unwrap_or_default(),
            parts_placed: r.parts_placed as i32,
            utilisation: r.utilisation as f64,
            improvement: r.is_improvement,
            final_: r.is_final,
            timestamp: r.timestamp as i64,
            error_message: r.error_message.clone(),
        }
    }
}

impl From<generated::NestingResponse> for SqsNestingResponse {
    fn from(g: generated::NestingResponse) -> Self {
        SqsNestingResponse {
            correlation_id: g.correlation_id,
            first_page_svg_url: g.first_page_svg_url,
            last_page_svg_url: g.last_page_svg_url,
            sheets: g.sheets.map(|v| v.max(0) as usize),
            sheets_total: g.sheets_total.map(|v| v.max(0) as usize),
            page_svg_urls: if g.page_svg_urls.is_empty() {
                None
            } else {
                Some(g.page_svg_urls)
            },
            pages: if g.pages.is_empty() {
                None
            } else {
                Some(g.pages)
            },
            parts_placed: g.parts_placed.max(0) as usize,
            utilisation: g.utilisation as f32,
            is_improvement: g.improvement,
            is_final: g.final_,
            timestamp: g.timestamp.max(0) as u64,
            error_message: g.error_message,
        }
    }
}

impl Serialize for SqsNestingResponse {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        generated::NestingResponse::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SqsNestingResponse {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let generated = generated::NestingResponse::deserialize(deserializer)?;
        Ok(SqsNestingResponse::from(generated))
    }
}
