//! Storage configuration: one place resolves bucket, region, endpoint, addressing
//! mode and upload ACL, and builds the S3 client from them.
//!
//! This exists as a module rather than a dozen inline `env::var` calls because the
//! endpoint has THREE consumers, not one: the SDK client, the result URL we hand
//! back to the backend (`S3Settings::object_url`), and the S3-URL parser
//! (`S3Settings::endpoint_host`). Resolving it in three places is how they drift.
//!
//! Every value is deliberately inert when absent AND when empty — see `resolve`.
//! That is what lets `deploy/k8s/deployment-staging.yaml` render an always-present
//! env key from an unset GitHub variable and still behave exactly as before.

use anyhow::{anyhow, Context, Result};
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Region;
use aws_sdk_s3::types::ObjectCannedAcl;
use aws_sdk_s3::Client as S3Client;
use log::{info, warn};

/// Where the endpoint override came from. Startup-log only; at a cutover this
/// string is most of the diagnosis.
pub const ENDPOINT_ENV_S3: &str = "AWS_ENDPOINT_URL_S3";
pub const ENDPOINT_ENV_GLOBAL: &str = "AWS_ENDPOINT_URL";

/// Resolved storage configuration. Build with [`S3Settings::from_env`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Settings {
    pub bucket: String,
    /// Also the SigV4 signing region. Flipping the endpoint without this gives
    /// `SignatureDoesNotMatch`, not a routing error — easy to misdiagnose.
    pub region: String,
    /// Normalised: trailing `/` stripped. `None` means real AWS S3.
    endpoint_url: Option<String>,
    /// `host[:port]` of `endpoint_url`, precomputed for `parse_s3_url_with`.
    endpoint_host: Option<String>,
    pub force_path_style: bool,
    /// `None` (the default) sends no ACL at all, exactly as before this existed.
    pub upload_acl: Option<ObjectCannedAcl>,
    endpoint_source: Option<&'static str>,
}

/// Read one variable, treating empty and whitespace-only as absent.
///
/// The empty case is load-bearing, not defensive: `envsubst` renders an unset
/// GitHub variable as `""`, so `value: "${JAGUA_STAGING_S3_ENDPOINT}"` reaches the
/// pod as a present-but-empty key on every AWS-mode deploy.
fn env_opt(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| v.trim().to_string())
}

/// Accepts the spellings an operator actually types. Anything else is an error
/// rather than a silent default: a typo here would flip addressing mode and
/// corrupt every result URL we advertise, which is worse than an unready pod.
fn parse_bool(key: &str, raw: &str) -> Result<bool> {
    match raw.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(anyhow!(
            "{key} must be one of true/1/yes/on or false/0/no/off, got {other:?}"
        )),
    }
}

/// Split a normalised endpoint into `(scheme, authority)`.
///
/// `endpoint_url` is validated at construction, so the `://` is always present;
/// the fallback keeps this total rather than panicking on a hand-built value.
fn split_endpoint(endpoint: &str) -> (&str, &str) {
    match endpoint.split_once("://") {
        Some((scheme, authority)) => (scheme, authority),
        None => ("https", endpoint),
    }
}

/// `host[:port]` of an endpoint: scheme stripped, any path dropped.
fn host_of(endpoint: &str) -> String {
    let (_, authority) = split_endpoint(endpoint);
    authority
        .split('/')
        .next()
        .unwrap_or(authority)
        .to_ascii_lowercase()
}

/// A region string that no AWS region looks like (`eu-north-1`, `ap-east-1`, …).
/// Used only to warn about a half-applied storage flip.
fn looks_like_aws_region(region: &str) -> bool {
    match region.rsplit_once('-') {
        Some((prefix, suffix)) => {
            !prefix.is_empty() && !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit())
        }
        None => false,
    }
}

impl S3Settings {
    /// Pure resolver: no environment, no network, no I/O. `get` must already
    /// treat empty as absent. This is the unit-testable core of the module.
    pub fn from_source<F>(get: F) -> Result<Self>
    where
        F: Fn(&str) -> Option<String>,
    {
        // No default on purpose: guessing a bucket name would write results
        // somewhere nobody is reading.
        let bucket = get("S3_BUCKET").context("S3_BUCKET is required")?;
        // `AWS_DEFAULT_REGION` is checked second because the SDK's own chain honours
        // it, and `build_client` sets the region explicitly — without this, setting
        // only the legacy name would silently fall back to eu-north-1.
        let region = get("AWS_REGION")
            .or_else(|| get("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|| "eu-north-1".to_string());

        // `AWS_ENDPOINT_URL_S3` is the name cutl-infra's storage handoff specifies
        // and scopes the override to S3; `AWS_ENDPOINT_URL` is the older global one
        // the local MinIO harness sets (scripts/cargo-docker.sh). Both keep working.
        let (endpoint_url, endpoint_source) = match get(ENDPOINT_ENV_S3) {
            Some(v) => (Some(v), Some(ENDPOINT_ENV_S3)),
            None => match get(ENDPOINT_ENV_GLOBAL) {
                Some(v) => (Some(v), Some(ENDPOINT_ENV_GLOBAL)),
                None => (None, None),
            },
        };
        let endpoint_url = match endpoint_url {
            Some(v) => {
                if !v.contains("://") {
                    return Err(anyhow!(
                        "{} must include a scheme, got {v:?}",
                        endpoint_source.unwrap_or(ENDPOINT_ENV_S3)
                    ));
                }
                Some(v.trim_end_matches('/').to_string())
            }
            None => None,
        };

        let force_path_style = match get("S3_FORCE_PATH_STYLE") {
            Some(v) => parse_bool("S3_FORCE_PATH_STYLE", &v)?,
            // An endpoint override points at something that is not the real S3,
            // and only the real S3 has the wildcard DNS virtual-host addressing
            // needs. MinIO in particular has none.
            None => endpoint_url.is_some(),
        };

        let upload_acl = match get("S3_UPLOAD_ACL") {
            // `try_parse` rejects unknown variants; `from` would silently accept a
            // typo as an Unknown and then have S3 reject every upload at runtime.
            Some(v) => Some(ObjectCannedAcl::try_parse(&v).map_err(|_| {
                anyhow!(
                    "S3_UPLOAD_ACL must be one of {:?}, got {v:?}",
                    ObjectCannedAcl::values()
                )
            })?),
            None => None,
        };

        Ok(Self {
            bucket,
            endpoint_host: endpoint_url.as_deref().map(host_of),
            region,
            endpoint_url,
            force_path_style,
            upload_acl,
            endpoint_source,
        })
    }

    pub fn from_env() -> Result<Self> {
        Self::from_source(env_opt)
    }

    /// Explicit constructor for tests and callers that already have the values.
    /// `force_path_style: None` applies the same derived default as `from_source`.
    pub fn new(
        bucket: impl Into<String>,
        region: impl Into<String>,
        endpoint_url: Option<String>,
        force_path_style: Option<bool>,
    ) -> Self {
        let endpoint_url = endpoint_url.map(|v| v.trim_end_matches('/').to_string());
        Self {
            bucket: bucket.into(),
            region: region.into(),
            endpoint_host: endpoint_url.as_deref().map(host_of),
            force_path_style: force_path_style.unwrap_or(endpoint_url.is_some()),
            endpoint_url,
            upload_acl: None,
            endpoint_source: None,
        }
    }

    pub fn endpoint_url(&self) -> Option<&str> {
        self.endpoint_url.as_deref()
    }

    /// `host[:port]` of the configured endpoint, for endpoint-aware URL parsing.
    pub fn endpoint_host(&self) -> Option<&str> {
        self.endpoint_host.as_deref()
    }

    /// The URL we advertise back to the backend for an uploaded object.
    ///
    /// `bucket` is a parameter rather than `self.bucket` because a request may
    /// override the bucket while the endpoint and region stay worker config.
    pub fn object_url(&self, bucket: &str, key: &str) -> String {
        match (&self.endpoint_url, self.force_path_style) {
            (Some(endpoint), true) => format!("{endpoint}/{bucket}/{key}"),
            (Some(endpoint), false) => {
                let (scheme, authority) = split_endpoint(endpoint);
                format!("{scheme}://{bucket}.{authority}/{key}")
            }
            // Byte-identical to the pre-S3Settings string. This is the flip-back
            // path and is pinned by a test.
            (None, false) => format!("https://{bucket}.s3.{}.amazonaws.com/{key}", self.region),
            (None, true) => format!("https://s3.{}.amazonaws.com/{bucket}/{key}", self.region),
        }
    }

    /// Credential-free one-line startup summary.
    pub fn describe(&self) -> String {
        let endpoint = match (&self.endpoint_url, self.endpoint_source) {
            (Some(url), Some(src)) => format!("{url} (from {src})"),
            (Some(url), None) => url.clone(),
            (None, _) => "<aws default>".to_string(),
        };
        let acl = match &self.upload_acl {
            Some(acl) => acl.as_str().to_string(),
            None => "<none>".to_string(),
        };
        format!(
            "bucket={} region={} endpoint={} path_style={} acl={}",
            self.bucket, self.region, endpoint, self.force_path_style, acl
        )
    }

    /// Build the S3 client. Separate from `from_source` so config resolution stays
    /// synchronous and unit-testable.
    pub async fn build_client(&self) -> S3Client {
        // A half-applied flip — bucket and region moved to VK, endpoint forgotten —
        // otherwise produces plausible-but-wrong `s3.ru-msk.amazonaws.com` URLs and
        // no error until every request fails.
        if self.endpoint_url.is_none() && !looks_like_aws_region(&self.region) {
            warn!(
                "AWS_REGION={} does not look like an AWS region but no endpoint override is set. \
                 If this should be VK Object Storage, {ENDPOINT_ENV_S3} is missing.",
                self.region
            );
        }

        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            // Setting the region explicitly short-circuits the default chain. With
            // no region anywhere the SDK falls through to IMDS at 169.254.169.254,
            // which is not routable from a VK pod — a startup hang.
            .region(Region::new(self.region.clone()))
            .load()
            .await;

        let mut builder = aws_sdk_s3::config::Builder::from(&sdk_config);
        // UNCONDITIONAL, and it must stay that way. `From<&SdkConfig>` scrapes
        // AWS_ENDPOINT_URL_S3 out of the environment itself (aws-sdk-s3 config.rs,
        // `service_config_key("S3", "AWS_ENDPOINT_URL", ...)`), and that path has no
        // emptiness check — an empty value becomes an empty endpoint and every
        // request fails. Passing `None` here clears it. Do NOT rewrite this as
        // `if let Some(ep) = ... { builder.set_endpoint_url(ep) }`.
        builder.set_endpoint_url(self.endpoint_url.clone());
        builder.set_force_path_style(Some(self.force_path_style));

        info!("S3 client: {}", self.describe());
        S3Client::from_conf(builder.build())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Mirrors `env_opt`: empty and whitespace-only are absent. Using a map rather
    /// than `set_var` keeps these tests free of cross-test races.
    fn source(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |key: &str| {
            map.get(key)
                .filter(|v| !v.trim().is_empty())
                .map(|v| v.trim().to_string())
        }
    }

    fn settings(pairs: &[(&str, &str)]) -> S3Settings {
        S3Settings::from_source(source(pairs)).expect("settings should resolve")
    }

    #[test]
    fn bucket_is_required() {
        let err = S3Settings::from_source(source(&[])).unwrap_err();
        assert!(err.to_string().contains("S3_BUCKET"), "{err}");
    }

    #[test]
    fn empty_bucket_is_absent() {
        let err = S3Settings::from_source(source(&[("S3_BUCKET", "  ")])).unwrap_err();
        assert!(err.to_string().contains("S3_BUCKET"), "{err}");
    }

    #[test]
    fn region_defaults_and_empty_is_absent() {
        assert_eq!(settings(&[("S3_BUCKET", "b")]).region, "eu-north-1");
        // The empty case is the regression: `unwrap_or_else` on a bare `env::var`
        // let an envsubst-rendered "" through as the region.
        assert_eq!(
            settings(&[("S3_BUCKET", "b"), ("AWS_REGION", "")]).region,
            "eu-north-1"
        );
        assert_eq!(
            settings(&[("S3_BUCKET", "b"), ("AWS_REGION", "ru-msk")]).region,
            "ru-msk"
        );
    }

    #[test]
    fn aws_default_region_is_the_second_choice() {
        assert_eq!(
            settings(&[("S3_BUCKET", "b"), ("AWS_DEFAULT_REGION", "ap-east-1")]).region,
            "ap-east-1"
        );
        // The primary name still wins when both are set.
        assert_eq!(
            settings(&[
                ("S3_BUCKET", "b"),
                ("AWS_REGION", "ru-msk"),
                ("AWS_DEFAULT_REGION", "ap-east-1"),
            ])
            .region,
            "ru-msk"
        );
    }

    #[test]
    fn s3_specific_endpoint_wins_over_global() {
        let s = settings(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", "https://hb.ru-msk.vkcs.cloud"),
            ("AWS_ENDPOINT_URL", "http://minio:9000"),
        ]);
        assert_eq!(s.endpoint_url(), Some("https://hb.ru-msk.vkcs.cloud"));
        assert_eq!(s.endpoint_source, Some(ENDPOINT_ENV_S3));
    }

    #[test]
    fn empty_s3_endpoint_falls_back_to_global() {
        // Exactly what the rendered staging manifest plus the local MinIO harness
        // produce: the _S3 key is present but empty, the global one is real.
        let s = settings(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", ""),
            ("AWS_ENDPOINT_URL", "http://minio:9000"),
        ]);
        assert_eq!(s.endpoint_url(), Some("http://minio:9000"));
        assert_eq!(s.endpoint_source, Some(ENDPOINT_ENV_GLOBAL));
        assert!(s.force_path_style);
    }

    #[test]
    fn both_endpoints_empty_means_aws() {
        let s = settings(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", ""),
            ("AWS_ENDPOINT_URL", ""),
        ]);
        assert_eq!(s.endpoint_url(), None);
        assert_eq!(s.endpoint_host(), None);
        assert!(!s.force_path_style);
        assert!(s.upload_acl.is_none());
    }

    #[test]
    fn endpoint_requires_a_scheme() {
        let err = S3Settings::from_source(source(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", "hb.x"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("scheme"), "{err}");
    }

    #[test]
    fn force_path_style_override_wins_both_ways() {
        let forced_off = settings(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", "https://hb.ru-msk.vkcs.cloud"),
            ("S3_FORCE_PATH_STYLE", "false"),
        ]);
        assert!(!forced_off.force_path_style);

        let forced_on = settings(&[("S3_BUCKET", "b"), ("S3_FORCE_PATH_STYLE", "true")]);
        assert!(forced_on.force_path_style);
    }

    #[test]
    fn force_path_style_accepts_the_usual_spellings() {
        for raw in ["true", "TRUE", "1", "yes", "On"] {
            assert!(
                settings(&[("S3_BUCKET", "b"), ("S3_FORCE_PATH_STYLE", raw)]).force_path_style,
                "{raw} should parse as true"
            );
        }
        for raw in ["false", "FALSE", "0", "no", "Off"] {
            assert!(
                !settings(&[
                    ("S3_BUCKET", "b"),
                    ("AWS_ENDPOINT_URL_S3", "http://minio:9000"),
                    ("S3_FORCE_PATH_STYLE", raw),
                ])
                .force_path_style,
                "{raw} should parse as false"
            );
        }
    }

    #[test]
    fn force_path_style_rejects_garbage() {
        let err = S3Settings::from_source(source(&[
            ("S3_BUCKET", "b"),
            ("S3_FORCE_PATH_STYLE", "path-style"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("S3_FORCE_PATH_STYLE"), "{err}");
    }

    #[test]
    fn upload_acl_parses_and_rejects_garbage() {
        let s = settings(&[("S3_BUCKET", "b"), ("S3_UPLOAD_ACL", "public-read")]);
        assert_eq!(s.upload_acl, Some(ObjectCannedAcl::PublicRead));

        let err = S3Settings::from_source(source(&[
            ("S3_BUCKET", "b"),
            ("S3_UPLOAD_ACL", "world-read"),
        ]))
        .unwrap_err();
        assert!(err.to_string().contains("S3_UPLOAD_ACL"), "{err}");
    }

    #[test]
    fn endpoint_host_derivation() {
        let cases = [
            ("https://hb.ru-msk.vkcs.cloud/", "hb.ru-msk.vkcs.cloud"),
            ("http://minio:9000", "minio:9000"),
            ("http://localhost:4566/", "localhost:4566"),
            ("https://HB.RU-MSK.VKCS.CLOUD", "hb.ru-msk.vkcs.cloud"),
        ];
        for (endpoint, expected) in cases {
            let s = settings(&[("S3_BUCKET", "b"), ("AWS_ENDPOINT_URL_S3", endpoint)]);
            assert_eq!(s.endpoint_host(), Some(expected), "endpoint {endpoint}");
        }
    }

    #[test]
    fn object_url_aws_virtual_host_is_unchanged() {
        // Pins the exact legacy string. This is the flip-back regression guard.
        let s = S3Settings::new("cutl-staging-uploads-ap-east-1", "ap-east-1", None, None);
        assert_eq!(
            s.object_url("cutl-staging-uploads-ap-east-1", "nesting/abc/first-page.svg"),
            "https://cutl-staging-uploads-ap-east-1.s3.ap-east-1.amazonaws.com/nesting/abc/first-page.svg"
        );
    }

    #[test]
    fn object_url_covers_the_whole_matrix() {
        let vk = S3Settings::new(
            "cutl-staging-data",
            "ru-msk",
            Some("https://hb.ru-msk.vkcs.cloud".into()),
            None,
        );
        assert_eq!(
            vk.object_url("cutl-staging-data", "nesting/abc/f.svg"),
            "https://hb.ru-msk.vkcs.cloud/cutl-staging-data/nesting/abc/f.svg"
        );

        let vk_vhost = S3Settings::new(
            "cutl-staging-data",
            "ru-msk",
            Some("https://hb.ru-msk.vkcs.cloud".into()),
            Some(false),
        );
        assert_eq!(
            vk_vhost.object_url("cutl-staging-data", "nesting/abc/f.svg"),
            "https://cutl-staging-data.hb.ru-msk.vkcs.cloud/nesting/abc/f.svg"
        );

        let aws_path = S3Settings::new("b", "eu-north-1", None, Some(true));
        assert_eq!(
            aws_path.object_url("b", "k/f.svg"),
            "https://s3.eu-north-1.amazonaws.com/b/k/f.svg"
        );

        let minio = S3Settings::new(
            "cutl-test-uploads",
            "ru-msk",
            Some("http://minio:9000".into()),
            None,
        );
        assert_eq!(
            minio.object_url("cutl-test-uploads", "k/f.svg"),
            "http://minio:9000/cutl-test-uploads/k/f.svg"
        );
    }

    #[test]
    fn trailing_slash_endpoint_does_not_double_slash() {
        let s = settings(&[
            ("S3_BUCKET", "b"),
            ("AWS_ENDPOINT_URL_S3", "https://hb.ru-msk.vkcs.cloud/"),
        ]);
        assert_eq!(s.endpoint_url(), Some("https://hb.ru-msk.vkcs.cloud"));
        assert_eq!(
            s.object_url("b", "k.svg"),
            "https://hb.ru-msk.vkcs.cloud/b/k.svg"
        );
    }

    #[test]
    fn describe_names_the_endpoint_source() {
        let s = settings(&[
            ("S3_BUCKET", "cutl-staging-data"),
            ("AWS_REGION", "ru-msk"),
            ("AWS_ENDPOINT_URL_S3", "https://hb.ru-msk.vkcs.cloud"),
        ]);
        assert_eq!(
            s.describe(),
            "bucket=cutl-staging-data region=ru-msk \
             endpoint=https://hb.ru-msk.vkcs.cloud (from AWS_ENDPOINT_URL_S3) \
             path_style=true acl=<none>"
        );
    }

    #[test]
    fn aws_region_shape_detection() {
        for region in ["eu-north-1", "ap-east-1", "us-gov-west-1"] {
            assert!(looks_like_aws_region(region), "{region}");
        }
        for region in ["ru-msk", "vk", "ru-msk-"] {
            assert!(!looks_like_aws_region(region), "{region}");
        }
    }
}
