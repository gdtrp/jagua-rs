//! Round-trip against a real S3-compatible endpoint that is not AWS.
//!
//! MinIO stands in for VK Object Storage here: both are reached by pointing
//! `AWS_ENDPOINT_URL_S3` at a custom host, and neither has the wildcard DNS that
//! virtual-host addressing needs. What this proves is the part unit tests cannot —
//! that the client `S3Settings::build_client` returns actually talks to the
//! configured endpoint, and that `object_url` and `parse_s3_url_with` agree well
//! enough for a result URL to be handed back and read again.
//!
//! `#[ignore]`d like every other broker/service-backed test in this crate; run with
//! `make compose-up` followed by `-- --ignored`.

use aws_sdk_s3::primitives::ByteStream;
use jagua_sqs_processor::S3Settings;

const ENDPOINT: &str = "http://localhost:9200";
const BUCKET: &str = "cutl-test-uploads";

fn minio_settings() -> S3Settings {
    // Same static credentials the compose harness provisions.
    std::env::set_var("AWS_ACCESS_KEY_ID", "jagua-test");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "jagua-test-secret");
    // `ru-msk` on purpose: a region string the SDK cannot map to a real AWS
    // endpoint, so if the override were ever dropped this would fail loudly
    // instead of quietly succeeding against AWS.
    S3Settings::new(BUCKET, "ru-msk", Some(ENDPOINT.to_string()), None)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the MinIO harness: make compose-up"]
async fn put_and_get_round_trip_through_the_configured_endpoint() {
    let s3 = minio_settings();
    assert!(
        s3.force_path_style,
        "an endpoint override must derive path-style addressing"
    );

    let client = s3.build_client().await;
    let key = "nesting/s3-endpoint-it/page-0.svg";
    let body = b"<svg xmlns=\"http://www.w3.org/2000/svg\"/>".to_vec();

    client
        .put_object()
        .bucket(BUCKET)
        .key(key)
        .body(ByteStream::from(body.clone()))
        .content_type("image/svg+xml")
        .set_acl(s3.upload_acl.clone())
        .send()
        .await
        .expect("PUT against the configured endpoint should succeed");

    // The URL we would hand back to the backend.
    let url = s3.object_url(BUCKET, key);
    assert_eq!(url, format!("{ENDPOINT}/{BUCKET}/{key}"));

    // That the URL parses back to this bucket/key is pinned by the unit test
    // `test_object_url_round_trips_through_the_parser`; here we only need the
    // object to actually be readable from the endpoint we wrote it to.
    let got = client
        .get_object()
        .bucket(BUCKET)
        .key(key)
        .send()
        .await
        .expect("GET of the object we just wrote should succeed");
    let bytes = got.body.collect().await.expect("body").into_bytes();
    assert_eq!(bytes.as_ref(), body.as_slice());
}

/// The state every AWS-mode staging deploy actually ships: the manifest always
/// renders `AWS_ENDPOINT_URL_S3`, and with no repository variable set it is empty.
/// The SDK reads that variable itself and does not check for emptiness, so this is
/// the case that would break every request if `build_client` stopped clearing it.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the MinIO harness: make compose-up"]
async fn empty_endpoint_env_does_not_leak_into_the_client() {
    std::env::set_var("AWS_ENDPOINT_URL_S3", "");
    let s3 = minio_settings();
    let client = s3.build_client().await;

    // Endpoint still honoured despite the empty variable being present.
    client
        .put_object()
        .bucket(BUCKET)
        .key("nesting/s3-endpoint-it/empty-env.svg")
        .body(ByteStream::from(b"<svg/>".to_vec()))
        .send()
        .await
        .expect("an empty AWS_ENDPOINT_URL_S3 must not override the configured endpoint");

    std::env::remove_var("AWS_ENDPOINT_URL_S3");
}
