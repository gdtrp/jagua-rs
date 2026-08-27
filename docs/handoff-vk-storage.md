# Handoff: dxf-processor + jagua-rs changes for the VK storage migration

Written 2026-08-27, from the cutl-infra side. Companion to
`handoff-backend-vk-storage.md` (read its "What already exists" table - all of
it applies here too). Staging's file storage moves from the AWS Hong Kong
bucket (`cutl-staging-uploads-ap-east-1`) to VK Object Storage
(`cutl-staging-data` at `https://hb.ru-msk.vkcs.cloud`); the HK data is already
fully synced to the VK bucket and verified.

Both workers are S3-heavy (dxf-processor downloads an input and uploads a
result on essentially every message). The change for each is deliberately
small: point the SDK at a different S3-compatible endpoint.

## The env contract (staging values)

Must be **inert when absent** - the images deploy before the cutover and keep
using AWS HK until the env flip.

```
AWS_ENDPOINT_URL_S3=https://hb.ru-msk.vkcs.cloud
AWS_REGION=ru-msk
BUCKET_NAME=cutl-staging-data
```

Credentials keep their existing delivery and names: the `VK_S3_SECRETS` GitHub
environment secret provides `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`. At
cutover, `cutl-infra/scripts/vk-github-secrets.sh staging --s3-source=vk`
swaps the values from the committed s3-upload-user AWS keys to the
Terraform-minted `cutl-staging-app` VK account - same names, new values,
flipped TOGETHER with the bucket/endpoint env (a mismatch means every S3 call
fails on the next redeploy).

## gdtrp/dxf-processor

1. **Endpoint override.** Go SDK v2 `config.LoadDefaultConfig` honors
   `AWS_ENDPOINT_URL_S3` natively in recent versions - CHECK THE PINNED
   VERSION of `aws-sdk-go-v2/config`; if too old, read the env var and set
   `BaseEndpoint` on the S3 client options.
2. **The `amazonaws` host predicate at `pkg/service/service.go:300`.** It
   string-matches `amazonaws` to decide whether a URL is an S3 URL at all
   (RFC-001 section 3.5 item 4). It must also recognize the VK storage hosts
   (`hb.ru-msk.vkcs.cloud` / `vkcs.cloud`, both path-style and virtual-host
   forms) - or better, become endpoint-config-driven so the next migration
   does not repeat this. Until fixed, any presigned/absolute VK URL arriving
   in a message is misclassified.
3. Virtual-host addressing against VK is confirmed working; path-style also
   works (`https://hb.ru-msk.vkcs.cloud/cutl-staging-data/<key>` was verified
   anonymously against a public object).

## gdtrp/jagua-rs

1. **Endpoint override.** Verify the pinned Rust `aws-config` version honors
   the `AWS_ENDPOINT_URL_S3` env var; older crates need
   `aws_config::from_env().endpoint_url(...)` wired manually from the env var.
2. Nothing else: jagua reads and writes by bucket+key through the SDK. The
   rule from `RelayBucketStack` stands unchanged on VK: resolve objects by
   bucket+key through the SDK, never by fetching a CDN URL.

## Verification (both workers, before cutover)

With the VK creds (`terraform output -raw app_s3_access_key_id` /
`app_s3_secret_access_key` in `cutl-infra/environments/staging-vk`) and the
env block above, a worker pointed at `cutl-staging-data` must be able to GET an
existing object (the full staging keyspace is already there, e.g. any key it
recently processed on AWS) and PUT a result. No ACL work is needed in the
workers UNLESS one of them writes objects that browsers fetch via the CDN
(`*/public/*` keys): those need `x-amz-acl: public-read` on the PUT - check
whether dxf-processor writes any such keys; the backend handoff explains why
(VK has no bucket policies; an object without the per-object ACL 403s through
the CDN).

At the staging cutover the workers get the env flip + redeploy and the smoke
test is an end-to-end calculation (upload -> Kafka -> process -> result
readable). Nothing in the Kafka contract changes.
