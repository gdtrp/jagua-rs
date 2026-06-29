---
name: project_asyncapi_codegen
description: API-first AsyncAPI contract — SQS wire types generated from the spec via typify in build.rs
metadata: 
  node_type: memory
  type: project
  originSessionId: ef5df3e2-0247-47d5-b52a-bc6211f50a8f
---

The jagua-rs SQS worker is **API-first**: the wire contract lives in
`cutl-backend/docs/asyncapi/jagua-rs.yaml` (the single source of truth), NOT in hand-written Rust
DTOs. Done 2026-06-25.

**Spec source (UPDATED 2026-06-29, branch `chore/consume-cutl-schemas`).** The spec is no longer
vendored/committed: `jagua-sqs-processor/asyncapi/jagua-rs.yaml` is **git-ignored** and synced from
the `gdtrp/cutl-schemas` repo by `scripts/sync-schema.sh` (`gh api … contents/asyncapi/jagua-rs.yaml`,
needs gh auth). Run it before building the processor; CI does it automatically. The older
`make sync-spec` (copies from a local `../cutl-backend` checkout) is the previous path and may be stale.

**Codegen flow.** `jagua-sqs-processor/build.rs` reads the synced copy at
`jagua-sqs-processor/asyncapi/jagua-rs.yaml`, lifts `components.schemas` into a draft-07 JSON
Schema (rewriting `$ref`s), and runs **typify** to emit `generated.rs` in `OUT_DIR` (included via
`src/generated.rs`). The spec is vendored (not read from `../cutl-backend`) because the Dockerfile
build context only copies the jagua-rs repo. typify needs `schemars 0.8` (its pinned major), not 1.x.

**Dockerfile must COPY `build.rs` + `asyncapi/`.** They live at the crate root / its own dir, NOT
under `src/`, so the hand-rolled COPY list in `jagua-sqs-processor/Dockerfile` needs explicit
`COPY jagua-sqs-processor/build.rs …` and `COPY jagua-sqs-processor/asyncapi …` BEFORE the
dep-build phase (the dummy-source `cargo build` runs build.rs too, so the spec must be present then).
Miss them and the container build fails with `OUT_DIR not defined` / `cannot find NestingRequest in
generated` (build.rs never ran). A host `cargo build` hides this — always verify `make build` (docker).

**Make targets (repo root):** `make sync-spec` copies the spec from `$CUTL_BACKEND`
(default `../cutl-backend`). `make codegen` = sync-spec + `touch` the spec + `cargo build`
(forces build.rs to re-run typify) — the one command to regenerate the Rust wire types after the
contract changes. build.rs also auto-regenerates on any normal `cargo build` when the vendored spec
changes (`cargo:rerun-if-changed`). The **Java** models are regenerated separately in cutl-backend
with `npm run gen:async`.

**Domain-type reuse.** build.rs uses typify `with_replacement` so the generated types reuse the
jagua-utils serde types instead of duplicating them: `OffcutPolicy`/`Offcut`/`OffcutVertex` and
`NestingResponsePage`→`PageResult`, `NestingPlacement`→`PlacedPartInfo`. This keeps the offcut wire
byte-identical to jagua-utils' own (tested) serde. `OffcutRect`/`OffcutPoly` subschemas are dropped
and replaced types stubbed in build.rs so typify emits no dead code.

**Boundary.** `src/wire.rs` maps generated wire types ⇄ the ergonomic `SqsNestingRequest`/
`SqsNestingResponse`/`SvgPartSpec` (kept in processor.rs with `f32`/`usize` fields). Their
`Serialize`/`Deserialize` delegate to the generated types, so `processor.rs` is untouched but the
wire is spec-governed. Request deserialize strips JSON `null`s recursively (cancellation messages
send `binWidth`/`parts`/`amountOfRotations` as null; `null` always means absent here).

**Offcut must stay a FLAT schema, never `oneOf`/discriminator.** The worker's wire is already flat
(serde internally-tagged enum → `{"kind":"RECT",x,y,w,h}` / `{"kind":"POLY",vertices[,holes]}`).
The spec models `Offcut` as a single flat object with all-optional geometry + a `kind`
enum field — NOT a `oneOf`+discriminator. A discriminated union makes Modelina emit `@JsonSubTypes`
named by class (`OffcutRect`/`OffcutPoly`), which Jackson can't route on the real `RECT`/`POLY`
values (the same limitation that made `scraper` spec-only) and won't map to the flat Java DTO.
Enumerated fields are named enum schemas (`OffcutKind`, `OffcutShape`) for strong typing.

**Spec = code (decisions).** Where spec and worker disagreed, the spec was amended to match the
worker: Offcut keeps `kind:RECT|POLY` + `holes`, `OffcutShape` has `RECTANGLE_MERGED`; `maxSeconds` documents the 600s
clamp; `NestingRequestPart.svgUrl`/`amountOfParts` are required; `cancelled` is optional (absent ⇒
false); legacy `svgUrl`/`svgBase64`/`amountOfParts`/`outputQueueUrl` are kept as `deprecated`
extensions. After editing the spec: `npx asyncapi validate` + `npm run gen:async` (regenerates Java
models in cutl-backend) — both must be committed in the **cutl-backend** repo separately.

Related: [[project_offcut_feature]], [[project_grain_direction]], [[feedback_no_modify_library]].
