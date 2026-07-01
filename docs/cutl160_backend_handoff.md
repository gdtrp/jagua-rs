# CUTL-160 — Backend / Frontend Handoff

Worker-side work for [CUTL-160](https://tracker.yandex.ru/CUTL-160) is **done and merged on the
jagua-rs side**. This note hands off the small contract + UI follow-ups to backend and frontend.
Full design: [`docs/rfcs/CUTL-160-nesting-optimization.md`](rfcs/CUTL-160-nesting-optimization.md).

## TL;DR

The jagua-rs worker now classifies each request by the **incoming part shapes** and routes bulk
rectangular / triangular / mixed parts to fast **deterministic** packers (grid / periodic / pairing /
mixed), keeping the stochastic LBF nester only for genuinely irregular inputs. Net effects the UI
will observe:

- **Identical full sheets + one remainder** for bulk parts (no more *N different* sheets).
- **Much faster** results for the common cases (closed-form, no optimizer loop).
- **Deterministic** output (same request → same layout).
- **"Max per sheet" is now consistent** with the actual sheets (the 44-vs-45 bug is gone).
- A new **`sheetsTotal`** response field enabling a **determinate progress bar**.

**No request-side change.** The worker picks the packer itself — callers send nothing new.

## Contract change (one field)

Schema: `gdtrp/cutl-schemas` → `asyncapi/jagua-rs.yaml`, bumped **1.0.0 → 1.1.0**
(PR: https://github.com/gdtrp/cutl-schemas/pull/1).

```yaml
# NestingResponse (additive, optional)
sheetsTotal:
  type: integer
  format: int32
  # Estimated total sheets, known up front for the deterministic packers.
  # Absent for the general LBF path.
```

That is the **only** wire change. `NestingResponsePage` / `NestingPlacement` / the request schema are
unchanged — the new packers emit the identical response shape.

## Release / publish

Per `cutl-schemas/CONSUMING.md`, nothing generated is committed; CI publishes on a **version tag**:

1. Merge cutl-schemas PR #1 to `main`.
2. Tag **`v1.1.0`** → CI publishes `com.cutl.schemas:*` jars (GitHub Packages) and the Rust/Go release
   tarballs.

## Backend (Java)

1. Bump the `com.cutl.schemas:cutl-schemas-async` / `-models` / `-server` dependency `1.0.0 → 1.1.0`
   in `pom.xml` once `v1.1.0` is published.
2. `NestingResponse` gains `getSheetsTotal(): Integer` (nullable). **Forward it as-is** to the
   frontend alongside the existing `sheets` (current count). No other code change required —
   `sheetsTotal` is null on the LBF path, which the frontend treats as "indeterminate".

## Frontend

1. **Determinate progress bar** (CUTL-160 #1): when `sheetsTotal != null`, render
   `percent = round(100 * sheets / sheetsTotal)` from the streamed improvement messages; fall back to
   the existing indeterminate spinner when it is null (LBF path).
2. **"Max parts per sheet"**: keep showing it, but it will now always be ≥ every sheet's count
   (consistency fix). The `maxFit` value and the real nest now come from the same deterministic
   stencil.
3. **Identical sheets**: the page list will contain runs of byte-identical sheets (one stencil
   repeated) + a remainder page — fine to render as-is; the per-page SVG URLs are unchanged in shape.

## Worker side (already implemented, for reference)

- `jagua-utils`: new `classify` / `grid` / `periodic` / `pairing` / `mixed` / `render` modules;
  `nest_auto` routes by shape. `NestingResult.sheets_total_estimate` set by the deterministic paths.
- `jagua-sqs-processor`: `nest()` → `nest_auto`, `maxFit` → `nest_max_fit_auto`; response carries
  `sheets_total` → wire `sheetsTotal`. Vendored spec + typify regen done; tests green.

## Still open (tracked, not blocking this handoff)

- Confirm identical pages reuse **one S3 object/URL** (blank-preview / «проблемы с прогрузкой», #3).
- WS-5 "1+3" co-pack of *different* types on the dominant sheets (today only the remainder is
  co-packed); irregular-single-part periodic-LBF stencil.
