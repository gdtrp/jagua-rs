---
name: project_cutl198_fill_direction
description: CUTL-198 fill direction / start corner / one RECT remnant — decisions, semantics, where it lives (fill.rs), what is deliberately ignored
metadata:
  type: project
---

CUTL-198 (Sep 2026): `fillDirection` (HORIZONTAL|VERTICAL|STAIRCASE) + `startCorner` on the nesting request
(cutl-schemas v1.66.0, asyncapi 1.3.0). Implemented in `jagua-utils/src/svg_nesting/fill.rs`, carried on
`AdaptiveNestingStrategy::with_layout`, routed in `classify.rs` before the classifier. Commits JG-198-1..3 on `main`
(2026-09-18). Handoff: `docs/cutl198_fill_direction_jagua_handoff.md`.

Decisions confirmed with the user (2026-09-17):
- HORIZONTAL = the packed block **grows along x** (vertical strips advancing in x); remnant = one full-height strip
  on the far x side. VERTICAL is the transpose. The handoff's "rows stacking away" wording was inconsistent; the
  remnant formula + the frontend example (1140×1500 on 3000×1500) decided it.
- `startCorner` is honoured only for HORIZONTAL/VERTICAL and **ignored under STAIRCASE** (reflecting bbox cells is
  unsafe for pairing/lattice/LBF where parts share cells or interlock). Engine (0,0) renders top-left.
- Pairing/lattice/irregular fall back to the bbox packer under a fill direction; sub-minimum remnant omitted; remnant
  starts `spacing` after the block; no offcutPolicy ⇒ any positive remnant still emitted.
- An explicit per-part `allowedRotations` wins over `amountOfRotations == 0` on the fill path.

**Why:** product (Nikita) wants one "direction" control and a single rectangular remnant the editor shows; the
remnant travels as the page's first RECT offcut (backend ADR), so the worker must keep `offcuts` through the
`pagesUrl` offload (`slim_pages_for_wire` in processor.rs).

**How to apply:** never touch STAIRCASE behaviour (cutl198_defaults pins byte-identity); any new fast path that
wants the corner must place parts in disjoint bbox cells. Local librdkafka build breaks after an Xcode/SDK change
because of a stale cmake cache — `rm -rf target/debug/build/rdkafka-sys-*/out/build` fixes it (docker builder can't
mount /Volumes/Projects here). See [[project_cutl160_nesting_rfc]], [[project_offcut_feature]].
