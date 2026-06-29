---
name: project_grain_direction
description: Grain direction control — per-part allowedRotations (integer degrees) added to nesting request
metadata: 
  node_type: memory
  type: project
  originSessionId: 716b00f9-7965-43a9-a990-0f2bb39d0c10
---

Grain direction control = constraining a part's allowed rotations. Implemented entirely in jagua-utils + jagua-sqs-processor (no jagua-rs/lbf changes — see [[feedback_no_modify_library]]).

Request contract: `SvgPartSpec.allowedRotations: Option<Vec<i32>>` (whole degrees, camelCase JSON), e.g. `[0]` locks fully, `[0,180]` keeps grain on one axis but allows flip. Per-part, optional; when set it overrides the request-level `amountOfRotations` for that part. Absent/`null` → falls back to `amountOfRotations` (unchanged). Only on the multi-part `parts[]` format, not legacy single-part. `[0]`/`[]` → 0° only.

Plumbing: SvgPartSpec (int degrees) → PartInput.allowed_rotations (`Option<Vec<f32>>`, degrees) → resolve_rotation_range() builds `RotationRange::Discrete` (degrees→radians) per Item in both SimpleNestingStrategy and AdaptiveNestingStrategy. fit_orientations() makes the bbox fit pre-check grain-aware. Engine already honored per-item RotationRange via lbf UniformRotDistr.

Helpers live in jagua-utils/src/svg_nesting/strategy.rs. Tests: test_grain_direction_locks_single_orientation, test_grain_direction_allows_only_flip in adaptive_strategy_test.rs.

Note: adaptive tests test_complex_svg_with_timeout and test_rounded_rect_with_holes_packs_21_parts are flaky under parallel `cargo test` (time-budgeted optimizer + CPU contention); pass in isolation / single-threaded. Pre-existing, unrelated to grain.
