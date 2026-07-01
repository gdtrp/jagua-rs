---
name: project_maxfit_parallel_budget
description: How adaptive max_fit nesting is tuned (parallel + hard 55s cap + predictive between-wave guard) and why
metadata: 
  node_type: memory
  type: project
  originSessionId: e3c6ff16-3db0-44a0-8e0c-ff704afe47f4
---

The adaptive nesting strategy (`jagua-utils/src/svg_nesting/strategy/adaptive.rs`) was reworked (2026-05/06) so "nest maximum parts on a sheet" (max_fit) is fast, bounded, and dense. FINAL design after much iteration — the user deliberately pruned earlier heuristics down to this core:

**Core (current) design:**
- **Parallel seed-waves**: independent optimization runs (different seeds/ls_frac) run concurrently via rayon `into_par_iter` in `run_parallel_wave`, keeping the best. Runs are `Send+Sync` and independent. This is the main speedup and the thing to preserve.
- **Time budget (UPDATED 2026-06-01)**: the hard-coded 42s max_fit cap (`MAX_FIT_TIME_BUDGET_SECS`) was **removed**. Budget now comes from the per-request SQS field **`maxSeconds`** (`SqsNestingRequest.max_seconds: Option<u64>`, clamped to a 600s ceiling): when set it drives BOTH the processor's cooperative `execution_timeout` AND the strategy's internal `deadline` (via `AdaptiveNestingStrategy::with_time_budget(Duration)`), for both the normal and max_fit paths. When **unset**, `nest_inner` falls back to the 600s ceiling (`MAX_TOTAL_OPTIMIZATION_SECONDS`) for BOTH paths — so **max_fit with no `maxSeconds` now runs up to 600s, not 42s**. The old 60s-frontend protection is gone from the worker; callers needing fast max_fit must set `maxSeconds` (e.g. 53). Time checks still compare against the absolute `Instant` deadline; the predictive between-wave guard is unchanged.
- **Predictive BETWEEN-wave guard** ("approximate execution time for the next cap"): before each wave, if `now + max_single_run_secs >= deadline`, return the best result already found. Waves are NOT cut mid-solve — cutting LBF mid-run yields a sparse partial layout (measured prod-2 304→200). So a wave always runs to completion; the guard just refuses to *start* one that won't finish. Consequence: one full wave can slightly exceed the budget on slow hardware — 42s is chosen so deadline+wave still lands < 60s (prod-2 wave ≈ 40-43s).
- **Rotations hardcoded to 1 for max_fit** (`rotations = if amount_of_rotations==0 {0} else {1}`). The user chose this: it's faster and, for thin/irregular parts, 1 orientation often packs densest anyway. (A progressive 1→2→4 rotation-tier scheme was tried and removed.)
- **Count cap**: max_fit's saturated 10_000 copies clamped to `ceil(bin_area/part_area)+2` so runs don't churn over impossible copies.
- **Global routine limiter**: `GLOBAL_ROUTINES` map keyed by per-call `execution_id` caps total concurrent routines across all in-flight executions at `MAX_GLOBAL_ROUTINES=100`, reserved per-wave via RAII `RoutineReservation`. Per-wave fan-out clamped to 32; `NEST_RUN_PARALLELISM` env overrides.
- **Escalate then converge** (the non-max_fit / within-budget path): no-improvement waves bump `loops` (cap `MAX_ESCALATED_LOOPS=7`); stop after `MAX_BATCHES_WITHOUT_IMPROVEMENT=5` non-improving waves. Counter is per-execution.
- **Always return best-ever**: `best_result` only overwritten on improvement; every exit path returns it (predictive guard, convergence, all-placed, barren-wave error path).

**FINAL measured (1500x3000, spacing 10):** prod-1 = 48 parts / ~37s; prod-2 = **304 parts / ~43s** (both < 60s). Required floor: prod-2 > 300 ✓.

**Tried and REMOVED (don't reintroduce without reason):**
- Sample ramping (start ~1000, grow per-wave by time feedback): helped prod-1 but for high-fit parts (prod-2) wave time is dominated by ITEM COUNT not samples (~16s even at 250 samples), so starting low just gave a sparser result (251 vs 304). Full samples + 1 wave wins for high-count.
- Rectangular grid fast-path (`try_rectangular_grid`): a bbox grid is a lower bound, not optimum (gave 71 vs optimiser 72); conflicts with max-quality.
- Mid-solve deadline cancellation: produced sparse partial layouts (200).

**Lints (2026-05-31):** `clippy::all` warn + `unsafe_code=forbid` on jagua-utils, `clippy::all` on jagua-sqs-processor, atop existing `warnings=deny`. Run via `rustup run stable cargo clippy` (the ~/.cargo/bin shim is stale). Remaining warnings are in `lbf` (not editable).

Regression tests (`jagua-utils/tests/adaptive_strategy_test.rs`): `test_max_fit_prod_1_terminates_quickly`, `test_max_fit_prod_2_terminates_quickly` (asserts >300 parts AND <60s), `test_max_fit_repro_1800_rounded_rect_cutouts` (>=64; was 72 pre-cap — the cap costs a few parts here). Fixtures `jagua-sqs-processor/tests/testdata/prod-1.svg`, `prod-2.svg`.

Constraint: only `jagua-utils` and `jagua-sqs-processor` may be modified — see [[feedback_no_modify_library]].
