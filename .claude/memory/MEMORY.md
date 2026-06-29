# Memory Index

- [feedback_no_modify_library.md](feedback_no_modify_library.md) — Only modify jagua-utils and jagua-sqs-processor, never jagua-rs or lbf
- [project_maxfit_parallel_budget.md](project_maxfit_parallel_budget.md) — max_fit nesting: parallel seed-waves + 60s budget; why (SQS visibility) and the tuning knobs
- [project_offcut_feature.md](project_offcut_feature.md) — CUTL offcuts (JG-OFF-1/2 done, 3 pending): free-space detection in jagua-utils + offcutPolicy request wiring
- [project_grain_direction.md](project_grain_direction.md) — Grain direction control: per-part allowedRotations (int degrees) in nesting request, implemented in jagua-utils + sqs-processor
- [project_asyncapi_codegen.md](project_asyncapi_codegen.md) — API-first: SQS wire types generated from AsyncAPI spec via typify in build.rs (vendored spec, make sync-spec, jagua-utils type reuse)
