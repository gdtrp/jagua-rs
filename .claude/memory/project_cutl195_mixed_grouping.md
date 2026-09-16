---
name: project_cutl195_mixed_grouping
description: CUTL-195 (Sep 2026) — group-mode nesting re-nested every sheet; fix generalised mixed.rs to any 2–4 types with per-type stencils + band-packed leftovers; follow-ups listed
metadata: 
  node_type: memory
  type: project
  originSessionId: 424bc1cf-5c2d-4fe6-8d4b-2750b21acfe0
  modified: 2026-09-16T14:52:16.016Z
---

CUTL-195 (filed 2026-09-14 by QA, staging project 7d7027ea…): two part types that each nest into
identical sheets alone (100×60 rounded rect → grid, Ø120 circle → lattice) produced 10 unrelated
sheets in group mode. Cause: `MixedFewTypes` required *all* types rectangular, so the mix fell to
LBF (107 s, 60-run cap). Fixed on 2026-09-16 in jagua-utils only: `classify::single_sheet_stencil`
(shared grid/pairing/lattice router), `nest_mixed` repeats each type's stencil for its full sheets,
leftovers band-packed (strips may split across sheets); rect-only mixes keep the old shelf packer;
irregular mixes with no full sheet still go to LBF. Regression test: `jagua-utils/tests/cutl195_mixed_irregular.rs`.

**Why:** shop wants identical dominant sheets ("зачем кроить лист 9 заново, если лист 8 готов");
cutl-backend groups API pages by identity, so different sheets show as quantity 1 each.

**How to apply:** follow-ups the user may raise: (1) "1+3" scheme (co-pack types on dominant
sheets) still not done; (2) `MIXED_MAX_TYPES = 4` — shift tasks with >4 types still go to LBF;
(3) remainder quality is stencil-strip based, not LBF. Staging request SVGs are public at
`hb.ru-msk.vkcs.cloud/cutl-staging-data/shared/public/calculation/<id>/result.svg`; Loki
`{app="jagua", environment="staging"}` shows which path ran ("Reached maximum total runs" ⇒ LBF).
Related: [[project_cutl160_nesting_rfc]].
