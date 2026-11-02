# sparrow-oracle (RFC-030 N10)

Sparrow 0.3.0 (JeroenGar/sparrow 6f7b9d3, MIT, `LICENSE`) on jagua-rs 2.0.0 (upstream main
4661c08e), with the decision trace the Go port is compared against.

Changes to Sparrow, all instrumentation or the 2.0.0 migration, none a fix:

- the 2.0.0 migration: `quadtree_depth`, `cd_threshold`, `n_ff_piers` dropped from the config,
  `layout_cde_matches_fresh`, no `SvgDrawOptions::quadtree`;
- `src/oracle.rs`: the trace (per-worker evaluation hashes, the iteration hash with "chk" lines,
  the phase lines) and `CountTerminator`, which stops a phase after an evaluation budget;
- hooks that call it in `optimizer/{worker,separator,explore,compress,lbf}.rs`;
- `src/bin/oracle.rs`: one run; simplification off, compression's decay `FailureBased`.

Build and run (its own workspace, outside the fork's):

    docker run --rm -v "$PWD:/src" -w /src rust:1.90-bookworm cargo build --release --bin oracle
    target/release/oracle <input.json> <trace.jsonl> <final.json> <seed> <expl evals> <cmpr evals> [workers]
