# Quality-Map Uncertainty Full Runbook

Status: seed-7 release audit passed; remaining frozen core seeds released

## Seed-7 release audit

The returned `quality_map_uncertainty_full_seed7_artifacts.zip` passed the
independent local release audit on 2026-07-18. Its authorized SHA-256 is:

```text
9aa4b4c7f752fc684a304fbc6138242d3931f54527e089379d8759bae2990e8e
```

The audit verified all 476 manifest records and their hashes, the frozen source
commit and runtime, all eight external and five OMBRIA training runs, the full
446-chip external manifest, 86,625 external per-chip rows, 1,650 external raw
summaries, 301 OMBRIA raw summaries, matched controls, zero-valid-label handling,
resume provenance, and current-versus-prior logs. All 11 audit sections passed.

This result releases seeds `13`, `21`, `29`, and `37` with no protocol changes.
It does **not** release seed-7 scores for manuscript use: all five core seeds,
the separate published-architecture gate, merge audit, paired uncertainty, and
Article decision gates remain required.

## Pre-score evaluator correction

The first seed-7 attempt stopped before writing any Sen1Floods11 evaluation
table. One official chip had zero valid hand-label pixels, while the evaluator
incorrectly assumed that every chip had at least one valid target pixel and
raised `ValueError: reference must not be empty` during valid-domain quality
accounting.

The authorized correction is restricted to that empty evaluation domain:

- the chip remains in the frozen 446-chip manifest;
- it contributes zero valid-domain confusion counts;
- `valid_target_pixels=0` and `has_valid_target=false` remain explicit in the
  per-chip table;
- its undefined valid-domain mean probability is written as an empty value,
  not `NaN`;
- data preparation, perturbations, route definitions, training, checkpoints,
  seeds, rates, splits, and OMBRIA evaluation are unchanged.

The failure was reproduced before the fix by the dedicated all-invalid-chip
test and the same test passed afterward. Existing checkpoints from the failed
attempt may be resumed because both training implementations and their frozen
configuration are byte-identical. The previous failed `run.log` is retained
under `prior_attempts/`; the current attempt receives a fresh log so the final
gate can distinguish historical failure evidence from current execution.

## Why Full is sharded

The Smoke run completed the complete code path but used one seed, two epochs,
52 external chips, and a reduced condition grid. Scaling all five seeds into a
single Kaggle session creates an unnecessary session-loss risk. Full is split
only by model seed; no scientific factor is removed or altered.

## Core shard contract

Each of seeds `7`, `13`, `21`, `29`, and `37` runs:

- OMBRIA: five routes, 25 epochs, the 5 x 5 false-available/false-unavailable
  surface, and three deterministic repetitions for the four fusion routes;
- Sen1Floods11: all 446 SCL-matched chips, all eight routes, 25 epochs, official
  test plus separate Bolivia evaluation, 25 independent conditions, 14
  structured conditions, 14 exact valid-domain matched controls, complete
  optical absence, and three repetitions;
- clean-validation checkpoint selection only;
- file-level SHA-256 manifests, checkpoints, trajectories, raw confusion
  counts, per-chip rows, per-event rows, and explicit decision gates.

One external seed shard must contain 550 seed-condition rows and 1650 raw
condition-repetition summaries. One OMBRIA seed shard must contain 101 response
cells and 301 raw summaries. Neither shard gate permits scientific
interpretation.

## Execution order

1. Run seed `7` and return
   `quality_map_uncertainty_full_seed7_artifacts.zip`.
2. Audit runtime, hashes, coverage, matched controls, convergence, and output
   size locally.
3. Seed `7` passed the release audit; run seeds `13`, `21`, `29`, and `37`
   without protocol changes using their supplied immutable notebooks.
4. Run the separate official-architecture SMAGNet adaptation shard or record
   its prespecified reproducibility failure before inspecting merged claims.
5. Merge only complete, disjoint seed shards; recompute all summaries from raw
   rows rather than concatenating already rounded tables.
6. Apply the Article success/stop gates and only then generate manuscript
   figures, tables, and claims.

## Stop conditions

Stop a shard and return its final traceback if CUDA, download, checksum,
checkpoint, finite-metric, condition-count, or packaging gates fail. Do not
reduce epochs, routes, chips, rates, repetitions, or structured conditions to
make a failing job finish.
