# Quality-Map Uncertainty Kaggle Operator Guide

## Smoke only

1. Create a fresh Kaggle notebook with a GPU accelerator and Internet enabled.
2. Import `notebooks/kaggle_quality_uncertainty_smoke.ipynb` from the frozen
   GitHub branch and choose **Run All**.
3. Do not edit seeds, routes, sample counts, epochs, or error rates.
4. If a cell fails, stop and return the final traceback. Do not start Full.
5. When the notebook finishes, download:

```text
quality_map_uncertainty_smoke_artifacts.zip
```

The notebook checks CUDA, runs all unit tests, executes the OMBRIA 3 x 3
two-mask Smoke, prepares 52 outcome-independent Sen1Floods11 chips, trains all
eight external routes, evaluates the independent and structured error paths,
checks both SCL providers, and verifies every packaged file hash.

Smoke scores are pipeline checks and are prohibited from the manuscript. Keep
the Kaggle output version until the local ZIP audit passes.

## Smoke audit result

The returned Smoke artifact passed the local Full-authorization audit. The
authorized SHA-256 is:

```text
32ebcd1d8bfa5cadcf9b007985548ae7d03b9ecb1015ea41b92e93b12b47e67e
```

This authorizes execution of the frozen Full protocol; it does not authorize
scientific use of Smoke scores.

## Full seed shards

Full is run as five immutable seed shards to remain inside Kaggle session
limits. Run only the supplied notebook for the requested seed and return its
ZIP without renaming internal files. The first shard is seed `7`; its exact
return artifact is:

```text
quality_map_uncertainty_full_seed7_artifacts.zip
```

Each seed shard contains the full OMBRIA 5 x 5 surface, the full 446-chip
Sen1Floods11/SCL experiment, three perturbation repetitions, checkpoints, raw
counts, per-chip/per-event rows, and file hashes. A passing seed shard remains
scientifically incomplete. Seeds `13`, `21`, `29`, and `37` are released after
the seed-7 runtime and artifact audit confirm that the operational split fits
Kaggle reliably.

## Full interpretation hold

Do not merge or interpret one shard manually. Manuscript evidence is generated
only after all five core seed artifacts, the separate SMAGNet gate, the final
merge audit, paired uncertainty, and stop/success gates are complete.
