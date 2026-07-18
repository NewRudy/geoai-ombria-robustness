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

## Full hold

Do not run or construct an ad hoc Full notebook. Full is generated from
`docs/QUALITY_MAP_UNCERTAINTY_PROTOCOL.md` only after the returned Smoke archive
passes local integrity, completeness, numerical, and scope audits. The Full
source commit, model seeds, five-by-five rate grid, structured conditions,
checkpoint rule, and published-baseline decision are frozen before Full starts.
