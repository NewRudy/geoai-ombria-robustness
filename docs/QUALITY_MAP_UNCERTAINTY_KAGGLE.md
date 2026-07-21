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
scientifically incomplete.

## Seed-7 audit result and released notebooks

The returned seed-7 artifact passed all 11 local release-audit sections. Its
authorized SHA-256 is:

```text
9aa4b4c7f752fc684a304fbc6138242d3931f54527e089379d8759bae2990e8e
```

Run the remaining frozen shards one at a time, each in a fresh Kaggle GPU
session, using exactly these notebooks and returning the correspondingly named
ZIP:

| Seed | Notebook | Return artifact |
|---:|---|---|
| 13 | `notebooks/kaggle_quality_uncertainty_full_seed13.ipynb` | `quality_map_uncertainty_full_seed13_artifacts.zip` |
| 21 | `notebooks/kaggle_quality_uncertainty_full_seed21.ipynb` | `quality_map_uncertainty_full_seed21_artifacts.zip` |
| 29 | `notebooks/kaggle_quality_uncertainty_full_seed29.ipynb` | `quality_map_uncertainty_full_seed29_artifacts.zip` |
| 37 | `notebooks/kaggle_quality_uncertainty_full_seed37.ipynb` | `quality_map_uncertainty_full_seed37_artifacts.zip` |

Do not edit source commit, routes, epochs, rates, repetitions, datasets, or
checkpoint rules. If a shard fails, return its final traceback rather than
reducing the protocol.

## Core Full completion

All five frozen core shards (`7`, `13`, `21`, `29`, and `37`) have now passed
the local shard-set gate. Their paired core analysis is generated offline; do
not manually recompute or selectively copy shard scores from Kaggle.

## Official SMAGNet Smoke

The remaining published-architecture gate starts with one frozen Smoke run.
Create a fresh Kaggle notebook with GPU and Internet enabled, import
`notebooks/kaggle_quality_uncertainty_smagnet_smoke.ipynb`, and choose **Run
All**. Do not edit the pinned experiment commit, official SMAGNet commit, seed,
epochs, subset sizes, or conditions.

If an earlier attempt stopped in `ensure_cuda_compat.py`, discard that Session
before importing the current notebook. The failed process may have partially
replaced the preinstalled PyTorch stack. The revised preflight records the
NVIDIA device, compute capability, installed torch distribution, and free disk;
it refuses a CPU Session before changing packages and frees an incompatible
preinstalled CUDA stack before installing the P100-compatible build.

Download and return exactly:

```text
quality_map_uncertainty_smagnet_smoke_artifacts.zip
```

This run imports the byte-verified official architecture, trains the dual
outputs for two epochs on the frozen 24/12 subset, evaluates 16 conditions on
the 12/4 test/Bolivia subset, and verifies every packaged file hash. Its scores
are pipeline-only and prohibited from the manuscript.

The returned Smoke archive passed all 10 independent local audit checks on
2026-07-21. Its SHA-256 is
`eedaf8027e5720ff1ee72f39bc98f12e56a82928fb13a988f2bfe96075c1b0e9`.
Smoke scores remain pipeline-only and prohibited from the manuscript.

Seed 7 is now released as:

```text
notebooks/kaggle_quality_uncertainty_smagnet_full_seed7.ipynb
```

Run it in a fresh Kaggle GPU Session with Internet enabled and download exactly:

```text
quality_map_uncertainty_smagnet_full_seed7_artifacts.zip
```

The notebook uses 200 epochs, all 446 prepared records, 54 conditions, and
three perturbation repetitions. It is resumable within the same live Session.
Do not alter the pinned source, official SMAGNet commit, seed, epochs, records,
conditions, or repetitions. The seed-7 scores remain scientifically
uninterpretable until all five Full shards pass independent audit and are
paired offline with the frozen seed-matched Sentinel-1 reference. Seeds 13,
21, 29, and 37 remain held until the returned seed-7 Full archive passes its
local gate. Formal manuscript Results remain on hold until the complete set and
post-analysis claim-evidence audit pass.
