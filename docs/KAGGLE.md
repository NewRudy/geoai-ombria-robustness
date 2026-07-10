# Kaggle Operator Guide

The notebooks in `notebooks/` are the supported Kaggle entrypoints. They clone the immutable `v0.1.0-confirmatory` tag from GitHub, install dependencies, compile-check the scripts, execute the matrix, verify the result archive, and expose a download link.

## One-time Kaggle settings

1. Open a new Kaggle Notebook.
2. Enable a GPU accelerator.
3. Enable Internet access.
4. Import the smoke notebook from this repository.

No GitHub token is required because this repository contains code only and is public. Never paste Kaggle or GitHub credentials into a notebook cell.

## First run: smoke gate

Import `notebooks/kaggle_confirmatory_smoke.ipynb`, then choose **Run All**.

The smoke gate uses one model seed and two epochs. It checks data retrieval, training, evaluation, per-chip export, figure generation, and artifact packaging. Its scores are not scientific evidence.

Download:

```text
results/ombria_2021_confirmatory_artifacts.zip
```

If any cell fails, stop and return the final traceback. Do not proceed to the full run.

## Second run: locked full matrix

After the smoke archive is reviewed, start a fresh Kaggle session, import `notebooks/kaggle_confirmatory_full.ipynb`, and choose **Run All**.

The full notebook uses three model seeds, 25 epochs, the locked split and perturbation seeds, and the full route/state matrix. Do not edit those parameters after inspecting results.

## Returned artifact

The archive contains summary tables, per-chip metrics, evaluation configurations, split manifests, and high-resolution qualitative panels. Model checkpoints are excluded to keep the archive manageable.
