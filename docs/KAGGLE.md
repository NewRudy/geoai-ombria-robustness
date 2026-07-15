# Kaggle Operator Guide

## v0.3 quality-gated method follow-up

Use `notebooks/kaggle_quality_gated_v3_smoke.ipynb` first. It clones the pinned `v0.3.0-quality-gated` tag, verifies CUDA with a real convolution, runs all repository tests, trains the four core method/control routes for two epochs, evaluates the fixed sensor states, checks the prespecified-contrast exporter, and verifies the returned archive.

Smoke is a pipeline gate only. Do not quote, graph, or use its scores in the manuscript.

After the Smoke archive passes review, open a fresh Kaggle session and import `notebooks/kaggle_quality_gated_v3_full.ipynb`. Full runs seven routes, five model seeds, 25 epochs, both checkpoint policies, and all eight sensor states. Its route matrix and interpretation thresholds are frozen in `docs/QUALITY_GATED_FUSION_V3_PROTOCOL.md`; do not change them after inspecting any Full result.

One-time settings:

1. Enable a GPU accelerator.
2. Enable Internet access.
3. Import the Smoke notebook and choose **Run All**.

Return this file after Smoke:

```text
results/ombria_quality_gated_v3_artifacts.zip
```

The archive includes selected weights, their hashes, training configurations, raw event and chip metrics, paired contrast tables, the decision-gate JSON, runtime provenance, and a file-level hash manifest. Full also includes selected-chip error maps and aligned-versus-shifted effective-gate panels. Keep the Kaggle output version until the archive has been downloaded and reviewed locally.

Full is expected to take roughly 5--7 hours on a P100; this is an estimate, and the actual duration is retained in `run.log`.

## v0.2 sensor-state follow-up

Use `notebooks/kaggle_sensor_state_v2_smoke.ipynb` first and `notebooks/kaggle_sensor_state_v2_full.ipynb` only after the smoke artifact passes review. Both clone the immutable tag, verify a real CUDA convolution, run the seven-route protocol, verify the returned archive, and show direct links for the evidence package.

- Smoke: one seed, two epochs, clean-selected checkpoint policy only; pipeline validation, not evidence.
- Full: five seeds, 25 epochs, both clean- and robustness-selected checkpoint policies.
- Expected P100 time: approximately 4–6 hours for Full; actual time is recorded in the log.
- Full output: `results/ombria_sensor_state_v2_artifacts.zip`.

The Full archive deliberately includes the exact selected weights. It will be substantially larger than the v0.1.5 evidence-only archive. Keep the Kaggle output version until the local download and archive manifest both verify.

The seven routes, independent random streams, two checkpoint rules, and interpretation boundary are defined in `docs/SENSOR_STATE_V2_PROTOCOL.md`. Do not edit the Full matrix after seeing Smoke or Full outcome scores.

## Historical v0.1.5 workflow

The v0.1.5 notebooks clone the immutable `v0.1.5-confirmatory` tag from GitHub, install dependencies, verify CUDA compatibility, compile-check the scripts, execute the historical matrix, verify the result archive, and expose a download link. Their setup cell changes to `/kaggle/working` before replacing an earlier clone, so rerunning the notebook cannot delete its own current working directory.

Kaggle P100 uses Pascal compute capability `sm_60`. Current CUDA 12.8 PyTorch wheels can omit that architecture, producing `no kernel image is available for execution on the device`. The notebooks detect this condition and install the official PyTorch 2.7.1 CUDA 12.6 build with its complete pinned CUDA dependency set, including cuSPARSELt, then run a real `Conv2d` CUDA gate before any experiment starts. T4 and other already-supported GPUs keep the installed PyTorch build. If an older notebook left a partial 2.7.1 installation that cannot import, the compatibility helper repairs it before the gate.

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

The archive contains summary tables, per-chip metrics, evaluation configurations, split manifests, per-epoch training metrics, high-resolution qualitative panels, the complete console log, a Python environment freeze, runtime and experiment manifests, and SHA-256 hashes for every packaged evidence file. Each evaluation configuration also records the exact checkpoint hash and byte size, and the packager verifies those links before emitting a checkpoint manifest. Model weights are excluded to keep the archive manageable. The notebook's final cell verifies ZIP CRC and both file and checkpoint manifests before displaying the download link.
