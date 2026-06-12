# OMBRIA Sentinel-1/Sentinel-2 Robustness Audit

This repository contains the reproducible code package for the manuscript:

> Auditing the Robustness of Multitemporal Sentinel-1/Sentinel-2 Fusion for Flood Mapping under Controlled Optical Degradation

The study evaluates whether multitemporal Sentinel-1/Sentinel-2 fusion for flood mapping remains reliable when Sentinel-2 inputs are partially degraded, noisy, or absent. It uses the public OMBRIA dataset and reports a controlled robustness audit rather than a new state-of-the-art architecture.

## Repository Status

This is a cleaned academic code repository prepared for manuscript submission. It intentionally excludes raw data, model checkpoints, Kaggle cache files, notebook scratch work, and large training artifacts.

Included:

- OMBRIA data loading utilities.
- Compact U-Net training and evaluation script.
- Robustness-matrix execution script.
- Summary, analysis, plotting, and qualitative-panel export scripts.
- Reported summary tables and the main robustness figure.

Excluded:

- Raw OMBRIA data.
- Trained model checkpoints.
- Cloud execution logs and intermediate artifacts.
- Manuscript drafting files.

## Main Result Summary

The manuscript reports three-seed mean results. Clean Sentinel-1/Sentinel-2 fusion gives the strongest clean-input IoU, but clean multimodal training is brittle under complete Sentinel-2 removal. Lightweight degradation training improves robustness under controlled partial/noisy optical degradation, while complete Sentinel-2 absence is better handled by an explicit bitemporal Sentinel-1 fallback.

Key reported values:

- Clean multimodal, clean input: IoU `0.6834`, F1 `0.8053`.
- Light degradation training, clean input: IoU `0.6540`, F1 `0.7831`.
- Clean multimodal, all S2 missing: IoU `0.0466`, F1 `0.0790`.
- Light degradation training, all S2 missing: IoU `0.3124`, F1 `0.4479`.
- S1 bitemporal fallback: IoU `0.4092`, F1 `0.5642`.

See:

- `results/tables/main_results.md`
- `results/tables/seed_level_uncertainty.md`
- `results/tables/fallback_policy.md`
- `results/figures/figure_robustness_tradeoff.svg`

## Environment

Python `3.10` or newer is recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the available CUDA runtime. Kaggle Tesla T4 runs used PyTorch with CUDA and batch size `8`.

## Data

The code expects the OMBRIA dataset in:

```text
external/OMBRIA/
```

The matrix script will clone the public OMBRIA repository automatically if the path does not exist:

```bash
ROOT=external/OMBRIA bash scripts/run_ombria_followup_matrix.sh
```

If the automatic clone route is unavailable, download OMBRIA manually and keep the same directory layout:

```text
external/OMBRIA/
  OmbriaS1/
    train/
    test/
  OmbriaS2/
    train/
    test/
```

## Reproducing the Robustness Matrix

Run the full three-seed matrix:

```bash
EPOCHS=25 BATCH_SIZE=8 BASE_CHANNELS=16 SEEDS="7 13 21" \
RUNS_DIR=results/runs/ombria_robustness \
bash scripts/run_ombria_followup_matrix.sh
```

The script trains and evaluates:

- clean multimodal Sentinel-1/Sentinel-2 fusion;
- lightweight Sentinel-2 degradation training;
- balanced Sentinel-2 degradation training;
- clean, patch, noise, post-S2-missing, and all-S2-missing test conditions.

The S1-only fallback can be trained with:

```bash
python scripts/train_ombria_unet.py \
  --root external/OMBRIA \
  --variant s1_bitemporal \
  --epochs 25 \
  --batch-size 8 \
  --base-channels 16 \
  --seed 7 \
  --out-dir results/runs/ombria_s1_fallback
```

Repeat the S1 fallback command for seeds `13` and `21`, then summarize the run directory with `scripts/summarize_ombria_runs.py`.

## Controlled Sentinel-2 Degradation Modes

The study uses controlled stress tests:

- `patch_after`: zero-valued patches in post-event Sentinel-2.
- `noise_after`: post-event Sentinel-2 replaced by random uniform noise.
- `zero_after`: post-event Sentinel-2 replaced by zeros.
- `zero_all`: both pre-event and post-event Sentinel-2 replaced by zeros.

These are not real cloud masks and should not be interpreted as operational cloud-robustness evidence.

## Repository Structure

```text
configs/                         Configuration examples.
data/                            Placeholder only; raw data is not committed.
docs/                            Data, reproducibility, and result notes.
results/figures/                 Lightweight reported figure assets.
results/tables/                  Reported summary tables.
scripts/                         Training, evaluation, analysis, and plotting scripts.
src/geoai_ombria_robustness/     Dataset and utility code.
```

## Citation

If this repository supports a published article, please cite the article and this code archive. A `CITATION.cff` file is included and should be updated with the final DOI after acceptance or archive deposit.

## License

Code is released under the MIT License unless institutional policy requires a different license before public release.
