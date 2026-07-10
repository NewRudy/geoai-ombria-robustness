# OMBRIA Sentinel-1/Sentinel-2 Robustness Audit

Reproducible code for the manuscript:

> Auditing the Robustness of Multitemporal Sentinel-1/Sentinel-2 Fusion for Flood Mapping under Controlled Optical Degradation

The project audits how multitemporal Sentinel-1/Sentinel-2 flood segmentation behaves when the Sentinel-2 stream is clean, synthetically occluded, noisy, partially missing, or completely absent. It is an OMBRIA-only controlled robustness study, not evidence of operational deployment, observed-cloud robustness, state-of-the-art performance, or universal generalization.

## Repository scope

Included:

- OMBRIA data loading and controlled Sentinel-2 stress generation;
- compact U-Net training and validation-only checkpoint selection;
- exploratory public-split evaluation scripts;
- locked 2021 event-held-out confirmation scripts;
- global and per-chip metric export;
- matched quality-map ablation;
- high-resolution qualitative panels with an S1-only reference;
- Kaggle smoke and full-run notebooks.

Excluded:

- raw OMBRIA data;
- model checkpoints and training caches;
- credentials and Kaggle tokens;
- manuscript drafts and private submission metadata.

## Current exploratory evidence

The manuscript's current public-split matrix reports three seed-controlled runs. Selected mean IoU values are:

| Route and state | IoU |
| --- | ---: |
| Clean multimodal, clean input | 0.6821 |
| Light degradation training, clean input | 0.6521 |
| Clean multimodal, all S2 missing | 0.0447 |
| Light degradation training, all S2 missing | 0.3689 |
| S1-only reference | 0.5071 |

These results are exploratory because the public test matrix informed route comparison during development. The separately released 2021 event folders are reserved for the locked confirmation described below.

## Kaggle: click-to-run workflow

Two notebooks are provided:

- [`notebooks/kaggle_confirmatory_smoke.ipynb`](notebooks/kaggle_confirmatory_smoke.ipynb): one model seed and two epochs; checks the complete runtime path but is not scientific evidence.
- [`notebooks/kaggle_confirmatory_full.ipynb`](notebooks/kaggle_confirmatory_full.ipynb): three model seeds and 25 epochs; run only after the smoke gate succeeds.

In Kaggle, enable a GPU and Internet access, import the appropriate notebook from this repository, and choose **Run All**. Each notebook clones the immutable `v0.1.4-confirmatory` tag, so the executed code does not drift with the default branch. The notebook first returns to `/kaggle/working`, making a repeated **Run All** safe even when the previous run left the kernel inside the cloned directory. It also detects whether the installed PyTorch wheel contains the active GPU architecture; on P100 (`sm_60`) it automatically installs the official PyTorch 2.7.1 CUDA 12.6 build together with its pinned CUDA runtime dependencies before running a real CUDA convolution gate. A partially installed 2.7.1 stack from an interrupted or older notebook run is repaired automatically.

The returned archive is:

```text
results/ombria_2021_confirmatory_artifacts.zip
```

The archive retains per-epoch training metrics, runtime and experiment manifests, a package hash manifest, the complete console log, and an environment freeze alongside the evaluation tables and figures. Model checkpoints remain excluded to keep the handoff manageable.

See [`docs/KAGGLE.md`](docs/KAGGLE.md) for the short operator guide.

## Locked 2021 event-held-out confirmation

Training uses only the released `OmbriaS1/train` and `OmbriaS2/train` folders. Confirmation uses the four separately released event folders under `2021/`:

| Event folder | Matched chips |
| --- | ---: |
| ALBANIA | 22 |
| FRANCE | 88 |
| GUYANA | 30 |
| TIMOR | 10 |
| **Total** | **150** |

The locked protocol uses:

- split seed `20260710`, independent of model seeds `7`, `13`, and `21`;
- perturbation seed `20260710` and three fixed realizations for stochastic states;
- validation IoU only for checkpoint selection;
- globally accumulated TP/FP/FN/TN metrics plus per-chip rows;
- identical degradation schedules with and without binary quality maps;
- a bitemporal S1-only reference for every tested Sentinel-2 state.

Read [`docs/CONFIRMATORY_PROTOCOL.md`](docs/CONFIRMATORY_PROTOCOL.md) before changing any locked parameter.

## Local execution

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Smoke gate:

```bash
MODE=smoke EPOCHS=2 bash scripts/run_confirmatory_event_matrix.sh
```

Full locked run:

```bash
MODE=full EPOCHS=25 bash scripts/run_confirmatory_event_matrix.sh
```

The runner fetches OMBRIA at commit `38a490355f76da8ce27ed051138f03f3492a6e46` and verifies the required train and 2021 event folders before starting.

## Controlled Sentinel-2 states

- `none`: unmodified input.
- `patch_after`: eight zero-valued post-event patches.
- `cloud_after_30`, `cloud_after_50`, `cloud_after_70`: synthetic zero-valued elliptical occlusion masks.
- `noise_after`: post-event Sentinel-2 replaced with uniform random noise.
- `zero_after`: post-event Sentinel-2 set to zero.
- `zero_all`: both Sentinel-2 timestamps set to zero.

The cloud-like masks are synthetic occlusion and must not be relabeled as observed clouds, cloud shadows, or atmospheric-correction errors.

## Repository structure

```text
configs/                         Configuration examples
data/                            Placeholder only; raw data is not committed
docs/                            Data, protocol, and reproducibility notes
notebooks/                       Kaggle smoke and full-run entrypoints
results/tables/                  Audited exploratory summary tables
scripts/                         Training, evaluation, export, and runner scripts
src/geoai_ombria_robustness/     Dataset and degradation utilities
```

## Citation and license

Citation metadata is available in [`CITATION.cff`](CITATION.cff). Code is released under the MIT License. OMBRIA remains subject to the terms and citation requirements of its original repository and publication.
