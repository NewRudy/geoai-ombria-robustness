# OMBRIA Sentinel-1/Sentinel-2 Sensor-State Audit

Reproducible code for the manuscript:

> Controlled Sensor-State Stress Testing of Multitemporal Sentinel-1/Sentinel-2 Flood Mapping: An Event-Held-Out OMBRIA Audit

The project tests how multitemporal Sentinel-1/Sentinel-2 flood segmentation changes when the Sentinel-2 stream is clean, synthetically occluded, noisy, or zero-filled to simulate absence. It is an OMBRIA-only controlled stress test, not evidence of observed-cloud robustness, operational deployment, state-of-the-art performance, or universal generalization.

## Current scientific reset: quality-map uncertainty

The current work no longer presents QGSF as the main method innovation. The v0.3 Full run did not show a reproducible advantage over aligned quality concatenation, and the published SMAGNet study now substantially overlaps the original spatial-mask/gated-fusion framing.

The new experiment separates two variables that oracle-mask studies conflate:

- the reference availability map used to degrade optical content;
- the imperfect quality map supplied to the fusion model.

It measures false-available and false-unavailable errors independently and reports fusion performance relative to an explicit S1-only reference. OMBRIA provides the controlled discovery surface. A pinned Sen1Floods11 manifest maps all 446 hand-labeled chips to Sentinel-2 L2A SCL assets for an external geospatial quality-proxy workflow. SCL remains an operational proxy, not human cloud truth.

The frozen research questions, eight capacity-controlled routes, published-baseline gate, Full matrix, and Article stop criteria are defined in [`docs/QUALITY_MAP_UNCERTAINTY_PROTOCOL.md`](docs/QUALITY_MAP_UNCERTAINTY_PROTOCOL.md).

Run the new Smoke notebook first:

- [`notebooks/kaggle_quality_uncertainty_smoke.ipynb`](notebooks/kaggle_quality_uncertainty_smoke.ipynb)
- [`manifests/sen1floods11_scl_manifest.json`](manifests/sen1floods11_scl_manifest.json)

The Smoke trains one seed for two epochs on OMBRIA and on an outcome-independent Sen1Floods11 subset. It evaluates a 3 × 3 quality-error grid, exercises all eight external routes, tests structured-versus-matched-random errors, verifies Earth Search and Planetary Computer SCL access, and packages an 11-event alignment audit. Reference quality requires both an available SCL class and a valid pixel in the official Sentinel-2 chip. Raw Sen1Floods11 rasters are cached for execution and are not redistributed. The workflow returns:

```text
results/quality_map_uncertainty_smoke_artifacts.zip
```

Smoke scores validate execution only and cannot enter the manuscript.

## v0.3 completed pilot

v0.3 tested a falsifiable QGSF method candidate. The architecture uses separate S1 and shared bitemporal S2 encoders, sanitizes unavailable optical inputs, and applies hard availability-constrained learned gates at three scales. When both S2 quality maps are zero, every optical term supplied to fusion is exactly zero and the network follows its S1-driven path.

The Full matrix compares:

- clean and corruption-trained early fusion;
- quality maps concatenated as input channels;
- aligned QGSF;
- the same QGSF architecture with prevalence-preserving shifted quality maps;
- S1-only and S2-only references.

The architecture, matched controls, paired contrasts, and decision thresholds are frozen in [`docs/QUALITY_GATED_FUSION_V3_PROTOCOL.md`](docs/QUALITY_GATED_FUSION_V3_PROTOCOL.md). The completed Full result did not support architectural superiority, so v0.3 is retained as audited pilot evidence rather than the current submission claim.

## Released v0.1.5 evidence

The corrected v0.1.5 evidence archive contains the five-route, three-run, eight-state evaluation over 150 chips from four released 2021 OMBRIA events. The statistical correction recomputes `df = 2` Student-t intervals from the original raw confusion counts; model means are unchanged and no retraining was performed.

- [v0.1.5-confirmatory release](https://github.com/NewRudy/geoai-ombria-robustness/releases/tag/v0.1.5-confirmatory)
- Evidence archive SHA-256: `dc0e1bf1c27cca1dcb208bc19afa8bff68152347dc4a8958deace128eb69ca6f`

The historical v0.1.4 artifact did not retain model weights. That limitation is explicit in the release manifest.

## v0.2 protocol-frozen follow-up

v0.2 repairs the strongest design and traceability limitations before final manuscript freeze:

- seven routes: clean multimodal, light degradation, matched-distribution control, known-quality maps, same-width mislocalized-quality control, S1-only, and S2-only;
- five model repeats (`7`, `13`, `21`, `29`, `37`);
- independent model, minibatch-order, training-corruption, and evaluation-perturbation random streams;
- training corruption fixed by epoch, chip ID, and corruption seed rather than dataset call order;
- exact applied corruption masks for quality channels;
- primary clean-selected and prespecified robustness-selected checkpoints;
- exact checkpoint weights and hashes in the Full artifact;
- train/validation exact and thumbnail-similarity duplicate audit;
- selected-chip probability arrays and FP/FN error-map panels.

The same four outcome events have already been inspected. v0.2 is therefore a follow-up/sensitivity analysis, not a new independent confirmation. See [`docs/SENSOR_STATE_V2_PROTOCOL.md`](docs/SENSOR_STATE_V2_PROTOCOL.md).

## Controlled Sentinel-2 states

- `none`: unmodified input;
- `patch_after`: eight zero-valued post-event patches;
- `cloud_after_30`, `cloud_after_50`, `cloud_after_70`: synthetic opaque elliptical occlusion;
- `noise_after`: post-event Sentinel-2 replaced with uniform noise;
- `zero_after`: post-event Sentinel-2 set to zero;
- `zero_all`: both Sentinel-2 timestamps set to zero.

Cloud-like occlusion is synthetic. Zero filling simulates input absence without removing network channels.

## Kaggle click-to-run workflow

The current entrypoint is `notebooks/kaggle_quality_uncertainty_smoke.ipynb`. It verifies a real CUDA convolution, runs the tests, executes the two-mask OMBRIA and Sen1Floods11 Smoke protocols, checks both SCL providers, verifies the external decision gate and packaged file hashes, and exposes the result archive.

```bash
bash scripts/run_quality_uncertainty_smoke.sh
```

The v0.3 historical entrypoints remain `notebooks/kaggle_quality_gated_v3_smoke.ipynb` and `notebooks/kaggle_quality_gated_v3_full.ipynb`. Both clone the pinned `v0.3.1-quality-gated` tag and reproduce the completed pilot.

For historical v0.3 reproduction, its Smoke uses one seed, two epochs, and four core routes only. Run that Full workflow only when reproducing the completed pilot; it is not the current experiment.

```bash
MODE=smoke bash scripts/run_quality_gated_v3_matrix.sh
MODE=full bash scripts/run_quality_gated_v3_matrix.sh
```

The returned archive is:

```text
results/ombria_quality_gated_v3_artifacts.zip
```

The historical v0.1.5 and v0.2 notebooks remain available. The v0.2 entrypoints are `notebooks/kaggle_sensor_state_v2_smoke.ipynb` and `notebooks/kaggle_sensor_state_v2_full.ipynb`.

The v0.2 runner supports:

```bash
MODE=smoke EPOCHS=2 bash scripts/run_sensor_state_v2_matrix.sh
MODE=full EPOCHS=25 bash scripts/run_sensor_state_v2_matrix.sh
```

Smoke uses one seed and the clean-selected checkpoint policy. Full uses five seeds and both clean- and robustness-selected policies. The returned archive is:

```text
results/ombria_sensor_state_v2_artifacts.zip
```

Kaggle P100 uses Pascal `sm_60`. The notebook reuses the repository's CUDA compatibility gate, which installs the official PyTorch 2.7.1 CUDA 12.6 stack only when the active wheel cannot execute a real CUDA convolution.

## Data boundary

Training uses matched chips under `OmbriaS1/train` and `OmbriaS2/train`. Evaluation uses:

| Event | Chips |
| --- | ---: |
| Albania | 22 |
| France | 88 |
| Guyana | 30 |
| Timor | 10 |
| **Total** | **150** |

The public PNG release does not include a chip-to-scene or geospatial grouping manifest. The validation split remains chip-level; the near-duplicate audit does not substitute for spatial grouping.

## Local setup and checks

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The runner fetches OMBRIA at commit `38a490355f76da8ce27ed051138f03f3492a6e46` and verifies the required train and 2021 event folders before starting.

## Repository scope

Included:

- OMBRIA loading and controlled Sentinel-2 stress generation;
- compact U-Net training and validation-only checkpoint selection;
- global, per-event, and per-chip metric export;
- quality-map and modality controls;
- capacity-controlled quality-gated sensor-state fusion;
- prespecified paired method contrasts and an outcome-independent decision gate;
- evidence packaging with checkpoint and file hashes;
- Kaggle runtime compatibility checks.

Excluded:

- raw OMBRIA data;
- credentials and access tokens;
- private manuscript and submission metadata.

Citation metadata is in [`CITATION.cff`](CITATION.cff). Code is MIT licensed. OMBRIA remains subject to its original terms and citation requirements.
