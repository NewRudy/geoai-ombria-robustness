# OMBRIA Sentinel-1/Sentinel-2 Sensor-State Audit

Reproducible code for the manuscript:

> Controlled Sensor-State Stress Testing of Multitemporal Sentinel-1/Sentinel-2 Flood Mapping: An Event-Held-Out OMBRIA Audit

The project tests how multitemporal Sentinel-1/Sentinel-2 flood segmentation changes when the Sentinel-2 stream is clean, synthetically occluded, noisy, or zero-filled to simulate absence. It is an OMBRIA-only controlled stress test, not evidence of observed-cloud robustness, operational deployment, state-of-the-art performance, or universal generalization.

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

The v0.1.5 smoke and Full notebooks remain historical entrypoints. The v0.2 entrypoints are `notebooks/kaggle_sensor_state_v2_smoke.ipynb` and `notebooks/kaggle_sensor_state_v2_full.ipynb`; both run `scripts/run_sensor_state_v2_matrix.sh` from a pinned Git tag. In Kaggle, enable a GPU and Internet access, import the appropriate notebook, and choose **Run All**.

The runner supports:

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
- evidence packaging with checkpoint and file hashes;
- Kaggle runtime compatibility checks.

Excluded:

- raw OMBRIA data;
- credentials and access tokens;
- private manuscript and submission metadata.

Citation metadata is in [`CITATION.cff`](CITATION.cff). Code is MIT licensed. OMBRIA remains subject to its original terms and citation requirements.
