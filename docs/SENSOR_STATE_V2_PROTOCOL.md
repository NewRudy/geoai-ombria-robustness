# Sensor-State v0.2 Follow-up Protocol

## Status and inferential boundary

This protocol is frozen before the v0.2 Full run. It is a follow-up and sensitivity analysis over the same four OMBRIA 2021 outcome events used in v0.1.4, not an independent confirmation. It repairs prespecified design and traceability limitations without claiming analyst-unseen events, cross-dataset generalization, or observed-cloud robustness.

## Fixed data and repeat structure

- Training pool: matched `OmbriaS1/train` and `OmbriaS2/train` chips.
- Validation split: fixed chip split generated with seed `20260710`.
- Outcome events: Albania (22 chips), France (88), Guyana (30), and Timor (10).
- Model seeds: `7`, `13`, `21`, `29`, and `37`.
- Independent per-repeat streams:
  - model initialization: model seed;
  - minibatch order: model seed + `200000`;
  - training corruption: model seed + `300000`;
  - evaluation perturbations: fixed seed `20260710`.
- Training corruption is a deterministic function of epoch, chip ID, and corruption seed. It is independent of dataset call order and worker scheduling.

The public OMBRIA PNG release does not provide a chip-to-scene or geospatial grouping manifest. The fixed validation split therefore remains chip-level. An exact and thumbnail-similarity train/validation audit is recorded, but it cannot establish scene-level or spatial independence.

## Seven routes

1. Clean eight-channel S1/S2 multimodal training.
2. Light multimodal degradation training.
3. Eight-channel matched-distribution control.
4. Ten-channel known-corruption quality-map route.
5. Ten-channel mislocalized-quality control with the same corrupted imagery, model width, initialization seed, minibatch order, and corruption realization as route 4; spatial quality maps are shifted while unavailable-pixel prevalence is preserved.
6. Bitemporal S1-only reference.
7. Bitemporal S2-only reference.

Quality maps are produced from the applied perturbation mask. They are not inferred from zero-valued image pixels. The known-quality route remains an oracle-information diagnostic, not a deployable quality detector.

## Eight test states

`none`, `patch_after`, `cloud_after_30`, `cloud_after_50`, `cloud_after_70`, `noise_after`, `zero_after`, and `zero_all`.

Cloud-like masks are synthetic opaque zero-valued occlusion. Zero filling simulates input absence without changing network topology.

## Checkpoint policies

- Primary: `best_clean.pt`, the epoch with maximum clean validation IoU.
- Prespecified sensitivity: `best_robust.pt`, the epoch with maximum mean validation IoU across `none`, `cloud_after_50`, and `zero_after`.

Both policies are evaluated over the complete seven-route/eight-state/five-repeat matrix. No outcome-event prediction is used for checkpoint selection.

## Metrics and uncertainty

- Primary aggregation: confusion counts pooled over all 150 event chips.
- Diagnostics: per-event, per-chip, equal-event, chip-macro, event-omission, and route-difference summaries.
- Stochastic test states use three matched perturbation realizations per repeat; deterministic states use one.
- Repetitions are averaged within model seed before reporting the across-run mean and two-sided Student-t interval with `df = 4`.
- Intervals describe five-run training variability under one split and one evaluation-perturbation panel. They do not quantify spatial, split, event-sampling, perturbation-distribution, or population uncertainty.

## Evidence preservation

The Full package includes both selected checkpoints for every route/repeat, their SHA-256 hashes, checkpoint-selection records, evaluation configurations, raw confusion counts, per-chip rows, training trajectories, split and near-duplicate audits, runtime/environment provenance, selected-chip float16 probability arrays converted from float32 inference, FP/FN error-map panels, and a file-level hash manifest.

## Decision rule

The v0.2 run can strengthen or weaken v0.1.4 conclusions. It cannot be labeled independent confirmation. Manuscript changes must report effect sizes, leading margins, aggregation sensitivity, and run stability; nominal leaders are not statistical winners.
