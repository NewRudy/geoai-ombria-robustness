# Locked Confirmatory Protocol

## Purpose

Test whether the route-ordering pattern observed on the exploratory 70-chip OMBRIA public test split persists on separately released 2021 flood events. This protocol is confirmatory for OMBRIA only.

## Locked data boundary

- Training pool: matched chips in `OmbriaS1/train` and `OmbriaS2/train` only.
- Validation: fixed 15% split generated with seed `20260710`.
- Confirmation: `2021/ALBANIA` (22 chips), `FRANCE` (88), `GUYANA` (30), and `TIMOR` (10).
- Total confirmation set: 150 matched chips.
- OMBRIA source revision: `38a490355f76da8ce27ed051138f03f3492a6e46`.

The 2021 labels must not be used for training, checkpoint selection, threshold selection, route selection, or parameter tuning.

## Locked seeds and routes

- Model seeds: `7`, `13`, `21`.
- Split seed: `20260710`.
- Perturbation seed: `20260710`.
- Stochastic perturbation repetitions: three per model seed.
- Checkpoint: highest validation IoU only.

Routes:

1. Clean multimodal training.
2. Light degradation training.
3. Matched degradation schedule without quality maps.
4. The identical matched schedule with two binary S2 quality maps.
5. Bitemporal S1-only reference.

Routes 3 and 4 form the quality-map comparison. Because the input dimensionality changes, describe it as a matched architectural-input ablation rather than a perfect causal experiment.

## Locked test states

`none`, `patch_after`, `cloud_after_30`, `cloud_after_50`, `cloud_after_70`, `noise_after`, `zero_after`, and `zero_all`.

Cloud-like masks are synthetic zero-valued occlusion and are not evidence of observed-cloud or cloud-shadow robustness.

## Metrics

- Primary: flood IoU from globally accumulated TP/FP/FN counts across all 150 chips.
- Secondary: F1, precision, recall, and accuracy from the same global counts.
- Diagnostic: per-event global metrics and per-chip metrics.
- Aggregation: average perturbation repetitions within each model seed, then report the mean and a two-sided 95% Student-t interval across the three model seeds (df = 2). With the split and perturbation panel fixed, this interval describes model-initialization and training-order variation only; it is not an event-, split-, perturbation-, spatial-, or population-level interval.
- No null-hypothesis significance test is planned for three model seeds.

## Qualitative export

Select one chip per event by median positive ground-truth flood fraction, without viewing model outputs. Export one high-resolution panel per state with post-event S2 false color, reference mask, clean-route probability, light-route probability, matched-training control probability, matched quality-map probability, S1-only probability, and a shared 0–1 color scale.

## Evidence package

The returned archive must contain per-epoch training metrics for every route and seed, split and evaluation configurations, per-event and per-chip metrics, qualitative panels, a complete console log, runtime and experiment manifests, a Python environment freeze, and SHA-256 hashes for every packaged evidence file. Each evaluation configuration records the hash and byte size of its validation-selected checkpoint; packaging verifies those records against the expected route-by-seed checkpoint paths and emits a checkpoint manifest without including the weights. Packaging fails if any expected route, seed, state, trajectory, checkpoint identity, or evaluation link is absent.

## Interpretation rule

- Retain a route-ordering claim only if its direction is reproduced in the pooled event-held-out result and is not driven solely by one event.
- Weaken or remove any claim that reverses.
- Report event heterogeneity explicitly.
- Preserve the S1 comparison boundary based on measured ordering, not a predetermined router policy.
- Do not describe the result as cross-dataset or operational generalization.
