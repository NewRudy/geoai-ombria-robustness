# Quality-Map Uncertainty and Fusion-Safety Protocol

Status: frozen before the first quality-map uncertainty Smoke outcome

Target: *Remote Sensing*, Article

Primary boundary: controlled OMBRIA discovery plus Sen1Floods11/SCL external replication

## Scientific question

The completed v0.3 Full run is retained as a controlled pilot. It showed that
aligned optical-availability information matters under partial synthetic
occlusion, but it did not show that the QGSF hard-gated architecture is better
than simple quality concatenation. The new study therefore asks:

> When the optical-quality map is imperfect, under what error conditions is
> Sentinel-1/Sentinel-2 fusion still safer than a separately trained
> Sentinel-1-only reference?

Let `M` be the reference optical-availability map and `M_hat` the map supplied
to a model. Optical content is governed by `M`; only `M_hat` is perturbed. This
separation prevents a model from receiving an oracle description of the
degradation applied to its optical input.

## Falsifiable hypotheses

1. False-available errors cause greater fusion regret than conservative
   false-unavailable errors.
2. Translation, dilation, and erosion cause more harm than random errors with
   exactly matched false-available and false-unavailable counts.
3. Each fusion strategy has a measurable zero-gain boundary relative to the
   Sentinel-1-only reference.
4. Quality-error-aware training reduces worst-case Sentinel-1-relative regret
   or expands the descriptive safe region without losing more than `0.01`
   clean IoU.

Every hypothesis may fail. A failed hypothesis is reported as a result and is
not replaced after inspecting Full scores.

## Dataset roles

### OMBRIA controlled discovery

- Synthetic optical degradation is called `cloud-like occlusion`.
- `M` is the exact applied degradation mask.
- The dense false-available by false-unavailable response surface is primary.
- The audited v0.3 Full artifact is pilot evidence, not external validation.

### Sen1Floods11 external replication

- The pinned manifest contains all 446 hand-labeled chips.
- Official splits contain 252 train, 89 validation, and 90 test chips.
- The 15-chip Bolivia file is reported separately as a small event-held-out
  audit, not broad geographic validation.
- Reference quality is an available Sentinel-2 L2A SCL class intersected with
  the official S2Hand chip valid-data mask.
- SCL is an operational proxy, not human cloud truth.

## Capacity-controlled route matrix

1. `s1_reference`: independently trained safety reference.
2. `early_fusion`: S1/S2 fusion without explicit quality information.
3. `early_fusion_dropout`: early fusion with optical modality dropout.
4. `quality_concat`: quality map appended to the input.
5. `quality_concat_error_aware`: the same architecture trained with independent
   quality-map errors.
6. `hard_quality_gate`: oracle-style hard masking and gated fusion.
7. `hard_quality_gate_error_aware`: the same hard-gated architecture trained
   with independent quality-map errors.
8. `soft_quality_prior_error_aware`: supplied quality is a correctable prior;
   a separate structural-absence flag preserves the complete-optical-absence
   boundary.

All capacity-controlled routes use the same data split, optimization rule,
validation-only checkpoint selection, base width, and model seeds.

## Strong published baseline gate

SMAGNet is the closest published method and cannot be omitted silently.

- Official repository: `https://github.com/ASUcicilab/SMAGNet`
- Frozen official commit: `4371df08e6ca3b9d71c0385ad57b589830469a0c`
- License: MIT
- Preferred Full use: adapt and retrain the official architecture on the
  frozen Sen1Floods11 inputs, splits, and error conditions.
- Allowed fallback: a clearly labeled mechanism-matched implementation only
  if the official model cannot be adapted reproducibly within Kaggle runtime
  or memory. The failure and environment must be recorded before Full scores
  are inspected.

SMAGNet is reported as a separate published-architecture comparison, not as a
capacity-controlled causal contrast. Encoder pretraining, parameter count, and
all deviations from the official defaults are reported explicitly.

## Smoke gate

Smoke is an execution test and never scientific evidence.

- Model seed: `7`
- Epochs: `2`
- Independent rates: `{0.0, 0.2, 0.4} x {0.0, 0.2, 0.4}`
- Sen1Floods11 subset: 24 train, 12 validation, 12 test, and 4 Bolivia chips,
  selected by an outcome-independent event-stratified hash rule.
- Routes: all eight capacity-controlled routes.
- Structured code paths: 5% translation, unavailable-region dilation and
  erosion at radius 8, plus an exact-count random control for each.
- Complete optical absence: evaluated separately.

Smoke passes only if data preparation, CUDA training, checkpoint hashing,
matrix evaluation, per-chip/per-event export, S1-relative summaries, and ZIP
hash verification all pass. Smoke values are prohibited from the manuscript.

## Full matrix

- Model seeds: `7`, `13`, `21`, `29`, `37`
- Epochs: `25`
- Independent rates: `{0.0, 0.05, 0.10, 0.20, 0.40}` on both error axes
- Perturbation realizations: three deterministic sample-level repetitions
- Sen1Floods11 evaluation: official test and Bolivia reported separately
- Structured translations: 5% and 10% of chip width in four cardinal
  directions; for 512-pixel chips these are 26 and 51 pixels
- Morphology: unavailable-region dilation and erosion at radii 4, 8, and 16
- Every structured condition has a random control with exact
  false-available/false-unavailable counts matched both over the whole quality
  map and inside the valid segmentation-evaluation domain
- Complete optical absence is outside the local error surface

The Full notebook is released to the operator only after the Smoke archive is
audited. No route, rate, seed, split, or checkpoint rule may be changed after
Full outcome inspection.

## Primary estimands

- Flood IoU is primary; F1, precision, and recall are secondary.
- Chip-global and event-equal results are both retained.
- For condition `c`, the primary safety contrast is
  `delta_S1(c) = IoU_fusion(c) - IoU_S1`.
- Report the two-dimensional response surface, zero-gain contour,
  Sentinel-1-relative regret `max(0, -delta_S1)`, and the descriptive fraction
  of the frozen grid whose lower paired interval is non-negative.
- Primary uncertainty is a paired hierarchical bootstrap over events and chips
  within model seed. Seed-level intervals are a sensitivity analysis.

## Article success and stop gates

The Article story may proceed only if:

1. the error response is reproducible across five model seeds;
2. at least one central direction transfers from OMBRIA to Sen1Floods11;
3. test, event-equal, and Bolivia results are all reported;
4. false-available and false-unavailable effects are reported even if their
   expected asymmetry fails; and
5. the closest published baseline is run or its prespecified adaptation
   failure is documented.

An error-aware method claim additionally requires a paired improvement in
worst-case regret or safe-region area, a clean-IoU loss no greater than `0.01`,
and preservation of the explicit complete-absence Sentinel-1 boundary.

Stop or retarget the paper if the response surfaces are dominated by seed
noise, no central direction transfers across datasets, or the external quality
proxy cannot be reproduced. Artifact completeness alone never satisfies a
scientific gate.

## Claim-to-evidence plan

| Candidate Article claim | Required evidence | If unsupported |
| --- | --- | --- |
| False-available and false-unavailable errors have different costs | Five-seed 5 x 5 surfaces on OMBRIA and direction check on Sen1Floods11 | Report a symmetric or inconclusive response; remove the asymmetry claim |
| Spatial organization matters beyond global error rate | Paired translation/dilation/erosion minus valid-domain count-matched random controls | Treat error rate, not spatial structure, as the supported explanation |
| Fusion has a measurable S1 safety boundary | `delta_S1` response surface, zero contour, and paired uncertainty | Report that the boundary cannot be localized reliably |
| Error-aware training expands safe operation | Matched architecture on/off contrasts, worst-case regret, safe-grid fraction, and clean-IoU tolerance | Keep the paper as an empirical reliability study; make no method claim |

The corresponding main figures are: data and two-mask definition; error
taxonomy and frozen protocol; the S1-relative response surface; cross-dataset
and Bolivia interval evidence; and outcome-independently selected failure
cases. Figures are generated only from audited Full evidence.

## Evidence contract

Every run must retain source commit, environment, runtime, selected records,
input and checkpoint hashes, training trajectories, checkpoint selection,
condition definitions, requested and realized quality-error rates, raw
confusion counts, per-chip rows, per-event rows, seed summaries, and a
file-level ZIP manifest. Raw Sen1Floods11 imagery is cached for execution but
is not redistributed in the result archive.
