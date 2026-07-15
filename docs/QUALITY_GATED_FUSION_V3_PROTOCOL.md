# Quality-Gated Sensor-State Fusion v0.3 Protocol

## Frozen status and evidence boundary

This protocol is written before the v0.3 Smoke or Full outcome scores are
inspected. The existing v0.2 sensor-state audit remains the problem and
mechanism evidence; v0.3 prospectively evaluates a new fusion architecture.
The same four released OMBRIA 2021 events have already been inspected, so the
v0.3 run is a method follow-up, not an independent confirmation or evidence of
cross-dataset generalization.

Synthetic opaque masks are described as **cloud-like occlusion**. Quality maps
come from the applied perturbation mask and remain oracle inputs in this study.
The study does not claim observed-cloud detection, operational deployment,
state-of-the-art performance, or universal robustness.

## Paper story

Multimodal flood segmentation should not treat Sentinel-2 as equally reliable
in every pixel and acquisition. Clean optical evidence should be used; locally
unavailable optical evidence should be suppressed at the corresponding
locations and scales; and complete optical absence should reduce the model to a
Sentinel-1-driven path. The proposed Quality-Gated Sensor-State Fusion (QGSF)
module encodes this policy in the network rather than asking an early-fusion
U-Net to infer it from concatenated channels.

The paper has three prospective contributions:

1. A spatial, bitemporal sensor-state formulation for Sentinel-1/Sentinel-2
   fusion under clean, partially unavailable, and completely unavailable
   optical inputs.
2. A hard availability-constrained, learned multi-scale fusion mechanism with
   an exact S1-only limiting behavior when both Sentinel-2 quality maps are
   zero.
3. A matched control design that separates corruption augmentation, access to
   quality information, gate architecture, and spatial alignment.

Only contributions supported by the Full results may be stated as findings.

## Method modules

| Module | Forward process | Why it is needed | Verifiable property |
| --- | --- | --- | --- |
| Bitemporal optical sanitization | Multiply pre/post S2 by their aligned availability maps before encoding | Explicitly unavailable values, including noise-filled inputs, should not enter the optical encoder | Changing S2 values cannot change inference when both quality maps are zero |
| Shared optical and radar encoders | Encode S2 pre/post with shared weights and S1 pre/post with a radar branch | Keeps temporal S2 representations comparable while retaining an S1 path | Pre/post S2 use the same parameters; S1 remains present at every fusion scale |
| Multi-scale quality gates | Downsample each quality map by area and multiply it by a learned radar/optical compatibility gate | Pixel availability must remain a hard upper bound after spatial aggregation | Every gate is in `[0, quality]`; unavailable regions have exactly zero optical contribution |
| Fused decoder | Concatenate radar, gated pre-S2, and gated post-S2 features at three scales, then decode with U-Net skips | Uses optical evidence only where admitted while preserving spatial detail | With both S2 qualities zero, all optical terms supplied to fusion are exactly zero |

## Forward definition

Let `R` be the two-channel bitemporal Sentinel-1 input, `O_t` the RGB
Sentinel-2 input for time `t` in `{pre, post}`, and `Q_t` its binary
availability map. The optical input is first sanitized:

```text
O_t_tilde = Q_t * O_t
```

At encoder scale `l`, the radar feature is `F_R^l`, the shared optical encoder
produces `F_t^l`, and area downsampling produces `D_l(Q_t)`. The effective gate
is

```text
A_t^l = D_l(Q_t) * sigmoid(G_l([F_R^l, F_t^l, D_l(Q_t)]))
```

and the fused feature is

```text
F^l = H_l([F_R^l, A_pre^l * F_pre^l, A_post^l * F_post^l]).
```

The multiplication by `D_l(Q_t)` is a hard constraint, not a learned
preference. If both quality maps are zero, both optical contributions are zero
at every scale and the prediction depends only on the S1 path.

The final convolution of each learned gate is initialized with zero weights and
bias `2.0`, so an available optical feature starts with a multiplier of about
`0.88` while unavailable features remain exactly zero. Training may reduce or
increase the learned compatibility factor but cannot override unavailability.

For `base_channels=16`, the default modality-branch width is 4. The resulting
QGSF-U-Net has 506,507 trainable parameters versus 484,449 for the 10-channel
early-fusion U-Net, a 4.6% increase. Exact counts are written to every training
configuration.

## Fixed input layout

The quality-gated architecture accepts exactly ten channels:

1. S2 before RGB: channels 0--2;
2. S2 after RGB: channels 3--5;
3. S1 before and after: channels 6--7;
4. S2 before and after availability: channels 8--9.

## Full route matrix

All multimodal corruption-trained routes use the same
`quality_matched_light` corruption schedule, split seed, perturbation seed, and
model seeds.

| Route | Architecture | Quality input | Training purpose |
| --- | --- | --- | --- |
| `clean` | Early-fusion U-Net | None | Clean-training reference |
| `matched_control` | Early-fusion U-Net | None | Corruption-augmentation control |
| `quality_concat` | Early-fusion U-Net | Aligned | Tests quality as extra input channels |
| `quality_gated` | QGSF-U-Net | Aligned | Proposed method |
| `gated_misaligned` | QGSF-U-Net | Shifted with prevalence preserved | Spatial-localization control with identical capacity |
| `s1_reference` | Early-fusion U-Net | None | Explicit complete-S2-absence reference |
| `s2_reference` | Early-fusion U-Net | None | Optical-only reference |

Full uses model seeds `7`, `13`, `21`, `29`, and `37`, 25 epochs, clean- and
robustness-selected checkpoints, and the eight fixed states from v0.2. Smoke
uses one seed and two epochs only to validate execution and packaging; its
scores are excluded from scientific claims.

## Prespecified contrasts and interpretation gates

The primary corruption composite averages per-seed IoU over `patch_after`,
`cloud_after_30`, `cloud_after_50`, and `cloud_after_70` after first averaging
perturbation repetitions within each seed.

1. **Architecture effect:** `quality_gated - quality_concat` on the primary
   composite.
2. **Localization effect:** `quality_gated - gated_misaligned` on the primary
   composite.
3. **Information effect:** `quality_gated - matched_control` on the primary
   composite.
4. **Clean preservation:** clean-state `quality_gated - quality_concat`.
5. **Fallback consistency:** zero-all `quality_gated - s1_reference`.

All paired differences are reported by model seed with a two-sided Student-t
interval (`df=4`). Interpretation is fixed as follows:

- A superiority statement requires a positive mean and a positive lower 95%
  run-level interval for the relevant paired contrast.
- A weaker consistency statement may be used only when the mean is positive
  and at least four of five seed-level differences are positive; it must not be
  called statistical superiority.
- Clean behavior is considered practically preserved only when the mean clean
  IoU loss relative to `quality_concat` is no worse than `-0.02`.
- Fallback behavior is considered consistent with the S1 reference only when
  the absolute mean zero-all IoU gap is at most `0.02`; this is a descriptive
  tolerance, not a formal equivalence test.
- If the architecture, localization, or information contrast fails, the
  manuscript remains a sensor-state audit and QGSF is reported as an
  unsuccessful or inconclusive follow-up rather than as the paper's
  innovation.

## Planned manuscript structure if the gates pass

1. Introduction: reliability mismatch across clean, partial, and absent S2.
2. Related work: multimodal flood mapping, missing-modality robustness, and
   quality-aware remote-sensing fusion.
3. Method: sensor-state formulation, QGSF module, and exact fallback property.
4. Experiments: matched controls, event-held-out protocol, and prespecified
   contrasts.
5. Results: clean/partial/absent regimes plus alignment ablation and gate maps.
6. Discussion: oracle-quality boundary, one-dataset/backbone scope, and the
   separate need for an observed quality estimator.
