# Official SMAGNet Adaptation Protocol

Status: frozen before the official-architecture Smoke outcome

Role: closest published architecture gate for the quality-map uncertainty Article

## Official identity

- Repository: `https://github.com/ASUcicilab/SMAGNet`
- Commit: `4371df08e6ca3b9d71c0385ad57b589830469a0c`
- Official `src/smagnet.py` SHA-256: `daf00d0533ca7865b4bd7b47404f1c0fa42e4a0bdc70706dee45bedcc1420f25`
- License: MIT; frozen license SHA-256: `4261bd84b3a36788cb1bb4e25d3f59a2cf2ac79abb93cb45cf09fc043b39265c`
- Paper: Lee and Li, *ISPRS Journal of Photogrammetry and Remote Sensing* 232 (2026), 492-508, `https://doi.org/10.1016/j.isprsjprs.2025.12.023`
- Runtime architecture dependency: `segmentation-models-pytorch==0.5.0`, matching the official environment file.

The official repository releases the model definition and an inference notebook, but no training program. The architecture is imported from the byte-verified official source without modifying it. Training and checkpoint selection are reconstructed from Equations 8-9 and Section 4.2 of the paper, and every adaptation is recorded below.

## Preserved official architecture and optimization

- Two ResNet-50 encoders: randomly initialized SAR and ImageNet-initialized MSI.
- Five spatially masked adaptive gate blocks using the official invalid-mask convention.
- One weight-shared decoder with channels `{256, 128, 64, 32, 16}`.
- Separate fused and SAR outputs.
- Equal dual-path objective: `0.5 * BCE(SAR) + 0.5 * BCE(fused)`.
- Adam, learning rate `5e-4`, weight decay `0.0`.
- Random 256 x 256 crop and horizontal/vertical flip during training.
- Full duration of 200 epochs.
- Lowest clean-validation dual-path BCE checkpoint.
- Validation precision-recall threshold that maximizes pixel IoU.

The verified architecture contains 56,035,958 trainable parameters. Under a completely unavailable optical state, the supplied quality map is zero, the official invalid mask is one, every masked gate is zero, and fused logits must equal the shared SAR-path logits within `1e-6`.

## Sen1Floods11 adaptation

- Inputs retain the frozen two Sentinel-1 and four Sentinel-2 bands. The optical layout is reordered from the repository's B/G/R/NIR convention to the official R/G/B/NIR convention, then every SAR and optical band is standardized with statistics computed from the frozen Sen1Floods11 training records only. This follows the paper's training-set standardization while replacing its C2S-MS statistics.
- The supplied official `spatial_mask` is `1 - M_hat`, because SMAGNet uses one for invalid pixels while this study defines one for available pixels.
- Reference training uses the SCL-available map intersected with the official Sentinel-2 chip's valid-data support.
- At evaluation, only `M_hat` is perturbed; the underlying optical chip is unchanged. Complete optical absence remains a separate state that zeros Sentinel-2 and supplies an all-invalid map.
- The frozen 252/89 training/validation records replace the C2S-MS split. Test contains 90 chips and Bolivia remains a separate 15-chip audit.
- Invalid Sen1Floods11 target pixels are excluded from both BCE paths and all metrics.
- Each 512 x 512 validation or evaluation chip is divided into four non-overlapping 256 x 256 patches, following the official paper.
- A micro-batch of four with four-step gradient accumulation preserves the official effective batch size of 16 on a 16 GB Kaggle GPU.
- Automatic mixed precision is enabled and recorded as an execution deviation.

SMAGNet is a published-architecture comparison, not a capacity-controlled causal contrast. Its parameter count, ImageNet initialization, optimizer, checkpoint rule, and threshold selection differ from the small U-Net family and must be reported explicitly.

## Smoke gate

Smoke uses seed 7, two epochs, the frozen 24/12/12/4 train/validation/test/Bolivia subset, 16 error conditions, and one perturbation repetition. It must verify the official source and license hashes, CUDA execution, dual-path training, clean-validation checkpointing, validation threshold selection, the complete-absence identity, all test and Bolivia conditions, per-chip/per-event exports, finite metrics, and artifact hashes. Smoke scores are prohibited from the manuscript.

## Full gate

After an independently returned Smoke archive passes local audit, Full is released as five immutable seed shards: 7, 13, 21, 29, and 37. Every shard uses 200 epochs, all 446 prepared records, the 54 frozen independent/structured/matched/absence conditions, three perturbation repetitions, and separate test and Bolivia outputs. A seed shard remains scientifically uninterpretable until all five official-architecture shards pass and are paired offline with the already frozen seed-matched Sentinel-1 reference.

The allowed fallback remains unchanged: a mechanism-matched reimplementation is considered only after an official-source adaptation failure is reproduced and documented before any Full SMAGNet score is inspected. Runtime inconvenience or an unfavorable result is not a fallback trigger.
