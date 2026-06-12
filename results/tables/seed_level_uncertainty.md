| Test S2 degradation | Training | n | IoU mean | IoU 95% CI | F1 mean | F1 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | Clean multimodal | 3 | 0.6834 | [0.6613, 0.7055] | 0.8053 | [0.7864, 0.8242] |
| Clean | Light degradation training | 3 | 0.6540 | [0.6279, 0.6801] | 0.7831 | [0.7640, 0.8022] |
| Clean | Balanced degradation training | 3 | 0.6432 | [0.5866, 0.6998] | 0.7747 | [0.7322, 0.8172] |
| Patch | Clean multimodal | 3 | 0.5598 | [0.4907, 0.6289] | 0.7025 | [0.6377, 0.7673] |
| Patch | Light degradation training | 3 | 0.5799 | [0.5344, 0.6254] | 0.7251 | [0.6906, 0.7596] |
| Patch | Balanced degradation training | 3 | 0.5530 | [0.4656, 0.6404] | 0.7006 | [0.6194, 0.7818] |
| Noise | Clean multimodal | 3 | 0.3620 | [0.2549, 0.4691] | 0.5202 | [0.4223, 0.6181] |
| Noise | Light degradation training | 3 | 0.4186 | [0.4097, 0.4275] | 0.5832 | [0.5715, 0.5949] |
| Noise | Balanced degradation training | 3 | 0.3868 | [0.2631, 0.5105] | 0.5493 | [0.4154, 0.6832] |
| Post-S2 missing | Clean multimodal | 3 | 0.4057 | [0.3796, 0.4318] | 0.5594 | [0.5442, 0.5746] |
| Post-S2 missing | Light degradation training | 3 | 0.4373 | [0.3948, 0.4798] | 0.5979 | [0.5574, 0.6384] |
| Post-S2 missing | Balanced degradation training | 3 | 0.4195 | [0.3514, 0.4876] | 0.5769 | [0.5091, 0.6447] |
| All S2 missing | Clean multimodal | 3 | 0.0466 | [-0.0918, 0.1850] | 0.0790 | [-0.1411, 0.2991] |
| All S2 missing | Light degradation training | 3 | 0.3124 | [0.2568, 0.3680] | 0.4479 | [0.3776, 0.5182] |
| All S2 missing | Balanced degradation training | 3 | 0.3451 | [0.2914, 0.3988] | 0.4895 | [0.4157, 0.5633] |
| S1 fallback | S1 bitemporal | 3 | 0.4092 | [0.3019, 0.5165] | 0.5642 | [0.4566, 0.6718] |

Note: Confidence intervals are seed-level 95% t intervals over three seeds (df = 2). They summarize training stochasticity and should not be interpreted as per-chip or spatial bootstrap intervals.
