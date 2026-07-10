| Test S2 degradation | Training | n | IoU mean | IoU 95% CI | F1 mean | F1 95% CI |
| --- | --- | --- | --- | --- | --- | --- |
| Clean | Clean multimodal | 3 | 0.6821 | [0.6734, 0.6908] | 0.8037 | [0.7970, 0.8104] |
| Clean | Light degradation training | 3 | 0.6521 | [0.6248, 0.6794] | 0.7809 | [0.7600, 0.8018] |
| Clean | Quality-aware composite training | 3 | 0.6594 | [0.6425, 0.6763] | 0.7862 | [0.7725, 0.7999] |
| Patch | Clean multimodal | 3 | 0.5621 | [0.4950, 0.6292] | 0.7044 | [0.6418, 0.7670] |
| Patch | Light degradation training | 3 | 0.5760 | [0.5643, 0.5877] | 0.7203 | [0.7049, 0.7357] |
| Patch | Quality-aware composite training | 3 | 0.5787 | [0.5335, 0.6239] | 0.7270 | [0.6940, 0.7600] |
| Cloud-like 30 | Clean multimodal | 3 | 0.5861 | [0.5262, 0.6460] | 0.7247 | [0.6703, 0.7791] |
| Cloud-like 30 | Light degradation training | 3 | 0.5942 | [0.5778, 0.6106] | 0.7350 | [0.7181, 0.7519] |
| Cloud-like 30 | Quality-aware composite training | 3 | 0.5999 | [0.5368, 0.6630] | 0.7434 | [0.6964, 0.7904] |
| Cloud-like 50 | Clean multimodal | 3 | 0.5343 | [0.4618, 0.6068] | 0.6797 | [0.6067, 0.7527] |
| Cloud-like 50 | Light degradation training | 3 | 0.5538 | [0.5384, 0.5692] | 0.7015 | [0.6811, 0.7219] |
| Cloud-like 50 | Quality-aware composite training | 3 | 0.5494 | [0.4724, 0.6264] | 0.7030 | [0.6416, 0.7644] |
| Cloud-like 70 | Clean multimodal | 3 | 0.4848 | [0.4287, 0.5409] | 0.6355 | [0.5707, 0.7003] |
| Cloud-like 70 | Light degradation training | 3 | 0.5121 | [0.4967, 0.5275] | 0.6650 | [0.6412, 0.6888] |
| Cloud-like 70 | Quality-aware composite training | 3 | 0.4961 | [0.4206, 0.5716] | 0.6567 | [0.5911, 0.7223] |
| Noise | Clean multimodal | 3 | 0.3721 | [0.2196, 0.5246] | 0.5301 | [0.3781, 0.6821] |
| Noise | Light degradation training | 3 | 0.4193 | [0.3696, 0.4690] | 0.5808 | [0.5271, 0.6345] |
| Noise | Quality-aware composite training | 3 | 0.3565 | [0.1968, 0.5162] | 0.5056 | [0.3421, 0.6691] |
| Post-S2 missing | Clean multimodal | 3 | 0.4058 | [0.3742, 0.4374] | 0.5601 | [0.5435, 0.5767] |
| Post-S2 missing | Light degradation training | 3 | 0.4434 | [0.4414, 0.4454] | 0.6007 | [0.5858, 0.6156] |
| Post-S2 missing | Quality-aware composite training | 3 | 0.3931 | [0.3168, 0.4694] | 0.5520 | [0.4747, 0.6293] |
| All S2 missing | Clean multimodal | 3 | 0.0447 | [-0.0487, 0.1381] | 0.0764 | [-0.0826, 0.2354] |
| All S2 missing | Light degradation training | 3 | 0.3689 | [0.2658, 0.4720] | 0.5169 | [0.3964, 0.6374] |
| All S2 missing | Quality-aware composite training | 3 | 0.3481 | [0.2460, 0.4502] | 0.4908 | [0.3663, 0.6153] |
| S1 fallback | S1 bitemporal | 3 | 0.5071 | [0.4996, 0.5146] | 0.6606 | [0.6519, 0.6693] |

Note: Confidence intervals are seed-level 95% t intervals over three seeds (df = 2). The seed controls the train/validation split, model training, and stochastic test perturbation, so the intervals summarize run-level variability rather than training stochasticity alone. They should not be interpreted as per-chip, event-level, or spatial-bootstrap intervals.
