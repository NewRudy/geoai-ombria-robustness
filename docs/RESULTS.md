# Reported Results

The reported manuscript results are stored as lightweight summary artifacts:

- `results/tables/main_results.md`
- `results/tables/seed_level_uncertainty.md`
- `results/tables/fallback_policy.md`
- `results/figures/figure_robustness_tradeoff.svg`

The central result is a sensor-availability policy:

1. Use multimodal Sentinel-1/Sentinel-2 inference when optical observations are available.
2. Prefer lightweight degradation-trained multimodal inference under partial or noisy Sentinel-2 degradation.
3. Use a bitemporal Sentinel-1 fallback when Sentinel-2 is completely absent.
