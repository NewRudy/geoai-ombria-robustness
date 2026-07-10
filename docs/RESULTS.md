# Reported Results

The reported manuscript results are stored as lightweight summary artifacts:

- `results/tables/main_results.md`
- `results/tables/seed_level_uncertainty.md`
- `results/tables/fallback_policy.md`
- `results/tables/publication_upgrade_baselines.md`

The exploratory public-split result supports conditional route comparison rather than an automated policy:

1. Clean multimodal inference has the highest mean under clean inputs.
2. Degradation-trained multimodal routes retain the highest means under patch and cloud-like 30/50 states.
3. Light degradation training and S1-only inference are near-tied at the cloud-like 70 target.
4. S1-only inference has the highest mean under noisy post-event S2, missing post-event S2, and complete S2 removal.

These comparisons are exploratory until the locked 2021 event-held-out matrix is complete. The repository does not implement or validate an automated router.
