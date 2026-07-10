# Reproducibility Notes

## Reported Settings

- Dataset: OMBRIA.
- Model seeds: `7`, `13`, `21`.
- Fixed train/validation split seed: `20260710`.
- Fixed test-perturbation seed: `20260710`.
- Epochs: `25`.
- Batch size: `8`.
- Base channels: `16`.
- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Checkpoint selection: highest validation IoU, saved as `best_model.pt`.
- Metrics: globally accumulated IoU, F1, precision, recall, and accuracy, plus per-chip rows.

## Main Command

```bash
MODE=full EPOCHS=25 BATCH_SIZE=8 BASE_CHANNELS=16 \
bash scripts/run_confirmatory_event_matrix.sh
```

## Important Limitations

The degradation modes are controlled synthetic stress tests. They are not real cloud, cloud-shadow, or atmospheric-correction simulations. The confirmatory uncertainty is based on three model seeds with fixed perturbation repetitions and should not be interpreted as cross-dataset, spatial, or operational uncertainty.
