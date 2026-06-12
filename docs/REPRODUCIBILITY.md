# Reproducibility Notes

## Reported Settings

- Dataset: OMBRIA.
- Seeds: `7`, `13`, `21`.
- Epochs: `25`.
- Batch size: `8`.
- Base channels: `16`.
- Optimizer: AdamW.
- Learning rate: `1e-3`.
- Checkpoint selection: highest validation IoU, saved as `best_model.pt`.
- Metrics: IoU, F1, precision, recall, accuracy.

## Main Command

```bash
EPOCHS=25 BATCH_SIZE=8 BASE_CHANNELS=16 SEEDS="7 13 21" \
RUNS_DIR=results/runs/ombria_robustness \
bash scripts/run_ombria_followup_matrix.sh
```

## Important Limitations

The degradation modes are controlled synthetic stress tests. They are not real cloud or atmospheric-degradation simulations. The reported uncertainty is based on three random seeds and should not be interpreted as event-level, spatial, or operational uncertainty.
