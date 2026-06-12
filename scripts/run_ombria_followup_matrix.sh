#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-external/OMBRIA}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
SEEDS="${SEEDS:-7 13 21}"
RUNS_DIR="${RUNS_DIR:-results/runs/ombria_robustness}"
TRAIN_MODES="${TRAIN_MODES:-modality_dropout_light modality_dropout_balanced}"
PYTHON="${PYTHON:-python}"

mkdir -p external
if [ ! -d "$ROOT" ]; then
  git clone --depth 1 https://github.com/geodrak/OMBRIA.git "$ROOT"
fi

"$PYTHON" scripts/train_ombria_unet.py --root "$ROOT" --variant multimodal --dry-run

for seed in $SEEDS; do
  clean_checkpoint="$RUNS_DIR/multimodal_none_seed${seed}/best_model.pt"
  if [ ! -f "$clean_checkpoint" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --variant multimodal \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$seed"
  fi

  for mode in none zero_after zero_all noise_after patch_after; do
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --variant multimodal \
      --degrade-s2 "$mode" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$seed" \
      --eval-checkpoint "$clean_checkpoint"
  done

  for train_mode in $TRAIN_MODES; do
    robust_checkpoint="$RUNS_DIR/multimodal_none_train-${train_mode}_seed${seed}/best_model.pt"
    if [ ! -f "$robust_checkpoint" ]; then
      "$PYTHON" scripts/train_ombria_unet.py \
        --root "$ROOT" \
        --out-dir "$RUNS_DIR" \
        --variant multimodal \
        --train-degrade-s2 "$train_mode" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        --seed "$seed"
    fi

    for mode in none zero_after zero_all noise_after patch_after; do
      "$PYTHON" scripts/train_ombria_unet.py \
        --root "$ROOT" \
        --out-dir "$RUNS_DIR" \
        --variant multimodal \
        --degrade-s2 "$mode" \
        --train-degrade-s2 "$train_mode" \
        --batch-size "$BATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        --seed "$seed" \
        --eval-checkpoint "$robust_checkpoint"
    done
  done
done

"$PYTHON" scripts/summarize_ombria_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out results/tables/ombria_followup_run_summary.csv

"$PYTHON" scripts/analyze_ombria_robustness.py \
  --summary results/tables/ombria_followup_run_summary.csv \
  --out-csv results/tables/ombria_followup_robustness_summary.csv \
  --out-md results/tables/ombria_followup_robustness_summary.md

"$PYTHON" scripts/plot_ombria_robustness.py \
  --summary results/tables/ombria_followup_robustness_summary.csv \
  --out results/figures/ombria_followup_robustness.png

"$PYTHON" - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out = Path("results/ombria_followup_artifacts.zip")
paths = [
    Path("results/tables/ombria_followup_run_summary.csv"),
    Path("results/tables/ombria_followup_robustness_summary.csv"),
    Path("results/tables/ombria_followup_robustness_summary.md"),
    Path("results/figures/ombria_followup_robustness.png"),
]
with ZipFile(out, "w", ZIP_DEFLATED) as zf:
    for path in paths:
        if path.exists():
            zf.write(path)
print(f"wrote {out}")
PY

cat results/tables/ombria_followup_robustness_summary.md
