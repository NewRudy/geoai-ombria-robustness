#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-external/OMBRIA}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
SEEDS="${SEEDS:-7 13 21}"
RUNS_DIR="${RUNS_DIR:-results/runs/publication_upgrade}"
PYTHON="${PYTHON:-python}"

MULTIMODAL_TRAIN_MODES="${MULTIMODAL_TRAIN_MODES:-modality_dropout_light quality_dropout_light}"
EVAL_MODES="${EVAL_MODES:-none patch_after cloud_after_30 cloud_after_50 cloud_after_70 noise_after zero_after zero_all}"

mkdir -p external results/tables results/figures
if [ ! -d "$ROOT" ]; then
  git clone --depth 1 https://github.com/geodrak/OMBRIA.git "$ROOT"
fi

"$PYTHON" scripts/train_ombria_unet.py --root "$ROOT" --variant multimodal --dry-run
"$PYTHON" scripts/train_ombria_unet.py --root "$ROOT" --variant s1_bitemporal --dry-run
"$PYTHON" scripts/train_ombria_unet.py --root "$ROOT" --variant s2_bitemporal --dry-run
"$PYTHON" scripts/train_ombria_unet.py --root "$ROOT" --variant multimodal --s2-quality binary --dry-run

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

  for mode in $EVAL_MODES; do
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

  for train_mode in $MULTIMODAL_TRAIN_MODES; do
    quality_arg=()
    quality_suffix=""
    if [ "$train_mode" = "quality_dropout_light" ]; then
      quality_arg=(--s2-quality binary)
      quality_suffix="_quality-binary"
    fi

    robust_checkpoint="$RUNS_DIR/multimodal${quality_suffix}_none_train-${train_mode}_seed${seed}/best_model.pt"
    if [ ! -f "$robust_checkpoint" ]; then
      "$PYTHON" scripts/train_ombria_unet.py \
        --root "$ROOT" \
        --out-dir "$RUNS_DIR" \
        --variant multimodal \
        "${quality_arg[@]}" \
        --train-degrade-s2 "$train_mode" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        --seed "$seed"
    fi

    for mode in $EVAL_MODES; do
      "$PYTHON" scripts/train_ombria_unet.py \
        --root "$ROOT" \
        --out-dir "$RUNS_DIR" \
        --variant multimodal \
        "${quality_arg[@]}" \
        --degrade-s2 "$mode" \
        --train-degrade-s2 "$train_mode" \
        --batch-size "$BATCH_SIZE" \
        --base-channels "$BASE_CHANNELS" \
        --seed "$seed" \
        --eval-checkpoint "$robust_checkpoint"
    done
  done

  s1_checkpoint="$RUNS_DIR/s1_bitemporal_none_seed${seed}/best_model.pt"
  if [ ! -f "$s1_checkpoint" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --variant s1_bitemporal \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$seed"
  fi

  "$PYTHON" scripts/train_ombria_unet.py \
    --root "$ROOT" \
    --out-dir "$RUNS_DIR" \
    --variant s1_bitemporal \
    --degrade-s2 none \
    --batch-size "$BATCH_SIZE" \
    --base-channels "$BASE_CHANNELS" \
    --seed "$seed" \
    --eval-checkpoint "$s1_checkpoint"

  s2_checkpoint="$RUNS_DIR/s2_bitemporal_none_seed${seed}/best_model.pt"
  if [ ! -f "$s2_checkpoint" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --variant s2_bitemporal \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$seed"
  fi

  for mode in $EVAL_MODES; do
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --variant s2_bitemporal \
      --degrade-s2 "$mode" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$seed" \
      --eval-checkpoint "$s2_checkpoint"
  done
done

"$PYTHON" scripts/summarize_ombria_runs.py \
  --runs-dir "$RUNS_DIR" \
  --out results/tables/publication_upgrade_run_summary.csv

"$PYTHON" scripts/analyze_ombria_robustness.py \
  --summary results/tables/publication_upgrade_run_summary.csv \
  --out-csv results/tables/publication_upgrade_robustness_summary.csv \
  --out-md results/tables/publication_upgrade_robustness_summary.md

"$PYTHON" scripts/analyze_publication_baselines.py \
  --summary results/tables/publication_upgrade_run_summary.csv \
  --out-csv results/tables/publication_upgrade_baseline_summary.csv \
  --out-md results/tables/publication_upgrade_baseline_summary.md

"$PYTHON" scripts/plot_ombria_robustness.py \
  --summary results/tables/publication_upgrade_robustness_summary.csv \
  --out results/figures/publication_upgrade_robustness.png

seed_for_panels="$(printf '%s\n' $SEEDS | head -n 1)"
clean_checkpoint="$RUNS_DIR/multimodal_none_seed${seed_for_panels}/best_model.pt"
robust_checkpoint="$RUNS_DIR/multimodal_none_train-modality_dropout_light_seed${seed_for_panels}/best_model.pt"
if [ -f "$clean_checkpoint" ] && [ -f "$robust_checkpoint" ]; then
  "$PYTHON" scripts/export_ombria_prediction_panels.py \
    --root "$ROOT" \
    --clean-checkpoint "$clean_checkpoint" \
    --robust-checkpoint "$robust_checkpoint" \
    --seed "$seed_for_panels" \
    --num-samples 4 \
    --modes none patch_after cloud_after_50 cloud_after_70 noise_after zero_after zero_all \
    --out-dir results/figures/publication_upgrade_qualitative
fi

"$PYTHON" - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

out = Path("results/publication_upgrade_artifacts.zip")
paths = [
    Path("results/tables/publication_upgrade_run_summary.csv"),
    Path("results/tables/publication_upgrade_robustness_summary.csv"),
    Path("results/tables/publication_upgrade_robustness_summary.md"),
    Path("results/tables/publication_upgrade_baseline_summary.csv"),
    Path("results/tables/publication_upgrade_baseline_summary.md"),
    Path("results/figures/publication_upgrade_robustness.png"),
]
paths.extend(Path("results/figures/publication_upgrade_qualitative").glob("*.png"))
paths.extend(Path("results/runs/publication_upgrade").glob("*/config.json"))
with ZipFile(out, "w", ZIP_DEFLATED) as zf:
    for path in paths:
        if path.exists():
            zf.write(path)
print(f"wrote {out}")
PY

cat results/tables/publication_upgrade_robustness_summary.md
