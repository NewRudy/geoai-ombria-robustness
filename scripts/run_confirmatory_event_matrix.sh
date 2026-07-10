#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-external/OMBRIA}"
OMBRIA_COMMIT="${OMBRIA_COMMIT:-38a490355f76da8ce27ed051138f03f3492a6e46}"
MODE="${MODE:-smoke}"
EPOCHS="${EPOCHS:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
SPLIT_SEED="${SPLIT_SEED:-20260710}"
PERTURB_SEED="${PERTURB_SEED:-20260710}"
PYTHON="${PYTHON:-python}"
RUNS_DIR="${RUNS_DIR:-results/confirmatory/runs}"
EVAL_DIR="${EVAL_DIR:-results/confirmatory/evaluations}"

case "$MODE" in
  smoke)
    SEEDS="${SEEDS:-7}"
    ;;
  full)
    SEEDS="${SEEDS:-7 13 21}"
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

EVAL_MODES="${EVAL_MODES:-none patch_after cloud_after_30 cloud_after_50 cloud_after_70 noise_after zero_after zero_all}"

"$PYTHON" scripts/check_cuda_runtime.py

mkdir -p external "$RUNS_DIR" "$EVAL_DIR" results/confirmatory/tables results/confirmatory/figures
if [ ! -d "$ROOT" ]; then
  git init -q "$ROOT"
  git -C "$ROOT" remote add origin https://github.com/geodrak/OMBRIA.git
  git -C "$ROOT" fetch --depth 1 origin "$OMBRIA_COMMIT"
  git -C "$ROOT" checkout -q --detach FETCH_HEAD
fi
for required in OmbriaS1/train OmbriaS2/train 2021/ALBANIA 2021/FRANCE 2021/GUYANA 2021/TIMOR; do
  if [ ! -d "$ROOT/$required" ]; then
    echo "Missing required OMBRIA path: $ROOT/$required" >&2
    exit 2
  fi
done

train_route() {
  local checkpoint="$1"
  shift
  if [ ! -f "$checkpoint" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --split-seed "$SPLIT_SEED" \
      "$@"
  fi
}

evaluate_route() {
  local route="$1"
  local variant="$2"
  local quality="$3"
  local checkpoint="$4"
  local seed="$5"
  local mode="$6"
  local repetitions=1
  case "$mode" in
    patch_after|cloud_after_30|cloud_after_50|cloud_after_70|noise_after)
      repetitions=3
      ;;
  esac
  local out="$EVAL_DIR/${route}/seed${seed}/${mode}"
  if [ ! -f "$out/summary_metrics.csv" ]; then
    "$PYTHON" scripts/evaluate_ombria_2021_events.py \
      --root "$ROOT" \
      --checkpoint "$checkpoint" \
      --route "$route" \
      --variant "$variant" \
      --s2-quality "$quality" \
      --degrade-s2 "$mode" \
      --model-seed "$seed" \
      --perturb-seed "$PERTURB_SEED" \
      --perturb-repetitions "$repetitions" \
      --batch-size "$BATCH_SIZE" \
      --out-dir "$out"
  fi
}

for seed in $SEEDS; do
  clean="$RUNS_DIR/multimodal_none_seed${seed}/best_model.pt"
  light="$RUNS_DIR/multimodal_none_train-modality_dropout_light_seed${seed}/best_model.pt"
  matched_control="$RUNS_DIR/multimodal_none_train-quality_matched_light_seed${seed}/best_model.pt"
  matched_quality="$RUNS_DIR/multimodal_quality-binary_none_train-quality_matched_light_seed${seed}/best_model.pt"
  s1="$RUNS_DIR/s1_bitemporal_none_seed${seed}/best_model.pt"

  train_route "$clean" \
    --variant multimodal --seed "$seed"
  train_route "$light" \
    --variant multimodal --train-degrade-s2 modality_dropout_light --seed "$seed"
  train_route "$matched_control" \
    --variant multimodal --train-degrade-s2 quality_matched_light --seed "$seed"
  train_route "$matched_quality" \
    --variant multimodal --s2-quality binary --train-degrade-s2 quality_matched_light --seed "$seed"
  train_route "$s1" \
    --variant s1_bitemporal --seed "$seed"

  for eval_mode in $EVAL_MODES; do
    evaluate_route clean multimodal none "$clean" "$seed" "$eval_mode"
    evaluate_route light multimodal none "$light" "$seed" "$eval_mode"
    evaluate_route matched_control multimodal none "$matched_control" "$seed" "$eval_mode"
    evaluate_route matched_quality multimodal binary "$matched_quality" "$seed" "$eval_mode"
    evaluate_route s1_reference s1_bitemporal none "$s1" "$seed" "$eval_mode"
  done
done

"$PYTHON" scripts/summarize_confirmatory_events.py \
  --evaluations-dir "$EVAL_DIR" \
  --out-csv results/confirmatory/tables/event_heldout_summary.csv \
  --out-md results/confirmatory/tables/event_heldout_summary.md

panel_seed="$(printf '%s\n' $SEEDS | head -n 1)"
"$PYTHON" scripts/export_confirmatory_event_panels.py \
  --root "$ROOT" \
  --clean-checkpoint "$RUNS_DIR/multimodal_none_seed${panel_seed}/best_model.pt" \
  --light-checkpoint "$RUNS_DIR/multimodal_none_train-modality_dropout_light_seed${panel_seed}/best_model.pt" \
  --quality-checkpoint "$RUNS_DIR/multimodal_quality-binary_none_train-quality_matched_light_seed${panel_seed}/best_model.pt" \
  --s1-checkpoint "$RUNS_DIR/s1_bitemporal_none_seed${panel_seed}/best_model.pt" \
  --perturb-seed "$PERTURB_SEED" \
  --out-dir results/confirmatory/figures

"$PYTHON" - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("results/confirmatory")
out = Path("results/ombria_2021_confirmatory_artifacts.zip")
with ZipFile(out, "w", ZIP_DEFLATED) as archive:
    for pattern in ("tables/*", "figures/*", "evaluations/**/summary_metrics.csv", "evaluations/**/per_chip_metrics.csv", "evaluations/**/evaluation_config.json", "runs/*/config.json", "runs/*/splits.json"):
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                archive.write(path)
print(out)
PY

cat results/confirmatory/tables/event_heldout_summary.md
