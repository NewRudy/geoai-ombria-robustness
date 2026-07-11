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
RESULT_ROOT="${RESULT_ROOT:-results/sensor_state_v2}"
RUNS_DIR="$RESULT_ROOT/runs"
EVAL_DIR="$RESULT_ROOT/evaluations"
LOG_PATH="$RESULT_ROOT/run.log"

case "$MODE" in
  smoke)
    SEEDS="${SEEDS:-7}"
    CHECKPOINT_POLICIES="${CHECKPOINT_POLICIES:-clean}"
    ;;
  full)
    SEEDS="${SEEDS:-7 13 21 29 37}"
    CHECKPOINT_POLICIES="${CHECKPOINT_POLICIES:-clean robust}"
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

ROUTES="clean light matched_control matched_quality mislocalized_quality s1_reference s2_reference"
EVAL_MODES="${EVAL_MODES:-none patch_after cloud_after_30 cloud_after_50 cloud_after_70 noise_after zero_after zero_all}"

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1
export PYTHONUNBUFFERED=1
echo "=== sensor-state v0.2 session $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE seeds=$SEEDS ==="

"$PYTHON" scripts/check_cuda_runtime.py --json-out "$RESULT_ROOT/runtime_manifest.json"
"$PYTHON" -m pip freeze > "$RESULT_ROOT/environment_freeze.txt"
"$PYTHON" scripts/write_experiment_manifest.py \
  --out "$RESULT_ROOT/experiment_manifest.json" \
  --mode "$MODE" \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --split-seed "$SPLIT_SEED" \
  --perturb-seed "$PERTURB_SEED" \
  --eval-modes $EVAL_MODES \
  --routes $ROUTES \
  --checkpoint-policies $CHECKPOINT_POLICIES \
  --ombria-commit "$OMBRIA_COMMIT"

mkdir -p external "$RUNS_DIR" "$EVAL_DIR" "$RESULT_ROOT/tables" "$RESULT_ROOT/figures"
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
  local run_dir="$1"
  local seed="$2"
  shift 2
  if [ ! -f "$run_dir/best_clean.pt" ] || [ ! -f "$run_dir/best_robust.pt" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --split-seed "$SPLIT_SEED" \
      --eval-perturb-seed "$PERTURB_SEED" \
      --seed "$seed" \
      --loader-seed "$((seed + 200000))" \
      --corruption-seed "$((seed + 300000))" \
      --robust-val-modes none cloud_after_50 zero_after \
      "$@"
  fi
}

evaluate_route() {
  local route="$1"
  local variant="$2"
  local quality="$3"
  local checkpoint="$4"
  local checkpoint_policy="$5"
  local seed="$6"
  local mode="$7"
  local repetitions=1
  case "$mode" in
    patch_after|cloud_after_30|cloud_after_50|cloud_after_70|noise_after)
      repetitions=3
      ;;
  esac
  local out="$EVAL_DIR/$checkpoint_policy/$route/seed${seed}/$mode"
  if [ ! -f "$out/summary_metrics.csv" ]; then
    "$PYTHON" scripts/evaluate_ombria_2021_events.py \
      --root "$ROOT" \
      --checkpoint "$checkpoint" \
      --checkpoint-policy "$checkpoint_policy" \
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
  clean_dir="$RUNS_DIR/multimodal_none_seed${seed}"
  light_dir="$RUNS_DIR/multimodal_none_train-modality_dropout_light_seed${seed}"
  control_dir="$RUNS_DIR/multimodal_none_train-quality_matched_light_seed${seed}"
  quality_dir="$RUNS_DIR/multimodal_quality-binary_none_train-quality_matched_light_seed${seed}"
  mislocalized_dir="$RUNS_DIR/multimodal_quality-mislocalized_none_train-quality_matched_light_seed${seed}"
  s1_dir="$RUNS_DIR/s1_bitemporal_none_seed${seed}"
  s2_dir="$RUNS_DIR/s2_bitemporal_none_seed${seed}"

  train_route "$clean_dir" "$seed" --variant multimodal
  train_route "$light_dir" "$seed" --variant multimodal --train-degrade-s2 modality_dropout_light
  train_route "$control_dir" "$seed" --variant multimodal --train-degrade-s2 quality_matched_light
  train_route "$quality_dir" "$seed" --variant multimodal --s2-quality binary --train-degrade-s2 quality_matched_light
  train_route "$mislocalized_dir" "$seed" --variant multimodal --s2-quality mislocalized --train-degrade-s2 quality_matched_light
  train_route "$s1_dir" "$seed" --variant s1_bitemporal
  train_route "$s2_dir" "$seed" --variant s2_bitemporal

  for checkpoint_policy in $CHECKPOINT_POLICIES; do
    for eval_mode in $EVAL_MODES; do
      evaluate_route clean multimodal none "$clean_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route light multimodal none "$light_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route matched_control multimodal none "$control_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route matched_quality multimodal binary "$quality_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route mislocalized_quality multimodal mislocalized "$mislocalized_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route s1_reference s1_bitemporal none "$s1_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
      evaluate_route s2_reference s2_bitemporal none "$s2_dir/best_${checkpoint_policy}.pt" "$checkpoint_policy" "$seed" "$eval_mode"
    done
  done
done

"$PYTHON" scripts/summarize_confirmatory_events.py \
  --evaluations-dir "$EVAL_DIR" \
  --out-csv "$RESULT_ROOT/tables/event_heldout_summary.csv" \
  --out-md "$RESULT_ROOT/tables/event_heldout_summary.md"

"$PYTHON" scripts/audit_split_near_duplicates.py \
  --root "$ROOT" \
  --split-json "$clean_dir/splits.json" \
  --out-json "$RESULT_ROOT/split_near_duplicate_audit.json" \
  --out-csv "$RESULT_ROOT/tables/train_val_nearest_pairs.csv"

panel_seed="$(printf '%s\n' $SEEDS | head -n 1)"
for checkpoint_policy in $CHECKPOINT_POLICIES; do
  "$PYTHON" scripts/export_sensor_state_v2_probabilities.py \
    --root "$ROOT" \
    --runs-dir "$RUNS_DIR" \
    --seed "$panel_seed" \
    --checkpoint-policy "$checkpoint_policy" \
    --perturb-seed "$PERTURB_SEED" \
    --modes cloud_after_50 zero_after zero_all \
    --out-dir "$RESULT_ROOT/figures"
done

"$PYTHON" scripts/package_confirmatory_artifacts.py \
  --root "$RESULT_ROOT" \
  --out results/ombria_sensor_state_v2_artifacts.zip \
  --include-checkpoints

cat "$RESULT_ROOT/tables/event_heldout_summary.md"
