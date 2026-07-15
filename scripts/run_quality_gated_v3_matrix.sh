#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-external/OMBRIA}"
OMBRIA_COMMIT="${OMBRIA_COMMIT:-38a490355f76da8ce27ed051138f03f3492a6e46}"
MODE="${MODE:-smoke}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-16}"
SPLIT_SEED="${SPLIT_SEED:-20260710}"
PERTURB_SEED="${PERTURB_SEED:-20260710}"
PYTHON="${PYTHON:-python}"
RESULT_ROOT="${RESULT_ROOT:-results/quality_gated_v3_${MODE}}"
RUNS_DIR="$RESULT_ROOT/runs"
EVAL_DIR="$RESULT_ROOT/evaluations"
LOG_PATH="$RESULT_ROOT/run.log"
EVAL_MODES="none patch_after cloud_after_30 cloud_after_50 cloud_after_70 noise_after zero_after zero_all"
PRIMARY_MODES="patch_after cloud_after_30 cloud_after_50 cloud_after_70"

case "$MODE" in
  smoke)
    EPOCHS="${EPOCHS:-2}"
    SEEDS="7"
    CHECKPOINT_POLICIES="clean"
    ROUTES="matched_control quality_concat quality_gated gated_misaligned"
    ;;
  full)
    EPOCHS="${EPOCHS:-25}"
    SEEDS="7 13 21 29 37"
    CHECKPOINT_POLICIES="clean robust"
    ROUTES="clean matched_control quality_concat quality_gated gated_misaligned s1_reference s2_reference"
    ;;
  *)
    echo "MODE must be smoke or full" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1
export PYTHONUNBUFFERED=1
echo "=== quality-gated v0.3 session $(date -u +%Y-%m-%dT%H:%M:%SZ) mode=$MODE seeds=$SEEDS ==="

"$PYTHON" scripts/check_cuda_runtime.py --json-out "$RESULT_ROOT/runtime_manifest.json"
"$PYTHON" -m pip freeze > "$RESULT_ROOT/environment_freeze.txt"

run_directory_templates=()
route_specs=()
for route in $ROUTES; do
  run_directory_templates+=("$route=${route}_seed{seed}")
  case "$route" in
    clean|matched_control)
      route_specs+=("$route=early_fusion_unet,multimodal,none")
      ;;
    quality_concat)
      route_specs+=("$route=early_fusion_unet,multimodal,binary")
      ;;
    quality_gated)
      route_specs+=("$route=quality_gated_fusion,multimodal,binary")
      ;;
    gated_misaligned)
      route_specs+=("$route=quality_gated_fusion,multimodal,mislocalized")
      ;;
    s1_reference)
      route_specs+=("$route=early_fusion_unet,s1_bitemporal,none")
      ;;
    s2_reference)
      route_specs+=("$route=early_fusion_unet,s2_bitemporal,none")
      ;;
  esac
done

if [ -f "$RESULT_ROOT/experiment_manifest.json" ]; then
  "$PYTHON" - "$RESULT_ROOT/experiment_manifest.json" "$MODE" "$EPOCHS" "$BATCH_SIZE" "$BASE_CHANNELS" "$SEEDS" "$ROUTES" "$CHECKPOINT_POLICIES" "$EVAL_MODES" "$SPLIT_SEED" "$PERTURB_SEED" "$OMBRIA_COMMIT" <<'PY'
import json
import sys

(
    manifest_path,
    mode,
    epochs,
    batch_size,
    base_channels,
    seeds,
    routes,
    policies,
    eval_modes,
    split_seed,
    perturb_seed,
    ombria_commit,
) = sys.argv[1:]
manifest = json.load(open(manifest_path))
expected = {
    "mode": mode,
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "base_channels": int(base_channels),
    "model_seeds": [int(value) for value in seeds.split()],
    "routes": routes.split(),
    "checkpoint_policies": policies.split(),
    "evaluation_modes": eval_modes.split(),
    "split_seed": int(split_seed),
    "perturb_seed": int(perturb_seed),
    "ombria_commit": ombria_commit,
}
mismatches = {
    key: {"existing": manifest.get(key), "requested": value}
    for key, value in expected.items()
    if manifest.get(key) != value
}
if mismatches:
    raise SystemExit(
        "Refusing an incompatible resume. Use a fresh RESULT_ROOT. "
        + json.dumps(mismatches, sort_keys=True)
    )
print("resume configuration gate: pass")
PY
fi

"$PYTHON" scripts/write_experiment_manifest.py \
  --out "$RESULT_ROOT/experiment_manifest.json" \
  --mode "$MODE" \
  --protocol quality-gated-v3 \
  --schema-version v3 \
  --seeds $SEEDS \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --split-seed "$SPLIT_SEED" \
  --perturb-seed "$PERTURB_SEED" \
  --eval-modes $EVAL_MODES \
  --primary-modes $PRIMARY_MODES \
  --routes $ROUTES \
  --checkpoint-policies $CHECKPOINT_POLICIES \
  --run-directory-template "${run_directory_templates[@]}" \
  --route-spec "${route_specs[@]}" \
  --ombria-commit "$OMBRIA_COMMIT"

mkdir -p external "$RUNS_DIR" "$EVAL_DIR" "$RESULT_ROOT/tables" "$RESULT_ROOT/figures"
if [ ! -d "$ROOT" ]; then
  git init -q "$ROOT"
  git -C "$ROOT" remote add origin https://github.com/geodrak/OMBRIA.git
  git -C "$ROOT" fetch --depth 1 origin "$OMBRIA_COMMIT"
  git -C "$ROOT" checkout -q --detach FETCH_HEAD
fi
for required in OmbriaS1/train OmbriaS1/test OmbriaS2/train OmbriaS2/test 2021/ALBANIA 2021/FRANCE 2021/GUYANA 2021/TIMOR; do
  if [ ! -d "$ROOT/$required" ]; then
    echo "Missing required OMBRIA path: $ROOT/$required" >&2
    exit 2
  fi
done

train_route() {
  local route="$1"
  local seed="$2"
  local run_dir="$RUNS_DIR/${route}_seed${seed}"
  local route_args=()
  case "$route" in
    clean)
      route_args=(--variant multimodal)
      ;;
    matched_control)
      route_args=(--variant multimodal --train-degrade-s2 quality_matched_light)
      ;;
    quality_concat)
      route_args=(--variant multimodal --s2-quality binary --train-degrade-s2 quality_matched_light)
      ;;
    quality_gated)
      route_args=(--variant multimodal --s2-quality binary --architecture quality_gated_fusion --train-degrade-s2 quality_matched_light)
      ;;
    gated_misaligned)
      route_args=(--variant multimodal --s2-quality mislocalized --architecture quality_gated_fusion --train-degrade-s2 quality_matched_light)
      ;;
    s1_reference)
      route_args=(--variant s1_bitemporal)
      ;;
    s2_reference)
      route_args=(--variant s2_bitemporal)
      ;;
    *)
      echo "Unknown route: $route" >&2
      exit 2
      ;;
  esac
  if [ ! -f "$run_dir/best_clean.pt" ] || [ ! -f "$run_dir/best_robust.pt" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --run-name "${route}_seed${seed}" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --split-seed "$SPLIT_SEED" \
      --eval-perturb-seed "$PERTURB_SEED" \
      --seed "$seed" \
      --loader-seed "$((seed + 200000))" \
      --corruption-seed "$((seed + 300000))" \
      --robust-val-modes none cloud_after_50 zero_after \
      "${route_args[@]}"
  fi
}

evaluate_route() {
  local route="$1"
  local seed="$2"
  local checkpoint_policy="$3"
  local mode="$4"
  local variant="multimodal"
  local quality="none"
  case "$route" in
    quality_concat|quality_gated)
      quality="binary"
      ;;
    gated_misaligned)
      quality="mislocalized"
      ;;
    s1_reference)
      variant="s1_bitemporal"
      ;;
    s2_reference)
      variant="s2_bitemporal"
      ;;
  esac
  local repetitions=1
  case "$mode" in
    patch_after|cloud_after_30|cloud_after_50|cloud_after_70|noise_after)
      repetitions=3
      ;;
  esac
  local checkpoint="$RUNS_DIR/${route}_seed${seed}/best_${checkpoint_policy}.pt"
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
  for route in $ROUTES; do
    train_route "$route" "$seed"
  done
  for checkpoint_policy in $CHECKPOINT_POLICIES; do
    for eval_mode in $EVAL_MODES; do
      for route in $ROUTES; do
        evaluate_route "$route" "$seed" "$checkpoint_policy" "$eval_mode"
      done
    done
  done
done

"$PYTHON" scripts/summarize_confirmatory_events.py \
  --evaluations-dir "$EVAL_DIR" \
  --out-csv "$RESULT_ROOT/tables/event_heldout_summary.csv" \
  --out-md "$RESULT_ROOT/tables/event_heldout_summary.md"

"$PYTHON" scripts/summarize_quality_gated_v3.py \
  --evaluations-dir "$EVAL_DIR" \
  --manifest "$RESULT_ROOT/experiment_manifest.json" \
  --out-csv "$RESULT_ROOT/tables/prespecified_contrasts.csv" \
  --out-md "$RESULT_ROOT/tables/prespecified_contrasts.md" \
  --decision-json "$RESULT_ROOT/decision_gate.json"

if [ "$MODE" = "full" ]; then
  audit_split="$RUNS_DIR/clean_seed7/splits.json"
else
  audit_split="$RUNS_DIR/quality_concat_seed7/splits.json"
fi
"$PYTHON" scripts/audit_split_near_duplicates.py \
  --root "$ROOT" \
  --split-json "$audit_split" \
  --out-json "$RESULT_ROOT/split_near_duplicate_audit.json" \
  --out-csv "$RESULT_ROOT/tables/train_val_nearest_pairs.csv"

if [ "$MODE" = "full" ]; then
  for checkpoint_policy in $CHECKPOINT_POLICIES; do
    "$PYTHON" scripts/export_sensor_state_v2_probabilities.py \
      --root "$ROOT" \
      --runs-dir "$RUNS_DIR" \
      --manifest "$RESULT_ROOT/experiment_manifest.json" \
      --seed 7 \
      --checkpoint-policy "$checkpoint_policy" \
      --perturb-seed "$PERTURB_SEED" \
      --modes cloud_after_50 zero_after zero_all \
      --out-dir "$RESULT_ROOT/figures"
    "$PYTHON" scripts/export_quality_gate_panels.py \
      --root "$ROOT" \
      --runs-dir "$RUNS_DIR" \
      --seed 7 \
      --checkpoint-policy "$checkpoint_policy" \
      --perturb-seed "$PERTURB_SEED" \
      --mode cloud_after_50 \
      --out-dir "$RESULT_ROOT/figures"
  done
fi

"$PYTHON" scripts/package_confirmatory_artifacts.py \
  --root "$RESULT_ROOT" \
  --out results/ombria_quality_gated_v3_artifacts.zip \
  --include-checkpoints

cat "$RESULT_ROOT/tables/prespecified_contrasts.md"
