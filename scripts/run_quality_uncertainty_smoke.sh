#!/usr/bin/env bash
set -eo pipefail

ROOT=external/OMBRIA
OMBRIA_COMMIT=38a490355f76da8ce27ed051138f03f3492a6e46
PYTHON=$(command -v python)
RESULT_ROOT=results/quality_uncertainty_smoke
EPOCHS=2
BATCH_SIZE=8
BASE_CHANNELS=16
SEED=7
SPLIT_SEED=20260710
PERTURB_SEED=20260716
RATES="0 0.2 0.4"
ROUTES="hard_oracle hard_error_aware concat_error_aware soft_error_aware s1_reference"
RUNS_DIR=$RESULT_ROOT/runs
EVALUATIONS_DIR=$RESULT_ROOT/evaluations

mkdir -p "$RESULT_ROOT" "$RUNS_DIR" "$EVALUATIONS_DIR" "$RESULT_ROOT/tables"
exec > >(tee -a "$RESULT_ROOT/run.log") 2>&1
export PYTHONUNBUFFERED=1

echo "=== quality-map uncertainty smoke $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PYTHON" scripts/check_cuda_runtime.py --json-out "$RESULT_ROOT/runtime_manifest.json"
"$PYTHON" -m pip freeze > "$RESULT_ROOT/environment_freeze.txt"

if [ ! -d "$ROOT" ]; then
  mkdir -p "$(dirname "$ROOT")"
  git init -q "$ROOT"
  git -C "$ROOT" remote add origin https://github.com/geodrak/OMBRIA.git
  git -C "$ROOT" fetch --depth 1 origin "$OMBRIA_COMMIT"
  git -C "$ROOT" checkout -q --detach FETCH_HEAD
fi
for required in OmbriaS1/train OmbriaS1/test OmbriaS2/train OmbriaS2/test; do
  if [ ! -d "$ROOT/$required" ]; then
    echo "Missing required OMBRIA path: $ROOT/$required" >&2
    exit 2
  fi
done

cp manifests/sen1floods11_scl_manifest.json "$RESULT_ROOT/"
"$PYTHON" scripts/smoke_sen1floods11_scl.py \
  --manifest manifests/sen1floods11_scl_manifest.json \
  --work-dir "$RESULT_ROOT/sen1floods11_sample" \
  --out-json "$RESULT_ROOT/sen1floods11_scl_smoke.json"

"$PYTHON" - "$RESULT_ROOT/experiment_manifest.json" "$EPOCHS" "$BATCH_SIZE" \
  "$BASE_CHANNELS" "$SEED" "$SPLIT_SEED" "$PERTURB_SEED" "$RATES" \
  "$ROUTES" "$OMBRIA_COMMIT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone

(
    output,
    epochs,
    batch_size,
    base_channels,
    seed,
    split_seed,
    perturb_seed,
    rates,
    routes,
    ombria_commit,
) = sys.argv[1:]
document = {
    "schema": "geoai-quality-map-uncertainty-smoke-v1",
    "mode": "smoke",
    "pipeline_only": True,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "ombria_commit": ombria_commit,
    "epochs": int(epochs),
    "batch_size": int(batch_size),
    "base_channels": int(base_channels),
    "seed": int(seed),
    "split_seed": int(split_seed),
    "perturb_seed": int(perturb_seed),
    "quality_error_rates": [float(value) for value in rates.split()],
    "routes": routes.split(),
    "content_degradation": "cloud_after_50",
    "boundary": (
        "Smoke scores validate execution only. OMBRIA occlusion is cloud-like "
        "and Sen1Floods11 SCL is an operational quality proxy."
    ),
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

train_route() {
  local route="$1"
  local run_dir="$RUNS_DIR/"$route"_seed$SEED"
  local route_args=""
  case "$route" in
    hard_oracle)
      route_args="--variant multimodal --s2-quality binary --architecture quality_gated_fusion --train-degrade-s2 quality_matched_light"
      ;;
    hard_error_aware)
      route_args="--variant multimodal --s2-quality binary --architecture quality_gated_fusion --train-degrade-s2 quality_matched_light --train-quality-error-rates 0 0.2 0.4"
      ;;
    concat_error_aware)
      route_args="--variant multimodal --s2-quality binary --architecture early_fusion_unet --train-degrade-s2 quality_matched_light --train-quality-error-rates 0 0.2 0.4"
      ;;
    soft_error_aware)
      route_args="--variant multimodal --s2-quality binary --architecture soft_quality_prior_fusion --train-degrade-s2 quality_matched_light --train-quality-error-rates 0 0.2 0.4"
      ;;
    s1_reference)
      route_args="--variant s1_bitemporal"
      ;;
    *)
      echo "Unknown route: $route" >&2
      exit 2
      ;;
  esac
  if [ ! -f "$run_dir/best_clean.pt" ]; then
    "$PYTHON" scripts/train_ombria_unet.py \
      --root "$ROOT" \
      --out-dir "$RUNS_DIR" \
      --run-name "$route"_seed"$SEED" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --base-channels "$BASE_CHANNELS" \
      --seed "$SEED" \
      --split-seed "$SPLIT_SEED" \
      --eval-perturb-seed "$PERTURB_SEED" \
      --loader-seed "$((SEED + 200000))" \
      --corruption-seed "$((SEED + 300000))" \
      --quality-error-seed "$((SEED + 400000))" \
      --robust-val-modes none cloud_after_50 \
      $route_args
  fi
}

for route in $ROUTES; do
  train_route "$route"
done

evaluate_cell() {
  local route="$1"
  local false_available="$2"
  local false_unavailable="$3"
  local fa_key
  local fu_key
  fa_key=$(printf '%s' "$false_available" | tr '.' 'p')
  fu_key=$(printf '%s' "$false_unavailable" | tr '.' 'p')
  local out="$EVALUATIONS_DIR/$route/fa"$fa_key"_fu$fu_key"
  if [ ! -f "$out/summary_metrics.csv" ]; then
    "$PYTHON" scripts/evaluate_ombria_quality_uncertainty.py \
      --root "$ROOT" \
      --checkpoint "$RUNS_DIR/"$route"_seed$SEED/best_clean.pt" \
      --route "$route" \
      --content-degradation cloud_after_50 \
      --false-available-rate "$false_available" \
      --false-unavailable-rate "$false_unavailable" \
      --perturb-seed "$PERTURB_SEED" \
      --repetitions 1 \
      --batch-size "$BATCH_SIZE" \
      --out-dir "$out"
  fi
}

evaluate_cell s1_reference 0 0
for route in hard_oracle hard_error_aware concat_error_aware soft_error_aware; do
  for false_available in $RATES; do
    for false_unavailable in $RATES; do
      evaluate_cell "$route" "$false_available" "$false_unavailable"
    done
  done
done

"$PYTHON" scripts/summarize_quality_uncertainty.py \
  --evaluations-dir "$EVALUATIONS_DIR" \
  --out-csv "$RESULT_ROOT/tables/response_surface.csv" \
  --out-md "$RESULT_ROOT/tables/response_surface.md"

"$PYTHON" scripts/package_quality_uncertainty_artifacts.py \
  --root "$RESULT_ROOT" \
  --out results/ombria_quality_uncertainty_smoke_artifacts.zip

echo "Smoke archive: results/ombria_quality_uncertainty_smoke_artifacts.zip"
