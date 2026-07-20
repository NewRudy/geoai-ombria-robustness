#!/usr/bin/env bash
set -eo pipefail

PYTHON=$(command -v python)
RESULT_ROOT=results/quality_uncertainty_smagnet_smoke
DATA_ROOT=external/Sen1Floods11_quality_uncertainty
SMAGNET_ROOT=external/SMAGNet_4371df0
SOURCE_MANIFEST=manifests/sen1floods11_scl_manifest.json
OFFICIAL_MANIFEST="$RESULT_ROOT/official_source_manifest.json"
ARTIFACT=results/quality_map_uncertainty_smagnet_smoke_artifacts.zip

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/run.log") 2>&1
export PYTHONUNBUFFERED=1

echo "=== official SMAGNet quality-uncertainty smoke $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PYTHON" scripts/check_cuda_runtime.py --json-out "$RESULT_ROOT/runtime_manifest.json"
"$PYTHON" -m pip freeze > "$RESULT_ROOT/environment_freeze.txt"
"$PYTHON" scripts/fetch_official_smagnet.py \
  --checkout "$SMAGNET_ROOT" \
  --manifest-out "$OFFICIAL_MANIFEST"

"$PYTHON" scripts/run_sen1floods11_smagnet.py \
  --mode smoke \
  --seed 7 \
  --source-manifest "$SOURCE_MANIFEST" \
  --data-root "$DATA_ROOT" \
  --smagnet-checkout "$SMAGNET_ROOT" \
  --official-source-manifest "$OFFICIAL_MANIFEST" \
  --result-root "$RESULT_ROOT" \
  --workers 4 \
  --micro-batch-size 4 \
  --gradient-accumulation 4 \
  --amp

"$PYTHON" scripts/package_quality_uncertainty_artifacts.py \
  --root "$RESULT_ROOT" \
  --out "$ARTIFACT"

echo "SMAGNet Smoke archive: $ARTIFACT"
