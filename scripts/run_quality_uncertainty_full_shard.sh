#!/usr/bin/env bash
set -eo pipefail

SEED="${1:-}"
case "$SEED" in
  7|13|21|29|37) ;;
  *)
    echo "Usage: $0 {7|13|21|29|37}" >&2
    exit 2
    ;;
esac

PYTHON=$(command -v python)
OMBRIA_ROOT=external/OMBRIA
OMBRIA_COMMIT=38a490355f76da8ce27ed051138f03f3492a6e46
SEN1_ROOT=external/Sen1Floods11_quality_uncertainty
RESULT_ROOT="results/quality_uncertainty_full_seed${SEED}"
ARCHIVE="results/quality_map_uncertainty_full_seed${SEED}_artifacts.zip"

mkdir -p "$RESULT_ROOT"
exec > >(tee -a "$RESULT_ROOT/run.log") 2>&1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== quality-map uncertainty Full seed $SEED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PYTHON" scripts/check_cuda_runtime.py --json-out "$RESULT_ROOT/runtime_manifest.json"
"$PYTHON" -m pip freeze > "$RESULT_ROOT/environment_freeze.txt"

if [ ! -d "$OMBRIA_ROOT/.git" ]; then
  mkdir -p "$(dirname "$OMBRIA_ROOT")"
  git init -q "$OMBRIA_ROOT"
  git -C "$OMBRIA_ROOT" remote add origin https://github.com/geodrak/OMBRIA.git
  git -C "$OMBRIA_ROOT" fetch --depth 1 origin "$OMBRIA_COMMIT"
  git -C "$OMBRIA_ROOT" checkout -q --detach FETCH_HEAD
fi
ACTUAL_OMBRIA_COMMIT=$(git -C "$OMBRIA_ROOT" rev-parse HEAD)
if [ "$ACTUAL_OMBRIA_COMMIT" != "$OMBRIA_COMMIT" ]; then
  echo "OMBRIA commit mismatch: $ACTUAL_OMBRIA_COMMIT" >&2
  exit 2
fi

cp manifests/sen1floods11_scl_manifest.json "$RESULT_ROOT/"
cp manifests/quality_uncertainty_smoke_authorization.json "$RESULT_ROOT/"
cp manifests/quality_uncertainty_core_equivalence.json "$RESULT_ROOT/"

"$PYTHON" - "$RESULT_ROOT/full_shard_plan.json" "$SEED" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from geoai_ombria_robustness.quality_uncertainty_full import build_full_shard_plan

output = Path(sys.argv[1])
seed = int(sys.argv[2])
document = {
    **build_full_shard_plan(seed).to_dict(),
    "source_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "smoke_authorization": "quality_uncertainty_smoke_authorization.json",
    "core_equivalence": "quality_uncertainty_core_equivalence.json",
    "published_architecture_gate": {
        "method": "SMAGNet",
        "official_repository": "https://github.com/ASUcicilab/SMAGNet",
        "official_commit": "4371df08e6ca3b9d71c0385ad57b589830469a0c",
        "license": "MIT",
        "status": "separate_shard_required_before_scientific_interpretation",
    },
}
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY

"$PYTHON" scripts/run_ombria_quality_uncertainty_full.py \
  --root "$OMBRIA_ROOT" \
  --result-root "$RESULT_ROOT/ombria/seed$SEED" \
  --seed "$SEED" \
  --workers 0

"$PYTHON" scripts/run_sen1floods11_quality_uncertainty.py \
  --mode full \
  --source-manifest manifests/sen1floods11_scl_manifest.json \
  --data-root "$SEN1_ROOT" \
  --result-root "$RESULT_ROOT/sen1floods11" \
  --workers 4 \
  --seeds "$SEED"

"$PYTHON" - "$RESULT_ROOT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seed = int(sys.argv[2])
runtime = json.loads((root / "runtime_manifest.json").read_text())
smoke = json.loads((root / "quality_uncertainty_smoke_authorization.json").read_text())
equivalence = json.loads((root / "quality_uncertainty_core_equivalence.json").read_text())
ombria = json.loads(
    (root / "ombria" / f"seed{seed}" / "ombria_decision_gate.json").read_text()
)
external = json.loads(
    (root / "sen1floods11" / "sen1floods11_decision_gate.json").read_text()
)
passed = (
    runtime["cuda_conv2d_gate"] == "pass"
    and runtime["repository_dirty_tracked"] is False
    and smoke["status"] == "pass"
    and smoke["audit"]["full_authorized"] is True
    and equivalence["status"] == "pass"
    and ombria["status"] == "pass"
    and ombria["active_seed"] == seed
    and external["status"] == "pass"
    and external["active_seeds"] == [seed]
    and external["shard_complete"] is True
    and external["all_full_seeds_present"] is False
)
gate = {
    "schema": "geoai-quality-map-uncertainty-full-shard-gate-v1",
    "status": "pass" if passed else "fail",
    "active_seed": seed,
    "smoke_authorized": smoke["audit"]["full_authorized"],
    "smoke_core_equivalence": equivalence["status"],
    "cuda_conv2d_gate": runtime["cuda_conv2d_gate"],
    "ombria_gate": ombria["status"],
    "sen1floods11_gate": external["status"],
    "remaining_core_seeds": [value for value in [7, 13, 21, 29, 37] if value != seed],
    "published_architecture_gate": "pending_separate_shard",
    "scientific_interpretation_allowed": False,
    "claim_boundary": (
        "This complete seed shard remains incomplete scientific evidence until "
        "all five core seeds, the SMAGNet gate, and post-run audits pass."
    ),
}
(root / "full_shard_decision_gate.json").write_text(
    json.dumps(gate, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(gate, indent=2))
if not passed:
    raise SystemExit("Full shard decision gate failed")
PY

"$PYTHON" scripts/package_quality_uncertainty_artifacts.py \
  --root "$RESULT_ROOT" \
  --out "$ARCHIVE"

echo "Full shard archive: $ARCHIVE"
