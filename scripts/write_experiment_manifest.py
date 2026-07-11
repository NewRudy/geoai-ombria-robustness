from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--base-channels", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--perturb-seed", type=int, required=True)
    parser.add_argument("--eval-modes", nargs="+", required=True)
    parser.add_argument(
        "--routes",
        nargs="+",
        default=(
            "clean",
            "light",
            "matched_control",
            "matched_quality",
            "s1_reference",
        ),
    )
    parser.add_argument("--checkpoint-policies", nargs="+", default=("clean",))
    parser.add_argument("--ombria-commit", required=True)
    return parser.parse_args()


def git_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def main() -> None:
    args = parse_args()
    manifest = {
        "schema": "geoai-ombria-confirmatory-experiment-v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "model_seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_channels": args.base_channels,
        "split_seed": args.split_seed,
        "perturb_seed": args.perturb_seed,
        "evaluation_modes": args.eval_modes,
        "routes": args.routes,
        "checkpoint_policies": args.checkpoint_policies,
        "ombria_commit": args.ombria_commit,
        "repository_commit": git_output(["git", "rev-parse", "HEAD"]),
        "repository_release": git_output(
            ["git", "describe", "--tags", "--exact-match"]
        ),
        "checkpoint_selection": {
            "clean": "maximum clean validation IoU",
            "robust": "maximum mean validation IoU across clean, cloud_after_50, and zero_after",
            "test_event_use": "test events are never evaluated during training or checkpoint selection",
        },
        "metric_aggregation": "global confusion counts across all 2021 event pixels, with per-chip rows retained",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"experiment manifest: {args.out}")


if __name__ == "__main__":
    main()
