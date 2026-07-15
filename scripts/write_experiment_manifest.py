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
    parser.add_argument(
        "--protocol",
        default="sensor-state-v2",
        help="Human-readable frozen protocol identifier.",
    )
    parser.add_argument(
        "--schema-version", choices=("v2", "v3"), default="v2"
    )
    parser.add_argument(
        "--run-directory-template",
        nargs="*",
        default=(),
        metavar="ROUTE=TEMPLATE",
    )
    parser.add_argument(
        "--route-spec",
        nargs="*",
        default=(),
        metavar="ROUTE=ARCHITECTURE,VARIANT,QUALITY",
    )
    parser.add_argument("--primary-modes", nargs="*", default=())
    return parser.parse_args()


def git_output(command: list[str]) -> str | None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else None


def parse_assignments(values: list[str] | tuple[str, ...], label: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use KEY=VALUE, got {value!r}")
        key, assigned = value.split("=", 1)
        if not key or not assigned or key in assignments:
            raise ValueError(f"Invalid or duplicate {label}: {value!r}")
        assignments[key] = assigned
    return assignments


def main() -> None:
    args = parse_args()
    run_directory_templates = parse_assignments(
        args.run_directory_template, "run-directory-template"
    )
    raw_route_specs = parse_assignments(args.route_spec, "route-spec")
    route_specs: dict[str, dict[str, str]] = {}
    for route, value in raw_route_specs.items():
        fields = value.split(",")
        if len(fields) != 3:
            raise ValueError(
                "route-spec values must be ARCHITECTURE,VARIANT,QUALITY"
            )
        route_specs[route] = dict(
            zip(("architecture", "variant", "s2_quality"), fields, strict=True)
        )
    if run_directory_templates and set(run_directory_templates) != set(args.routes):
        raise ValueError("Run-directory templates must cover the exact route set")
    if route_specs and set(route_specs) != set(args.routes):
        raise ValueError("Route specs must cover the exact route set")
    manifest = {
        "schema": f"geoai-ombria-confirmatory-experiment-{args.schema_version}",
        "protocol": args.protocol,
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
        "run_directory_templates": run_directory_templates or None,
        "route_specs": route_specs or None,
        "primary_modes": args.primary_modes,
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
