from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_artifact_audit import (  # noqa: E402
    audit_quality_uncertainty_smoke_artifact,
    render_quality_uncertainty_audit_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a returned quality-map uncertainty Smoke ZIP."
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument(
        "--alignment-visual-status",
        choices=("not-reviewed", "pass", "fail"),
        default="not-reviewed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_quality_uncertainty_smoke_artifact(
        args.archive,
        alignment_visual_status=args.alignment_visual_status,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(
        render_quality_uncertainty_audit_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    if not report["decision"]["full_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
