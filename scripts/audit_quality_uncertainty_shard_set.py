from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.quality_uncertainty_shard_set_audit import (  # noqa: E402
    audit_quality_uncertainty_shard_set_artifacts,
    render_quality_uncertainty_shard_set_audit_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit split and evidence compatibility across partial or complete "
            "quality-map uncertainty Full seed shards."
        )
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="SEED=ZIP",
        help="Repeat once for each returned frozen seed shard.",
    )
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    parsed_artifacts: list[tuple[int, Path]] = []
    for value in args.artifact:
        try:
            seed_text, path_text = value.split("=", 1)
            parsed_artifacts.append((int(seed_text), Path(path_text)))
        except (TypeError, ValueError):
            parser.error(f"invalid --artifact {value!r}; expected SEED=ZIP")
    args.artifact = parsed_artifacts
    return args


def main() -> None:
    args = parse_args()
    report = audit_quality_uncertainty_shard_set_artifacts(
        args.artifact,
        code_root=args.code_root,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(
        render_quality_uncertainty_shard_set_audit_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report["decision"], indent=2, ensure_ascii=False))
    if not report["decision"]["subset_preflight_authorized"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
