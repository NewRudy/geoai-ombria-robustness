from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from geoai_ombria_robustness.smagnet_adapter import (  # noqa: E402
    OFFICIAL_SMAGNET_COMMIT,
    OFFICIAL_SMAGNET_REPOSITORY,
    render_source_manifest,
    validate_official_smagnet_checkout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the frozen official SMAGNet source."
    )
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    return parser.parse_args()


def git_output(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def normalize_repository_url(value: str) -> str:
    normalized = value.rstrip("/")
    return normalized[:-4] if normalized.endswith(".git") else normalized


def fetch(checkout: Path) -> None:
    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git path: {checkout}")
    if not checkout.exists():
        checkout.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(
            ["git", "remote", "add", "origin", OFFICIAL_SMAGNET_REPOSITORY],
            cwd=checkout,
            check=True,
        )
    remotes = git_output("remote", cwd=checkout).splitlines()
    if "origin" not in remotes:
        subprocess.run(
            ["git", "remote", "add", "origin", OFFICIAL_SMAGNET_REPOSITORY],
            cwd=checkout,
            check=True,
        )
    origin = git_output("remote", "get-url", "origin", cwd=checkout)
    if normalize_repository_url(origin) != normalize_repository_url(
        OFFICIAL_SMAGNET_REPOSITORY
    ):
        raise RuntimeError("existing SMAGNet checkout has an unexpected origin")
    subprocess.run(
        ["git", "fetch", "--depth", "1", "origin", OFFICIAL_SMAGNET_COMMIT],
        cwd=checkout,
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "-q", "--detach", "FETCH_HEAD"],
        cwd=checkout,
        check=True,
    )
    head = git_output("rev-parse", "HEAD", cwd=checkout)
    if head != OFFICIAL_SMAGNET_COMMIT:
        raise RuntimeError(f"official checkout resolved to unexpected commit {head}")
    if git_output("status", "--short", cwd=checkout):
        raise RuntimeError("official checkout is dirty after fetch")


def main() -> None:
    args = parse_args()
    fetch(args.checkout)
    manifest = validate_official_smagnet_checkout(args.checkout)
    manifest["checkout_commit"] = git_output("rev-parse", "HEAD", cwd=args.checkout)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(render_source_manifest(manifest), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
