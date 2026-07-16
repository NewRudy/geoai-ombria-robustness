from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.root.is_dir():
        raise FileNotFoundError(args.root)
    files = [
        path
        for path in sorted(args.root.rglob("*"))
        if path.is_file() and path.resolve() != args.out.resolve()
    ]
    if not files:
        raise RuntimeError(f"No artifacts found under {args.root}")
    manifest = {
        "schema": "geoai-quality-map-uncertainty-artifacts-v1",
        "root": args.root.name,
        "files": [
            {
                "path": path.relative_to(args.root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.out, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(
                path,
                arcname=f"{args.root.name}/{path.relative_to(args.root)}",
            )
        archive.writestr(
            f"{args.root.name}/artifact_manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        )
    with ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Artifact ZIP integrity check failed")
    print(f"Wrote {args.out} with {len(files)} files")


if __name__ == "__main__":
    main()
