from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_FIELDS = ("s1_before", "s1_after", "s2_before", "s2_after")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact and visually near train/validation chip pairs."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--near-threshold", type=float, default=0.9995)
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def resolved_path(value: str, root: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    candidates = (root / path, root.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(value)


def exact_fingerprint(sample: dict[str, object], root: Path) -> str:
    digest = hashlib.sha256()
    for field in (*IMAGE_FIELDS, "s2_mask"):
        path = resolved_path(str(sample[field]), root)
        digest.update(field.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def image_feature(sample: dict[str, object], root: Path) -> np.ndarray:
    features: list[np.ndarray] = []
    for field in IMAGE_FIELDS:
        path = resolved_path(str(sample[field]), root)
        array = np.asarray(
            Image.open(path).convert("L").resize((16, 16)), dtype=np.float32
        ).reshape(-1)
        scale = float(array.std())
        features.append((array - float(array.mean())) / max(scale, 1.0))
    feature = np.concatenate(features)
    norm = float(np.linalg.norm(feature))
    return feature / max(norm, 1e-12)


def main() -> None:
    args = parse_args()
    split = json.loads(args.split_json.read_text())
    train = split["train"]
    val = split["val"]
    train_hashes = {
        exact_fingerprint(sample, args.root): str(sample["chip_id"]) for sample in train
    }
    val_hashes = {
        exact_fingerprint(sample, args.root): str(sample["chip_id"]) for sample in val
    }
    exact = sorted(set(train_hashes) & set(val_hashes))

    train_features = np.stack([image_feature(sample, args.root) for sample in train])
    val_features = np.stack([image_feature(sample, args.root) for sample in val])
    similarities = val_features @ train_features.T
    nearest_indexes = np.argmax(similarities, axis=1)
    rows = sorted(
        (
            {
                "val_chip_id": str(val[index]["chip_id"]),
                "nearest_train_chip_id": str(train[int(train_index)]["chip_id"]),
                "cosine_similarity": f"{float(similarities[index, train_index]):.8f}",
            }
            for index, train_index in enumerate(nearest_indexes)
        ),
        key=lambda row: float(row["cosine_similarity"]),
        reverse=True,
    )
    flagged = [
        row for row in rows if float(row["cosine_similarity"]) >= args.near_threshold
    ]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "val_chip_id",
                "nearest_train_chip_id",
                "cosine_similarity",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows[: args.top_k])

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(
            {
                "schema": "geoai-ombria-split-near-duplicate-audit-v1",
                "train_chips": len(train),
                "validation_chips": len(val),
                "exact_cross_split_duplicates": [
                    {
                        "train_chip_id": train_hashes[fingerprint],
                        "validation_chip_id": val_hashes[fingerprint],
                    }
                    for fingerprint in exact
                ],
                "near_duplicate_threshold": args.near_threshold,
                "near_duplicate_pairs_at_or_above_threshold": flagged,
                "feature_definition": "concatenated per-image standardized 16x16 grayscale S1-before/S1-after/S2-before/S2-after thumbnails; cosine similarity",
                "scope_boundary": "Diagnostic for exact or visually near chip duplication only; it cannot establish scene-level or spatial independence without source geolocation metadata.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "exact_duplicates": len(exact),
                "near_pairs": len(flagged),
                "maximum_similarity": float(rows[0]["cosine_similarity"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
