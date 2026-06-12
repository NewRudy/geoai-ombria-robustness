from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_last_metrics(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("results/runs/ombria"))
    parser.add_argument("--out", type=Path, default=Path("results/tables/ombria_run_summary.csv"))
    args = parser.parse_args()

    records = []
    for run_dir in sorted(args.runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        with config_path.open() as f:
            config = json.load(f)

        record = {
            "run": run_dir.name,
            "record_type": "train",
            "variant": config.get("variant", ""),
            "degrade_s2": config.get("degrade_s2", ""),
            "train_degrade_s2": config.get("train_degrade_s2", "none"),
            "s2_quality": config.get("s2_quality", "none"),
            "seed": config.get("seed", ""),
            "epochs": config.get("epochs", ""),
            "batch_size": config.get("batch_size", ""),
            "base_channels": config.get("base_channels", ""),
        }

        eval_path = run_dir / "eval_metrics.json"
        if eval_path.exists():
            with eval_path.open() as f:
                metrics = json.load(f)
            record["record_type"] = "eval"
            if metrics.get("checkpoint_epochs") is not None:
                record["epochs"] = metrics["checkpoint_epochs"]
            if metrics.get("checkpoint_batch_size") is not None:
                record["batch_size"] = metrics["checkpoint_batch_size"]
            if metrics.get("checkpoint_base_channels") is not None:
                record["base_channels"] = metrics["checkpoint_base_channels"]
            if metrics.get("checkpoint_train_degrade_s2") is not None:
                record["train_degrade_s2"] = metrics["checkpoint_train_degrade_s2"]
            if metrics.get("checkpoint_s2_quality") is not None:
                record["s2_quality"] = metrics["checkpoint_s2_quality"]
            for key, value in metrics.items():
                if key.startswith("test_"):
                    record[key] = value
        else:
            last = read_last_metrics(run_dir / "metrics.csv")
            if last:
                for key, value in last.items():
                    if key == "epoch" or key.startswith("test_") or key.startswith("val_"):
                        record[key] = value
        records.append(record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for record in records for key in record})
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
