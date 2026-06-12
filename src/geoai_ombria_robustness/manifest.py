from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


KNOWN_LAYERS = ("S1", "S2", "QC")


@dataclass(frozen=True)
class ChipFile:
    event: str
    chip_id: str
    layer: str
    path: Path


def parse_sen1floods11_name(path: Path) -> Optional[ChipFile]:
    """Parse names like Bolivia_103757_S2.tif into event/chip/layer fields."""
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return None

    parts = path.stem.split("_")
    if len(parts) < 3:
        return None

    layer = parts[-1]
    if layer not in KNOWN_LAYERS:
        return None

    event = parts[0]
    chip_id = "_".join(parts[1:-1])
    if not chip_id:
        return None

    return ChipFile(event=event, chip_id=chip_id, layer=layer, path=path)


def build_manifest(root: Path) -> list[ChipFile]:
    """Collect parseable Sen1Floods11 chip files under a local mirror."""
    return [
        chip
        for path in sorted(root.rglob("*.tif"))
        if (chip := parse_sen1floods11_name(path)) is not None
    ]
