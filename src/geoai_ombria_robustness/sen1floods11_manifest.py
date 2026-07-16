from __future__ import annotations

import csv
import io
import json
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Iterable


SEN1FLOODS11_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
PLANETARY_COMPUTER_SEARCH_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1/search"
)
SCL_UNAVAILABLE_CLASSES = (0, 1, 3, 8, 9, 10, 11)
SPLIT_FILES = {
    "train": "flood_train_data.csv",
    "validation": "flood_valid_data.csv",
    "test": "flood_test_data.csv",
    "bolivia": "flood_bolivia_data.csv",
}
EVENT_METADATA_ALIASES = {
    "Mekong": "Cambodia",
}


@dataclass(frozen=True)
class SclAsset:
    item_id: str
    href: str
    mgrs_tile: str
    datetime: str
    processing_baseline: str
    sequence: int
    cloud_cover: float | None
    bbox: tuple[float, float, float, float]
    provider: str = "earth-search"


@dataclass(frozen=True)
class Sen1Floods11ManifestRecord:
    chip_id: str
    event: str
    split: str
    bbox: tuple[float, float, float, float]
    s1_date: str
    s2_date: str
    s1_url: str
    s2_url: str
    label_url: str
    source_stac_url: str
    scl_assets: tuple[SclAsset, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox"] = list(self.bbox)
        result["scl_assets"] = [
            {**asdict(asset), "bbox": list(asset.bbox)}
            for asset in self.scl_assets
        ]
        return result


def read_url(url: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "geoai-ombria-robustness/0.4"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _read_json(url: str) -> dict[str, Any]:
    return json.loads(read_url(url))


def load_official_split_map() -> dict[str, str]:
    split_map: dict[str, str] = {}
    split_root = f"{SEN1FLOODS11_BASE}/splits/flood_handlabeled"
    for split, filename in SPLIT_FILES.items():
        rows = csv.reader(
            io.StringIO(read_url(f"{split_root}/{filename}").decode("utf-8"))
        )
        for row in rows:
            if not row:
                continue
            chip_id = row[0].removesuffix("_S1Hand.tif")
            if chip_id in split_map:
                raise ValueError(f"Chip {chip_id} appears in multiple official splits")
            split_map[chip_id] = split
    return split_map


def load_event_metadata() -> dict[str, dict[str, Any]]:
    metadata = _read_json(f"{SEN1FLOODS11_BASE}/Sen1Floods11_Metadata.geojson")
    result: dict[str, dict[str, Any]] = {}
    for feature in metadata["features"]:
        properties = feature["properties"]
        event = str(properties["location"])
        if event in result:
            raise ValueError(f"Duplicate event metadata for {event}")
        result[event] = properties
    return result


def load_hand_labeled_source_items() -> dict[str, dict[str, Any]]:
    archive = zipfile.ZipFile(
        io.BytesIO(read_url(f"{SEN1FLOODS11_BASE}/catalog.zip"))
    )
    prefix = "catalog/sen1floods11_hand_labeled_source/"
    items: dict[str, dict[str, Any]] = {}
    for name in archive.namelist():
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        if name.endswith("/collection.json"):
            continue
        item = json.loads(archive.read(name))
        items[str(item["id"])] = item
    return items


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "geoai-ombria-robustness/0.4",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120.0) as response:
        return json.load(response)


def _search_sentinel2_l2a(
    bbox: tuple[float, float, float, float],
    acquisition_date: str,
    endpoint: str,
    provider: str,
) -> list[dict[str, Any]]:
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": list(bbox),
        "datetime": (
            f"{acquisition_date}T00:00:00Z/"
            f"{acquisition_date}T23:59:59Z"
        ),
        "limit": 100,
    }
    response = _post_json(endpoint, payload)
    features: list[dict[str, Any]] = []
    pages = 0
    while True:
        pages += 1
        if pages > 100:
            raise RuntimeError("Earth Search pagination exceeded 100 pages")
        for feature in response.get("features", []):
            feature["_quality_proxy_provider"] = provider
            features.append(feature)
        next_links = [
            link
            for link in response.get("links", [])
            if link.get("rel") == "next"
        ]
        if not next_links:
            break
        next_link = next_links[0]
        href = str(next_link["href"])
        if str(next_link.get("method") or "GET").upper() == "POST":
            response = _post_json(
                href,
                dict(next_link.get("body") or payload),
            )
        else:
            response = _read_json(href)
    return features


def search_sentinel2_l2a(
    bbox: tuple[float, float, float, float],
    acquisition_date: str,
) -> list[dict[str, Any]]:
    return _search_sentinel2_l2a(
        bbox,
        acquisition_date,
        EARTH_SEARCH_URL,
        "earth-search",
    )


def search_planetary_computer_l2a(
    bbox: tuple[float, float, float, float],
    acquisition_date: str,
) -> list[dict[str, Any]]:
    return _search_sentinel2_l2a(
        bbox,
        acquisition_date,
        PLANETARY_COMPUTER_SEARCH_URL,
        "planetary-computer",
    )


def _bbox_union(
    bboxes: Iterable[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    bboxes = list(bboxes)
    if not bboxes:
        raise ValueError("At least one bbox is required")
    return (
        min(bbox[0] for bbox in bboxes),
        min(bbox[1] for bbox in bboxes),
        max(bbox[2] for bbox in bboxes),
        max(bbox[3] for bbox in bboxes),
    )


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _processing_key(item: dict[str, Any]) -> tuple[float, int, str, str]:
    properties = item.get("properties", {})
    baseline_raw = str(properties.get("s2:processing_baseline") or "0")
    try:
        baseline = float(baseline_raw)
    except ValueError:
        baseline = 0.0
    return (
        baseline,
        int(properties.get("s2:sequence") or 0),
        str(properties.get("created") or ""),
        str(item.get("id") or ""),
    )


def _mgrs_tile(item: dict[str, Any]) -> str:
    properties = item.get("properties", {})
    explicit = properties.get("s2:mgrs_tile")
    if explicit:
        return str(explicit)
    item_id = str(item["id"])
    parts = item_id.split("_")
    for part in parts:
        if len(part) == 6 and part.startswith("T") and part[1:3].isdigit():
            return part[1:]
    if len(parts) >= 3 and len(parts[1]) == 5:
        return parts[1]
    raise ValueError(f"Cannot determine MGRS tile for {item_id}")


def select_scl_assets(
    chip_bbox: tuple[float, float, float, float],
    candidates: Iterable[dict[str, Any]],
) -> tuple[SclAsset, ...]:
    """Select one pinned processing version per intersecting MGRS tile."""

    intersecting = []
    for item in candidates:
        assets = item.get("assets", {})
        scl = assets.get("scl") or assets.get("SCL")
        if scl and _intersection_area(chip_bbox, tuple(item["bbox"])) > 0.0:
            intersecting.append(item)
    by_tile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in intersecting:
        by_tile[_mgrs_tile(item)].append(item)

    selected: list[SclAsset] = []
    for tile, items in sorted(by_tile.items()):
        item = max(items, key=_processing_key)
        properties = item.get("properties", {})
        assets = item.get("assets", {})
        scl = assets.get("scl") or assets.get("SCL")
        if scl is None:
            raise RuntimeError(f"Selected item {item['id']} has no SCL asset")
        selected.append(
            SclAsset(
                item_id=str(item["id"]),
                href=str(scl["href"]),
                mgrs_tile=tile,
                datetime=str(properties.get("datetime") or ""),
                processing_baseline=str(
                    properties.get("s2:processing_baseline") or ""
                ),
                sequence=int(properties.get("s2:sequence") or 0),
                cloud_cover=(
                    float(properties["eo:cloud_cover"])
                    if properties.get("eo:cloud_cover") is not None
                    else None
                ),
                bbox=tuple(float(value) for value in item["bbox"]),
                provider=str(
                    item.get("_quality_proxy_provider") or "earth-search"
                ),
            )
        )
    return tuple(selected)


def build_manifest_records() -> list[Sen1Floods11ManifestRecord]:
    split_map = load_official_split_map()
    event_metadata = load_event_metadata()
    source_items = load_hand_labeled_source_items()

    if set(source_items) != set(split_map):
        missing_splits = sorted(set(source_items) - set(split_map))
        missing_items = sorted(set(split_map) - set(source_items))
        raise RuntimeError(
            "Official split/catalog mismatch: "
            f"missing_splits={missing_splits[:5]}, "
            f"missing_items={missing_items[:5]}"
        )

    items_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chip_id, item in source_items.items():
        event = chip_id.rsplit("_", 1)[0]
        items_by_event[event].append(item)

    candidates_by_event: dict[str, list[dict[str, Any]]] = {}
    fallback_candidates_by_event: dict[str, list[dict[str, Any]]] = {}
    for event, items in sorted(items_by_event.items()):
        metadata_event = EVENT_METADATA_ALIASES.get(event, event)
        if metadata_event not in event_metadata:
            raise RuntimeError(f"No event metadata found for {event}")
        union = _bbox_union(tuple(item["bbox"]) for item in items)
        s2_date = str(event_metadata[metadata_event]["s2_date"]).replace("/", "-")
        candidates_by_event[event] = search_sentinel2_l2a(union, s2_date)
        if any(
            not select_scl_assets(
                tuple(float(value) for value in item["bbox"]),
                candidates_by_event[event],
            )
            for item in items
        ):
            fallback_candidates_by_event[event] = (
                search_planetary_computer_l2a(union, s2_date)
            )

    records: list[Sen1Floods11ManifestRecord] = []
    data_root = f"{SEN1FLOODS11_BASE}/data/flood_events/HandLabeled"
    catalog_root = (
        f"{SEN1FLOODS11_BASE}/catalog/"
        "sen1floods11_hand_labeled_source"
    )
    for chip_id, item in sorted(source_items.items()):
        event = chip_id.rsplit("_", 1)[0]
        metadata = event_metadata[EVENT_METADATA_ALIASES.get(event, event)]
        bbox = tuple(float(value) for value in item["bbox"])
        scl_assets = select_scl_assets(
            bbox,
            candidates_by_event[event],
        )
        if not scl_assets:
            scl_assets = select_scl_assets(
                bbox,
                fallback_candidates_by_event.get(event, ()),
            )
        records.append(
            Sen1Floods11ManifestRecord(
                chip_id=chip_id,
                event=event,
                split=split_map[chip_id],
                bbox=bbox,
                s1_date=str(metadata["s1_date"]).replace("/", "-"),
                s2_date=str(metadata["s2_date"]).replace("/", "-"),
                s1_url=f"{data_root}/S1Hand/{chip_id}_S1Hand.tif",
                s2_url=f"{data_root}/S2Hand/{chip_id}_S2Hand.tif",
                label_url=f"{data_root}/LabelHand/{chip_id}_LabelHand.tif",
                source_stac_url=f"{catalog_root}/{chip_id}/{chip_id}.json",
                scl_assets=scl_assets,
            )
        )
    return records


def build_manifest_document(
    records: Iterable[Sen1Floods11ManifestRecord],
) -> dict[str, Any]:
    records = list(records)
    split_counts: dict[str, int] = defaultdict(int)
    event_counts: dict[str, int] = defaultdict(int)
    unmatched: list[str] = []
    multiple_assets: list[str] = []
    for record in records:
        split_counts[record.split] += 1
        event_counts[record.event] += 1
        if not record.scl_assets:
            unmatched.append(record.chip_id)
        if len(record.scl_assets) > 1:
            multiple_assets.append(record.chip_id)

    return {
        "schema": "geoai-sen1floods11-scl-manifest-v1",
        "generated_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "sources": {
            "sen1floods11_version": "v1.1",
            "sen1floods11_base": SEN1FLOODS11_BASE,
            "earth_search_url": EARTH_SEARCH_URL,
            "planetary_computer_search_url": PLANETARY_COMPUTER_SEARCH_URL,
            "sentinel2_collection": "sentinel-2-l2a",
        },
        "selection_rule": (
            "Exact event S2 date; positive chip intersection; one item per "
            "MGRS tile; prefer Earth Search, use Planetary Computer only for "
            "otherwise unmatched chips; within a provider prefer highest "
            "processing baseline, sequence, created timestamp, then item ID."
        ),
        "scl_unavailable_classes": list(SCL_UNAVAILABLE_CLASSES),
        "summary": {
            "record_count": len(records),
            "matched_count": len(records) - len(unmatched),
            "match_fraction": (
                (len(records) - len(unmatched)) / len(records)
                if records
                else 0.0
            ),
            "split_counts": dict(sorted(split_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
            "unmatched_chip_ids": unmatched,
            "multiple_asset_chip_ids": multiple_assets,
        },
        "records": [record.to_dict() for record in records],
    }
