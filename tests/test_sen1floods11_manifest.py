from __future__ import annotations

import unittest

from geoai_ombria_robustness.sen1floods11_manifest import (
    SclAsset,
    Sen1Floods11ManifestRecord,
    build_manifest_document,
    select_scl_assets,
)


def candidate(
    item_id: str,
    bbox: list[float],
    tile: str,
    baseline: str,
    sequence: int,
) -> dict:
    return {
        "id": item_id,
        "bbox": bbox,
        "assets": {"scl": {"href": f"https://example.test/{item_id}/SCL.tif"}},
        "properties": {
            "datetime": "2019-09-18T11:00:31Z",
            "s2:mgrs_tile": tile,
            "s2:processing_baseline": baseline,
            "s2:sequence": sequence,
            "created": f"202{sequence}-01-01T00:00:00Z",
            "eo:cloud_cover": 12.5,
        },
    }


class Sen1Floods11ManifestTests(unittest.TestCase):
    def test_selection_prefers_latest_processing_per_tile(self) -> None:
        chip_bbox = (-0.8, 38.0, -0.7, 38.1)
        candidates = [
            candidate("old", [-1.0, 37.5, 0.0, 39.0], "30SXH", "02.13", 0),
            candidate("new", [-1.0, 37.5, 0.0, 39.0], "30SXH", "05.00", 1),
            candidate("outside", [10.0, 10.0, 11.0, 11.0], "31AAA", "05.00", 1),
        ]
        selected = select_scl_assets(chip_bbox, candidates)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].item_id, "new")
        self.assertEqual(selected[0].mgrs_tile, "30SXH")

    def test_selection_keeps_intersecting_adjacent_tiles(self) -> None:
        chip_bbox = (0.9, 0.1, 1.1, 0.9)
        selected = select_scl_assets(
            chip_bbox,
            [
                candidate("left", [0.0, 0.0, 1.0, 1.0], "30AAA", "05.00", 1),
                candidate("right", [1.0, 0.0, 2.0, 1.0], "31AAA", "05.00", 1),
            ],
        )
        self.assertEqual([asset.item_id for asset in selected], ["left", "right"])

    def test_manifest_summary_reports_unmatched_records(self) -> None:
        asset = SclAsset(
            item_id="item",
            href="https://example.test/SCL.tif",
            mgrs_tile="30SXH",
            datetime="2019-09-18T11:00:31Z",
            processing_baseline="05.00",
            sequence=1,
            cloud_cover=2.0,
            bbox=(-1.0, 37.0, 0.0, 39.0),
            provider="earth-search",
        )
        common = {
            "event": "Spain",
            "bbox": (-0.8, 38.0, -0.7, 38.1),
            "s1_date": "2019-09-17",
            "s2_date": "2019-09-18",
            "s1_url": "s1",
            "s2_url": "s2",
            "label_url": "label",
            "source_stac_url": "stac",
        }
        records = [
            Sen1Floods11ManifestRecord(
                chip_id="Spain_1",
                split="train",
                scl_assets=(asset,),
                **common,
            ),
            Sen1Floods11ManifestRecord(
                chip_id="Spain_2",
                split="test",
                scl_assets=(),
                **common,
            ),
        ]
        document = build_manifest_document(records)
        self.assertEqual(document["summary"]["record_count"], 2)
        self.assertEqual(document["summary"]["matched_count"], 1)
        self.assertEqual(document["summary"]["unmatched_chip_ids"], ["Spain_2"])


if __name__ == "__main__":
    unittest.main()
