from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT / "notebooks" / "kaggle_quality_uncertainty_smagnet_full_seed7.ipynb"
)
AUTHORIZATION = (
    ROOT / "manifests" / "quality_uncertainty_smagnet_smoke_authorization.json"
)
SOURCE_COMMIT = "8b5a4f9ed7d0393a3b9259451f7e7dd3089f5d64"
SMOKE_SHA256 = (
    "eedaf8027e5720ff1ee72f39bc98f12e56a82928fb13a988f2bfe96075c1b0e9"
)


class SmagnetFullNotebookTest(unittest.TestCase):
    def test_smoke_authorization_is_explicit_and_non_scientific(self) -> None:
        document = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
        self.assertEqual(document["status"], "pass")
        self.assertEqual(document["artifact"]["sha256"], SMOKE_SHA256)
        self.assertEqual(document["experiment_source"]["commit"], SOURCE_COMMIT)
        self.assertTrue(document["audit"]["full_authorized"])
        self.assertFalse(document["audit"]["smoke_scores_publishable"])
        self.assertFalse(document["audit"]["scientific_interpretation_allowed"])
        self.assertEqual(document["release"]["released_full_seed"], 7)

    def test_notebook_is_frozen_parseable_and_output_free(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(len(notebook["cells"]), 5)
        combined = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        for token in (
            SOURCE_COMMIT,
            SMOKE_SHA256,
            "quality_uncertainty_smagnet_full_seed7",
            "quality_map_uncertainty_smagnet_full_seed7_artifacts.zip",
            "gate['condition_count'] == 54",
            "gate['repetitions'] == 3",
            "configuration['epochs'] == 200",
            "gate['scientific_interpretation_allowed'] is False",
        ):
            self.assertIn(token, combined)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
