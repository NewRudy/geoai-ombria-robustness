from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_quality_uncertainty_smoke.ipynb"
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"
SOURCE_COMMIT = "3e97f93bdea999e1781869b74842a88050f9bdb1"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build() -> dict[str, object]:
    setup = f"""from pathlib import Path
import os, shutil, subprocess, sys

WORKING = Path('/kaggle/working')
project = WORKING / 'geoai-ombria-robustness'
os.chdir(WORKING)
if project.exists():
    shutil.rmtree(project)
project.mkdir(parents=True)
subprocess.run(
    ['git', '-C', str(project), 'init', '-q'],
    check=True,
)
subprocess.run(
    ['git', '-C', str(project), 'remote', 'add', 'origin', {REPOSITORY!r}],
    check=True,
)
subprocess.run(
    ['git', '-C', str(project), 'fetch', '--depth', '1', 'origin', {SOURCE_COMMIT!r}],
    check=True,
)
subprocess.run(
    ['git', '-C', str(project), 'checkout', '-q', '--detach', 'FETCH_HEAD'],
    check=True,
)
os.chdir(project)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert commit == {SOURCE_COMMIT!r}, (commit, {SOURCE_COMMIT!r})
print('source commit:', commit)
"""
    dependencies = """subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 'rasterio>=1.4'],
    check=True,
)
subprocess.run([sys.executable, 'scripts/ensure_cuda_compat.py'], check=True)
subprocess.run([sys.executable, 'scripts/check_cuda_runtime.py'], check=True)
test_env = os.environ.copy()
test_env['PYTHONPATH'] = str(project / 'src')
subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    check=True,
    env=test_env,
)
"""
    run = """env = os.environ.copy()
subprocess.run(
    ['bash', 'scripts/run_quality_uncertainty_smoke.sh'],
    check=True,
    env=env,
)
"""
    verify = """import hashlib, json
from zipfile import ZipFile
from IPython.display import FileLink, display

artifact = project / 'results' / 'quality_map_uncertainty_smoke_artifacts.zip'
with ZipFile(artifact) as archive:
    assert archive.testzip() is None
    assert not any(
        name.endswith(('_S1Hand.tif', '_S2Hand.tif', '_LabelHand.tif'))
        for name in archive.namelist()
    ), 'Raw Sen1Floods11 rasters must not be redistributed'
    manifest_name = next(
        name for name in archive.namelist()
        if name.endswith('/artifact_manifest.json')
    )
    manifest = json.loads(archive.read(manifest_name))
    root = manifest['root']
    for record in manifest['files']:
        name = root + '/' + record['path']
        assert hashlib.sha256(archive.read(name)).hexdigest() == record['sha256']
    smoke_name = next(
        name for name in archive.namelist()
        if name.endswith('/sen1floods11_scl_smoke.json')
    )
    scl_smoke = json.loads(archive.read(smoke_name))
    assert scl_smoke['status'] == 'pass'
    assert scl_smoke['manifest_match_fraction'] >= 0.95
    alignment_name = next(
        name for name in archive.namelist()
        if name.endswith('/sen1floods11_alignment_audit.json')
    )
    alignment = json.loads(archive.read(alignment_name))
    assert alignment['automated_status'] == 'pass'
    assert alignment['selected_count'] >= 11
    external_gate_name = next(
        name for name in archive.namelist()
        if name.endswith('/sen1floods11/sen1floods11_decision_gate.json')
    )
    external_gate = json.loads(archive.read(external_gate_name))
    assert external_gate['status'] == 'pass'
    assert external_gate['pipeline_only'] is True
    assert external_gate['expected_training_runs'] == 8
    assert external_gate['complete_training_runs'] == 8
    preparation_name = next(
        name for name in archive.namelist()
        if name.endswith('/sen1floods11/sen1floods11_preparation_report.json')
    )
    preparation = json.loads(archive.read(preparation_name))
    assert preparation['status'] == 'pass'
    assert preparation['record_count'] == 52
print('artifact MB:', round(artifact.stat().st_size / 1024**2, 1))
print('files verified:', len(manifest['files']))
print('alignment panel:', alignment['panel'])
print('external training runs:', external_gate['complete_training_runs'])
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(
                "# Quality-map uncertainty smoke\n\n"
                "This pipeline validates the new two-mask experiment: optical "
                "content is degraded using a reference availability map, while "
                "the model receives an independently corrupted quality map. It "
                "also downloads an outcome-independent Sen1Floods11 subset, "
                "trains all eight frozen routes, evaluates independent and "
                "structured quality-map errors, verifies both SCL providers, "
                "and packages an 11-event alignment panel. Raw dataset rasters "
                "are not included in the result ZIP.\n\n"
                "**Smoke scores are pipeline checks, not paper evidence.**\n"
            ),
            code(setup),
            code(dependencies),
            code(run),
            code(verify),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
