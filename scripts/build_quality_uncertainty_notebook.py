from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_quality_uncertainty_smoke.ipynb"
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"
SOURCE_COMMIT = "98b1e75ee2df71863d6ae6b257cf1a6e0a60ccbb"


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
subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    check=True,
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

artifact = project / 'results' / 'ombria_quality_uncertainty_smoke_artifacts.zip'
with ZipFile(artifact) as archive:
    assert archive.testzip() is None
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
print('artifact MB:', round(artifact.stat().st_size / 1024**2, 1))
print('files verified:', len(manifest['files']))
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(
                "# Quality-map uncertainty smoke\n\n"
                "This pipeline validates the new two-mask experiment: optical "
                "content is degraded using a reference availability map, while "
                "the model receives an independently corrupted quality map. It "
                "also verifies pinned Sen1Floods11 SCL assets from both Earth "
                "Search and Planetary Computer.\n\n"
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
