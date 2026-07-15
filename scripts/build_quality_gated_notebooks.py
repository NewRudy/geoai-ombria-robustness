from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
SOURCE_REF = "v0.3.1-quality-gated"
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"


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


def notebook(mode: str) -> dict[str, object]:
    if mode == "smoke":
        title = "OMBRIA QGSF v0.3 smoke gate"
        description = (
            "This one-seed/two-epoch run validates the quality-gated method, "
            "event evaluation, prespecified contrast export, and artifact packaging. "
            "Its scores are not scientific evidence."
        )
        run_note = "Expected to be much shorter than Full; actual time is recorded."
    else:
        title = "OMBRIA QGSF v0.3 locked Full matrix"
        description = (
            "Run only after the smoke archive passes review. This notebook executes "
            "the frozen five-seed route/state matrix. Do not change routes, seeds, "
            "modes, or decision thresholds after inspecting results."
        )
        run_note = "Allow roughly 5--7 hours on a P100; actual time is recorded."
    setup = f"""from pathlib import Path
import os, shutil, subprocess, sys

WORKING = Path('/kaggle/working')
REPO_URL = {REPOSITORY!r}
SOURCE_REF = {SOURCE_REF!r}
project = WORKING / 'geoai-ombria-robustness'
os.chdir(WORKING)
if project.exists():
    shutil.rmtree(project)
subprocess.run(['git', 'clone', '--depth', '1', '--branch', SOURCE_REF, REPO_URL, str(project)], check=True)
os.chdir(project)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('locked source:', SOURCE_REF, commit)
"""
    dependencies = """subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy>=1.24', 'pillow>=10.0', 'matplotlib>=3.7'], check=True)
subprocess.run([sys.executable, 'scripts/ensure_cuda_compat.py'], check=True)
subprocess.run([sys.executable, 'scripts/check_cuda_runtime.py'], check=True)
scripts = [str(path) for path in Path('scripts').glob('*.py')]
subprocess.run([sys.executable, '-m', 'py_compile', *scripts], check=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'], check=True)
"""
    run = f"""env = os.environ.copy()
env.update({{'MODE': {mode!r}, 'PYTHON': sys.executable}})
subprocess.run(['bash', 'scripts/run_quality_gated_v3_matrix.sh'], check=True, env=env)
"""
    verify = f"""import hashlib, json
from zipfile import ZipFile
from IPython.display import FileLink, display

artifact = project / 'results' / 'ombria_quality_gated_v3_artifacts.zip'
with ZipFile(artifact) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    artifact_manifest_name = next(name for name in names if name.endswith('artifact_manifest.json'))
    artifact_manifest = json.loads(archive.read(artifact_manifest_name))
    for record in artifact_manifest['files']:
        assert record['path'] in names
        assert hashlib.sha256(archive.read(record['path'])).hexdigest() == record['sha256']
    checkpoint_manifest_name = next(name for name in names if name.endswith('checkpoint_manifest.json'))
    checkpoint_manifest = json.loads(archive.read(checkpoint_manifest_name))
    assert checkpoint_manifest['weights_included'] is True
    decision_manifest_name = next(name for name in names if name.endswith('decision_gate.json'))
    decision = json.loads(archive.read(decision_manifest_name))['decision']
{'assert decision["status"] == "pipeline_only"' if mode == 'smoke' else 'print("prespecified decision:", json.dumps(decision, indent=2))'}
print('artifact MB:', round(artifact.stat().st_size / 1024**2, 1))
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(f"# {title}\n\n{description}\n\n{run_note}\n"),
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
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("smoke", "full"):
        path = NOTEBOOK_DIR / f"kaggle_quality_gated_v3_{mode}.ipynb"
        path.write_text(json.dumps(notebook(mode), indent=1) + "\n")
        print(path)


if __name__ == "__main__":
    main()
