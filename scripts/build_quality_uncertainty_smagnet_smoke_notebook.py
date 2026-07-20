from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_quality_uncertainty_smagnet_smoke.ipynb"
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"
SOURCE_COMMIT = "ccc4aea4487558d5cfa9f98fda5ab2d1f58e2797"
OFFICIAL_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
OFFICIAL_SOURCE_SHA256 = (
    "daf00d0533ca7865b4bd7b47404f1c0fa42e4a0bdc70706dee45bedcc1420f25"
)


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
import hashlib, json, os, shutil, subprocess, sys

WORKING = Path('/kaggle/working')
project = WORKING / 'geoai-ombria-robustness-smagnet-smoke-ccc4aea'
os.chdir(WORKING)
if project.exists():
    shutil.rmtree(project)
project.mkdir(parents=True)
subprocess.run(['git', '-C', str(project), 'init', '-q'], check=True)
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
assert subprocess.run(['git', 'diff', '--quiet']).returncode == 0
assert subprocess.run(['git', 'diff', '--cached', '--quiet']).returncode == 0
print('frozen experiment commit:', commit)
"""
    dependencies = """subprocess.run(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        'rasterio>=1.4',
        'segmentation-models-pytorch==0.5.0',
        'scikit-learn>=1.3',
        'tensorboard',
        'pandas',
        'tqdm',
        'matplotlib',
    ],
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
env['PYTHONPATH'] = str(project / 'src')
try:
    subprocess.run(
        ['bash', 'scripts/run_quality_uncertainty_smagnet_smoke.sh'],
        check=True,
        env=env,
    )
except subprocess.CalledProcessError:
    log = project / 'results' / 'quality_uncertainty_smagnet_smoke' / 'run.log'
    if log.exists():
        print('\\n===== CURRENT SMAGNET SMOKE LOG TAIL =====')
        print('\\n'.join(log.read_text(errors='replace').splitlines()[-300:]))
    raise
"""
    verify = f"""import csv, io
from zipfile import ZipFile
from IPython.display import FileLink, display

artifact = project / 'results' / 'quality_map_uncertainty_smagnet_smoke_artifacts.zip'

def archive_sha256(archive, name):
    digest = hashlib.sha256()
    with archive.open(name) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

with ZipFile(artifact) as archive:
    names = archive.namelist()
    assert archive.testzip() is None
    assert len(names) == len(set(names))
    assert not any(
        name.endswith(('_S1Hand.tif', '_S2Hand.tif', '_LabelHand.tif'))
        for name in names
    ), 'Raw Sen1Floods11 rasters must not be redistributed'
    manifest_name = next(
        name for name in names if name.endswith('/artifact_manifest.json')
    )
    manifest = json.loads(archive.read(manifest_name))
    root = manifest['root']
    expected_names = {{root + '/' + record['path'] for record in manifest['files']}}
    assert expected_names == set(names) - {{manifest_name}}
    for record in manifest['files']:
        name = root + '/' + record['path']
        assert archive.getinfo(name).file_size == record['bytes']
        assert archive_sha256(archive, name) == record['sha256']

    def document(suffix):
        name = next(value for value in names if value.endswith(suffix))
        return json.loads(archive.read(name))

    runtime = document('/runtime_manifest.json')
    gate = document('/published_architecture_gate.json')
    plan = document('/experiment_plan.json')
    selected = document('/sen1floods11_selected_manifest.json')
    configuration = document('/runs/smagnet_official_seed7/config.json')
    normalization = document('/runs/smagnet_official_seed7/normalization.json')
    checkpoint_manifest = document(
        '/runs/smagnet_official_seed7/checkpoint_manifest.json'
    )
    source = gate['official_source']
    fallback = gate['fallback_boundary']

    assert runtime['repository_commit'] == {SOURCE_COMMIT!r}
    assert runtime['repository_dirty_tracked'] is False
    assert runtime['cuda_conv2d_gate'] == 'pass'
    assert gate['status'] == 'pass'
    assert gate['mode'] == 'smoke'
    assert gate['pipeline_only'] is True
    assert gate['model_seed'] == 7
    assert gate['source_commit'] == {SOURCE_COMMIT!r}
    assert gate['scientific_interpretation_allowed'] is False
    assert source['commit'] == {OFFICIAL_COMMIT!r}
    assert source['source_sha256'] == {OFFICIAL_SOURCE_SHA256!r}
    assert gate['training']['epochs'] == 2
    assert gate['training']['effective_batch_size'] == 16
    assert gate['training']['parameter_count'] == 56035958
    assert gate['condition_count'] == 16
    assert gate['repetitions'] == 1
    assert gate['summary_rows'] == {{'bolivia': 16, 'test': 16}}
    assert gate['per_chip_rows'] == {{'bolivia': 64, 'test': 192}}
    assert fallback['status'] == 'pass'
    assert fallback['maximum_fused_sar_logit_difference'] == 0.0
    assert fallback['maximum_masked_gate'] == 0.0
    assert plan['planned_full_seeds'] == [7, 13, 21, 29, 37]
    assert plan['pipeline_only'] is True
    assert configuration['train_count'] == 24
    assert configuration['validation_count'] == 12
    assert configuration['validation_patches'] == 48
    assert configuration['normalization'] == normalization
    assert normalization['source'] == 'frozen training records only'
    assert normalization['optical_order'] == [
        'B4_red', 'B3_green', 'B2_blue', 'B8_nir'
    ]

    split_counts = {{}}
    for record in selected['records']:
        split = str(record['split'])
        split_counts[split] = split_counts.get(split, 0) + 1
    assert split_counts == {{'bolivia': 4, 'test': 12, 'train': 24, 'validation': 12}}

    checkpoint_name = next(
        name for name in names
        if name.endswith('/runs/smagnet_official_seed7/best_validation_loss.pt')
    )
    assert archive_sha256(archive, checkpoint_name) == checkpoint_manifest[
        'best_checkpoint_sha256'
    ]

    for split, expected_rows in (('test', 16), ('bolivia', 16)):
        summary_name = next(
            name for name in names
            if name.endswith(f'/evaluations/seed7/{{split}}/summary_metrics.csv')
        )
        rows = list(csv.DictReader(io.StringIO(archive.read(summary_name).decode())))
        assert len(rows) == expected_rows
        assert all(row['route'] == 'smagnet_official' for row in rows)

print('artifact GB:', round(artifact.stat().st_size / 1024**3, 3))
print('manifest files verified:', len(manifest['files']))
print('official source:', source['commit'])
print('SMAGNet trainable parameters:', gate['training']['parameter_count'])
print('Smoke scientific interpretation allowed:', gate['scientific_interpretation_allowed'])
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(
                "# Official SMAGNet quality-map uncertainty Smoke\n\n"
                "This frozen pipeline imports the byte-verified official SMAGNet "
                "architecture, adapts its published dual-path training protocol "
                "to the frozen Sen1Floods11 splits, and exercises the complete "
                "quality-map uncertainty evaluation path. It uses seed 7, two "
                "epochs, 24/12/12/4 records, 16 conditions, and one perturbation "
                "repetition.\n\n"
                "**This is a published-architecture execution gate. Smoke scores "
                "are prohibited from the manuscript. Do not change the source "
                "commit, official commit, seed, epochs, samples, or conditions.**\n"
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
