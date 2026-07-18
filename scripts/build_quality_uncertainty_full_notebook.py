from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "kaggle_quality_uncertainty_full_seed7.ipynb"
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"
SOURCE_COMMIT = "abf6a792ba158ca2302850f3234097e06f9a1d8e"
SEED = 7


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
import hashlib, json, os, subprocess, sys

WORKING = Path('/kaggle/working')
legacy_project = WORKING / 'geoai-ombria-robustness-full-seed7-d25bc67'
fresh_project = WORKING / 'geoai-ombria-robustness-full-seed7-abf6a79'
project = legacy_project if legacy_project.exists() else fresh_project
os.chdir(WORKING)
if not project.exists():
    project.mkdir(parents=True)
    subprocess.run(['git', '-C', str(project), 'init', '-q'], check=True)
    subprocess.run(
        ['git', '-C', str(project), 'remote', 'add', 'origin', {REPOSITORY!r}],
        check=True,
    )
assert (project / '.git').is_dir(), project
assert subprocess.run(
    ['git', '-C', str(project), 'diff', '--quiet']
).returncode == 0, 'Tracked source edits would make resume provenance ambiguous'
assert subprocess.run(
    ['git', '-C', str(project), 'diff', '--cached', '--quiet']
).returncode == 0, 'Staged source edits would make resume provenance ambiguous'
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

smoke = json.loads(
    (project / 'manifests/quality_uncertainty_smoke_authorization.json').read_text()
)
assert smoke['status'] == 'pass'
assert smoke['artifact']['sha256'] == '32ebcd1d8bfa5cadcf9b007985548ae7d03b9ecb1015ea41b92e93b12b47e67e'
assert smoke['audit']['full_authorized'] is True
assert smoke['scientific_interpretation_allowed'] is False

equivalence = json.loads(
    (project / 'manifests/quality_uncertainty_core_equivalence.json').read_text()
)
assert equivalence['status'] == 'pass-with-authorized-evaluation-hotfix'
for relative, expected in equivalence['byte_identical_files'].items():
    actual = hashlib.sha256((project / relative).read_bytes()).hexdigest()
    assert actual == expected, (relative, actual, expected)
for relative, exception in equivalence['authorized_exceptions'].items():
    actual = hashlib.sha256((project / relative).read_bytes()).hexdigest()
    assert actual == exception['full_sha256'], (relative, actual, exception)
    assert exception['regression_status'] == 'pass'
    assert exception['training_reuse_allowed'] is True
print('Full source commit:', commit)
print('Smoke authorization: pass')
print('Byte-identical core files:', len(equivalence['byte_identical_files']))
print('Authorized pre-score evaluator hotfixes:', len(equivalence['authorized_exceptions']))
result_root = project / 'results' / 'quality_uncertainty_full_seed7'
print('Resuming existing seed-7 results:', result_root.exists())
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
    run = f"""env = os.environ.copy()
env['PYTHONPATH'] = str(project / 'src')
try:
    subprocess.run(
        ['bash', 'scripts/run_quality_uncertainty_full_shard.sh', {str(SEED)!r}],
        check=True,
        env=env,
    )
except subprocess.CalledProcessError:
    log = project / 'results' / 'quality_uncertainty_full_seed7' / 'run.log'
    if log.exists():
        print('\\n===== CURRENT RUN LOG TAIL =====')
        print('\\n'.join(log.read_text(errors='replace').splitlines()[-250:]))
    raise
"""
    verify = f"""import csv, io
from zipfile import ZipFile
from IPython.display import FileLink, display

artifact = project / 'results' / 'quality_map_uncertainty_full_seed{SEED}_artifacts.zip'
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
        payload = archive.read(name)
        assert len(payload) == record['bytes']
        assert hashlib.sha256(payload).hexdigest() == record['sha256']

    def document(suffix):
        name = next(value for value in names if value.endswith(suffix))
        return json.loads(archive.read(name))

    runtime = document('/runtime_manifest.json')
    top_gate = document('/full_shard_decision_gate.json')
    ombria_gate = document('/ombria/seed7/ombria_decision_gate.json')
    external_gate = document('/sen1floods11/sen1floods11_decision_gate.json')
    preparation = document('/sen1floods11/sen1floods11_preparation_report.json')
    equivalence = document('/quality_uncertainty_core_equivalence.json')
    assert runtime['repository_commit'] == {SOURCE_COMMIT!r}
    assert runtime['repository_dirty_tracked'] is False
    assert runtime['cuda_conv2d_gate'] == 'pass'
    assert top_gate['status'] == 'pass'
    assert top_gate['active_seed'] == {SEED}
    assert top_gate['scientific_interpretation_allowed'] is False
    assert top_gate['authorized_evaluation_hotfix'] == 'pass'
    assert equivalence['status'] == 'pass-with-authorized-evaluation-hotfix'
    assert ombria_gate['status'] == 'pass'
    assert ombria_gate['raw_summary_rows'] == 301
    assert ombria_gate['response_surface_rows'] == 101
    assert external_gate['status'] == 'pass'
    assert external_gate['mode'] == 'full'
    assert external_gate['pipeline_only'] is False
    assert external_gate['active_seeds'] == [{SEED}]
    assert external_gate['planned_seeds'] == [7, 13, 21, 29, 37]
    assert external_gate['expected_training_runs'] == 8
    assert external_gate['complete_training_runs'] == 8
    assert external_gate['seed_condition_rows'] == 550
    assert external_gate['shard_complete'] is True
    assert external_gate['all_full_seeds_present'] is False
    assert preparation['status'] == 'pass'
    assert preparation['record_count'] == 446
    zero_valid_ids = {{
        str(record['chip_id'])
        for record in preparation['records']
        if int(record['valid_target_pixels']) == 0
    }}
    assert zero_valid_ids, 'The Full-only empty-target edge case was not retained'

    external_raw_rows = 0
    ombria_raw_rows = 0
    zero_valid_metric_rows = 0
    for name in names:
        if name.endswith('/summary_metrics.csv'):
            rows = list(csv.DictReader(io.StringIO(archive.read(name).decode('utf-8'))))
            if '/sen1floods11/evaluations/' in name:
                external_raw_rows += len(rows)
            elif '/ombria/seed7/evaluations/' in name:
                ombria_raw_rows += len(rows)
        elif (
            '/sen1floods11/evaluations/' in name
            and name.endswith('/per_chip_metrics.csv')
        ):
            rows = csv.DictReader(io.StringIO(archive.read(name).decode('utf-8')))
            for row in rows:
                if str(row['chip_id']) not in zero_valid_ids:
                    continue
                zero_valid_metric_rows += 1
                assert int(row['valid_target_pixels']) == 0
                assert row['has_valid_target'] == 'False'
                assert row['mean_probability'] == ''
                assert row['valid_quality_false_available_rate'] == '0.0'
                assert row['valid_quality_false_unavailable_rate'] == '0.0'
    assert external_raw_rows == 1650, external_raw_rows
    assert ombria_raw_rows == 301, ombria_raw_rows
    assert zero_valid_metric_rows > 0

print('artifact GB:', round(artifact.stat().st_size / 1024**3, 3))
print('manifest files verified:', len(manifest['files']))
print('OMBRIA raw summaries:', ombria_raw_rows)
print('Sen1Floods11 raw summaries:', external_raw_rows)
print('Zero-valid-domain per-chip rows verified:', zero_valid_metric_rows)
print('Scientific interpretation allowed:', top_gate['scientific_interpretation_allowed'])
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(
                "# Quality-map uncertainty Full — seed 7 resume\n\n"
                "This immutable notebook resumes the first Full seed shard "
                "after the pre-score empty-target evaluator correction. It runs the "
                "complete OMBRIA and 446-chip Sen1Floods11/SCL matrices for "
                "model seed 7, including all eight external routes, 25 "
                "quality-error cells, structured and valid-domain matched "
                "controls, complete optical absence, and three deterministic "
                "perturbation repetitions.\n\n"
                "It reuses completed checkpoints only because the frozen training "
                "implementations are byte-identical; the failed log remains under "
                "`prior_attempts/`. The job is resumable inside the same Kaggle "
                "session. Do not "
                "edit the seed, routes, epochs, rates, repetitions, or sample "
                "counts. Return the ZIP even though one seed is not yet "
                "scientifically interpretable.\n"
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
