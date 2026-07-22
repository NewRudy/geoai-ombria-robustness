from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/NewRudy/geoai-ombria-robustness.git"
SOURCE_COMMIT = "8b5a4f9ed7d0393a3b9259451f7e7dd3089f5d64"
OFFICIAL_COMMIT = "4371df08e6ca3b9d71c0385ad57b589830469a0c"
OFFICIAL_SOURCE_SHA256 = (
    "daf00d0533ca7865b4bd7b47404f1c0fa42e4a0bdc70706dee45bedcc1420f25"
)
SMOKE_ARTIFACT_SHA256 = (
    "eedaf8027e5720ff1ee72f39bc98f12e56a82928fb13a988f2bfe96075c1b0e9"
)
RELEASED_SEEDS = (7, 13)
FULL_SHARD_AUTHORIZATIONS = {
    13: {
        "previous_seed": 7,
        "artifact_sha256": (
            "db64d42d53615301cb4818ec960f9a50cbb08a299ae28cf5f6668074215c36f7"
        ),
    }
}


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


def output_path(seed: int) -> Path:
    return (
        ROOT
        / "notebooks"
        / f"kaggle_quality_uncertainty_smagnet_full_seed{seed}.ipynb"
    )


def build(seed: int = 7) -> dict[str, object]:
    if seed not in RELEASED_SEEDS:
        raise ValueError(f"Released SMAGNet Full seeds are {RELEASED_SEEDS}")
    previous = FULL_SHARD_AUTHORIZATIONS.get(seed)
    prior_gate = ""
    if previous is not None:
        prior_gate = f"""
PRIOR_FULL_AUDIT = {{
    'status': 'pass',
    'seed': {previous['previous_seed']},
    'artifact_sha256': {previous['artifact_sha256']!r},
    'shard_accepted': True,
    'next_seed_authorized': True,
    'shard_scores_publishable': False,
}}
assert PRIOR_FULL_AUDIT['status'] == 'pass'
assert PRIOR_FULL_AUDIT['seed'] == {previous['previous_seed']}
assert PRIOR_FULL_AUDIT['shard_accepted'] is True
assert PRIOR_FULL_AUDIT['next_seed_authorized'] is True
assert PRIOR_FULL_AUDIT['shard_scores_publishable'] is False
print(
    'independently audited prior Full SHA-256:',
    PRIOR_FULL_AUDIT['artifact_sha256'],
)
"""
    setup = f"""from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys

WORKING = Path('/kaggle/working')
project = WORKING / 'geoai-ombria-robustness-smagnet-full-seed{seed}-8b5a4f9'
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
).returncode == 0, 'Tracked source edits make Full provenance ambiguous'
assert subprocess.run(
    ['git', '-C', str(project), 'diff', '--cached', '--quiet']
).returncode == 0, 'Staged source edits make Full provenance ambiguous'
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

SMOKE_AUDIT = {{
    'status': 'pass',
    'artifact_sha256': {SMOKE_ARTIFACT_SHA256!r},
    'full_authorized': True,
    'smoke_scores_publishable': False,
}}
assert SMOKE_AUDIT['status'] == 'pass'
assert SMOKE_AUDIT['full_authorized'] is True
assert SMOKE_AUDIT['smoke_scores_publishable'] is False
{prior_gate}
print('frozen experiment commit:', commit)
print('independently audited Smoke SHA-256:', SMOKE_AUDIT['artifact_sha256'])
print('Full seed released:', {seed})
print(
    'resuming existing result directory:',
    (project / 'results' / 'quality_uncertainty_smagnet_full_seed{seed}').exists(),
)
"""
    dependencies = """subprocess.run(
    [sys.executable, 'scripts/ensure_cuda_compat.py'],
    check=True,
)
subprocess.run([sys.executable, 'scripts/check_cuda_runtime.py'], check=True)
subprocess.run(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        'rasterio>=1.4',
        'scikit-learn>=1.3',
        'tensorboard',
        'pandas',
        'tqdm',
        'matplotlib',
        'timm>=0.9',
        'huggingface-hub>=0.24',
        'safetensors>=0.3.1',
        'pillow>=8',
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        '-m',
        'pip',
        'install',
        '-q',
        '--no-deps',
        'segmentation-models-pytorch==0.5.0',
    ],
    check=True,
)
subprocess.run([sys.executable, 'scripts/check_cuda_runtime.py'], check=True)
subprocess.run(
    [
        sys.executable,
        '-c',
        (
            "import segmentation_models_pytorch as smp; "
            "assert smp.__version__ == '0.5.0', smp.__version__; "
            "print('segmentation-models-pytorch:', smp.__version__)"
        ),
    ],
    check=True,
)
test_env = os.environ.copy()
test_env['PYTHONPATH'] = str(project / 'src')
subprocess.run(
    [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
    check=True,
    env=test_env,
)
"""
    run = f"""from datetime import datetime, timezone

result_root = project / 'results' / 'quality_uncertainty_smagnet_full_seed{seed}'
data_root = project / 'external' / 'Sen1Floods11_quality_uncertainty'
smagnet_root = project / 'external' / 'SMAGNet_4371df0'
source_manifest = project / 'manifests' / 'sen1floods11_scl_manifest.json'
official_manifest = result_root / 'official_source_manifest.json'
artifact = project / 'results' / 'quality_map_uncertainty_smagnet_full_seed{seed}_artifacts.zip'
result_root.mkdir(parents=True, exist_ok=True)
log_path = result_root / 'run.log'
env = os.environ.copy()
env['PYTHONPATH'] = str(project / 'src')
env['PYTHONUNBUFFERED'] = '1'

def run_logged(command):
    print('$', ' '.join(map(str, command)), flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            cwd=project,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

with log_path.open('a', encoding='utf-8') as log:
    log.write(
        '=== official SMAGNet quality-uncertainty Full seed {seed} '
        + datetime.now(timezone.utc).isoformat()
        + ' ===\\n'
    )

try:
    run_logged(
        [
            sys.executable,
            'scripts/check_cuda_runtime.py',
            '--json-out',
            result_root / 'runtime_manifest.json',
        ]
    )
    with (result_root / 'environment_freeze.txt').open(
        'w', encoding='utf-8'
    ) as freeze:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            cwd=project,
            env=env,
            stdout=freeze,
            check=True,
        )
    run_logged(
        [
            sys.executable,
            'scripts/fetch_official_smagnet.py',
            '--checkout',
            smagnet_root,
            '--manifest-out',
            official_manifest,
        ]
    )
    run_logged(
        [
            sys.executable,
            'scripts/run_sen1floods11_smagnet.py',
            '--mode',
            'full',
            '--seed',
            '{seed}',
            '--source-manifest',
            source_manifest,
            '--data-root',
            data_root,
            '--smagnet-checkout',
            smagnet_root,
            '--official-source-manifest',
            official_manifest,
            '--result-root',
            result_root,
            '--workers',
            '4',
            '--micro-batch-size',
            '4',
            '--gradient-accumulation',
            '4',
            '--amp',
        ]
    )
    run_logged(
        [
            sys.executable,
            'scripts/package_quality_uncertainty_artifacts.py',
            '--root',
            result_root,
            '--out',
            artifact,
        ]
    )
except subprocess.CalledProcessError:
    if log_path.exists():
        print('\\n===== CURRENT SMAGNET FULL LOG TAIL =====')
        print('\\n'.join(log_path.read_text(errors='replace').splitlines()[-300:]))
    raise
"""
    verify = f"""import csv
import io
from zipfile import ZipFile
from IPython.display import FileLink, display

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
    assert root == 'quality_uncertainty_smagnet_full_seed{seed}'
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
    configuration = document('/runs/smagnet_official_seed{seed}/config.json')
    normalization = document('/runs/smagnet_official_seed{seed}/normalization.json')
    checkpoint_manifest = document(
        '/runs/smagnet_official_seed{seed}/checkpoint_manifest.json'
    )
    source = gate['official_source']
    fallback = gate['fallback_boundary']

    assert runtime['repository_commit'] == {SOURCE_COMMIT!r}
    assert runtime['repository_dirty_tracked'] is False
    assert runtime['cuda_conv2d_gate'] == 'pass'
    assert gate['status'] == 'pass'
    assert gate['mode'] == 'full'
    assert gate['pipeline_only'] is False
    assert gate['model_seed'] == {seed}
    assert gate['source_commit'] == {SOURCE_COMMIT!r}
    assert gate['scientific_interpretation_allowed'] is False
    assert source['commit'] == {OFFICIAL_COMMIT!r}
    assert source['source_sha256'] == {OFFICIAL_SOURCE_SHA256!r}
    assert gate['training']['epochs'] == 200
    assert gate['training']['effective_batch_size'] == 16
    assert gate['training']['parameter_count'] == 56035958
    assert gate['condition_count'] == 54
    assert gate['repetitions'] == 3
    assert gate['summary_rows'] == {{'bolivia': 162, 'test': 162}}
    assert gate['per_chip_rows'] == {{'bolivia': 2430, 'test': 14580}}
    assert gate['finite_metrics'] is True
    assert fallback['status'] == 'pass'
    assert fallback['maximum_fused_sar_logit_difference'] == 0.0
    assert fallback['maximum_masked_gate'] == 0.0
    assert plan['planned_full_seeds'] == [7, 13, 21, 29, 37]
    assert plan['seed'] == {seed}
    assert plan['pipeline_only'] is False
    assert plan['condition_count'] == 54
    assert plan['repetitions'] == 3
    assert configuration['train_count'] == 252
    assert configuration['validation_count'] == 89
    assert configuration['validation_patches'] == 356
    assert configuration['epochs'] == 200
    assert configuration['seed'] == {seed}
    assert configuration['normalization'] == normalization
    assert normalization['source'] == 'frozen training records only'
    assert normalization['optical_order'] == [
        'B4_red', 'B3_green', 'B2_blue', 'B8_nir'
    ]

    split_counts = {{}}
    for record in selected['records']:
        split = str(record['split'])
        split_counts[split] = split_counts.get(split, 0) + 1
    assert split_counts == {{
        'bolivia': 15, 'test': 90, 'train': 252, 'validation': 89
    }}

    checkpoint_name = next(
        name for name in names
        if name.endswith('/runs/smagnet_official_seed{seed}/best_validation_loss.pt')
    )
    assert archive_sha256(archive, checkpoint_name) == checkpoint_manifest[
        'best_checkpoint_sha256'
    ]

    for split, expected_summary, expected_chips in (
        ('test', 162, 14580),
        ('bolivia', 162, 2430),
    ):
        summary_name = next(
            name for name in names
            if name.endswith(f'/evaluations/seed{seed}/{{split}}/summary_metrics.csv')
        )
        chip_name = next(
            name for name in names
            if name.endswith(f'/evaluations/seed{seed}/{{split}}/per_chip_metrics.csv')
        )
        rows = list(csv.DictReader(io.StringIO(archive.read(summary_name).decode())))
        chips = list(csv.DictReader(io.StringIO(archive.read(chip_name).decode())))
        assert len(rows) == expected_summary
        assert len(chips) == expected_chips
        assert all(row['route'] == 'smagnet_official' for row in rows)
        assert all(
            math.isfinite(float(row[metric]))
            for row in rows
            for metric in ('iou', 'f1', 'precision', 'recall', 'accuracy')
        )
        by_condition_chip_rep = {{
            (row['condition_id'], row['chip_id'], row['repetition']): row
            for row in chips
        }}
        for condition_id in (
            'translate_east_5pct',
            'translate_east_10pct',
            'translate_west_5pct',
            'translate_west_10pct',
            'translate_north_5pct',
            'translate_north_10pct',
            'translate_south_5pct',
            'translate_south_10pct',
            'dilate_unavailable_r4',
            'dilate_unavailable_r8',
            'dilate_unavailable_r16',
            'erode_unavailable_r4',
            'erode_unavailable_r8',
            'erode_unavailable_r16',
        ):
            matched_id = 'matched_random__' + condition_id
            structured_rows = [
                row for key, row in by_condition_chip_rep.items()
                if key[0] == condition_id
            ]
            assert structured_rows
            for structured in structured_rows:
                matched = by_condition_chip_rep[(
                    matched_id,
                    structured['chip_id'],
                    structured['repetition'],
                )]
                for field in (
                    'quality_false_available_rate',
                    'quality_false_unavailable_rate',
                    'valid_quality_false_available_rate',
                    'valid_quality_false_unavailable_rate',
                ):
                    assert float(structured[field]) == float(matched[field])

print('artifact GB:', round(artifact.stat().st_size / 1024**3, 3))
print('manifest files verified:', len(manifest['files']))
print('official source:', source['commit'])
print('SMAGNet trainable parameters:', gate['training']['parameter_count'])
print('Full seed complete:', gate['model_seed'])
print('Scientific interpretation allowed:', gate['scientific_interpretation_allowed'])
display(FileLink(str(artifact)))
"""
    return {
        "cells": [
            markdown(
                f"# Official SMAGNet quality-map uncertainty Full — seed {seed}\n\n"
                "This immutable notebook is released only after every preceding "
                "execution gate passed an independent local integrity and "
                "numerical audit. It trains the byte-verified "
                "official SMAGNet architecture for 200 epochs on all 252/89 "
                "training/validation records, then evaluates 54 frozen conditions "
                "with three repetitions on the 90-chip test split and independent "
                "15-chip Bolivia audit.\n\n"
                "Start from a fresh Kaggle GPU Session with Internet enabled. The "
                "job can resume inside the same live Session, but do not reuse a "
                "Session that previously failed during CUDA compatibility setup.\n\n"
                "**Do not change the source commit, official commit, seed, epochs, "
                "samples, conditions, or repetitions. Return the ZIP even though "
                "one Full shard is not scientifically interpretable until all five "
                "seeds pass independent local audit.**\n"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one authorized official-SMAGNet Full Kaggle notebook."
    )
    parser.add_argument("--seed", type=int, choices=RELEASED_SEEDS, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = output_path(args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build(args.seed), indent=1) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
