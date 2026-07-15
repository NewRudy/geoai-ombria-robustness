from __future__ import annotations

import json
import py_compile
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str) -> None:
        checks.append((name, condition, detail))

    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "docs/CONFIRMATORY_PROTOCOL.md",
        "docs/SENSOR_STATE_V2_PROTOCOL.md",
        "docs/QUALITY_GATED_FUSION_V3_PROTOCOL.md",
        "docs/KAGGLE.md",
        "notebooks/kaggle_confirmatory_smoke.ipynb",
        "notebooks/kaggle_confirmatory_full.ipynb",
        "notebooks/kaggle_sensor_state_v2_smoke.ipynb",
        "notebooks/kaggle_sensor_state_v2_full.ipynb",
        "notebooks/kaggle_quality_gated_v3_smoke.ipynb",
        "notebooks/kaggle_quality_gated_v3_full.ipynb",
        "scripts/train_ombria_unet.py",
        "scripts/evaluate_ombria_2021_events.py",
        "scripts/summarize_confirmatory_events.py",
        "scripts/export_confirmatory_event_panels.py",
        "scripts/run_confirmatory_event_matrix.sh",
        "scripts/run_sensor_state_v2_matrix.sh",
        "scripts/run_quality_gated_v3_matrix.sh",
        "scripts/summarize_quality_gated_v3.py",
        "scripts/export_quality_gate_panels.py",
        "scripts/audit_split_near_duplicates.py",
        "scripts/export_sensor_state_v2_probabilities.py",
        "scripts/ensure_cuda_compat.py",
        "scripts/check_cuda_runtime.py",
        "scripts/write_experiment_manifest.py",
        "scripts/package_confirmatory_artifacts.py",
        "src/geoai_ombria_robustness/ombria.py",
        "src/geoai_ombria_robustness/models.py",
    ]
    for relative in required:
        check(f"required: {relative}", (ROOT / relative).is_file(), relative)

    compile_errors: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")) + sorted(
        (ROOT / "src").glob("**/*.py")
    ):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            compile_errors.append(str(exc))
    check(
        "Python compile gate", not compile_errors, "; ".join(compile_errors) or "pass"
    )

    runner = ROOT / "scripts/run_confirmatory_event_matrix.sh"
    v2_runner = ROOT / "scripts/run_sensor_state_v2_matrix.sh"
    v3_runner = ROOT / "scripts/run_quality_gated_v3_matrix.sh"
    shell_errors = []
    for script in (runner, v2_runner, v3_runner):
        shell = subprocess.run(
            ["bash", "-n", str(script)], capture_output=True, text=True, check=False
        )
        if shell.returncode:
            shell_errors.append(f"{script.name}: {shell.stderr.strip()}")
    check("Shell syntax gate", not shell_errors, "; ".join(shell_errors) or "pass")

    notebook_expectations = {
        "notebooks/kaggle_confirmatory_smoke.ipynb": (
            '"MODE": "smoke"',
            '"EPOCHS": "2"',
            "v0.1.5-confirmatory",
        ),
        "notebooks/kaggle_confirmatory_full.ipynb": (
            '"MODE": "full"',
            '"EPOCHS": "25"',
            "v0.1.5-confirmatory",
        ),
        "notebooks/kaggle_sensor_state_v2_smoke.ipynb": (
            "'MODE': 'smoke'",
            "'EPOCHS': '2'",
            "v0.2.0-sensor-state",
        ),
        "notebooks/kaggle_sensor_state_v2_full.ipynb": (
            "'MODE': 'full'",
            "'EPOCHS': '25'",
            "v0.2.0-sensor-state",
        ),
        "notebooks/kaggle_quality_gated_v3_smoke.ipynb": (
            "'MODE': 'smoke'",
            "v0.3.1-quality-gated",
            "run_quality_gated_v3_matrix.sh",
            "decision_gate.json",
        ),
        "notebooks/kaggle_quality_gated_v3_full.ipynb": (
            "'MODE': 'full'",
            "v0.3.1-quality-gated",
            "run_quality_gated_v3_matrix.sh",
            "decision_gate.json",
        ),
    }
    for relative, expected in notebook_expectations.items():
        path = ROOT / relative
        try:
            notebook = json.loads(path.read_text())
            source = "".join(
                "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
            )
            valid = notebook.get("nbformat") == 4 and all(
                token in source for token in expected
            )
            valid = valid and "ensure_cuda_compat.py" in source
            valid = valid and "check_cuda_runtime.py" in source
            valid = valid and "artifact_manifest.json" in source
            valid = valid and "checkpoint_manifest.json" in source
            valid = valid and "archive.testzip()" in source
            valid = valid and "hashlib.sha256(archive.read" in source
            clone_source = "".join(notebook["cells"][1].get("source", []))
            chdir_index = clone_source.find("os.chdir(")
            remove_index = clone_source.find("shutil.rmtree(project)")
            rerun_safe = 0 <= chdir_index < remove_index
            valid = valid and rerun_safe
            detail = f"cells={len(notebook.get('cells', []))}, rerun_safe={rerun_safe}"
        except (OSError, json.JSONDecodeError) as exc:
            valid = False
            detail = str(exc)
        check(f"Notebook contract: {relative}", valid, detail)

    trainer = (ROOT / "scripts/train_ombria_unet.py").read_text()
    epoch_loop = trainer.split("for epoch in range", 1)[-1]
    check(
        "Validation-only checkpoint loop",
        "test_loader" not in epoch_loop and '"test_iou"' not in epoch_loop,
        "No test evaluation inside the epoch loop.",
    )
    check(
        "Independent seeds",
        "--split-seed" in trainer and "--eval-perturb-seed" in trainer,
        "Split and evaluation perturbations are independent of model seed.",
    )
    check(
        "Global metrics",
        'reduction="sum"' in trainer and "tp +=" in trainer,
        "Confusion counts are pooled across batches.",
    )
    check(
        "Matched quality-map schedule",
        '"quality_matched_light": [0.55, 0.10, 0.07, 0.07, 0.07, 0.07, 0.07]'
        in trainer,
        "The same degradation schedule can run with and without quality maps.",
    )

    evaluator = (ROOT / "scripts/evaluate_ombria_2021_events.py").read_text()
    check(
        "Locked events and per-chip export",
        all(event in evaluator for event in ("ALBANIA", "FRANCE", "GUYANA", "TIMOR"))
        and "per_chip_metrics.csv" in evaluator
        and '"event": "ALL"' in evaluator,
        "Four event folders, event/global metrics, and chip rows are required.",
    )
    check(
        "Checkpoint identity at evaluation time",
        '"checkpoint_sha256"' in evaluator and '"checkpoint_bytes"' in evaluator,
        "Each evaluation config records the exact checkpoint hash and size.",
    )

    runner_text = runner.read_text()
    check(
        "Pinned OMBRIA revision",
        "38a490355f76da8ce27ed051138f03f3492a6e46" in runner_text,
        "The upstream dataset checkout is immutable.",
    )

    packager = (ROOT / "scripts/package_confirmatory_artifacts.py").read_text()
    summarizer = (ROOT / "scripts/summarize_confirmatory_events.py").read_text()
    panel_exporter = (ROOT / "scripts/export_confirmatory_event_panels.py").read_text()
    check(
        "Paper-grade evidence package",
        all(
            token in packager
            for token in (
                '"runs/*/metrics.csv"',
                '"runtime_manifest.json"',
                '"experiment_manifest.json"',
                '"environment_freeze.txt"',
                '"run.log"',
                '"sha256"',
                '"checkpoint_manifest.json"',
                '"decision_gate.json"',
            )
        )
        and "4.302652729911275" in summarizer
        and "2.7764451051977987" in summarizer
        and "1.96 * stdev" not in summarizer
        and "not estimable from one seed" in summarizer
        and "RUN_DIR_TEMPLATES" in packager
        and "Checkpoint-to-evaluation traceability gate failed" in packager
        and "--control-checkpoint" in panel_exporter,
        "Training trajectories, Student-t intervals, Smoke NA handling, provenance, file hashes, checkpoint-to-evaluation hashes, and the matched control are retained.",
    )

    v2_runner_text = v2_runner.read_text()
    ombria_module = (ROOT / "src/geoai_ombria_robustness/ombria.py").read_text()
    check(
        "v0.2 controls and independent streams",
        all(
            token in v2_runner_text
            for token in (
                "7 13 21 29 37",
                "mislocalized_quality",
                "s2_reference",
                "best_${checkpoint_policy}.pt",
                "--include-checkpoints",
            )
        )
        and "stable_stream_seed" in trainer
        and "--loader-seed" in trainer
        and "--corruption-seed" in trainer
        and "best_robust.pt" in trainer
        and "degrade_s2_pair_with_quality" in ombria_module
        and "after.sum" not in ombria_module,
        "Five repeats, seven routes, two checkpoint policies, exact applied quality masks, and independent random streams are explicit.",
    )

    v3_runner_text = v3_runner.read_text()
    model_module = (ROOT / "src/geoai_ombria_robustness/models.py").read_text()
    v3_summary = (ROOT / "scripts/summarize_quality_gated_v3.py").read_text()
    check(
        "v0.3 quality-gated method contract",
        all(
            token in v3_runner_text
            for token in (
                "7 13 21 29 37",
                "quality_concat",
                "quality_gated",
                "gated_misaligned",
                "s1_reference",
                "s2_reference",
                "summarize_quality_gated_v3.py",
                "--include-checkpoints",
            )
        )
        and "s2_before = x[:, 0:3] * quality_before" in model_module
        and "return_gate_maps" in model_module
        and "return availability * learned" in model_module
        and "architecture_partial" in v3_summary
        and "localization_partial" in v3_summary
        and "information_partial" in v3_summary
        and "pipeline_only" in v3_summary,
        "QGSF sanitization, hard gates, matched controls, five seeds, fixed contrasts, and Smoke exclusion are explicit.",
    )

    cuda_compat = (ROOT / "scripts/ensure_cuda_compat.py").read_text()
    cuda_gate = (ROOT / "scripts/check_cuda_runtime.py").read_text()
    check(
        "P100 CUDA compatibility guard",
        all(
            token in cuda_compat
            for token in (
                "torch==2.7.1",
                "torchvision==0.22.1",
                "torchaudio==2.7.1",
                "https://download.pytorch.org/whl/cu126",
                "required_arch",
            )
        )
        and '"--no-deps"' not in cuda_compat
        and "nvidia-cusparselt-cu12" in cuda_compat
        and "torch.nn.Conv2d" in cuda_gate
        and "torch.cuda.synchronize" in cuda_gate,
        "The complete CUDA 12.6 stack is installed before a real CUDA Conv2d gate.",
    )

    tracked = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    forbidden_tracked = [
        path
        for path in tracked
        if path.startswith("external/")
        or path.startswith("results/runs/")
        or path.endswith((".pt", ".pth", ".ckpt", ".zip"))
    ]
    check(
        "No data/checkpoint/archive payload",
        not forbidden_tracked,
        str(forbidden_tracked),
    )

    secret_pattern = re.compile(
        r"(ghp_[A-Za-z0-9]+|github_pat_[A-Za-z0-9_]+|AKIA[0-9A-Z]{16}|BEGIN (?:RSA|OPENSSH|EC) PRIVATE KEY|/"
        + "Users/)",
        re.IGNORECASE,
    )
    secret_hits: list[str] = []
    for relative in tracked:
        path = ROOT / relative
        if not path.is_file() or path.suffix.lower() in {".png", ".tif", ".tiff"}:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if secret_pattern.search(text):
            secret_hits.append(relative)
    check("No credential/local-path patterns", not secret_hits, str(secret_hits))

    failed = [item for item in checks if not item[1]]
    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'BLOCKED'} | {name} | {detail}")
    if failed:
        raise SystemExit(1)
    print(f"repository audit: pass ({len(checks)} checks)")


if __name__ == "__main__":
    main()
