from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from optest.plan import PlanOptions, load_plan, run_plan

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "matmul_cpp"
PLAN_PATH = EXAMPLE_DIR / "plan.yaml"
RUNNER_PATH = EXAMPLE_DIR / "operator" / "build" / "matmul_runner"


@pytest.fixture(scope="session")
def matmul_runner() -> Path:
    """Build the C++ runner once for all matmul example tests."""

    if not shutil.which("cmake"):
        pytest.skip("cmake is required to build matmul example")
    subprocess.run(["bash", "build.sh"], cwd=EXAMPLE_DIR / "operator", check=True)
    if not RUNNER_PATH.exists():
        pytest.skip("matmul_runner binary missing after build")
    return RUNNER_PATH


def _write_plan(tmp_path: Path, overrides: dict) -> Path:
    data = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    data.update(overrides)
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return plan_path


def _override_backend_for_tmp(tmp_path: Path, data: dict, runner: Path) -> None:
    # Use absolute paths so the plan can live in a temp dir.
    data["inputs"] = [str(tmp_path / "in0.bin"), str(tmp_path / "in1.bin")]
    data["outputs"] = [str(tmp_path / "out0.bin")]
    backend = data["backends"][0]
    backend["workdir"] = str(EXAMPLE_DIR)
    backend["command"] = [
        str(runner),
        "--input0",
        "{input0}",
        "--input1",
        "{input1}",
        "--output0",
        "{output0}",
        "--dtype",
        "{dtype}",
        "--shapes",
        "{shapes}",
    ]


def test_matmul_example_reports_shape_mismatch(matmul_runner: Path, tmp_path: Path) -> None:
    data = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    _override_backend_for_tmp(tmp_path, data, matmul_runner)
    # Introduce an incorrect output shape to force a reshape failure.
    data["cases"] = [
        {
            "name": "bad_shape",
            "dtypes": ["float32", "float32"],
            "shapes": [{"inputs": [[2, 2], [2, 2]], "outputs": [[1, 1]]}],
        }
    ]
    plan_path = tmp_path / "plan_bad_shape.yaml"
    plan_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    plan = load_plan(str(plan_path))
    exit_code = run_plan(plan, PlanOptions(backend="cuda", chip="local"), use_color=False)
    assert exit_code == 1


def test_matmul_example_reports_unsupported_dtype(matmul_runner: Path, tmp_path: Path) -> None:
    data = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    _override_backend_for_tmp(tmp_path, data, matmul_runner)
    data["cases"] = [
        {
            "name": "bad_dtype",
            "dtypes": ["float16", "float16"],
            "shapes": [{"inputs": [[1, 1], [1, 1]], "outputs": [[1, 1]]}],
        }
    ]
    plan_path = tmp_path / "plan_bad_dtype.yaml"
    plan_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    plan = load_plan(str(plan_path))
    exit_code = run_plan(plan, PlanOptions(backend="cuda", chip="local"), use_color=False)
    assert exit_code == 1
