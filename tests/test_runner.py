from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from optest.backends import BackendDriver, backend_manager
from optest.core import BackendTarget, Tolerance
from optest.core.runner import TestRunner
from optest.plans.loader import build_execution_plan
from optest.plans.types import RunOptions


def _build_simple_case() -> RunOptions:
    return RunOptions(
        ops=("relu",),
        dtype_override=("float32",),
        shape_overrides={"input0": (1, 1, 2, 2)},
        backend="gpu",
        seed=0,
    )


def test_runner_executes_case_successfully() -> None:
    plan = build_execution_plan(_build_simple_case())
    runner = TestRunner(seed=0)
    results = runner.run(plan.cases)
    assert len(results) == 1
    assert results[0].status == "passed"
    assert results[0].comparison is not None
    tensor_metrics = results[0].comparison.tensors[0]
    assert tensor_metrics.passed
    assert tensor_metrics.mismatched == 0


class ScaleBackend(BackendDriver):
    """Custom backend used to verify extension pattern."""

    name = "scale"
    kind = "npu"
    chips = ("sim",)

    def run(self, case, inputs: Sequence[np.ndarray]):
        # emulate a backend that scales outputs by 2 then uses reference tolerance
        return tuple(tensor * 2 for tensor in inputs)


# Register extension backend once (safe across tests) if not already present
try:
    backend_manager.register(ScaleBackend())
except ValueError:
    pass


def test_runner_with_custom_backend_detects_mismatch() -> None:
    run_options = RunOptions(
        ops=("relu",),
        dtype_override=("float32",),
        shape_overrides={"input0": (1, 1, 2, 2)},
        backend="npu",
        chip="sim",
    )
    plan = build_execution_plan(run_options)
    runner = TestRunner(seed=1)
    result = runner.run(plan.cases)[0]
    assert result.status == "failed"
    assert result.comparison is not None
    assert result.comparison.passed is False
    tensor_metrics = result.comparison.tensors[0]
    assert tensor_metrics.mismatched == tensor_metrics.total


def double_reference(inputs, attrs):  # type: ignore[override]
    (x,) = inputs
    return (x * 2,)


class FixedGenerator:
    """Deterministic generator for golden dump tests."""

    def generate(self, case, rng):
        data = np.array([-1.0, 2.0], dtype=np.float32)
        return [data], None


def test_runner_with_ascend_command_backend() -> None:
    workdir = Path("tests/artifacts/ascend_backend")
    script = workdir / "produce.py"
    script_path = script.resolve()
    output_file = workdir / "output/output_z.bin"
    if output_file.exists():
        output_file.unlink()
    script.chmod(script.stat().st_mode | 0o111)
    run_options = RunOptions(
        ops=("relu",),
        dtype_override=("float32",),
        shape_overrides={"input0": (2,), "output0": (2,)},
        backend="npu",
        chip="ascend",
        attribute_overrides={
            "backend_config": {
                "ascend": {
                    "workdir": str(workdir),
                    "command": {"binary": str(script_path)},
                    "inputs": [
                        {"tensor": "input0", "path": "input/input_x.bin", "dtype": "float32"}
                    ],
                    "outputs": [
                        {"tensor": "output0", "path": "output/output_z.bin", "dtype": "float32"}
                    ],
                }
            }
        },
        reference_override="tests.test_runner:double_reference",
    )
    plan = build_execution_plan(run_options)
    runner = TestRunner(seed=2)
    result = runner.run(plan.cases)[0]
    assert result.status == "passed", result.error


def test_runner_writes_missing_golden_outputs(tmp_path: Path) -> None:
    workdir = tmp_path
    run_script = workdir / "run.py"
    run_script.write_text(
        "import numpy as np, pathlib;"
        "data=np.fromfile('input/input0.bin',dtype=np.float32);"
        "pathlib.Path('output').mkdir(parents=True, exist_ok=True);"
        "np.maximum(data,0).astype(np.float32).tofile('output/output0.bin')",
        encoding="utf-8",
    )
    run_script.chmod(run_script.stat().st_mode | 0o111)
    golden_file = workdir / "golden" / "output0.bin"
    run_options = RunOptions(
        ops=("relu",),
        dtype_override=("float32",),
        shape_overrides={"input0": (2,), "output0": (2,)},
        backend="npu",
        chip="ascend",
        attribute_overrides={
            "backend_config": {
                "ascend": {
                    "workdir": str(workdir),
                    "command": ["python3", str(run_script)],
                    "golden": {"output0": "golden/output0.bin"},
                }
            }
        },
        generator_override="tests.test_runner:FixedGenerator",
    )
    plan = build_execution_plan(run_options)
    runner = TestRunner(seed=5)
    result = runner.run(plan.cases)[0]
    assert result.status == "passed", result.error
    assert golden_file.exists()
    saved = np.fromfile(golden_file, dtype=np.float32).tolist()
    assert saved == [0.0, 2.0]


def test_runner_reports_ascend_command_failure(tmp_path: Path) -> None:
    workdir = tmp_path / "add_custom"
    shutil.copytree(Path("tmp/add_custom"), workdir)
    run_options = RunOptions(
        ops=("elementwise_add",),
        dtype_override=("float16", "float16"),
        shape_overrides={"input0": (8, 2048), "input1": (8, 2048), "output0": (8, 2048)},
        backend="npu",
        chip="ascend910b",
        attribute_overrides={
            "backend_config": {
                "ascend": {
                    "workdir": str(workdir),
                    "command": ["bash", "simulate_failure.sh"],
                    "inputs": [
                        {"tensor": "input0", "path": "input/input_x.bin", "dtype": "float16"},
                        {"tensor": "input1", "path": "input/input_y.bin", "dtype": "float16"},
                    ],
                    "outputs": [
                        {"tensor": "output0", "path": "output/output_z.bin", "dtype": "float16"}
                    ],
                }
            }
        },
    )
    plan = build_execution_plan(run_options)
    runner = TestRunner(seed=3)
    result = runner.run(plan.cases)[0]
    assert result.status == "error"
    assert result.error is not None
    assert "simulate_failure.sh intentionally stops" in result.error
