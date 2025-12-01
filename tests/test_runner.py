from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from op_tester.backends import BackendDriver, backend_manager
from op_tester.core import BackendTarget, Tolerance
from op_tester.core.runner import TestRunner
from op_tester.plans.loader import build_execution_plan
from op_tester.plans.types import RunOptions


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


def test_runner_with_ascend_command_backend() -> None:
    workdir = Path("tests/artifacts/ascend_backend")
    script = workdir / "produce.py"
    output_file = workdir / "output/output_z.bin"
    if output_file.exists():
        output_file.unlink()
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
                    "command": ["python3", str(script)],
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
