from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np

from optest.core import BackendTarget, TestCase, Tolerance
from optest.core.comparator import ComparisonResult, TensorComparisonResult, compare_outputs
from optest.core.results import CaseResult
from optest.plans.types import ExecutionPlan, RunSettings
from optest.registry import registry
from optest.reporting.json_reporter import JsonReporter
from optest.reporting.terminal import TerminalReporter


def test_json_reporter_writes_file(tmp_path_factory=None) -> None:
    descriptor = registry.get("relu")
    case = TestCase(
        descriptor=descriptor,
        dtype_spec=("float32",),
        shapes={"input0": (1, 1, 2, 2)},
        backend=BackendTarget(kind="gpu"),
        tolerance=Tolerance(),
    )
    comparison = ComparisonResult(
        passed=True,
        tensors=[
            TensorComparisonResult(
                passed=True,
                max_abs_error=0.0,
                max_rel_error=0.0,
                mismatched=0,
                total=4,
            )
        ],
    )
    result = CaseResult(case=case, status="passed", duration_s=0.001, comparison=comparison)
    plan = ExecutionPlan(
        cases=[case],
        settings=RunSettings(
            backend="gpu",
            chip=None,
            seed=0,
            fail_fast=False,
            report_format="json",
            report_path="tmp/report_output.json",
            color=True,
        ),
    )
    output_path = Path("tmp") / "report_output.json"
    reporter = JsonReporter(path=str(output_path))
    reporter.on_start(plan)
    reporter.on_case_result(result, index=1, total=1)
    with mock.patch("pathlib.Path.mkdir") as mock_mkdir, mock.patch(
        "pathlib.Path.write_text"
    ) as mock_write:
        reporter.on_complete([result])
        mock_mkdir.assert_called()
        mock_write.assert_called_once()
        written = mock_write.call_args.args[0]
        payload = json.loads(written)
        assert payload["summary"]["total"] == 1
        assert payload["cases"][0]["operator"] == "relu"
        assert payload["generated_at"].endswith("Z")


def test_terminal_reporter_renders_failure_details(capsys) -> None:
    descriptor = registry.get("relu")
    case = TestCase(
        descriptor=descriptor,
        dtype_spec=("float32",),
        shapes={"input0": (1, 1, 2, 2)},
        backend=BackendTarget(kind="gpu"),
        tolerance=Tolerance(),
    )
    actual = (np.array([[[[1.0, 2.5], [3.0, 4.0]]]], dtype=np.float32),)
    expected = (np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),)
    comparison = compare_outputs(case, actual, expected)
    result = CaseResult(
        case=case, status="failed", duration_s=0.002, comparison=comparison, seed=7
    )
    plan = ExecutionPlan(
        cases=[case],
        settings=RunSettings(
            backend="gpu",
            chip=None,
            seed=0,
            fail_fast=False,
            report_format="terminal",
            report_path=None,
            color=False,
        ),
    )
    reporter = TerminalReporter(use_color=False)
    reporter.on_start(plan)
    reporter.on_case_result(result, index=1, total=1)
    reporter.on_complete([result])
    output = capsys.readouterr().out
    assert "Failure details:" in output
    assert "seed=7" in output
    assert "tensor 0: mismatched 1/4" in output
    assert "idx=(0, 0, 0, 1)" in output
    assert "actual=2.5" in output
    assert "expected=2.0" in output
