from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from optest.core import BackendTarget, TestCase, Tolerance
from optest.core.comparator import ComparisonResult, TensorComparisonResult
from optest.core.results import CaseResult
from optest.plans.types import ExecutionPlan, RunSettings
from optest.registry import registry
from optest.reporting.json_reporter import JsonReporter


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
