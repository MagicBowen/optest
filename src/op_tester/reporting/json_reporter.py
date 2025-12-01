"""JSON reporter emitting structured execution results."""
from __future__ import annotations

import datetime as dt
import json
import pathlib
from typing import Any, Dict, Sequence

import click
from jsonschema import validate

from op_tester.core.results import CaseResult
from op_tester.plans.types import ExecutionPlan

from .base import Reporter
from .schema import JSON_SCHEMA_V1, SCHEMA_VERSION


class JsonReporter(Reporter):
    """Writes results to a JSON file validated against the schema."""

    def __init__(self, path: str) -> None:
        self._path = pathlib.Path(path)
        self._records: list[Dict[str, Any]] = []
        self._plan: ExecutionPlan | None = None
        self._start_time = 0.0

    def on_start(self, plan: ExecutionPlan) -> None:
        self._plan = plan
        self._records.clear()
        self._start_time = dt.datetime.utcnow().timestamp()

    def on_case_result(self, result: CaseResult, index: int, total: int) -> None:
        self._records.append(_case_to_dict(result))

    def on_complete(self, results: Sequence[CaseResult]) -> None:
        if self._plan is None:
            return
        total_duration = dt.datetime.utcnow().timestamp() - self._start_time
        payload = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "summary": _build_summary(self._plan, results, total_duration),
            "cases": self._records,
        }
        validate(instance=payload, schema=JSON_SCHEMA_V1)
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem protection
            raise RuntimeError(f"Failed to write JSON report to {self._path}: {exc}") from exc
        click.echo(f"JSON report written to {self._path}")


def _build_summary(plan: ExecutionPlan, results: Sequence[CaseResult], duration: float) -> Dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.status == "passed")
    failed = sum(1 for result in results if result.status == "failed")
    errors = sum(1 for result in results if result.status == "error")
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "backend": plan.settings.backend,
        "chip": plan.settings.chip,
        "seed": plan.settings.seed,
        "duration_s": duration,
    }


def _case_to_dict(result: CaseResult) -> Dict[str, Any]:
    case = result.case
    record: Dict[str, Any] = {
        "id": case.identifier(),
        "operator": case.descriptor.name,
        "status": result.status,
        "duration_ms": result.duration_s * 1000,
        "backend": case.backend.kind,
        "chip": case.backend.chip,
        "dtypes": list(case.dtype_spec),
        "shapes": {key: list(value) for key, value in case.shapes.items()},
        "tolerance": {
            "abs": case.tolerance.absolute,
            "rel": case.tolerance.relative,
        },
        "attributes": _jsonify(case.attributes),
    }
    if result.error:
        record["error"] = result.error
    if result.comparison:
        record["comparison"] = [
            {
                "tensor_index": index,
                "passed": tensor_result.passed,
                "max_abs_error": tensor_result.max_abs_error,
                "max_rel_error": tensor_result.max_rel_error,
                "mismatched": tensor_result.mismatched,
                "total": tensor_result.total,
            }
            for index, tensor_result in enumerate(result.comparison.tensors)
        ]
    return record


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
