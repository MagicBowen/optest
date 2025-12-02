"""Terminal reporter rendering progress and summaries."""
from __future__ import annotations

import time
from typing import Sequence

import click

from optest.core.results import CaseResult
from optest.plans.types import ExecutionPlan

from .base import Reporter


STATUS_COLORS = {
    "passed": "green",
    "failed": "red",
    "error": "yellow",
}


class TerminalReporter(Reporter):
    """Human-readable reporter that streams to stdout."""

    def __init__(self, *, use_color: bool = True) -> None:
        self._use_color = use_color
        self._start_time = 0.0
        self._plan: ExecutionPlan | None = None
        self._failures: list[tuple[int, CaseResult]] = []

    def on_start(self, plan: ExecutionPlan) -> None:
        self._plan = plan
        self._start_time = time.perf_counter()
        self._failures.clear()
        settings = plan.settings
        click.echo(
            self._styled(
                f"Starting run: {len(plan.cases)} case(s) on {settings.backend}"
                + (f":{settings.chip}" if settings.chip else "")
                + f" seed={settings.seed} fail_fast={settings.fail_fast}",
                force_color="cyan",
            )
        )

    def on_case_result(self, result: CaseResult, index: int, total: int) -> None:
        identifier = result.case.identifier()
        ms = result.duration_s * 1000
        status_text = self._styled(result.status.upper())
        click.echo(f"[{index}/{total}] {identifier} -> {status_text} ({ms:.2f} ms)")
        if not result.passed:
            self._failures.append((index, result))
            self._print_failure_details(result)

    def on_complete(self, results: Sequence[CaseResult]) -> None:
        duration = time.perf_counter() - self._start_time
        total = len(results)
        passed = sum(1 for result in results if result.status == "passed")
        failed = sum(1 for result in results if result.status == "failed")
        errors = sum(1 for result in results if result.status == "error")
        click.echo(
            self._styled(
                f"Summary: total={total} passed={passed} failed={failed} errors={errors} "
                f"duration={duration:.2f}s",
                force_color="cyan",
            )
        )
        if self._failures:
            click.echo(self._styled("Failure details:", force_color="red"))
            for index, result in self._failures:
                click.echo(f"  [{index}] {result.case.identifier()} -> {result.status}")
                self._print_failure_details(result, indent="    ")

    def _styled(self, text: str, *, force_color: str | None = None) -> str:
        if not self._use_color:
            return text
        color = force_color or STATUS_COLORS.get(text.lower(), None)
        if color:
            return click.style(text, fg=color)
        return text

    def _print_failure_details(self, result: CaseResult, *, indent: str = "    ") -> None:
        case = result.case
        shape_text = ", ".join(f"{k}={v}" for k, v in sorted(case.shapes.items()))
        seed_text = result.seed if result.seed is not None else "?"
        click.echo(
            f"{indent}backend={case.backend.label()} seed={seed_text} "
            f"dtypes={','.join(case.dtype_spec)} tol(abs={case.tolerance.absolute},rel={case.tolerance.relative})"
        )
        click.echo(f"{indent}shapes: {shape_text}")
        if result.error:
            click.echo(f"{indent}error: {result.error}")
            return
        comparison = result.comparison
        if comparison is None:
            click.echo(f"{indent}comparison data unavailable")
            return
        if comparison.message:
            click.echo(f"{indent}reason: {comparison.message}")
        for tensor_index, tensor_result in enumerate(comparison.tensors):
            if tensor_result.passed:
                continue
            percent = (
                (tensor_result.mismatched / tensor_result.total * 100)
                if tensor_result.total
                else 0.0
            )
            location = (
                f" idx={tensor_result.max_error_index}"
                if tensor_result.max_error_index is not None
                else ""
            )
            click.echo(
                f"{indent}tensor {tensor_index}: mismatched "
                f"{tensor_result.mismatched}/{tensor_result.total} ({percent:.2f}%) "
                f"max_abs={tensor_result.max_abs_error:.3e} "
                f"max_rel={tensor_result.max_rel_error:.3e}{location}"
            )
            if tensor_result.detail:
                click.echo(f"{indent}  detail: {tensor_result.detail}")
            if tensor_result.actual_value is not None and tensor_result.max_error_index is not None:
                click.echo(
                    f"{indent}  actual={tensor_result.actual_value} "
                    f"expected={tensor_result.expected_value} "
                    f"at {tensor_result.max_error_index}"
                )
