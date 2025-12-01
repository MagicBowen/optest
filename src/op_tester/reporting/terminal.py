"""Terminal reporter rendering progress and summaries."""
from __future__ import annotations

import time
from typing import Sequence

import click

from op_tester.core.results import CaseResult
from op_tester.plans.types import ExecutionPlan

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

    def on_start(self, plan: ExecutionPlan) -> None:
        self._plan = plan
        self._start_time = time.perf_counter()

    def on_case_result(self, result: CaseResult, index: int, total: int) -> None:
        identifier = result.case.identifier()
        ms = result.duration_s * 1000
        status_text = self._styled(result.status.upper())
        click.echo(f"[{index}/{total}] {identifier} -> {status_text} ({ms:.2f} ms)")
        if result.error:
            click.echo(f"    error: {result.error}")
        elif result.comparison and not result.comparison.passed:
            for tensor_index, tensor_result in enumerate(result.comparison.tensors):
                if tensor_result.passed:
                    continue
                click.echo(
                    "    tensor %d mismatched: count=%d max_abs=%.3e max_rel=%.3e"
                    % (
                        tensor_index,
                        tensor_result.mismatched,
                        tensor_result.max_abs_error,
                        tensor_result.max_rel_error,
                    )
                )

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

    def _styled(self, text: str, *, force_color: str | None = None) -> str:
        if not self._use_color:
            return text
        color = force_color or STATUS_COLORS.get(text.lower(), None)
        if color:
            return click.style(text, fg=color)
        return text
