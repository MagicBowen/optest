"""Reporter interface definitions."""
from __future__ import annotations

from typing import List, Sequence

from op_tester.core.results import CaseResult
from op_tester.plans.types import ExecutionPlan


class Reporter:
    """Interface for output renderers."""

    def on_start(self, plan: ExecutionPlan) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def on_case_result(self, result: CaseResult, index: int, total: int) -> None:  # pragma: no cover
        raise NotImplementedError

    def on_complete(self, results: Sequence[CaseResult]) -> None:  # pragma: no cover
        raise NotImplementedError


class ReportManager:
    """Dispatches lifecycle callbacks to multiple reporters."""

    def __init__(self, reporters: Sequence[Reporter]) -> None:
        self._reporters = list(reporters)

    def start(self, plan: ExecutionPlan) -> None:
        for reporter in self._reporters:
            reporter.on_start(plan)

    def handle_result(self, result: CaseResult, index: int, total: int) -> None:
        for reporter in self._reporters:
            reporter.on_case_result(result, index, total)

    def complete(self, results: Sequence[CaseResult]) -> None:
        for reporter in self._reporters:
            reporter.on_complete(results)

    def reporters(self) -> List[Reporter]:
        return list(self._reporters)
