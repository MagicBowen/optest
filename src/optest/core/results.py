"""Result data structures produced by the test runner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .comparator import ComparisonResult
from .models import TestCase


@dataclass
class CaseResult:
    """Outcome of executing a single test case."""

    case: TestCase
    status: str
    duration_s: float
    comparison: Optional[ComparisonResult] = None
    error: Optional[str] = None
    seed: Optional[int] = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"
