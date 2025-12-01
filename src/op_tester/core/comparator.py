"""Utilities for comparing backend outputs with reference tensors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .models import TestCase, Tolerance


@dataclass
class TensorComparisonResult:
    """Per-tensor comparison metrics."""

    passed: bool
    max_abs_error: float
    max_rel_error: float
    mismatched: int
    total: int


@dataclass
class ComparisonResult:
    """Aggregated comparison outcome for a test case."""

    passed: bool
    tensors: List[TensorComparisonResult] = field(default_factory=list)
    message: str | None = None


def compare_outputs(
    case: TestCase,
    actual: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
) -> ComparisonResult:
    if len(actual) != len(expected):
        return ComparisonResult(
            passed=False,
            message=(
                f"Output arity mismatch: got {len(actual)} tensors, expected {len(expected)}"
            ),
        )

    tolerance = case.tolerance
    metrics: list[TensorComparisonResult] = []
    overall_passed = True
    for idx, (act, exp) in enumerate(zip(actual, expected)):
        if act.shape != exp.shape:
            metrics.append(
                TensorComparisonResult(
                    passed=False,
                    max_abs_error=float("inf"),
                    max_rel_error=float("inf"),
                    mismatched=act.size,
                    total=act.size,
                )
            )
            overall_passed = False
            continue
        abs_diff, rel_diff = _diff_metrics(act, exp)
        close = np.isclose(act, exp, atol=tolerance.absolute, rtol=tolerance.relative)
        mismatched = int(close.size - int(np.count_nonzero(close)))
        metrics.append(
            TensorComparisonResult(
                passed=mismatched == 0,
                max_abs_error=abs_diff,
                max_rel_error=rel_diff,
                mismatched=mismatched,
                total=close.size,
            )
        )
        if mismatched:
            overall_passed = False

    return ComparisonResult(passed=overall_passed, tensors=metrics)


def _diff_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    act = actual.astype(np.float64)
    exp = expected.astype(np.float64)
    diff = np.abs(act - exp)
    max_abs = float(diff.max(initial=0.0))
    denom = np.maximum(np.abs(exp), 1e-12)
    rel = np.divide(diff, denom)
    max_rel = float(rel.max(initial=0.0))
    return max_abs, max_rel
