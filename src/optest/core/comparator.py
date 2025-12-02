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
    max_error_index: tuple[int, ...] | None = None
    actual_value: float | None = None
    expected_value: float | None = None
    detail: str | None = None


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
                    detail=f"shape mismatch: actual {act.shape}, expected {exp.shape}",
                )
            )
            overall_passed = False
            continue
        abs_diff, rel_diff, index, actual_value, expected_value = _diff_metrics(act, exp)
        close = np.isclose(act, exp, atol=tolerance.absolute, rtol=tolerance.relative)
        mismatched = int(close.size - int(np.count_nonzero(close)))
        metrics.append(
            TensorComparisonResult(
                passed=mismatched == 0,
                max_abs_error=abs_diff,
                max_rel_error=rel_diff,
                mismatched=mismatched,
                total=close.size,
                max_error_index=index,
                actual_value=actual_value,
                expected_value=expected_value,
            )
        )
        if mismatched:
            overall_passed = False

    return ComparisonResult(passed=overall_passed, tensors=metrics)


def _diff_metrics(
    actual: np.ndarray, expected: np.ndarray
) -> tuple[float, float, tuple[int, ...] | None, float | None, float | None]:
    act = actual.astype(np.float64)
    exp = expected.astype(np.float64)
    diff = np.abs(act - exp)
    if diff.size == 0:
        return 0.0, 0.0, None, None, None
    flat_index = int(np.argmax(diff))
    max_abs = float(diff.flat[flat_index])
    max_abs_index = tuple(np.unravel_index(flat_index, act.shape))
    denom = np.maximum(np.abs(exp), 1e-12)
    rel = np.divide(diff, denom)
    max_rel = float(rel.max(initial=0.0))
    return (
        max_abs,
        max_rel,
        max_abs_index,
        float(act.flat[flat_index]),
        float(exp.flat[flat_index]),
    )
