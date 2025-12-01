"""Test runner orchestrating generators, backends, and comparisons."""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Sequence

import numpy as np

from optest.backends import backend_manager
from optest.generators import GeneratorProtocol, RandomTensorGenerator, resolve_generator

from .comparator import ComparisonResult, compare_outputs
from .models import TestCase
from .references import resolve_reference
from .results import CaseResult


class TestRunner:
    """Executes a collection of test cases sequentially."""

    def __init__(self, *, seed: int = 0, fail_fast: bool = False) -> None:
        self._seed = seed
        self._fail_fast = fail_fast

    def run(
        self,
        cases: Sequence[TestCase],
        *,
        on_result: Optional[Callable[[CaseResult, int, int], None]] = None,
    ) -> List[CaseResult]:
        results: List[CaseResult] = []
        master_rng = np.random.default_rng(self._seed)
        total = len(cases)
        for index, case in enumerate(cases, start=1):
            case_seed = int(master_rng.integers(0, 2**32 - 1))
            case_rng = np.random.default_rng(case_seed)
            result = self._execute_case(case, case_rng)
            results.append(result)
            if on_result:
                on_result(result, index, total)
            if self._fail_fast and not result.passed:
                break
        return results

    def _execute_case(self, case: TestCase, rng: np.random.Generator) -> CaseResult:
        start = time.perf_counter()
        try:
            generator = self._resolve_generator(case)
            inputs, maybe_reference = generator.generate(case, rng)
            expected = maybe_reference
            if expected is None:
                reference_fn = resolve_reference(case.descriptor, case.reference_override)
                expected = reference_fn(inputs, case.attributes)
            driver = backend_manager.get_driver(case.backend.kind, case.backend.chip)
            outputs = driver.run(case, inputs)
            comparison = compare_outputs(case, outputs, expected)
            status = "passed" if comparison.passed else "failed"
            duration = time.perf_counter() - start
            return CaseResult(case=case, status=status, duration_s=duration, comparison=comparison)
        except Exception as exc:  # pragma: no cover - aggregated error path
            duration = time.perf_counter() - start
            return CaseResult(
                case=case,
                status="error",
                duration_s=duration,
                comparison=None,
                error=str(exc),
            )

    def _resolve_generator(self, case: TestCase) -> GeneratorProtocol:
        if case.generator_override:
            return resolve_generator(case.generator_override)
        if case.descriptor.default_generator:
            return resolve_generator(case.descriptor.default_generator)
        return RandomTensorGenerator()
