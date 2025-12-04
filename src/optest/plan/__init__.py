"""Plan loader and executor aligned with docs/optest_todos.md."""

from .loader import load_plan
from .models import (
    AssertionConfig,
    BackendConfig,
    CaseConfig,
    CaseShape,
    ExecutionPlan,
    GeneratorConfig,
    PlanOptions,
    ResolvedCase,
)
from .runner import run_plan

__all__ = [
    "AssertionConfig",
    "BackendConfig",
    "CaseConfig",
    "CaseShape",
    "ExecutionPlan",
    "GeneratorConfig",
    "PlanOptions",
    "ResolvedCase",
    "load_plan",
    "run_plan",
]
