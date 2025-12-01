"""Dataclasses representing execution plan structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from optest.core import BackendTarget, TestCase, Tolerance


@dataclass
class RunSettings:
    backend: str
    chip: Optional[str]
    seed: int
    fail_fast: bool
    report_format: str = "terminal"
    report_path: Optional[str] = None
    color: bool = True


@dataclass
class ExecutionPlan:
    cases: List[TestCase]
    settings: RunSettings


@dataclass
class CaseConfig:
    op: str
    dtypes: Optional[Tuple[str, ...]] = None
    shapes: Optional[Mapping[str, Tuple[int, ...]]] = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    generator: Optional[str] = None
    reference: Optional[str] = None
    tolerance: Optional[Tolerance] = None


@dataclass
class RunOptions:
    ops: Tuple[str, ...] = tuple()
    plan_path: Optional[str] = None
    dtype_override: Optional[Tuple[str, ...]] = None
    shape_overrides: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    attribute_overrides: Mapping[str, object] = field(default_factory=dict)
    backend: Optional[str] = None
    chip: Optional[str] = None
    seed: Optional[int] = None
    fail_fast: bool = False
    generator_override: Optional[str] = None
    reference_override: Optional[str] = None
    tolerance_override: Optional[Tolerance] = None
    report_format: Optional[str] = None
    report_path: Optional[str] = None
    color: Optional[bool] = None
