"""Data models for the redesigned plan format."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class GeneratorConfig:
    name: str
    source: Optional[Path] = None
    seed: Optional[int] = None
    params: Mapping[str, Any] = field(default_factory=dict)
    per_input: Mapping[int, "GeneratorConfig"] = field(default_factory=dict)
    constants: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssertionConfig:
    name: str
    source: Optional[Path] = None
    rtol: Optional[float] = None
    atol: Optional[float] = None
    metric: Optional[str] = None
    output_dtypes: Optional[Sequence[str]] = None
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandConfig:
    argv: Sequence[str]


@dataclass(frozen=True)
class BackendConfig:
    type: str
    chip: str
    workdir: Path
    env: Mapping[str, str]
    timeout: Optional[int]
    retries: int
    prepare: Sequence[CommandConfig]
    cleanup: Sequence[CommandConfig]
    command: CommandConfig
    only_cases: Sequence[str]
    skip_cases: Sequence[str]
    xfail_cases: Sequence[str]


@dataclass(frozen=True)
class CaseShape:
    inputs: Sequence[Sequence[int]]
    outputs: Sequence[Sequence[int]]


@dataclass(frozen=True)
class CaseBackends:
    only: Sequence[str] = field(default_factory=tuple)
    skip: Sequence[str] = field(default_factory=tuple)
    xfail: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class CaseConfig:
    name: str
    dtypes: Sequence[str]
    shapes: Sequence[CaseShape]
    generator: Optional[GeneratorConfig] = None
    assertion: Optional[AssertionConfig] = None
    inputs: Optional[Sequence[str]] = None
    outputs: Optional[Sequence[str]] = None
    backends: CaseBackends = field(default_factory=CaseBackends)
    tags: Sequence[str] = field(default_factory=tuple)
    priority: Optional[int] = None


@dataclass(frozen=True)
class ExecutionPlan:
    operator: str
    description: str
    inputs: Sequence[str]
    outputs: Sequence[str]
    generator: GeneratorConfig
    assertion: AssertionConfig
    backends: Sequence[BackendConfig]
    cases: Sequence[CaseConfig]
    cache: str
    tags: Sequence[str]
    priority: Optional[int]
    plan_dir: Path


@dataclass(frozen=True)
class ResolvedCase:
    plan: ExecutionPlan
    backend: BackendConfig
    case: CaseConfig
    shape: CaseShape
    case_index: int
    shape_index: int
    input_paths: Sequence[Path]
    output_paths: Sequence[Path]
    xfail: bool = False


@dataclass(frozen=True)
class AssertionResult:
    ok: bool
    details: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CaseRunResult:
    identifier: str
    status: str
    details: str = ""
    metrics: Mapping[str, Any] = field(default_factory=dict)
    xfail: bool = False


@dataclass(frozen=True)
class PlanOptions:
    backend: Optional[str] = None
    chip: Optional[str] = None
    cases: Sequence[str] = field(default_factory=tuple)
    tags: Sequence[str] = field(default_factory=tuple)
    skip_tags: Sequence[str] = field(default_factory=tuple)
    priority_max: Optional[int] = None
    cache: Optional[str] = None
    list_only: bool = False
