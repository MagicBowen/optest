"""Execution plan loader combining CLI options and YAML files."""
from __future__ import annotations

import pathlib
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from optest.core import BackendTarget, TestCase, Tolerance
from optest.registry import OperatorRegistry, registry

from .types import CaseConfig, ExecutionPlan, RunOptions, RunSettings


def build_execution_plan(
    options: RunOptions,
    *,
    operator_registry: OperatorRegistry | None = None,
) -> ExecutionPlan:
    operator_registry = operator_registry or registry
    raw = _load_yaml(options.plan_path)
    settings = _build_run_settings(options, raw)
    case_configs = _resolve_case_configs(options, raw)
    if not case_configs:
        raise ValueError("No operator test cases resolved from CLI or plan file")
    cases = [
        _materialize_case(cfg, settings, options, operator_registry)
        for cfg in case_configs
    ]
    return ExecutionPlan(cases=cases, settings=settings)


def _load_yaml(path: Optional[str]) -> Mapping[str, Any]:
    if not path:
        return {}
    data = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("Plan file must contain a mapping at the top level")
    return data


def _build_run_settings(options: RunOptions, raw: Mapping[str, Any]) -> RunSettings:
    backend = options.backend or raw.get("backend") or "gpu"
    chip = options.chip or raw.get("chip")
    seed = options.seed if options.seed is not None else raw.get("seed", 0)
    fail_fast = options.fail_fast or bool(raw.get("fail_fast", False))
    report_cfg = raw.get("report", {}) if isinstance(raw.get("report"), Mapping) else {}
    report_format = options.report_format or report_cfg.get("format", "terminal")
    report_path = options.report_path or report_cfg.get("path")
    color_default = bool(report_cfg.get("color", True)) if report_cfg else True
    color = options.color if options.color is not None else color_default
    return RunSettings(
        backend=backend,
        chip=chip,
        seed=int(seed),
        fail_fast=bool(fail_fast),
        report_format=report_format,
        report_path=report_path,
        color=color,
    )


def _resolve_case_configs(options: RunOptions, raw: Mapping[str, Any]) -> List[CaseConfig]:
    cases_raw = raw.get("cases")
    configs: List[CaseConfig] = []
    if isinstance(cases_raw, list):
        for entry in cases_raw:
            if not isinstance(entry, Mapping):
                raise ValueError("Each case entry must be a mapping")
            configs.extend(_parse_case_entry(entry))
    if options.ops:
        desired = list(options.ops)
        filtered = [cfg for cfg in configs if cfg.op in desired]
        known = {cfg.op for cfg in filtered}
        for op in desired:
            if op not in known:
                filtered.append(CaseConfig(op=op))
        configs = filtered
    if not configs:
        if not options.ops:
            raise ValueError("Specify operators via --op or provide cases in the plan file")
        configs = [CaseConfig(op=op) for op in options.ops]
    return configs


def _parse_case_entry(entry: Mapping[str, Any]) -> List[CaseConfig]:
    op = entry.get("op")
    if not isinstance(op, str):
        raise ValueError("Each case entry must specify an 'op' string")
    dtype_field = entry.get("dtypes") or entry.get("dtype")
    dtype_variants = _normalize_dtype_variants(dtype_field)
    shape_variants = _normalize_shape_variants(entry.get("shapes"))
    attributes = dict(entry.get("attributes", {}))
    for key, value in entry.items():
        if key in {"op", "dtypes", "dtype", "shapes", "attributes", "generator", "reference", "tolerance"}:
            continue
        if key not in attributes:
            attributes[key] = value
    tolerance = Tolerance.from_mapping(entry.get("tolerance")) if entry.get("tolerance") else None
    generator = entry.get("generator")
    reference = entry.get("reference")
    configs: List[CaseConfig] = []
    for dtype in dtype_variants:
        for shapes in shape_variants:
            configs.append(
                CaseConfig(
                    op=op,
                    dtypes=dtype,
                    shapes=shapes,
                    attributes=attributes,
                    generator=generator,
                    reference=reference,
                    tolerance=tolerance,
                )
            )
    return configs


def _materialize_case(
    config: CaseConfig,
    settings: RunSettings,
    options: RunOptions,
    operator_registry: OperatorRegistry,
) -> TestCase:
    descriptor = operator_registry.get(config.op)
    dtype = options.dtype_override or config.dtypes or _default_dtypes(descriptor)
    if len(dtype) != descriptor.num_inputs:
        raise ValueError(
            f"Operator '{descriptor.name}' expects {descriptor.num_inputs} dtypes, got {len(dtype)}"
        )
    shapes = dict(config.shapes or {})
    shapes.update(options.shape_overrides)
    if not shapes:
        shapes = _default_shapes(descriptor)
    tolerance = options.tolerance_override or config.tolerance or descriptor.default_tolerance
    attributes = dict(config.attributes)
    attributes.update(options.attribute_overrides)
    backend_target = BackendTarget(kind=settings.backend, chip=settings.chip)
    return TestCase(
        descriptor=descriptor,
        dtype_spec=tuple(dtype),
        shapes=shapes,
        backend=backend_target,
        tolerance=tolerance,
        attributes=attributes,
        generator_override=options.generator_override or config.generator,
        reference_override=options.reference_override or config.reference,
    )


def _default_dtypes(descriptor) -> tuple[str, ...]:
    if descriptor.dtype_variants:
        return descriptor.dtype_variants[0]
    return tuple("float32" for _ in range(descriptor.num_inputs))


def _default_shapes(descriptor) -> Dict[str, tuple[int, ...]]:
    return {f"input{idx}": (2, 2) for idx in range(descriptor.num_inputs)}


def _normalize_dtype_spec(field: Any) -> Optional[tuple[str, ...]]:
    if field is None:
        return None
    if isinstance(field, str):
        return tuple(part.strip() for part in field.split(",") if part.strip())
    if isinstance(field, Iterable):
        values = [str(item) for item in field]
        return tuple(values)
    raise TypeError("Unsupported dtype specification format")


def _normalize_dtype_variants(field: Any) -> List[Optional[tuple[str, ...]]]:
    if field is None:
        return [None]
    if isinstance(field, (list, tuple)) and field and not isinstance(field[0], (str, bytes)):
        return [_normalize_dtype_spec(item) for item in field]
    return [_normalize_dtype_spec(field)]


def _normalize_shapes(raw: Any) -> Dict[str, tuple[int, ...]]:
    if not raw:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("Shapes specification must be a mapping")
    result: Dict[str, tuple[int, ...]] = {}
    for key, value in raw.items():
        result[str(key)] = _coerce_shape(value)
    return result


def _normalize_shape_variants(raw: Any) -> List[Dict[str, tuple[int, ...]]]:
    if raw is None:
        return [{}]
    if isinstance(raw, list):
        return [_normalize_shapes(item) for item in raw]
    return [_normalize_shapes(raw)]


def _coerce_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [part for part in value.replace("X", "x").split("x") if part]
        if not parts:
            raise ValueError("Shape string must contain at least one dimension")
        return tuple(int(part) for part in parts)
    if isinstance(value, Iterable):
        dims = [int(v) for v in value]
        if not dims:
            raise ValueError("Shape iterable cannot be empty")
        return tuple(dims)
    if isinstance(value, int):
        return (value,)
    raise TypeError("Unsupported shape specification")
