"""YAML loader and validation for the redesigned plan format."""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml
from jsonschema import Draft7Validator, ValidationError

from .models import (
    AssertionConfig,
    BackendConfig,
    CaseBackends,
    CaseConfig,
    CaseShape,
    CommandConfig,
    ExecutionPlan,
    GeneratorConfig,
)

ALLOWED_BACKENDS = {"cann", "cuda"}


def load_plan(path: str) -> ExecutionPlan:
    """Load and validate a plan file."""
    plan_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(plan_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Plan file must contain a mapping at the top level")
    errors = sorted(_validator.iter_errors(raw), key=lambda e: e.path)
    if errors:
        messages = "; ".join(f"{'/'.join(map(str, err.path)) or 'root'}: {err.message}" for err in errors)
        raise ValueError(f"Plan schema validation failed: {messages}")
    operator = _require_str(raw, "operator")
    description = str(raw.get("description", ""))
    inputs = _parse_str_list(raw.get("inputs"))
    outputs = _parse_str_list(raw.get("outputs"))
    generator = _parse_generator(raw.get("generator"), plan_path.parent)
    assertion = _parse_assertion(raw.get("assertion"), plan_path.parent)
    backends = _parse_backends(raw.get("backends"), plan_path.parent)
    cases = _parse_cases(raw.get("cases"), plan_path.parent, inputs, outputs, generator, assertion)
    cache = raw.get("cache", "reuse")
    if cache not in {"reuse", "regen"}:
        raise ValueError("cache must be 'reuse' or 'regen'")
    tags = tuple(str(tag) for tag in raw.get("tags", []) or [])
    priority = raw.get("priority")
    if priority is not None:
        priority = int(priority)
    _validate_cases(inputs, outputs, cases)
    return ExecutionPlan(
        operator=operator,
        description=description,
        inputs=inputs,
        outputs=outputs,
        generator=generator,
        assertion=assertion,
        backends=backends,
        cases=cases,
        cache=cache,
        tags=tags,
        priority=priority,
        plan_dir=plan_path.parent,
    )


def _require_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing required string field '{key}'")
    text = value.strip()
    if not text:
        raise ValueError(f"Field '{key}' cannot be empty")
    return text


def _parse_str_list(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ValueError("inputs/outputs must be a list of strings")
    values = []
    for item in raw:
        if not isinstance(item, (str, Path)):
            raise ValueError("inputs/outputs entries must be strings")
        text = str(item).strip()
        if not text:
            raise ValueError("inputs/outputs entries cannot be empty")
        values.append(text)
    if not values:
        raise ValueError("inputs/outputs cannot be empty")
    return tuple(values)


def _parse_generator(raw: Any, base: Path, default: str = "builtin.random") -> GeneratorConfig:
    if raw is None:
        return GeneratorConfig(name=default)
    if isinstance(raw, str):
        return GeneratorConfig(name=raw)
    if not isinstance(raw, Mapping):
        raise ValueError("generator must be a string or mapping")
    name = str(raw.get("name", default))
    source = raw.get("source")
    source_path = base / source if source else None
    seed = raw.get("seed")
    params = raw.get("params") or {}
    per_input_raw = raw.get("per_input") or {}
    per_input: Dict[int, GeneratorConfig] = {}
    if isinstance(per_input_raw, Mapping):
        for key, value in per_input_raw.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError("generator.per_input keys must be integers") from exc
            per_input[idx] = _parse_generator(value, base, name)
    constants = raw.get("constants") or {}
    return GeneratorConfig(
        name=name,
        source=source_path.resolve() if source_path else None,
        seed=int(seed) if seed is not None else None,
        params=params,
        per_input=per_input,
        constants=constants,
    )


def _parse_assertion(raw: Any, base: Path, default: str = "builtin.identity") -> AssertionConfig:
    if raw is None:
        return AssertionConfig(name=default)
    if isinstance(raw, str):
        return AssertionConfig(name=raw)
    if not isinstance(raw, Mapping):
        raise ValueError("assertion must be a string or mapping")
    name = str(raw.get("name", default))
    source = raw.get("source")
    source_path = base / source if source else None
    rtol = raw.get("rtol")
    atol = raw.get("atol")
    metric = raw.get("metric")
    output_dtypes = raw.get("output_dtypes")
    if output_dtypes is not None:
        output_dtypes = tuple(str(v) for v in output_dtypes)
    params = raw.get("params") or {}
    return AssertionConfig(
        name=name,
        source=source_path.resolve() if source_path else None,
        rtol=float(rtol) if rtol is not None else None,
        atol=float(atol) if atol is not None else None,
        metric=str(metric) if metric is not None else None,
        output_dtypes=output_dtypes,
        params=params,
    )


def _parse_backends(raw: Any, base: Path) -> tuple[BackendConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("backends must be a non-empty list")
    backends: list[BackendConfig] = []
    seen_keys: set[tuple[str, str]] = set()
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Each backend entry must be a mapping")
        b_type = _require_str(entry, "type")
        if b_type not in ALLOWED_BACKENDS:
            raise ValueError(f"Unsupported backend type '{b_type}'")
        chip = _require_str(entry, "chip")
        key = (b_type, chip)
        if key in seen_keys:
            raise ValueError(f"Duplicate backend entry for type={b_type} chip={chip}")
        seen_keys.add(key)
        workdir_raw = entry.get("workdir")
        workdir = (base / workdir_raw).resolve() if workdir_raw else base
        env_raw = entry.get("env") or {}
        env = {str(k): str(v) for k, v in env_raw.items()}
        timeout = entry.get("timeout")
        timeout_value = int(timeout) if timeout is not None else None
        retries = int(entry.get("retries", 0))
        prepare = _parse_commands(entry.get("prepare"), base)
        cleanup = _parse_commands(entry.get("cleanup"), base)
        command = _parse_single_command(entry.get("command"))
        only_cases = tuple(str(x) for x in entry.get("only_cases", []) or [])
        skip_cases = tuple(str(x) for x in entry.get("skip_cases", []) or [])
        xfail_cases = tuple(str(x) for x in entry.get("xfail_cases", []) or [])
        backends.append(
            BackendConfig(
                type=b_type,
                chip=chip,
                workdir=Path(workdir),
                env=env,
                timeout=timeout_value,
                retries=retries,
                prepare=prepare,
                cleanup=cleanup,
                command=command,
                only_cases=only_cases,
                skip_cases=skip_cases,
                xfail_cases=xfail_cases,
            )
        )
    return tuple(backends)


def _parse_single_command(raw: Any) -> CommandConfig:
    argv = _normalize_command(raw)
    return CommandConfig(argv=argv)


def _parse_commands(raw: Any, base: Path) -> tuple[CommandConfig, ...]:
    if raw is None:
        return tuple()
    entries: list[Any]
    if isinstance(raw, (str, Path, Mapping, list)):
        entries = raw if isinstance(raw, list) else [raw]
    else:
        raise ValueError("prepare/cleanup must be a command or list of commands")
    commands: list[CommandConfig] = []
    for entry in entries:
        argv = _normalize_command(entry)
        commands.append(CommandConfig(argv=argv))
    return tuple(commands)


def _normalize_command(raw: Any) -> tuple[str, ...]:
    if raw is None:
        raise ValueError("command is required")
    if isinstance(raw, (str, Path)):
        return tuple(shlex.split(str(raw)))
    if isinstance(raw, Mapping):
        executable = raw.get("binary") or raw.get("executable")
        if not executable:
            raise ValueError("command mapping requires 'binary' or 'executable'")
        args = raw.get("args", [])
        if isinstance(args, (str, Path)):
            args_list = [str(args)]
        elif isinstance(args, list):
            args_list = [str(part) for part in args]
        else:
            raise ValueError("command args must be list or string")
        return tuple([str(executable)] + [str(a) for a in args_list])
    if isinstance(raw, (list, tuple)):
        return tuple(str(part) for part in raw)
    raise ValueError("command must be string, list, or mapping")


def _parse_cases(
    raw: Any,
    base: Path,
    default_inputs: Sequence[str],
    default_outputs: Sequence[str],
    default_generator: GeneratorConfig,
    default_assertion: AssertionConfig,
) -> tuple[CaseConfig, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("cases must be a non-empty list")
    cases: list[CaseConfig] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ValueError("case entries must be mappings")
        name = _require_str(entry, "name")
        dtypes_raw = entry.get("dtypes")
        if not isinstance(dtypes_raw, (list, tuple)) or not dtypes_raw:
            raise ValueError("case dtypes must be a non-empty list")
        dtypes = tuple(str(dt) for dt in dtypes_raw)
        shapes_raw = entry.get("shapes")
        if not isinstance(shapes_raw, list) or not shapes_raw:
            raise ValueError("case shapes must be a non-empty list")
        shapes: list[CaseShape] = []
        for shape_entry in shapes_raw:
            if not isinstance(shape_entry, Mapping):
                raise ValueError("case shapes entries must be mappings")
            inputs = _parse_shape_list(shape_entry.get("inputs"))
            outputs = _parse_shape_list(shape_entry.get("outputs"))
            shapes.append(CaseShape(inputs=inputs, outputs=outputs))
        generator = _parse_generator(entry.get("generator"), base, default_generator.name) if "generator" in entry else None
        assertion = _parse_assertion(entry.get("assertion"), base, default_assertion.name) if "assertion" in entry else None
        inputs_override = tuple(str(x) for x in entry.get("inputs", []) or []) or None
        outputs_override = tuple(str(x) for x in entry.get("outputs", []) or []) or None
        backend_filters = entry.get("backends") or {}
        if not isinstance(backend_filters, Mapping):
            raise ValueError("case.backends must be a mapping when provided")
        backend_spec = CaseBackends(
            only=tuple(str(x) for x in backend_filters.get("only", []) or []),
            skip=tuple(str(x) for x in backend_filters.get("skip", []) or []),
            xfail=tuple(str(x) for x in backend_filters.get("xfail", []) or []),
        )
        tags = tuple(str(tag) for tag in entry.get("tags", []) or [])
        priority = entry.get("priority")
        if priority is not None:
            priority = int(priority)
        cases.append(
            CaseConfig(
                name=name,
                dtypes=dtypes,
                shapes=tuple(shapes),
                generator=generator,
                assertion=assertion,
                inputs=inputs_override,
                outputs=outputs_override,
                backends=backend_spec,
                tags=tags,
                priority=priority,
            )
        )
    return tuple(cases)


def _parse_shape_list(raw: Any) -> tuple[tuple[int, ...], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("shapes inputs/outputs must be non-empty lists")
    shapes: list[tuple[int, ...]] = []
    for shape in raw:
        if isinstance(shape, (list, tuple)):
            dims = tuple(int(dim) for dim in shape)
        else:
            raise ValueError("shape entries must be lists")
        if not dims:
            raise ValueError("shape cannot be empty")
        shapes.append(dims)
    return tuple(shapes)


def _validate_cases(
    plan_inputs: Sequence[str],
    plan_outputs: Sequence[str],
    cases: Sequence[CaseConfig],
) -> None:
    for case in cases:
        expected_inputs = len(case.inputs or plan_inputs)
        expected_outputs = len(case.outputs or plan_outputs)
        if len(case.dtypes) != expected_inputs:
            raise ValueError(f"Case '{case.name}' dtypes length {len(case.dtypes)} does not match inputs {expected_inputs}")
        for idx, shape in enumerate(case.shapes):
            if len(shape.inputs) != expected_inputs:
                raise ValueError(
                    f"Case '{case.name}' shape index {idx} has {len(shape.inputs)} inputs, expected {expected_inputs}"
                )
            if len(shape.outputs) != expected_outputs:
                raise ValueError(
                    f"Case '{case.name}' shape index {idx} has {len(shape.outputs)} outputs, expected {expected_outputs}"
                )
        intersect = set(case.backends.only) & set(case.backends.skip)
        if intersect:
            raise ValueError(f"Case '{case.name}' has backends listed in both only and skip: {sorted(intersect)}")
        intersect = set(case.backends.skip) & set(case.backends.xfail)
        if intersect:
            raise ValueError(f"Case '{case.name}' has backends listed in both skip and xfail: {sorted(intersect)}")
PLAN_SCHEMA = {
    "type": "object",
    "required": ["operator", "inputs", "outputs", "backends", "cases"],
    "properties": {
        "operator": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "inputs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "outputs": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "generator": {"type": ["string", "object"]},
        "assertion": {"type": ["string", "object"]},
        "backends": {"type": "array", "minItems": 1},
        "cases": {"type": "array", "minItems": 1},
        "cache": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": ["number", "integer"]},
    },
}
_validator = Draft7Validator(PLAN_SCHEMA)
