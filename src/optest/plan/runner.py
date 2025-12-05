"""Executor for the redesigned plan format."""
from __future__ import annotations

import fnmatch
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
from colorama import Fore, Style, init as colorama_init
from jsonschema import Draft7Validator

from optest.operators import builtin_operators

from . import custom
from .models import AssertionConfig, AssertionResult, CaseRunResult, ExecutionPlan, GeneratorConfig, PlanOptions, ResolvedCase

# Registry of built-in operator classes keyed by normalized assertion name.
_BUILTIN_ASSERTION_REGISTRY: Dict[str, type[builtin_operators.BuiltinOperator]] = {}


def _normalize_builtin_key(name: str) -> str:
    """Normalize assertion.name to a registry key (case-insensitive, strip prefixes)."""

    text = name or ""
    if ":" in text:
        text = text.split(":", 1)[-1]
    if text.endswith(".run"):
        text = text[: -len(".run")]
    if "." in text:
        text = text.split(".")[-1]
    return text.lower()


def _populate_builtin_registry() -> None:
    if _BUILTIN_ASSERTION_REGISTRY:
        return
    for cls in builtin_operators.BUILTIN_OPERATOR_CLASSES:
        aliases = {
            cls.name.lower(),
            f"builtin.{cls.name}".lower(),
            cls.__name__.lower(),
        }
        for alias in aliases:
            _BUILTIN_ASSERTION_REGISTRY.setdefault(alias, cls)


def run_plan(
    plan: ExecutionPlan,
    options: PlanOptions,
    *,
    report_format: str = "terminal",
    report_path: str | None = None,
    use_color: bool = True,
) -> int:
    """Execute the plan; returns process exit code (0 success, 1 failures)."""

    colorama_init()
    resolved = _resolve_cases(plan, options)
    if options.list_only:
        for case in resolved:
            print(_format_case_identifier(case))
        return 0
    if not resolved:
        print("No cases matched the provided filters.")
        return 1
    cache_policy = options.cache or plan.cache
    results: list[CaseRunResult] = []
    for item in resolved:
        result = _execute_case(item, cache_policy)
        results.append(result)
        if report_format == "terminal":
            _print_result(result, use_color=use_color)
    failures = sum(1 for r in results if r.status in {"failed", "error", "xfail-pass"})
    if report_format == "terminal":
        _print_summary(results, failures, use_color=use_color)
    else:
        _write_json_report(results, report_path)
    return 0 if failures == 0 else 1


def _resolve_cases(plan: ExecutionPlan, options: PlanOptions) -> list[ResolvedCase]:
    matches: list[ResolvedCase] = []
    for backend_index, backend in enumerate(plan.backends):
        if options.backend and backend.type != options.backend:
            continue
        if options.chip and backend.chip != options.chip:
            continue
        for case_index, case in enumerate(plan.cases):
            if options.cases and not any(fnmatch.fnmatchcase(case.name, pattern) for pattern in options.cases):
                continue
            if options.tags and not set(options.tags) & set(case.tags):
                continue
            if options.skip_tags and set(options.skip_tags) & set(case.tags):
                continue
            priority = case.priority if case.priority is not None else plan.priority
            if options.priority_max is not None and priority is not None and priority > options.priority_max:
                continue
            if backend.only_cases and case.name not in backend.only_cases:
                continue
            if case.name in backend.skip_cases:
                continue
            if case.backends.only and backend.type not in case.backends.only:
                continue
            if backend.type in case.backends.skip:
                continue
            xfail = backend.type in case.backends.xfail or case.name in backend.xfail_cases
            input_files = tuple(_resolve_paths(case.inputs or plan.inputs, backend.workdir))
            output_files = tuple(_resolve_paths(case.outputs or plan.outputs, backend.workdir))
            for shape_index, shape in enumerate(case.shapes):
                matches.append(
                    ResolvedCase(
                        plan=plan,
                        backend=backend,
                        case=case,
                        shape=shape,
                        case_index=case_index,
                        shape_index=shape_index,
                        input_paths=input_files,
                        output_paths=output_files,
                        xfail=xfail,
                    )
                )
    return matches


def _resolve_paths(paths: Sequence[str], base: Path) -> Sequence[Path]:
    resolved: list[Path] = []
    for path in paths:
        p = Path(path)
        if not p.is_absolute():
            p = base / p
        resolved.append(p)
    return tuple(resolved)


def _execute_case(resolved: ResolvedCase, cache_policy: str) -> CaseRunResult:
    identifier = _format_case_identifier(resolved)
    try:
        generator = resolved.case.generator or resolved.plan.generator
        assertion = resolved.case.assertion or resolved.plan.assertion
        cache_policy = cache_policy or resolved.plan.cache
        inputs = _prepare_inputs(resolved, generator, cache_policy)
        _ensure_output_dirs(resolved.output_paths)
        _run_backend_commands(resolved)
        outputs = _load_outputs(resolved, assertion)
        assertion_result = _run_assertion(resolved, assertion, inputs, outputs)
        if assertion_result.ok:
            status = "xfail-pass" if resolved.xfail else "passed"
        else:
            status = "xfail" if resolved.xfail else "failed"
        return CaseRunResult(
            identifier=identifier,
            status=status,
            details=assertion_result.details,
            metrics=assertion_result.metrics,
            xfail=resolved.xfail,
        )
    except Exception as exc:
        status = "xfail" if resolved.xfail else "error"
        return CaseRunResult(
            identifier=identifier,
            status=status,
            details=str(exc),
            metrics={},
            xfail=resolved.xfail,
        )


def _format_case_identifier(resolved: ResolvedCase) -> str:
    shape_desc = f"shape{resolved.shape_index}"
    return f"{resolved.case.name}@{resolved.backend.type}:{resolved.backend.chip}/{shape_desc}"


def _prepare_inputs(resolved: ResolvedCase, generator_cfg: GeneratorConfig, cache_policy: str) -> Sequence[np.ndarray]:
    if cache_policy == "reuse" and all(path.exists() for path in resolved.input_paths):
        return _load_inputs(resolved)
    rng_seed = generator_cfg.seed
    rng = np.random.default_rng(rng_seed)
    if generator_cfg.source:
        _call_custom_generator(generator_cfg, resolved, rng)
        return _load_inputs(resolved)
    inputs: list[np.ndarray] = []
    for index, (path, shape, dtype) in enumerate(
        zip(resolved.input_paths, resolved.shape.inputs, resolved.case.dtypes)
    ):
        gen_cfg = generator_cfg.per_input.get(index, generator_cfg)
        arr = _generate_array(gen_cfg, shape, dtype, rng, resolved)
        path.parent.mkdir(parents=True, exist_ok=True)
        arr.astype(dtype).tofile(path)
        inputs.append(arr.astype(dtype))
    return tuple(inputs)


def _load_inputs(resolved: ResolvedCase) -> Sequence[np.ndarray]:
    arrays: list[np.ndarray] = []
    for path, shape, dtype in zip(resolved.input_paths, resolved.shape.inputs, resolved.case.dtypes):
        data = np.fromfile(path, dtype=dtype)
        arrays.append(data.reshape(shape))
    return tuple(arrays)


def _call_custom_generator(config: GeneratorConfig, resolved: ResolvedCase, rng: np.random.Generator) -> None:
    func = custom.load_from_source(config.source, config.name)  # type: ignore[arg-type]
    func(
        input_paths=[str(p) for p in resolved.input_paths],
        shapes={"inputs": [list(s) for s in resolved.shape.inputs], "outputs": [list(s) for s in resolved.shape.outputs]},
        dtypes=list(resolved.case.dtypes),
        params=config.params,
        seed=config.seed,
        constants=config.constants,
        rng=rng,
    )


def _generate_array(
    config: GeneratorConfig, shape: Sequence[int], dtype: str, rng: np.random.Generator, resolved: ResolvedCase
) -> np.ndarray | None:
    name = config.name
    low = float(config.params.get("low", -1.0))
    high = float(config.params.get("high", 1.0))
    constants = config.constants or {}
    if "value" in constants:
        return np.full(shape, constants["value"], dtype=dtype)
    if name.endswith("uniform"):
        return rng.uniform(low, high, size=shape).astype(dtype)
    if name.endswith("ones"):
        return np.ones(shape, dtype=dtype)
    if name.endswith("random"):
        data = rng.standard_normal(size=shape).astype(dtype)
    else:
        raise ValueError(
            f"Unknown generator '{name}'. "
            "Supported builtins: builtin.random, builtin.uniform, builtin.ones. "
            "For a custom generator, set both generator.name and generator.source."
        )
    scale = float(constants.get("scale", 1.0))
    shift = float(constants.get("shift", 0.0))
    return (data * scale + shift).astype(dtype)


def _ensure_output_dirs(paths: Sequence[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()


def _run_backend_commands(resolved: ResolvedCase) -> None:
    backend = resolved.backend
    tokens = _build_tokens(resolved)
    env = os.environ.copy()
    env.update(_render_env(backend.env, tokens))
    for cmd in backend.prepare:
        _run_command(cmd.argv, backend.workdir, env, tokens, backend.timeout)
    _run_command(backend.command.argv, backend.workdir, env, tokens, backend.timeout, backend.retries)
    for cmd in backend.cleanup:
        _run_command(cmd.argv, backend.workdir, env, tokens, backend.timeout)


def _run_command(
    argv: Sequence[str],
    workdir: Path,
    env: Mapping[str, str],
    tokens: Mapping[str, str],
    timeout: int | None,
    retries: int = 0,
) -> None:
    rendered = [_render_token(part, tokens) for part in argv]
    attempts = retries + 1
    last_exc: RuntimeError | None = None
    for attempt in range(attempts):
        proc = subprocess.run(
            rendered,
            cwd=str(workdir),
            env=dict(env),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return
        last_exc = RuntimeError(
            f"command '{' '.join(rendered)}' failed (code {proc.returncode}) "
            f"in {workdir}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    if last_exc:
        raise last_exc


def _build_tokens(resolved: ResolvedCase) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
    tokens["chip"] = resolved.backend.chip
    tokens["backend"] = resolved.backend.type
    tokens["case"] = resolved.case.name
    tokens["dtype"] = resolved.case.dtypes[0] if resolved.case.dtypes else ""
    tokens["dtypes"] = ",".join(resolved.case.dtypes)
    first_shape = resolved.shape.inputs[0] if resolved.shape.inputs else ()
    tokens["shape"] = "x".join(str(dim) for dim in first_shape)
    tokens["shapes"] = json.dumps({"inputs": resolved.shape.inputs, "outputs": resolved.shape.outputs})
    tokens["workdir"] = str(resolved.backend.workdir)
    tokens["inputs"] = ",".join(str(p) for p in resolved.input_paths)
    tokens["outputs"] = ",".join(str(p) for p in resolved.output_paths)
    for idx, path in enumerate(resolved.input_paths):
        tokens[f"input{idx}"] = str(path)
    for idx, path in enumerate(resolved.output_paths):
        tokens[f"output{idx}"] = str(path)
    return tokens


def _render_token(value: str, tokens: Mapping[str, str]) -> str:
    return _render_template(value, tokens, quote=True)


def _render_template(value: str, tokens: Mapping[str, str], *, quote: bool) -> str:
    if "{" in value and "}" in value:
        try:
            rendered = value.format(**tokens)
        except KeyError as exc:
            available = ", ".join(sorted(tokens.keys()))
            raise RuntimeError(
                f"Unknown token {exc} in value '{value}'. Available tokens: {available}"
            ) from exc
    else:
        rendered = value
    if quote and rendered:
        return shlex.quote(rendered)
    return rendered


def _render_env(values: Mapping[str, str], tokens: Mapping[str, str]) -> Dict[str, str]:
    rendered: Dict[str, str] = {}
    for key, value in values.items():
        rendered_key = _render_template(str(key), tokens, quote=False)
        rendered_value = _render_template(str(value), tokens, quote=False)
        rendered[rendered_key] = rendered_value
    return rendered


def _load_outputs(resolved: ResolvedCase, assertion: AssertionConfig) -> Sequence[np.ndarray]:
    dtypes = _resolve_output_dtypes(resolved, assertion)
    outputs: list[np.ndarray] = []
    for path, shape, dtype in zip(resolved.output_paths, resolved.shape.outputs, dtypes):
        if not path.exists():
            raise FileNotFoundError(f"expected output missing at {path} for case {resolved.case.name}")
        data = np.fromfile(path, dtype=dtype)
        outputs.append(data.reshape(shape))
    return tuple(outputs)


def _resolve_output_dtypes(resolved: ResolvedCase, assertion: AssertionConfig) -> Sequence[str]:
    if assertion.output_dtypes:
        if len(assertion.output_dtypes) != len(resolved.output_paths):
            raise ValueError("output_dtypes length must match outputs")
        return tuple(assertion.output_dtypes)
    # default: mirror case dtypes; if fewer dtypes than outputs, repeat last dtype
    if resolved.case.dtypes:
        values = list(resolved.case.dtypes)
        while len(values) < len(resolved.output_paths):
            values.append(values[-1])
        return tuple(values[: len(resolved.output_paths)])
    return tuple("float32" for _ in resolved.output_paths)


def _run_assertion(
    resolved: ResolvedCase,
    assertion: AssertionConfig,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
) -> AssertionResult:
    if assertion.source:
        func = custom.load_from_source(assertion.source, assertion.name)
        result = func(
            input_paths=[str(p) for p in resolved.input_paths],
            output_paths=[str(p) for p in resolved.output_paths],
            shapes={"inputs": [list(s) for s in resolved.shape.inputs], "outputs": [list(s) for s in resolved.shape.outputs]},
            dtypes=list(resolved.case.dtypes),
            output_dtypes=list(_resolve_output_dtypes(resolved, assertion)),
            params=assertion.params,
            rtol=assertion.rtol,
            atol=assertion.atol,
            metric=assertion.metric,
        )
        if isinstance(result, AssertionResult):
            return result
        if isinstance(result, tuple) and len(result) == 2:
            ok, details = result
            return AssertionResult(ok=bool(ok), details=str(details))
        raise TypeError("Custom assertion must return AssertionResult or (ok, details)")
    return _builtin_assertion(assertion, inputs, outputs, resolved)


def _builtin_assertion(
    assertion: AssertionConfig,
    inputs: Sequence[np.ndarray],
    outputs: Sequence[np.ndarray],
    resolved: ResolvedCase,
) -> AssertionResult:
    _populate_builtin_registry()
    name = assertion.name
    normalized = _normalize_builtin_key(name)
    if normalized == "identity":
        expected = outputs
        default_tol = None
    else:
        op_cls = _BUILTIN_ASSERTION_REGISTRY.get(normalized)
        if not op_cls:
            supported = ", ".join(sorted({cls.name for cls in _BUILTIN_ASSERTION_REGISTRY.values()}))
            return AssertionResult(
                ok=False,
                details=(
                    f"Unknown builtin assertion '{name}'. "
                    f"Supported builtins: {supported}. "
                    "For custom assertions, set both assertion.name and assertion.source."
                ),
            )
        expected = op_cls.run(inputs, assertion.params)
        default_tol = getattr(op_cls, "default_tolerance", None)
    rtol = assertion.rtol if assertion.rtol is not None else (default_tol.relative if default_tol else 1e-5)
    atol = assertion.atol if assertion.atol is not None else (default_tol.absolute if default_tol else 1e-4)
    metric_name = assertion.metric or "max_abs"
    ok, details, metrics = _compare_outputs(outputs, expected, rtol, atol, metric_name)
    return AssertionResult(ok=ok, details=details, metrics=metrics)


def _compare_outputs(
    outputs: Sequence[np.ndarray],
    expected: Sequence[np.ndarray],
    rtol: float,
    atol: float,
    metric: str,
) -> tuple[bool, str, Dict[str, Any]]:
    metrics: Dict[str, Any] = {}
    if len(outputs) != len(expected):
        return False, f"Output count mismatch (got {len(outputs)} expected {len(expected)})", metrics
    for idx, (got, want) in enumerate(zip(outputs, expected)):
        if got.shape != want.shape:
            return False, f"Output{idx} shape mismatch {got.shape} vs {want.shape}", metrics
        if not np.allclose(got, want, rtol=rtol, atol=atol):
            diff = np.abs(got - want)
            max_abs = float(np.max(diff))
            metrics[f"output{idx}_max_abs"] = max_abs
            if metric == "mean_abs":
                metrics[f"output{idx}_mean_abs"] = float(np.mean(diff))
            return False, f"Output{idx} mismatch (max_abs={max_abs})", metrics
    return True, "", metrics


def _print_result(result: CaseRunResult, *, use_color: bool = True) -> None:
    status = result.status
    label, color = _format_status(status, use_color=use_color)
    reset = Style.RESET_ALL if use_color else ""
    status_block = f"{color}{label:<11}{reset}"
    print(f"{status_block} {result.identifier}")
    if result.details:
        print(f"    detail: {result.details}")
    if result.metrics:
        metrics_text = ", ".join(f"{k}={v}" for k, v in result.metrics.items())
        print(f"    metrics: {metrics_text}")


def _print_summary(results: Sequence[CaseRunResult], failures: int, *, use_color: bool = True) -> None:
    total = len(results)
    summary_color = Fore.GREEN if failures == 0 and use_color else Fore.RED if use_color else ""
    reset = Style.RESET_ALL if use_color else ""
    passed = sum(1 for r in results if r.status == "passed")
    xfail = sum(1 for r in results if r.status.startswith("xfail"))
    failed = failures
    print(f"{summary_color}Summary{reset}: total={total} passed={passed} xfail={xfail} failed={failed}")


def _format_status(status: str, *, use_color: bool) -> tuple[str, str]:
    color = ""
    if not use_color:
        return status.upper(), color
    if status == "passed":
        color = Fore.GREEN
    elif status in {"failed", "error"}:
        color = Fore.RED
    elif status.startswith("xfail"):
        color = Fore.YELLOW
    label = {
        "passed": "PASS",
        "failed": "FAIL",
        "error": "ERROR",
        "xfail": "XFAIL",
        "xfail-pass": "XPASS",
    }.get(status, status.upper())
    return label, color


def _write_json_report(results: Sequence[CaseRunResult], path: str | None) -> None:
    payload = {
        "summary": {
            "total": len(results),
            "failures": sum(1 for r in results if r.status in {"failed", "error", "xfail-pass"}),
        },
        "cases": [
            {
                "id": r.identifier,
                "status": r.status,
                "details": r.details,
                "metrics": r.metrics,
                "xfail": r.xfail,
            }
            for r in results
        ],
    }
    REPORT_SCHEMA = {
        "type": "object",
        "required": ["summary", "cases"],
        "properties": {
            "summary": {
                "type": "object",
                "required": ["total", "failures"],
                "properties": {
                    "total": {"type": "integer"},
                    "failures": {"type": "integer"},
                },
            },
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "status", "details", "metrics", "xfail"],
                    "properties": {
                        "id": {"type": "string"},
                        "status": {"type": "string"},
                        "details": {"type": "string"},
                        "metrics": {"type": "object"},
                        "xfail": {"type": "boolean"},
                    },
                },
            },
        },
    }
    Draft7Validator(REPORT_SCHEMA).validate(payload)
    text = json.dumps(payload, indent=2)
    if path:
        Path(path).write_text(text, encoding="utf-8")
    else:
        print(text)
