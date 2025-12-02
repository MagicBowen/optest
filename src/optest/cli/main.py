"""CLI entry point for optest."""
from __future__ import annotations

import sys
import datetime as dt
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import yaml

from optest import __version__, bootstrap
from optest.core import Tolerance
from optest.core.runner import TestRunner
from optest.plans.loader import build_execution_plan
from optest.plans.types import ExecutionPlan, RunOptions
from optest.reporting import JsonReporter, ReportManager, TerminalReporter


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


class CliState:
    """Holds global CLI state (future expansion)."""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose


def _print_version(_: click.Context, __: click.Parameter, value: bool) -> None:
    if not value or click.get_current_context().resilient_parsing:
        return
    click.echo(f"optest {__version__}")
    raise click.exceptions.Exit()


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--verbose", is_flag=True, help="Enable verbose logging output.")
@click.option(
    "--version",
    is_flag=True,
    callback=_print_version,
    expose_value=False,
    is_eager=True,
    help="Show the optest version and exit.",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Top level CLI group for optest."""

    bootstrap()
    ctx.obj = CliState(verbose=verbose)


@cli.command()
@click.option(
    "--plan",
    "--config",
    "plan_path",
    type=click.Path(exists=True),
    help="YAML plan file describing test cases.",
)
@click.option(
    "-o",
    "--op",
    "ops",
    multiple=True,
    help="Operator(s) to run. Overrides plan file when provided.",
)
@click.option("--backend", type=click.Choice(["gpu", "npu"]), help="Backend kind to target.")
@click.option("--chip", type=str, help="Specific chip identifier, e.g. a100 or ascend310b.")
@click.option("--dtype", "dtype_option", type=str, help="Comma separated dtype list per operator input.")
@click.option(
    "--shape",
    "shape_options",
    multiple=True,
    help="Shape override in the form input0=2x3x4. Repeat for multiple tensors.",
)
@click.option(
    "--attr",
    "attr_options",
    multiple=True,
    help="Operator attribute override key=value (supports scalars, tuples, YAML literals).",
)
@click.option("--generator", "generator_override", type=str, help="Override generator dotted path.")
@click.option("--reference", "reference_override", type=str, help="Override reference implementation dotted path.")
@click.option("--tolerance", "tolerance_text", type=str, help="Override tolerance as abs=1e-4,rel=1e-5")
@click.option("--seed", type=int, default=None, help="Random seed for data generation.")
@click.option("--fail-fast", is_flag=True, help="Stop on first failure.")
@click.option(
    "--report",
    "report_format",
    type=click.Choice(["terminal", "json"]),
    help="Report format override (terminal default).",
)
@click.option("--report-path", type=str, help="When --report json, write to this path.")
@click.option("--no-color", is_flag=True, help="Disable ANSI colors in terminal output.")
@click.pass_obj
def run(
    state: CliState,
    plan_path: Optional[str],
    ops: Tuple[str, ...],
    backend: Optional[str],
    chip: Optional[str],
    dtype_option: Optional[str],
    shape_options: Tuple[str, ...],
    attr_options: Tuple[str, ...],
    generator_override: Optional[str],
    reference_override: Optional[str],
    tolerance_text: Optional[str],
    seed: Optional[int],
    fail_fast: bool,
    report_format: Optional[str],
    report_path: Optional[str],
    no_color: bool,
) -> None:
    """Execute operator test cases defined via CLI or plan files."""

    dtype_override = _parse_dtype_option(dtype_option)
    shape_overrides = _parse_shape_options(shape_options)
    attr_overrides = _parse_attr_options(attr_options)
    tolerance_override = _parse_tolerance_option(tolerance_text)
    run_options = RunOptions(
        ops=ops,
        plan_path=plan_path,
        dtype_override=dtype_override,
        shape_overrides=shape_overrides,
        attribute_overrides=attr_overrides,
        backend=backend,
        chip=chip,
        seed=seed,
        fail_fast=fail_fast,
        generator_override=generator_override,
        reference_override=reference_override,
        tolerance_override=tolerance_override,
        report_format=report_format,
        report_path=report_path,
        color=False if no_color else None,
    )
    try:
        plan = build_execution_plan(run_options)
    except Exception as exc:  # pragma: no cover - CLI error translation
        raise click.ClickException(str(exc)) from exc

    report_manager = _create_report_manager(plan)
    runner = TestRunner(seed=plan.settings.seed, fail_fast=plan.settings.fail_fast)
    try:
        report_manager.start(plan)
        results = runner.run(plan.cases, on_result=report_manager.handle_result)
        report_manager.complete(results)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if any(result.status != "passed" for result in results):
        raise click.exceptions.Exit(1)


def main(argv: Optional[list[str]] = None) -> int:
    """Program entry point for console_scripts shim."""

    argv = argv if argv is not None else sys.argv[1:]
    try:
        cli.main(args=argv, prog_name="optest", standalone_mode=True)
    except click.ClickException as err:  # pragma: no cover - click handles display
        err.show()
        return err.exit_code
    except SystemExit as exc:  # click may raise exit code
        return int(exc.code)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


def _parse_dtype_option(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    if not value:
        return None
    entries = [part.strip() for part in value.split(",") if part.strip()]
    if not entries:
        return None
    return tuple(entries)


def _parse_shape_options(specs: Tuple[str, ...]) -> Dict[str, Tuple[int, ...]]:
    shapes: Dict[str, Tuple[int, ...]] = {}
    for spec in specs:
        key, raw = _split_assignment(spec)
        shapes[key] = _parse_shape_dims(raw)
    return shapes


def _parse_shape_dims(raw: str) -> Tuple[int, ...]:
    normalized = raw.replace("X", "x").replace(",", "x")
    parts = [part.strip() for part in normalized.split("x") if part.strip()]
    if not parts:
        raise click.BadParameter(f"Invalid shape specification '{raw}'")
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:  # pragma: no cover - user input error
        raise click.BadParameter(f"Non-integer dimension in shape '{raw}'") from exc


def _parse_attr_options(specs: Tuple[str, ...]) -> Dict[str, object]:
    attrs: Dict[str, object] = {}
    for spec in specs:
        key, raw = _split_assignment(spec)
        attrs[key] = _parse_attr_value(raw)
    return attrs


def _parse_attr_value(raw: str) -> object:
    text = raw.strip()
    if not text:
        raise click.BadParameter("Attribute values cannot be empty")
    parts = [part for part in text.replace("X", "x").split("x") if part]
    if len(parts) > 1 and _all_int(parts):
        return tuple(int(part) for part in parts)
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _all_int(parts: list[str]) -> bool:
    try:
        [int(part) for part in parts]
        return True
    except ValueError:
        return False


def _parse_tolerance_option(text: Optional[str]) -> Optional[Tolerance]:
    if not text:
        return None
    entries = [segment.strip() for segment in text.split(",") if segment.strip()]
    data: Dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise click.BadParameter("Tolerance entries must be key=value")
        key, value = entry.split("=", 1)
        try:
            data[key.strip()] = float(value)
        except ValueError as exc:
            raise click.BadParameter(f"Invalid tolerance value '{value}'") from exc
    return Tolerance.from_mapping(data)


def _split_assignment(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise click.BadParameter(f"Expected key=value format, got '{spec}'")
    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise click.BadParameter(f"Malformed specification '{spec}'")
    return key, value


def _create_report_manager(plan: ExecutionPlan) -> ReportManager:
    reporters = [TerminalReporter(use_color=plan.settings.color)]
    if plan.settings.report_format == "json":
        path = plan.settings.report_path or _default_report_path()
        reporters.append(JsonReporter(path))
    return ReportManager(reporters)


def _default_report_path() -> str:
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return str(Path.cwd() / f"optest_report_{timestamp}.json")
