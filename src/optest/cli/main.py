"""CLI entry point for optest."""
from __future__ import annotations

import sys
from typing import Optional, Tuple

import click

from optest import __version__, bootstrap
from optest.plan import PlanOptions, load_plan, run_plan


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
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="YAML plan file (new format).",
)
@click.option("--backend", type=str, help="Backend type to run (overrides plan when multiple are present).")
@click.option("--chip", type=str, help="Chip identifier to run (overrides plan when multiple are present).")
@click.option("--cases", "case_filters", type=str, help="Comma-separated case filters (supports globs).")
@click.option("--tags", "tag_filters", type=str, help="Comma-separated tags to include.")
@click.option("--skip-tags", "skip_tag_filters", type=str, help="Comma-separated tags to skip.")
@click.option("--priority-max", type=int, help="Maximum priority to run.")
@click.option("--cache", "cache_policy", type=click.Choice(["reuse", "regen"]), help="Cache policy override.")
@click.option("--list", "list_only", is_flag=True, help="List matched cases without running.")
@click.option(
    "--report",
    "report_format",
    type=click.Choice(["terminal", "json"]),
    default="terminal",
    show_default=True,
    help="Report format (terminal by default).",
)
@click.option("--report-path", type=str, help="When --report json, write to this path.")
@click.option("--no-color", is_flag=True, help="Disable ANSI colors in terminal output.")
@click.pass_obj
def run(
    state: CliState,
    plan_path: Optional[str],
    backend: Optional[str],
    chip: Optional[str],
    report_format: Optional[str],
    report_path: Optional[str],
    no_color: bool,
    case_filters: Optional[str],
    tag_filters: Optional[str],
    skip_tag_filters: Optional[str],
    priority_max: Optional[int],
    cache_policy: Optional[str],
    list_only: bool,
) -> None:
    """Execute operator test cases defined via CLI or plan files."""

    assert plan_path  # required by click
    options = PlanOptions(
        backend=backend,
        chip=chip,
        cases=_split_csv(case_filters),
        tags=_split_csv(tag_filters),
        skip_tags=_split_csv(skip_tag_filters),
        priority_max=priority_max,
        cache=cache_policy,
        list_only=list_only,
    )
    try:
        plan = load_plan(plan_path)
        exit_code = run_plan(
            plan,
            options,
            report_format=report_format or "terminal",
            report_path=report_path,
            use_color=not no_color,
        )
    except Exception as exc:  # pragma: no cover - CLI error translation
        raise click.ClickException(str(exc)) from exc
    raise click.exceptions.Exit(exit_code)


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


def _split_csv(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(parts)


def _split_csv(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return tuple()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(parts)
