from click.testing import CliRunner

from optest.cli.main import cli


def test_cli_run_with_plan() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--op", "relu", "--no-color"])
    assert result.exit_code == 0, result.output


def test_cli_help_short_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["-h"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output
