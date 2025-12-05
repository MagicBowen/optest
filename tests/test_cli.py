from click.testing import CliRunner

from optest.cli.main import cli


def test_cli_help_short_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["-h"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "run" in result.output


def test_cli_run_plan(tmp_path) -> None:
    script = tmp_path / "adder.py"
    script.write_text(
        "import numpy as np;import argparse;"
        "p=argparse.ArgumentParser();"
        "p.add_argument('--input0');p.add_argument('--input1');p.add_argument('--output0');"
        "p.add_argument('--dtype');p.add_argument('--shape');"
        "a=p.parse_args();"
        "shape=tuple(int(x) for x in a.shape.split('x') if x);"
        "x=np.fromfile(a.input0,dtype=a.dtype).reshape(shape);"
        "y=np.fromfile(a.input1,dtype=a.dtype).reshape(shape);"
        "np.add(x,y).astype(a.dtype).tofile(a.output0)",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        f"""
operator: vec_add
inputs: ["in0.bin", "in1.bin"]
outputs: ["out0.bin"]
generator: {{name: builtin.ones}}
assertion: {{name: builtin.elementwise_add, rtol: 0, atol: 0}}
backends:
  - type: cuda
    chip: local
    workdir: {tmp_path.as_posix()}
    command: ["python", "{script.as_posix()}", "--input0", "{{input0}}", "--input1", "{{input1}}", "--output0", "{{output0}}", "--dtype", "{{dtype}}", "--shape", "{{shape}}"]
cases:
  - name: smoke
    dtypes: [float32, float32]
    shapes:
      - inputs: [[1, 2], [1, 2]]
        outputs: [[1, 2]]
    tags: ["smoke"]
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--plan",
            str(plan),
            "--backend",
            "cuda",
            "--chip",
            "local",
            "--tags",
            "smoke",
        ],
    )
    assert result.exit_code == 0, result.output


def test_cli_json_report(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    script = tmp_path / "adder.py"
    script.write_text(
        "import numpy as np;import argparse;"
        "p=argparse.ArgumentParser();"
        "p.add_argument('--input0');p.add_argument('--input1');p.add_argument('--output0');"
        "p.add_argument('--dtype');p.add_argument('--shape');"
        "a=p.parse_args();"
        "shape=tuple(int(x) for x in a.shape.split('x') if x);"
        "x=np.fromfile(a.input0,dtype=a.dtype).reshape(shape);"
        "y=np.fromfile(a.input1,dtype=a.dtype).reshape(shape);"
        "np.add(x,y).astype(a.dtype).tofile(a.output0)",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        f"""
operator: vec_add
inputs: ["in0.bin", "in1.bin"]
outputs: ["out0.bin"]
generator: {{name: builtin.ones}}
assertion: {{name: builtin.elementwise_add, rtol: 0, atol: 0}}
backends:
  - type: cuda
    chip: local
    workdir: {tmp_path.as_posix()}
    command: ["python", "{script.as_posix()}", "--input0", "{{input0}}", "--input1", "{{input1}}", "--output0", "{{output0}}", "--dtype", "{{dtype}}", "--shape", "{{shape}}"]
cases:
  - name: smoke
    dtypes: [float32, float32]
    shapes:
      - inputs: [[1, 2], [1, 2]]
        outputs: [[1, 2]]
""",
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--plan",
            str(plan),
            "--backend",
            "cuda",
            "--chip",
            "local",
            "--report",
            "json",
            "--report-path",
            str(report_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
