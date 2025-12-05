from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

from optest.plan import PlanOptions, load_plan, run_plan


def _write_plan(tmp_path: Path) -> Path:
    script = tmp_path / "adder.py"
    script.write_text(
        textwrap.dedent(
            """
            import argparse
            import numpy as np

            parser = argparse.ArgumentParser()
            parser.add_argument("--input0", required=True)
            parser.add_argument("--input1", required=True)
            parser.add_argument("--output0", required=True)
            parser.add_argument("--dtype", required=True)
            parser.add_argument("--shape", required=True)
            args = parser.parse_args()

            shape = tuple(int(part) for part in args.shape.replace("X", "x").split("x") if part)
            a = np.fromfile(args.input0, dtype=args.dtype).reshape(shape)
            b = np.fromfile(args.input1, dtype=args.dtype).reshape(shape)
            np.add(a, b).astype(args.dtype).tofile(args.output0)
            """
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.yaml"
    plan.write_text(
        textwrap.dedent(
            f"""
            operator: vector_add
            description: simple add
            inputs: ["data/in0.bin", "data/in1.bin"]
            outputs: ["out/out0.bin"]
            generator:
              name: builtin.ones
            assertion:
              name: builtin.elementwise_add
              rtol: 0
              atol: 0
            backends:
              - type: cuda
                chip: local
                workdir: {tmp_path.as_posix()}
                command: ["python", "{script.as_posix()}", "--input0", "{{input0}}", "--input1", "{{input1}}", "--output0", "{{output0}}", "--dtype", "{{dtype}}", "--shape", "{{shape}}"]
            cases:
              - name: smoke
                dtypes: [float32, float32]
                shapes:
                  - inputs: [[1, 4], [1, 4]]
                    outputs: [[1, 4]]
            """
        ),
        encoding="utf-8",
    )
    return plan


def test_run_plan_end_to_end(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_plan(str(plan_path))
    exit_code = run_plan(plan, PlanOptions(backend="cuda", chip="local"))
    assert exit_code == 0
    output_path = tmp_path / "out" / "out0.bin"
    data = np.fromfile(output_path, dtype="float32")
    assert np.allclose(data, np.full((4,), 2.0, dtype="float32"))


def test_builtin_assertions_and_backend_templates(tmp_path: Path) -> None:
    runner_script = tmp_path / "relu_runner.py"
    runner_script.write_text(
        textwrap.dedent(
            """
            import argparse
            import os
            import pathlib
            import numpy as np

            parser = argparse.ArgumentParser()
            parser.add_argument("--input0", required=True)
            parser.add_argument("--output0", required=True)
            parser.add_argument("--dtype", required=True)
            parser.add_argument("--shape", required=True)
            args = parser.parse_args()

            expected_env = f"{os.environ.get('BACKEND_NAME')}-{os.environ.get('CHIP_NAME')}"
            actual_env = os.environ.get("PLAN_ENV")
            if actual_env != expected_env:
                raise SystemExit(f"unexpected PLAN_ENV: {actual_env!r} vs {expected_env!r}")

            shape = tuple(int(part) for part in args.shape.replace("X", "x").split("x") if part)
            data = np.fromfile(args.input0, dtype=args.dtype).reshape(shape)
            np.maximum(data, 0).astype(args.dtype).tofile(args.output0)
            pathlib.Path("seen_env.txt").write_text(actual_env or "", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )

    prepare_script = tmp_path / "prepare.py"
    prepare_script.write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("case")
            parser.add_argument("shape")
            args = parser.parse_args()

            Path("prep.txt").write_text(f"{args.case}-{args.shape}", encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )

    cleanup_script = tmp_path / "cleanup.py"
    cleanup_script.write_text(
        textwrap.dedent(
            """
            import argparse
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("dtype")
            args = parser.parse_args()

            Path("cleanup.txt").write_text(args.dtype, encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )

    plan = tmp_path / "plan_relu.yaml"
    plan.write_text(
        textwrap.dedent(
            f"""
            operator: relu_op
            description: relu with templated backend values
            inputs: ["inputs/in0.bin"]
            outputs: ["outputs/out0.bin"]
            generator:
              name: builtin.random
              seed: 0
            assertion:
              name: builtin.relu
            backends:
              - type: cuda
                chip: local
                workdir: {tmp_path.as_posix()}
                env:
                  PLAN_ENV: "{{backend}}-{{chip}}"
                  BACKEND_NAME: "{{backend}}"
                  CHIP_NAME: "{{chip}}"
                prepare:
                  - ["python", "{prepare_script.as_posix()}", "{{case}}", "{{shape}}"]
                cleanup:
                  - ["python", "{cleanup_script.as_posix()}", "{{dtype}}"]
                command: ["python", "{runner_script.as_posix()}", "--input0", "{{input0}}", "--output0", "{{output0}}", "--dtype", "{{dtype}}", "--shape", "{{shape}}"]
            cases:
              - name: relu
                dtypes: [float32]
                shapes:
                  - inputs: [[1, 4]]
                    outputs: [[1, 4]]
            """
        ),
        encoding="utf-8",
    )

    loaded = load_plan(str(plan))
    exit_code = run_plan(loaded, PlanOptions(backend="cuda", chip="local"))
    assert exit_code == 0

    assert (tmp_path / "prep.txt").read_text(encoding="utf-8") == "relu-1x4"
    assert (tmp_path / "cleanup.txt").read_text(encoding="utf-8") == "float32"
    assert (tmp_path / "seen_env.txt").read_text(encoding="utf-8") == "cuda-local"
