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
