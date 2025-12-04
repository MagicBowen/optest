from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from optest.plan import PlanOptions, load_plan
from optest.plan import runner as plan_runner


def _write_plan(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_load_plan_basic(tmp_path: Path) -> None:
    plan_path = _write_plan(
        tmp_path,
        """
        operator: vector_add
        inputs: ["data/in0.bin", "data/in1.bin"]
        outputs: ["out/out0.bin"]
        backends:
          - type: cuda
            chip: local
            command: ["echo", "ok"]
        cases:
          - name: f32
            dtypes: [float32, float32]
            shapes:
              - inputs: [[1, 4], [1, 4]]
                outputs: [[1, 4]]
        """,
    )
    plan = load_plan(str(plan_path))
    assert plan.operator == "vector_add"
    assert plan.inputs == ("data/in0.bin", "data/in1.bin")
    assert plan.outputs == ("out/out0.bin",)
    assert len(plan.backends) == 1
    assert plan.backends[0].type == "cuda"
    assert plan.backends[0].chip == "local"
    assert len(plan.cases) == 1
    assert plan.cases[0].name == "f32"


def test_load_plan_validates_shapes(tmp_path: Path) -> None:
    plan_path = _write_plan(
        tmp_path,
        """
        operator: bad
        inputs: ["a.bin"]
        outputs: ["b.bin"]
        backends:
          - type: cuda
            chip: local
            command: ["echo", "ok"]
        cases:
          - name: bad_case
            dtypes: [float32, float32]
            shapes:
              - inputs: [[1], [1]]
                outputs: [[1]]
        """,
    )
    with pytest.raises(ValueError):
        load_plan(str(plan_path))


def test_resolve_cases_filters(tmp_path: Path) -> None:
    plan_path = _write_plan(
        tmp_path,
        """
        operator: vector_add
        inputs: ["a.bin", "b.bin"]
        outputs: ["c.bin"]
        tags: ["plan-tag"]
        backends:
          - type: cuda
            chip: local
            command: ["echo", "ok"]
            skip_cases: ["skipme"]
          - type: cann
            chip: ascend
            command: ["echo", "nope"]
        cases:
          - name: runme
            dtypes: [float32, float32]
            tags: ["smoke"]
            shapes:
              - inputs: [[1, 1], [1, 1]]
                outputs: [[1, 1]]
          - name: skipme
            dtypes: [float32, float32]
            shapes:
              - inputs: [[1, 1], [1, 1]]
                outputs: [[1, 1]]
        """,
    )
    plan = load_plan(str(plan_path))
    options = PlanOptions(backend="cuda", cases=("runme",), tags=("smoke",))
    resolved = plan_runner._resolve_cases(plan, options)
    assert len(resolved) == 1
    assert resolved[0].backend.type == "cuda"
    assert resolved[0].case.name == "runme"
