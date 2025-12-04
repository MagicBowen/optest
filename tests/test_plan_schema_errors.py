from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from optest.plan.loader import load_plan


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "plan.yaml"
    path.write_text(textwrap.dedent(content), encoding="utf-8")
    return path


def test_schema_missing_required_fields(tmp_path: Path) -> None:
    plan_path = _write(
        tmp_path,
        """
        operator: add
        inputs: ["a.bin"]
        # outputs missing
        backends: []
        cases: []
        """,
    )
    with pytest.raises(ValueError) as exc:
        load_plan(str(plan_path))
    assert "outputs" in str(exc.value)


def test_duplicate_backend_rejected(tmp_path: Path) -> None:
    plan_path = _write(
        tmp_path,
        """
        operator: add
        inputs: ["a.bin"]
        outputs: ["b.bin"]
        backends:
          - type: cuda
            chip: local
            command: ["echo", "ok"]
          - type: cuda
            chip: local
            command: ["echo", "ok"]
        cases:
          - name: c
            dtypes: [float32]
            shapes:
              - inputs: [[1]]
                outputs: [[1]]
        """,
    )
    with pytest.raises(ValueError) as exc:
        load_plan(str(plan_path))
    assert "Duplicate backend" in str(exc.value)
