import yaml

from optest.core import BackendTarget, Tolerance
from optest.plans.loader import build_execution_plan
from optest.plans.types import RunOptions


def test_build_execution_plan_with_cli_overrides() -> None:
    run_options = RunOptions(
        ops=("elementwise_add",),
        dtype_override=("float32", "float32"),
        shape_overrides={"input0": (2, 2), "input1": (2, 2)},
        backend="gpu",
        seed=123,
        fail_fast=True,
    )
    plan = build_execution_plan(run_options)
    assert plan.settings.backend == "gpu"
    assert plan.settings.seed == 123
    assert plan.settings.fail_fast is True
    case = plan.cases[0]
    assert case.descriptor.name == "elementwise_add"
    assert case.dtype_spec == ("float32", "float32")
    assert case.shapes["input0"] == (2, 2)
    assert case.backend.kind == "gpu"
    assert isinstance(case.tolerance, Tolerance)


def test_plan_loader_expands_shape_and_dtype_variants(tmp_path) -> None:
    plan_data = {
        "backend": "npu",
        "chip": "b",
        "cases": [
            {
                "op": "elementwise_add",
                "dtypes": [["float16", "float16"], ["float32", "float32"]],
                "shapes": [
                    {"input0": [1, 2], "input1": [1, 2], "output0": [1, 2]},
                    {"input0": [2, 2], "input1": [2, 2], "output0": [2, 2]},
                ],
                "attributes": {"backend_config": {"ascend": {"command": ["true"]}}},
            }
        ],
    }
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(yaml.safe_dump(plan_data), encoding="utf-8")
    options = RunOptions(plan_path=str(plan_path))
    plan = build_execution_plan(options)
    assert len(plan.cases) == 4
    assert [case.dtype_spec for case in plan.cases] == [
        ("float16", "float16"),
        ("float16", "float16"),
        ("float32", "float32"),
        ("float32", "float32"),
    ]
    assert [case.shapes["input0"] for case in plan.cases] == [(1, 2), (2, 2), (1, 2), (2, 2)]
    assert all(case.backend == BackendTarget(kind="npu", chip="b") for case in plan.cases)
