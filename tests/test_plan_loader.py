from op_tester.core import Tolerance
from op_tester.plans.loader import build_execution_plan
from op_tester.plans.types import RunOptions


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
