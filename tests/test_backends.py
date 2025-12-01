import pytest

from optest.backends.ascend import _extract_config
from optest.core import BackendTarget, OperatorDescriptor, TestCase, Tolerance


def _make_case(config) -> TestCase:
    descriptor = OperatorDescriptor(
        name="dummy",
        category="test",
        num_inputs=1,
        dtype_variants=(("float32",),),
    )
    return TestCase(
        descriptor=descriptor,
        dtype_spec=("float32",),
        shapes={"input0": (2,)},
        backend=BackendTarget(kind="npu"),
        tolerance=Tolerance(),
        attributes={"backend_config": config},
    )


def test_ascend_backend_requires_inputs_and_outputs() -> None:
    config = {"ascend": {"workdir": ".", "command": ["true"], "inputs": []}}
    case = _make_case(config)
    with pytest.raises(ValueError):
        _extract_config(case)
