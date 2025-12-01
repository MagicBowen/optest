from optest.core import BackendTarget, OperatorDescriptor, TestCase, Tolerance
from optest.core.comparator import compare_outputs

import numpy as np

def _make_case() -> TestCase:
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
        backend=BackendTarget(kind="gpu"),
        tolerance=Tolerance(absolute=1e-4, relative=1e-4),
    )


def test_compare_outputs_passes() -> None:
    case = _make_case()
    actual = (np.array([1.0, 2.0], dtype=np.float32),)
    expected = (np.array([1.0, 2.0], dtype=np.float32),)
    result = compare_outputs(case, actual, expected)
    assert result.passed
    assert result.tensors[0].mismatched == 0


def test_compare_outputs_detects_failure() -> None:
    case = _make_case()
    actual = (np.array([1.0, 3.0], dtype=np.float32),)
    expected = (np.array([1.0, 2.0], dtype=np.float32),)
    result = compare_outputs(case, actual, expected)
    assert not result.passed
    assert result.tensors[0].mismatched == 1
