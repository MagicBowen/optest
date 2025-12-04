import numpy as np
import numpy.testing as npt

from optest.operators import builtin_operators as ops


def test_logical_comparison_references() -> None:
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 0, 4], dtype=np.int32)
    (eq,) = ops.Equal.run((a, b), {})
    (gt,) = ops.Greater.run((a, b), {})
    (lt,) = ops.Less.run((a, b), {})
    (le,) = ops.LessEqual.run((a, b), {})
    (ge,) = ops.GreaterEqual.run((a, b), {})
    npt.assert_array_equal(eq, np.array([True, False, False]))
    npt.assert_array_equal(gt, np.array([False, True, False]))
    npt.assert_array_equal(lt, np.array([False, False, True]))
    npt.assert_array_equal(le, np.array([True, False, True]))
    npt.assert_array_equal(ge, np.array([True, True, False]))


def test_vector_references_cover_common_calculations() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.5, -1.0], [0.5, 2.0]], dtype=np.float32)
    (dot,) = ops.VectorDot.run((a, b), {})
    (norm,) = ops.VectorNorm.run((a,), {})
    (summed,) = ops.VectorSum.run((a,), {})
    npt.assert_allclose(dot, np.array([-0.5, 9.5], dtype=np.float32))
    npt.assert_allclose(norm, np.linalg.norm(a))
    npt.assert_allclose(summed, np.sum(a))


def test_softmax_reference_respects_axis() -> None:
    x = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    (out_axis1,) = ops.Softmax.run((x,), {"axis": 1})
    npt.assert_allclose(out_axis1.sum(axis=1), np.array([1.0, 1.0]))


def test_matmul_reference_matches_numpy() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    (result,) = ops.Matmul.run((a, b), {})
    npt.assert_allclose(result, np.matmul(a, b))


def test_reduction_references() -> None:
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    (sum_all,) = ops.ReduceSum.run((x,), {})
    (mean_keep,) = ops.ReduceMean.run((x,), {"axis": 0, "keepdims": True})
    npt.assert_allclose(sum_all, np.sum(x))
    npt.assert_allclose(mean_keep, np.mean(x, axis=0, keepdims=True))


def test_broadcast_and_sinh_references() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    (broadcasted,) = ops.BroadcastTo.run((x,), {"shape": (2, 2)})
    (sinh_out,) = ops.Sinh.run((x,), {})
    npt.assert_allclose(broadcasted, np.broadcast_to(x, (2, 2)))
    npt.assert_allclose(sinh_out, np.sinh(x))
