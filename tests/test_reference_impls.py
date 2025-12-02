import numpy as np
import numpy.testing as npt

from optest.operators import reference_impls as refs


def test_logical_comparison_references() -> None:
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 0, 4], dtype=np.int32)
    (eq,) = refs.equal_reference((a, b), {})
    (gt,) = refs.greater_reference((a, b), {})
    (lt,) = refs.less_reference((a, b), {})
    (le,) = refs.less_equal_reference((a, b), {})
    (ge,) = refs.greater_equal_reference((a, b), {})
    npt.assert_array_equal(eq, np.array([True, False, False]))
    npt.assert_array_equal(gt, np.array([False, True, False]))
    npt.assert_array_equal(lt, np.array([False, False, True]))
    npt.assert_array_equal(le, np.array([True, False, True]))
    npt.assert_array_equal(ge, np.array([True, True, False]))


def test_vector_references_cover_common_calculations() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.5, -1.0], [0.5, 2.0]], dtype=np.float32)
    (dot,) = refs.vector_dot_reference((a, b), {})
    (norm,) = refs.vector_norm_reference((a,), {})
    (summed,) = refs.vector_sum_reference((b,), {})
    npt.assert_allclose(dot, np.array([-0.5, 9.5], dtype=np.float32))
    npt.assert_allclose(norm, np.linalg.norm(a))
    npt.assert_allclose(summed, np.sum(b))


def test_softmax_reference_respects_axis() -> None:
    x = np.array([[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float32)
    (out_axis1,) = refs.softmax_reference((x,), {"axis": 1})
    (out_default,) = refs.softmax_reference((x,), {})
    expected_axis1 = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    expected_default = np.exp(x - np.max(x, axis=-1, keepdims=True)) / np.sum(
        np.exp(x - np.max(x, axis=-1, keepdims=True)), axis=-1, keepdims=True
    )
    npt.assert_allclose(out_axis1, expected_axis1, rtol=1e-6, atol=0.0)
    npt.assert_allclose(out_default, expected_default, rtol=1e-6, atol=0.0)


def test_matmul_reference_matches_numpy() -> None:
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b = np.array([[5, 6], [7, 8]], dtype=np.float32)
    (result,) = refs.matmul_reference((a, b), {})
    npt.assert_allclose(result, np.matmul(a, b))


def test_reduction_references() -> None:
    x = np.array([[1, 2], [3, 4]], dtype=np.float32)
    (sum_all,) = refs.reduce_sum_reference((x,), {})
    (sum_axis0,) = refs.reduce_sum_reference((x,), {"axis": 0, "keepdims": True})
    (mean_axis1,) = refs.reduce_mean_reference((x,), {"axis": 1})
    npt.assert_allclose(sum_all, np.sum(x))
    npt.assert_allclose(sum_axis0, np.sum(x, axis=0, keepdims=True))
    npt.assert_allclose(mean_axis1, np.mean(x, axis=1))


def test_broadcast_and_sinh_references() -> None:
    x = np.array([1.0, 2.0], dtype=np.float32)
    (broadcasted,) = refs.broadcast_to_reference((x,), {"shape": (2, 2)})
    (sinh_out,) = refs.sinh_reference((x,), {})
    npt.assert_allclose(broadcasted, np.broadcast_to(x, (2, 2)))
    npt.assert_allclose(sinh_out, np.sinh(x))
