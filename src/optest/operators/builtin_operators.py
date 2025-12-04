"""Built-in operators: descriptors plus NumPy reference implementations."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from optest.core import OperatorDescriptor, Tolerance

ArraySeq = Sequence[np.ndarray]
AttrMap = Mapping[str, object]

REF_PREFIX = "optest.operators.builtin_operators"

COMMON_FLOAT_DTYPES = ("float32", "float16")
ELEMENTWISE_INT_DTYPES = ("int8", "int16", "int32")
BOOL_DTYPES = ("bool",)

UNARY_FLOAT_DTYPES = tuple((dtype,) for dtype in COMMON_FLOAT_DTYPES)
UNARY_ELEMENTWISE_DTYPES = UNARY_FLOAT_DTYPES + tuple((dtype,) for dtype in ELEMENTWISE_INT_DTYPES)
BINARY_FLOAT_DTYPES = tuple((dtype, dtype) for dtype in COMMON_FLOAT_DTYPES)
BINARY_INT_DTYPES = tuple((dtype, dtype) for dtype in ELEMENTWISE_INT_DTYPES)
BINARY_ELEMENTWISE_DTYPES = BINARY_FLOAT_DTYPES + BINARY_INT_DTYPES
BINARY_LOGICAL_DTYPES = BINARY_ELEMENTWISE_DTYPES + tuple((dtype, dtype) for dtype in BOOL_DTYPES)

GEMM_DTYPES = BINARY_FLOAT_DTYPES + (("int8", "int8"),)
CONV_DTYPES = tuple((dtype, dtype) for dtype in COMMON_FLOAT_DTYPES)

POOL_DTYPES = UNARY_FLOAT_DTYPES
ACTIVATION_DTYPES = UNARY_ELEMENTWISE_DTYPES
REDUCTION_DTYPES = UNARY_ELEMENTWISE_DTYPES
REDUCTION_FLOAT_DTYPES = UNARY_FLOAT_DTYPES


class BuiltinOperator:
    """Base class for built-in operators."""

    name: str
    category: str
    num_inputs: int
    dtype_variants: tuple
    attribute_names: tuple = ()
    description: str = ""
    tags: tuple = ()
    default_tolerance: Tolerance = Tolerance()

    @classmethod
    def reference_path(cls) -> str:
        return f"{REF_PREFIX}:{cls.__name__}.run"

    @classmethod
    def descriptor(cls) -> OperatorDescriptor:
        return OperatorDescriptor(
            name=cls.name,
            category=cls.category,
            num_inputs=cls.num_inputs,
            dtype_variants=cls.dtype_variants,
            attribute_names=getattr(cls, "attribute_names", ()),
            description=cls.description,
            tags=getattr(cls, "tags", ()),
            default_tolerance=getattr(cls, "default_tolerance", Tolerance()),
            default_reference=cls.reference_path(),
        )


class ElementwiseAdd(BuiltinOperator):
    name = "elementwise_add"
    category = "tensor"
    num_inputs = 2
    dtype_variants = BINARY_ELEMENTWISE_DTYPES
    tags = ("elementwise", "tensor")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.add(a, b),)


class ElementwiseSub(BuiltinOperator):
    name = "elementwise_sub"
    category = "tensor"
    num_inputs = 2
    dtype_variants = BINARY_ELEMENTWISE_DTYPES
    tags = ("elementwise", "tensor")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.subtract(a, b),)


class ElementwiseMul(BuiltinOperator):
    name = "elementwise_mul"
    category = "tensor"
    num_inputs = 2
    dtype_variants = BINARY_ELEMENTWISE_DTYPES
    tags = ("elementwise", "tensor")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.multiply(a, b),)


class ElementwiseDiv(BuiltinOperator):
    name = "elementwise_div"
    category = "tensor"
    num_inputs = 2
    dtype_variants = BINARY_FLOAT_DTYPES
    tags = ("elementwise", "tensor")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.divide(a, b),)


class Equal(BuiltinOperator):
    name = "equal"
    category = "comparison"
    num_inputs = 2
    dtype_variants = BINARY_LOGICAL_DTYPES
    tags = ("comparison", "logical")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.equal(a, b),)


class Greater(BuiltinOperator):
    name = "greater"
    category = "comparison"
    num_inputs = 2
    dtype_variants = BINARY_LOGICAL_DTYPES
    tags = ("comparison", "logical")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.greater(a, b),)


class Less(BuiltinOperator):
    name = "less"
    category = "comparison"
    num_inputs = 2
    dtype_variants = BINARY_LOGICAL_DTYPES
    tags = ("comparison", "logical")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.less(a, b),)


class LessEqual(BuiltinOperator):
    name = "less_equal"
    category = "comparison"
    num_inputs = 2
    dtype_variants = BINARY_LOGICAL_DTYPES
    tags = ("comparison", "logical")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.less_equal(a, b),)


class GreaterEqual(BuiltinOperator):
    name = "greater_equal"
    category = "comparison"
    num_inputs = 2
    dtype_variants = BINARY_LOGICAL_DTYPES
    tags = ("comparison", "logical")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.greater_equal(a, b),)


class VectorDot(BuiltinOperator):
    name = "vector_dot"
    category = "linalg"
    num_inputs = 2
    dtype_variants = BINARY_ELEMENTWISE_DTYPES
    tags = ("vector", "dot", "reduction")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.sum(a * b, axis=-1),)


class VectorNorm(BuiltinOperator):
    name = "vector_norm"
    category = "linalg"
    num_inputs = 1
    dtype_variants = UNARY_FLOAT_DTYPES
    tags = ("vector", "reduction")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (np.linalg.norm(x),)


class VectorSum(BuiltinOperator):
    name = "vector_sum"
    category = "tensor"
    num_inputs = 1
    dtype_variants = UNARY_ELEMENTWISE_DTYPES
    tags = ("vector", "reduction")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (np.sum(x),)


class Gemm(BuiltinOperator):
    name = "gemm"
    category = "linalg"
    num_inputs = 2
    dtype_variants = GEMM_DTYPES
    attribute_names = ("m", "n", "k", "trans_a", "trans_b")
    tags = ("gemm", "matrix", "dense")
    default_tolerance = Tolerance(absolute=1e-4, relative=1e-5)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        trans_a = bool(attrs.get("trans_a", False))
        trans_b = bool(attrs.get("trans_b", False))
        if trans_a:
            a = np.swapaxes(a, -1, -2)
        if trans_b:
            b = np.swapaxes(b, -1, -2)
        result = np.matmul(a, b)
        return (result,)


class Matmul(BuiltinOperator):
    name = "matmul"
    category = "linalg"
    num_inputs = 2
    dtype_variants = GEMM_DTYPES
    tags = ("matmul", "matrix", "dense")
    default_tolerance = Tolerance(absolute=1e-4, relative=1e-5)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        a, b = inputs
        return (np.matmul(a, b),)


class Relu(BuiltinOperator):
    name = "relu"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    tags = ("activation", "relu")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (np.maximum(x, 0),)


class Sigmoid(BuiltinOperator):
    name = "sigmoid"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    tags = ("activation", "sigmoid")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (1 / (1 + np.exp(-x)),)


class Tanh(BuiltinOperator):
    name = "tanh"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    tags = ("activation", "tanh")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (np.tanh(x),)


class LeakyRelu(BuiltinOperator):
    name = "leaky_relu"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    attribute_names = ("alpha",)
    tags = ("activation", "relu")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        alpha = float(attrs.get("alpha", 0.01))
        return (np.where(x > 0, x, alpha * x),)


class Sinh(BuiltinOperator):
    name = "sinh"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    tags = ("activation", "sinh")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        return (np.sinh(x),)


class Softmax(BuiltinOperator):
    name = "softmax"
    category = "activation"
    num_inputs = 1
    dtype_variants = ACTIVATION_DTYPES
    attribute_names = ("axis",)
    tags = ("activation", "softmax")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        axis_value = attrs.get("axis", -1)
        axis = int(axis_value) if axis_value is not None else -1
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return (exp / np.sum(exp, axis=axis, keepdims=True),)


class ReduceSum(BuiltinOperator):
    name = "reduce_sum"
    category = "reduction"
    num_inputs = 1
    dtype_variants = REDUCTION_DTYPES
    attribute_names = ("axis", "keepdims")
    tags = ("reduction",)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        axis = attrs.get("axis", None)
        keepdims = bool(attrs.get("keepdims", False))
        return (np.sum(x, axis=axis, keepdims=keepdims),)


class ReduceMean(BuiltinOperator):
    name = "reduce_mean"
    category = "reduction"
    num_inputs = 1
    dtype_variants = REDUCTION_DTYPES
    attribute_names = ("axis", "keepdims")
    tags = ("reduction",)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        axis = attrs.get("axis", None)
        keepdims = bool(attrs.get("keepdims", False))
        return (np.mean(x, axis=axis, keepdims=keepdims),)


class BroadcastTo(BuiltinOperator):
    name = "broadcast_to"
    category = "tensor"
    num_inputs = 1
    dtype_variants = REDUCTION_FLOAT_DTYPES
    attribute_names = ("shape",)
    tags = ("broadcast",)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        (x,) = inputs
        shape = attrs.get("shape")
        if shape is None:
            raise ValueError("broadcast_to requires 'shape' attribute")
        target_shape = tuple(int(dim) for dim in shape)
        return (np.broadcast_to(x, target_shape),)


class MaxPool2d(BuiltinOperator):
    name = "maxpool2d"
    category = "pooling"
    num_inputs = 1
    dtype_variants = POOL_DTYPES
    attribute_names = ("kernel_size", "stride", "padding")
    tags = ("pool", "maxpool")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        return (pool2d(inputs[0], attrs, mode="max"),)


class AvgPool2d(BuiltinOperator):
    name = "avgpool2d"
    category = "pooling"
    num_inputs = 1
    dtype_variants = POOL_DTYPES
    attribute_names = ("kernel_size", "stride", "padding")
    tags = ("pool", "avgpool")

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        return (pool2d(inputs[0], attrs, mode="avg"),)


class Conv2d(BuiltinOperator):
    name = "conv2d"
    category = "convolution"
    num_inputs = 2
    dtype_variants = CONV_DTYPES
    attribute_names = ("stride", "dilation", "groups", "padding")
    tags = ("conv2d", "convolution")
    default_tolerance = Tolerance(absolute=1e-3, relative=1e-3)

    @staticmethod
    def run(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
        x, weight = inputs
        stride = _pair(attrs.get("stride"), default=(1, 1))
        dilation = _pair(attrs.get("dilation"), default=(1, 1))
        groups = int(attrs.get("groups", 1) or 1)
        padding = attrs.get("padding", (0, 0))

        x = np.asarray(x)
        weight = np.asarray(weight)

        pad_top, pad_bottom, pad_left, pad_right = _parse_padding(
            padding,
            input_hw=(x.shape[2], x.shape[3]),
            stride=stride,
            dilation=dilation,
            kernel_hw=(weight.shape[2], weight.shape[3]),
        )
        x_padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )

        n, c_in, h, w = x_padded.shape
        out_channels = weight.shape[0]
        c_in_per_group = c_in // groups
        c_out_per_group = out_channels // groups
        kernel_h, kernel_w = weight.shape[2:]
        dilated_kernel_h = (kernel_h - 1) * dilation[0] + 1
        dilated_kernel_w = (kernel_w - 1) * dilation[1] + 1
        out_h = (h - dilated_kernel_h) // stride[0] + 1
        out_w = (w - dilated_kernel_w) // stride[1] + 1

        output = np.zeros((n, out_channels, out_h, out_w), dtype=x.dtype)
        for batch in range(n):
            for group in range(groups):
                in_offset = group * c_in_per_group
                out_offset = group * c_out_per_group
                x_slice = x_padded[batch, in_offset : in_offset + c_in_per_group]
                w_slice = weight[out_offset : out_offset + c_out_per_group]
                for oc in range(c_out_per_group):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            h_start = oh * stride[0]
                            w_start = ow * stride[1]
                            accum = 0
                            for ic in range(c_in_per_group):
                                for dh in range(kernel_h):
                                    for dw in range(kernel_w):
                                        accum += w_slice[oc, ic, dh, dw] * x_slice[
                                            ic, h_start + dh * dilation[0], w_start + dw * dilation[1]
                                        ]
                            output[batch, oc + out_offset, oh, ow] = accum
        return (output,)


def pool2d(x: np.ndarray, attrs: AttrMap, mode: str) -> np.ndarray:
    kernel_size = _pair(attrs.get("kernel_size"), default=(2, 2))
    stride = _pair(attrs.get("stride"), default=kernel_size)
    padding = attrs.get("padding", (0, 0))
    pad_top, pad_bottom, pad_left, pad_right = _parse_padding(
        padding,
        input_hw=(x.shape[2], x.shape[3]),
        stride=stride,
        dilation=(1, 1),
        kernel_hw=kernel_size,
    )
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )
    n, c, h, w = x_padded.shape
    out_h = (h - kernel_size[0]) // stride[0] + 1
    out_w = (w - kernel_size[1]) // stride[1] + 1
    output = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for batch in range(n):
        for channel in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]
                    window = x_padded[
                        batch,
                        channel,
                        h_start : h_start + kernel_size[0],
                        w_start : w_start + kernel_size[1],
                    ]
                    if mode == "max":
                        output[batch, channel, oh, ow] = np.max(window)
                    else:
                        output[batch, channel, oh, ow] = np.mean(window)
    return output


def _pair(value, default=(1, 1)) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise TypeError("Expected int or pair for kernel/stride/dilation")


def _parse_padding(
    padding,
    *,
    input_hw: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    kernel_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    if padding is None or padding == "valid":
        return (0, 0, 0, 0)
    if isinstance(padding, str):
        padding = padding.lower()
        if padding == "same":
            return _same_padding(input_hw, stride, dilation, kernel_hw)
        raise ValueError(f"Unsupported padding string '{padding}'")
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if isinstance(padding, (tuple, list)):
        values = tuple(int(v) for v in padding)
        if len(values) == 2:
            return (values[0], values[0], values[1], values[1])
        if len(values) == 4:
            return values  # (top, bottom, left, right)
    raise TypeError("Unsupported padding specification")


def _same_padding(
    input_hw: tuple[int, int],
    stride: tuple[int, int],
    dilation: tuple[int, int],
    kernel_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    pad_pairs: list[tuple[int, int]] = []
    for dim, strd, dil, kernel in zip(input_hw, stride, dilation, kernel_hw):
        effective = (kernel - 1) * dil + 1
        out_dim = math.ceil(dim / strd)
        pad_needed = max((out_dim - 1) * strd + effective - dim, 0)
        pad_before = pad_needed // 2
        pad_after = pad_needed - pad_before
        pad_pairs.append((pad_before, pad_after))
    (top, bottom), (left, right) = pad_pairs
    return (top, bottom, left, right)


BUILTIN_OPERATOR_CLASSES = (
    ElementwiseAdd,
    ElementwiseSub,
    ElementwiseMul,
    ElementwiseDiv,
    Equal,
    Greater,
    Less,
    LessEqual,
    GreaterEqual,
    VectorDot,
    VectorNorm,
    VectorSum,
    Gemm,
    Matmul,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    Relu,
    Sigmoid,
    Tanh,
    LeakyRelu,
    Sinh,
    Softmax,
    ReduceSum,
    ReduceMean,
    BroadcastTo,
)
