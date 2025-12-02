"""Reference implementations for built-in operators using NumPy only."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


ArraySeq = Sequence[np.ndarray]
AttrMap = Mapping[str, object]


def elementwise_add_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.add(a, b),)


def elementwise_sub_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.subtract(a, b),)


def elementwise_mul_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.multiply(a, b),)


def elementwise_div_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.divide(a, b),)


def equal_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.equal(a, b),)


def greater_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.greater(a, b),)


def less_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.less(a, b),)


def less_equal_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.less_equal(a, b),)


def greater_equal_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.greater_equal(a, b),)


def vector_dot_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.sum(a * b, axis=-1),)


def vector_norm_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (np.linalg.norm(x),)


def vector_sum_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (np.sum(x),)


def gemm_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    trans_a = bool(attrs.get("trans_a", False))
    trans_b = bool(attrs.get("trans_b", False))
    if trans_a:
        a = np.swapaxes(a, -1, -2)
    if trans_b:
        b = np.swapaxes(b, -1, -2)
    result = np.matmul(a, b)
    return (result,)


def matmul_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    a, b = inputs
    return (np.matmul(a, b),)


def relu_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (np.maximum(x, 0),)


def sigmoid_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (1 / (1 + np.exp(-x)),)


def tanh_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (np.tanh(x),)


def leaky_relu_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    alpha = float(attrs.get("alpha", 0.01))
    return (np.where(x > 0, x, alpha * x),)


def sinh_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    return (np.sinh(x),)


def softmax_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    axis_value = attrs.get("axis", -1)
    axis = int(axis_value) if axis_value is not None else -1
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True),)


def reduce_sum_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    axis = attrs.get("axis", None)
    keepdims = bool(attrs.get("keepdims", False))
    return (np.sum(x, axis=axis, keepdims=keepdims),)


def reduce_mean_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    axis = attrs.get("axis", None)
    keepdims = bool(attrs.get("keepdims", False))
    return (np.mean(x, axis=axis, keepdims=keepdims),)


def broadcast_to_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    (x,) = inputs
    shape = attrs.get("shape")
    if shape is None:
        raise ValueError("broadcast_to requires 'shape' attribute")
    target_shape = tuple(int(dim) for dim in shape)
    return (np.broadcast_to(x, target_shape),)


def maxpool2d_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    return (pool2d(inputs[0], attrs, mode="max"),)


def avgpool2d_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
    return (pool2d(inputs[0], attrs, mode="avg"),)


def conv2d_reference(inputs: ArraySeq, attrs: AttrMap) -> ArraySeq:
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
                        acc = 0.0
                        for ic in range(c_in_per_group):
                            for kh in range(kernel_h):
                                for kw in range(kernel_w):
                                    in_h = h_start + kh * dilation[0]
                                    in_w = w_start + kw * dilation[1]
                                    acc += (
                                        x_slice[ic, in_h, in_w]
                                        * w_slice[oc, ic, kh, kw]
                                    )
                        output[batch, out_offset + oc, oh, ow] = acc
    return (output,)


def pool2d(x: np.ndarray, attrs: AttrMap, mode: str) -> np.ndarray:
    kernel = _pair(attrs.get("kernel_size"), default=(2, 2))
    stride = _pair(attrs.get("stride"), default=kernel)
    padding = attrs.get("padding", (0, 0))
    pad_top, pad_bottom, pad_left, pad_right = _parse_padding(
        padding,
        input_hw=(x.shape[2], x.shape[3]),
        stride=stride,
        dilation=(1, 1),
        kernel_hw=kernel,
    )
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
    )
    n, c, h, w = x_padded.shape
    out_h = (h - kernel[0]) // stride[0] + 1
    out_w = (w - kernel[1]) // stride[1] + 1
    output = np.zeros((n, c, out_h, out_w), dtype=x.dtype)
    for batch in range(n):
        for channel in range(c):
            for oh in range(out_h):
                for ow in range(out_w):
                    h_start = oh * stride[0]
                    w_start = ow * stride[1]
                    region = x_padded[
                        batch,
                        channel,
                        h_start : h_start + kernel[0],
                        w_start : w_start + kernel[1],
                    ]
                    if mode == "max":
                        output[batch, channel, oh, ow] = np.max(region)
                    else:
                        output[batch, channel, oh, ow] = np.mean(region)
    return output


def _pair(value: object | None, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected a pair for attribute value")
        return int(value[0]), int(value[1])
    raise TypeError("Unsupported attribute format for tuple conversion")


def _parse_padding(
    padding: object,
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
