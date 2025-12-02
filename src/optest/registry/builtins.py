"""Built-in operator descriptors shipped with optest."""
from __future__ import annotations

from optest.core import OperatorDescriptor, Tolerance

REF = "optest.operators.reference_impls"

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

BUILTIN_DESCRIPTORS = (
    OperatorDescriptor(
        name="elementwise_add",
        category="tensor",
        num_inputs=2,
        dtype_variants=BINARY_ELEMENTWISE_DTYPES,
        description="Elementwise tensor addition for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_add_reference",
    ),
    OperatorDescriptor(
        name="elementwise_sub",
        category="tensor",
        num_inputs=2,
        dtype_variants=BINARY_ELEMENTWISE_DTYPES,
        description="Elementwise tensor subtraction for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_sub_reference",
    ),
    OperatorDescriptor(
        name="elementwise_mul",
        category="tensor",
        num_inputs=2,
        dtype_variants=BINARY_ELEMENTWISE_DTYPES,
        description="Elementwise tensor multiplication for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_mul_reference",
    ),
    OperatorDescriptor(
        name="elementwise_div",
        category="tensor",
        num_inputs=2,
        dtype_variants=BINARY_FLOAT_DTYPES,
        description="Elementwise tensor division for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_div_reference",
    ),
    OperatorDescriptor(
        name="equal",
        category="comparison",
        num_inputs=2,
        dtype_variants=BINARY_LOGICAL_DTYPES,
        description="Elementwise equality comparison producing boolean outputs.",
        tags=("comparison", "logical"),
        default_reference=f"{REF}:equal_reference",
    ),
    OperatorDescriptor(
        name="greater",
        category="comparison",
        num_inputs=2,
        dtype_variants=BINARY_LOGICAL_DTYPES,
        description="Elementwise greater-than comparison producing boolean outputs.",
        tags=("comparison", "logical"),
        default_reference=f"{REF}:greater_reference",
    ),
    OperatorDescriptor(
        name="less",
        category="comparison",
        num_inputs=2,
        dtype_variants=BINARY_LOGICAL_DTYPES,
        description="Elementwise less-than comparison producing boolean outputs.",
        tags=("comparison", "logical"),
        default_reference=f"{REF}:less_reference",
    ),
    OperatorDescriptor(
        name="less_equal",
        category="comparison",
        num_inputs=2,
        dtype_variants=BINARY_LOGICAL_DTYPES,
        description="Elementwise less-than-or-equal comparison producing boolean outputs.",
        tags=("comparison", "logical"),
        default_reference=f"{REF}:less_equal_reference",
    ),
    OperatorDescriptor(
        name="greater_equal",
        category="comparison",
        num_inputs=2,
        dtype_variants=BINARY_LOGICAL_DTYPES,
        description="Elementwise greater-than-or-equal comparison producing boolean outputs.",
        tags=("comparison", "logical"),
        default_reference=f"{REF}:greater_equal_reference",
    ),
    OperatorDescriptor(
        name="vector_dot",
        category="linalg",
        num_inputs=2,
        dtype_variants=BINARY_ELEMENTWISE_DTYPES,
        description="Vector dot product (last-dimension contraction).",
        tags=("vector", "dot", "reduction"),
        default_reference=f"{REF}:vector_dot_reference",
    ),
    OperatorDescriptor(
        name="vector_norm",
        category="linalg",
        num_inputs=1,
        dtype_variants=UNARY_FLOAT_DTYPES,
        description="Vector L2 norm reduction.",
        tags=("vector", "reduction"),
        default_reference=f"{REF}:vector_norm_reference",
    ),
    OperatorDescriptor(
        name="vector_sum",
        category="tensor",
        num_inputs=1,
        dtype_variants=UNARY_ELEMENTWISE_DTYPES,
        description="Vector/tensor sum reduction over all elements.",
        tags=("vector", "reduction"),
        default_reference=f"{REF}:vector_sum_reference",
    ),
    OperatorDescriptor(
        name="gemm",
        category="linalg",
        num_inputs=2,
        dtype_variants=GEMM_DTYPES,
        attribute_names=("m", "n", "k", "trans_a", "trans_b"),
        description="General matrix multiplication.",
        tags=("gemm", "matrix", "dense"),
        default_tolerance=Tolerance(absolute=1e-4, relative=1e-5),
        default_reference=f"{REF}:gemm_reference",
    ),
    OperatorDescriptor(
        name="matmul",
        category="linalg",
        num_inputs=2,
        dtype_variants=GEMM_DTYPES,
        description="General matrix multiplication supporting batched inputs.",
        tags=("matmul", "matrix", "dense"),
        default_tolerance=Tolerance(absolute=1e-4, relative=1e-5),
        default_reference=f"{REF}:matmul_reference",
    ),
    OperatorDescriptor(
        name="conv2d",
        category="convolution",
        num_inputs=2,
        dtype_variants=CONV_DTYPES,
        attribute_names=("stride", "padding", "dilation", "groups"),
        description="2D convolution with NCHW layout.",
        tags=("conv", "spatial"),
        default_tolerance=Tolerance(absolute=5e-4, relative=5e-4),
        default_reference=f"{REF}:conv2d_reference",
    ),
    OperatorDescriptor(
        name="maxpool2d",
        category="pooling",
        num_inputs=1,
        dtype_variants=POOL_DTYPES,
        attribute_names=("kernel_size", "stride", "padding"),
        description="2D max pooling.",
        tags=("pool", "spatial"),
        default_reference=f"{REF}:maxpool2d_reference",
    ),
    OperatorDescriptor(
        name="avgpool2d",
        category="pooling",
        num_inputs=1,
        dtype_variants=POOL_DTYPES,
        attribute_names=("kernel_size", "stride", "padding"),
        description="2D average pooling.",
        tags=("pool", "spatial"),
        default_reference=f"{REF}:avgpool2d_reference",
    ),
    OperatorDescriptor(
        name="relu",
        category="activation",
        num_inputs=1,
        dtype_variants=ACTIVATION_DTYPES,
        description="Rectified Linear Unit activation.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:relu_reference",
    ),
    OperatorDescriptor(
        name="leaky_relu",
        category="activation",
        num_inputs=1,
        dtype_variants=ACTIVATION_DTYPES,
        attribute_names=("alpha",),
        description="Leaky ReLU activation with configurable alpha.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:leaky_relu_reference",
    ),
    OperatorDescriptor(
        name="sigmoid",
        category="activation",
        num_inputs=1,
        dtype_variants=UNARY_FLOAT_DTYPES,
        description="Sigmoid activation.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:sigmoid_reference",
    ),
    OperatorDescriptor(
        name="tanh",
        category="activation",
        num_inputs=1,
        dtype_variants=UNARY_FLOAT_DTYPES,
        description="Hyperbolic tangent activation.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:tanh_reference",
    ),
    OperatorDescriptor(
        name="softmax",
        category="activation",
        num_inputs=1,
        dtype_variants=UNARY_FLOAT_DTYPES,
        attribute_names=("axis",),
        description="Softmax activation along the specified axis (default last).",
        tags=("activation", "softmax"),
        default_reference=f"{REF}:softmax_reference",
    ),
    OperatorDescriptor(
        name="sinh",
        category="activation",
        num_inputs=1,
        dtype_variants=UNARY_FLOAT_DTYPES,
        description="Hyperbolic sine activation.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:sinh_reference",
    ),
    OperatorDescriptor(
        name="reduce_sum",
        category="reduction",
        num_inputs=1,
        dtype_variants=REDUCTION_DTYPES,
        attribute_names=("axis", "keepdims"),
        description="Reduction summation over axis or whole tensor.",
        tags=("reduction",),
        default_reference=f"{REF}:reduce_sum_reference",
    ),
    OperatorDescriptor(
        name="reduce_mean",
        category="reduction",
        num_inputs=1,
        dtype_variants=REDUCTION_FLOAT_DTYPES,
        attribute_names=("axis", "keepdims"),
        description="Reduction mean over axis or whole tensor.",
        tags=("reduction",),
        default_reference=f"{REF}:reduce_mean_reference",
    ),
    OperatorDescriptor(
        name="broadcast_to",
        category="tensor",
        num_inputs=1,
        dtype_variants=UNARY_ELEMENTWISE_DTYPES,
        attribute_names=("shape",),
        description="Broadcast tensor to target shape.",
        tags=("broadcast", "tensor"),
        default_reference=f"{REF}:broadcast_to_reference",
    ),
)
