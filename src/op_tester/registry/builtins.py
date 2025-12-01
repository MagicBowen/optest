"""Built-in operator descriptors shipped with op-tester."""
from __future__ import annotations

from op_tester.core import OperatorDescriptor, Tolerance

REF = "op_tester.operators.reference_impls"

COMMON_FLOAT_DTYPES = (
    ("float32",),
    ("float16",),
)

ELEMENTWISE_INT_DTYPES = (
    ("int8",),
    ("int16",),
    ("int32",),
)

GEMM_DTYPES = (
    ("float32", "float32"),
    ("float16", "float16"),
    ("int8", "int8"),
)

CONV_DTYPES = (
    ("float32", "float32"),
    ("float16", "float16"),
)

POOL_DTYPES = COMMON_FLOAT_DTYPES
ACTIVATION_DTYPES = COMMON_FLOAT_DTYPES + ELEMENTWISE_INT_DTYPES

BUILTIN_DESCRIPTORS = (
    OperatorDescriptor(
        name="elementwise_add",
        category="tensor",
        num_inputs=2,
        dtype_variants=COMMON_FLOAT_DTYPES + ELEMENTWISE_INT_DTYPES,
        description="Elementwise tensor addition for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_add_reference",
    ),
    OperatorDescriptor(
        name="elementwise_mul",
        category="tensor",
        num_inputs=2,
        dtype_variants=COMMON_FLOAT_DTYPES + ELEMENTWISE_INT_DTYPES,
        description="Elementwise tensor multiplication for two inputs.",
        tags=("elementwise", "tensor"),
        default_reference=f"{REF}:elementwise_mul_reference",
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
        name="sigmoid",
        category="activation",
        num_inputs=1,
        dtype_variants=COMMON_FLOAT_DTYPES,
        description="Sigmoid activation.",
        tags=("activation", "elementwise"),
        default_reference=f"{REF}:sigmoid_reference",
    ),
)
