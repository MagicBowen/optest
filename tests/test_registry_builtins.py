from optest.registry import registry
from optest.registry.builtins import BUILTIN_DESCRIPTORS


def test_builtin_dtype_variants_match_input_count() -> None:
    for descriptor in BUILTIN_DESCRIPTORS:
        for variant in descriptor.dtype_variants:
            assert (
                len(variant) == descriptor.num_inputs
            ), f"{descriptor.name} dtype variant {variant} does not match num_inputs={descriptor.num_inputs}"


def test_registry_includes_extended_operators() -> None:
    names = set(registry.names())
    expected = {
        "softmax",
        "matmul",
        "vector_dot",
        "vector_norm",
        "vector_sum",
        "sinh",
        "reduce_sum",
        "reduce_mean",
        "broadcast_to",
        "leaky_relu",
        "equal",
        "greater",
        "less",
        "less_equal",
        "greater_equal",
    }
    assert expected.issubset(names)
