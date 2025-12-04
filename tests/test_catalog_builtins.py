from optest.operators import catalog


def test_builtin_catalog_contains_expected_names() -> None:
    catalog.load_builtins()
    names = set(catalog.names())
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


def test_builtin_descriptors_match_input_counts() -> None:
    catalog.load_builtins()
    for desc in catalog.descriptors():
        for variant in desc.dtype_variants:
            assert len(variant) == desc.num_inputs
