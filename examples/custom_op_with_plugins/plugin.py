from optest.core import OperatorDescriptor
from optest.registry import registry


def register() -> None:
    descriptor = OperatorDescriptor(
        name="custom_square",
        category="math",
        num_inputs=1,
        dtype_variants=(("float32",), ("float16",), ("int32",)),
        description="Custom square operator registered via plugin",
        tags=("custom", "example"),
        default_reference="custom_plugins:square_reference",
        default_generator="custom_plugins:SequentialGenerator",
    )
    registry.register(descriptor)
