import numpy as np

from optest.generators import GeneratorProtocol, GeneratorOutput
from optest.core import TestCase


class SequentialGenerator(GeneratorProtocol):
    """Generate a single tensor with sequential values for reproducibility."""

    def generate(self, case: TestCase, rng: np.random.Generator) -> GeneratorOutput:
        shape = case.shapes.get("input0")
        if shape is None:
            raise ValueError("input0 shape is required")
        size = int(np.prod(shape))
        data = np.arange(size, dtype=case.dtype_spec[0]).reshape(shape)
        return [data], None


def square_reference(inputs, attrs):
    (x,) = inputs
    return (np.square(x),)
