"""Generator scaffolding for optest."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import numpy as np

from optest.core import TestCase
from optest.utils import import_string

GeneratorOutput = tuple[list[np.ndarray], Optional[list[np.ndarray]]]


class GeneratorProtocol(Protocol):
    """Protocol all generators must follow."""

    def generate(self, case: TestCase, rng: np.random.Generator) -> GeneratorOutput:
        ...


@dataclass
class RandomTensorGenerator:
    """Basic generator that emits random tensors for each input."""

    allow_negative: bool = True

    def generate(self, case: TestCase, rng: np.random.Generator) -> GeneratorOutput:
        inputs: list[np.ndarray] = []
        default_variant: tuple[str, ...]
        if case.descriptor.dtype_variants:
            default_variant = case.descriptor.dtype_variants[0]
        else:
            default_variant = tuple("float32" for _ in range(case.descriptor.num_inputs))
        for index in range(case.descriptor.num_inputs):
            shape_value = case.shapes.get(f"input{index}")
            if shape_value is None:
                raise ValueError(
                    f"Missing shape definition for input{index} in case {case.identifier()}"
                )
            shape = tuple(int(dim) for dim in shape_value)
            if index < len(case.dtype_spec):
                dtype_name = case.dtype_spec[index]
            elif index < len(default_variant):
                dtype_name = default_variant[index]
            else:
                dtype_name = default_variant[0]
            dtype = np.dtype(dtype_name)
            inputs.append(self._generate_tensor(rng, shape, dtype))
        return inputs, None

    def _generate_tensor(self, rng: np.random.Generator, shape: Sequence[int], dtype: np.dtype) -> np.ndarray:
        if dtype.kind in {"i", "u"}:
            low, high = (-128, 127)
            if dtype.itemsize == 2:
                low, high = (-32768, 32767)
            if dtype.itemsize >= 4:
                low, high = (-2 ** 31, 2 ** 31 - 1)
            return rng.integers(low, high, size=shape, dtype=dtype)
        if dtype.kind == "b":
            return rng.integers(0, 2, size=shape, dtype=dtype)
        if dtype.kind == "f":
            data = rng.standard_normal(size=shape)
            if not self.allow_negative:
                data = np.abs(data)
            return data.astype(dtype)
        raise TypeError(f"Unsupported dtype kind '{dtype}' for generator")


def resolve_generator(path: str) -> GeneratorProtocol:
    """Import and instantiate a generator from a dotted path string."""

    obj = import_string(path)
    if isinstance(obj, type):
        instance = obj()
    elif callable(obj) and not hasattr(obj, "generate"):
        instance = obj()
    else:
        instance = obj
    if not hasattr(instance, "generate"):
        raise TypeError(f"Generator '{path}' does not implement 'generate'")
    return instance  # type: ignore[return-value]
