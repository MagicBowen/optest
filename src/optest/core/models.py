"""Core dataclasses shared across optest subsystems."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple


BackendKind = str  # Alias for readability ("gpu" or "npu").


@dataclass(frozen=True)
class Tolerance:
    """Numerical tolerance definition for comparisons."""

    absolute: float = 1e-4
    relative: float = 1e-5

    @classmethod
    def from_mapping(cls, data: Optional[Mapping[str, Any]]) -> "Tolerance":
        if not data:
            return cls()
        return cls(
            absolute=float(data.get("abs", data.get("absolute", 1e-4))),
            relative=float(data.get("rel", data.get("relative", 1e-5))),
        )


@dataclass(frozen=True)
class BackendTarget:
    """Represents the backend selection for a test run."""

    kind: BackendKind
    chip: Optional[str] = None

    def label(self) -> str:
        if self.chip:
            return f"{self.kind}:{self.chip}"
        return self.kind


@dataclass(frozen=True)
class OperatorDescriptor:
    """Metadata describing an operator within the registry."""

    name: str
    category: str
    num_inputs: int
    num_outputs: int = 1
    dtype_variants: Tuple[Tuple[str, ...], ...] = tuple()
    supported_backends: Tuple[BackendKind, ...] = ("gpu", "npu")
    default_tolerance: Tolerance = field(default_factory=Tolerance)
    tags: Tuple[str, ...] = tuple()
    attribute_names: Tuple[str, ...] = tuple()
    description: str = ""
    default_generator: Optional[str] = None  # dotted path string override
    default_reference: Optional[str] = None

    def supports_backend(self, backend: BackendKind) -> bool:
        return backend in self.supported_backends


@dataclass
class TestCase:
    """Concrete description of work to execute for an operator."""

    descriptor: OperatorDescriptor
    dtype_spec: Tuple[str, ...]
    shapes: Mapping[str, Any]
    backend: BackendTarget
    tolerance: Tolerance
    attributes: Mapping[str, Any] = field(default_factory=dict)
    generator_override: Optional[str] = None
    reference_override: Optional[str] = None

    def identifier(self) -> str:
        dtype = ",".join(self.dtype_spec)
        shape_str = ";".join(f"{k}={v}" for k, v in sorted(self.shapes.items()))
        return f"{self.descriptor.name}[{dtype}]({shape_str})@{self.backend.label()}"


@dataclass(frozen=True)
class GeneratorSpec:
    """Declarative metadata about a generator implementation."""

    dotted_path: str
    description: str = ""
